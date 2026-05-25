"""
DQN Agent — Deep Q-Network implemented with NumPy only.

A lightweight DQN with:
  - 2-hidden-layer neural network (no PyTorch/TensorFlow needed)
  - Experience replay buffer
  - Epsilon-greedy exploration
  - Target network for stable training
"""

import json
import os
from pathlib import Path
from collections import deque

import numpy as np
from loguru import logger

from .environment import STATE_DIM, NUM_ACTIONS


# ── Neural Network (NumPy only) ─────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_deriv(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


class NeuralNetwork:
    """Simple 2-hidden-layer feedforward network in pure NumPy."""

    def __init__(self, input_dim: int, hidden1: int, hidden2: int, output_dim: int, lr: float = 0.001):
        self.lr = lr

        # He initialization
        self.W1 = np.random.randn(input_dim, hidden1).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.W2 = np.random.randn(hidden1, hidden2).astype(np.float32) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2, dtype=np.float32)
        self.W3 = np.random.randn(hidden2, output_dim).astype(np.float32) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(output_dim, dtype=np.float32)

        # Cache for backward pass
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x shape: (batch, input_dim) or (input_dim,)."""
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3  # linear output (Q-values)

        self._cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3}

        return z3[0] if single else z3

    def backward(self, grad_output: np.ndarray):
        """Backward pass and parameter update (SGD with gradient clipping)."""
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)

        c = self._cache
        batch_size = grad_output.shape[0]

        # Layer 3
        dW3 = (c["a2"].T @ grad_output) / batch_size
        db3 = grad_output.mean(axis=0)
        d_a2 = grad_output @ self.W3.T

        # Layer 2
        d_z2 = d_a2 * relu_deriv(c["z2"])
        dW2 = (c["a1"].T @ d_z2) / batch_size
        db2 = d_z2.mean(axis=0)
        d_a1 = d_z2 @ self.W2.T

        # Layer 1
        d_z1 = d_a1 * relu_deriv(c["z1"])
        dW1 = (c["x"].T @ d_z1) / batch_size
        db1 = d_z1.mean(axis=0)

        # Gradient clipping
        max_norm = 1.0
        for grad in [dW1, db1, dW2, db2, dW3, db3]:
            norm = np.linalg.norm(grad)
            if norm > max_norm:
                grad *= max_norm / norm

        # SGD update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

    def copy_from(self, other: "NeuralNetwork"):
        """Copy weights from another network (for target network sync)."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()

    def get_weights(self) -> dict:
        """Serialize weights to a JSON-compatible dict."""
        return {
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
            "W3": self.W3.tolist(), "b3": self.b3.tolist(),
        }

    def set_weights(self, weights: dict):
        """Load weights from a dict."""
        self.W1 = np.array(weights["W1"], dtype=np.float32)
        self.b1 = np.array(weights["b1"], dtype=np.float32)
        self.W2 = np.array(weights["W2"], dtype=np.float32)
        self.b2 = np.array(weights["b2"], dtype=np.float32)
        self.W3 = np.array(weights["W3"], dtype=np.float32)
        self.b3 = np.array(weights["b3"], dtype=np.float32)


# ── Experience Replay Buffer ────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size ring buffer storing (state, action, reward, next_state, done)."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            action,
            reward,
            np.array(next_state, dtype=np.float32),
            done,
        ))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ────────────────────────────────────────────────────────────────

# Default model path
MODEL_DIR = Path(__file__).parent.parent.parent.parent / "data"
DEFAULT_MODEL_PATH = MODEL_DIR / "rl_model.json"


class DQNAgent:
    """
    Deep Q-Network agent for supply chain disruption decision-making.

    Architecture:
        Input (8) → Dense(64, ReLU) → Dense(32, ReLU) → Output (4)

    Training uses:
        - Experience replay (buffer size 10k)
        - Target network (synced every 100 steps)
        - Epsilon-greedy exploration (1.0 → 0.05)
        - Gradient clipping (max norm 1.0)
    """

    def __init__(
        self,
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.997,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_sync: int = 100,
    ):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_sync = target_sync

        # Networks
        self.q_network = NeuralNetwork(STATE_DIM, 64, 32, NUM_ACTIONS, lr=lr)
        self.target_network = NeuralNetwork(STATE_DIM, 64, 32, NUM_ACTIONS, lr=lr)
        self.target_network.copy_from(self.q_network)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Counters
        self.train_steps = 0
        self.total_episodes = 0

        # Training history
        self.reward_history = []

    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        During inference (training=False), always pick the best Q-value.
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)

        q_values = self.q_network.forward(state)
        return int(np.argmax(q_values))

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for all actions."""
        return self.q_network.forward(state)

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """
        Sample a batch from replay buffer and perform one gradient step.
        Returns the mean loss.
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Current Q-values
        q_current = self.q_network.forward(states)  # (batch, 4)

        # Target Q-values (from target network)
        q_next = self.target_network.forward(next_states)  # (batch, 4)
        q_target_values = rewards + self.gamma * np.max(q_next, axis=1) * (1 - dones)

        # Compute gradient: only for the selected actions
        grad = np.zeros_like(q_current)
        for i in range(self.batch_size):
            a = actions[i]
            td_error = q_current[i, a] - q_target_values[i]
            grad[i, a] = td_error  # MSE gradient

        # Backward pass
        self.q_network.backward(grad)

        # Sync target network periodically
        self.train_steps += 1
        if self.train_steps % self.target_sync == 0:
            self.target_network.copy_from(self.q_network)

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Return mean loss
        loss = float(np.mean(grad ** 2))
        return loss

    def save(self, path: str = None):
        """Save model weights and training metadata to JSON."""
        path = path or str(DEFAULT_MODEL_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "weights": self.q_network.get_weights(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
            "total_episodes": self.total_episodes,
            "reward_history_last100": self.reward_history[-100:] if self.reward_history else [],
            "metadata": {
                "state_dim": STATE_DIM,
                "num_actions": NUM_ACTIONS,
                "architecture": "8 → 64 → 32 → 4",
                "gamma": self.gamma,
            },
        }

        with open(path, "w") as f:
            json.dump(data, f)

        logger.info("💾 RL model saved to {} ({} episodes)", path, self.total_episodes)

    def load(self, path: str = None) -> bool:
        """Load model weights from JSON. Returns True if successful."""
        path = path or str(DEFAULT_MODEL_PATH)

        if not os.path.exists(path):
            logger.warning("⚠️  RL model not found at {}", path)
            return False

        try:
            with open(path, "r") as f:
                data = json.load(f)

            self.q_network.set_weights(data["weights"])
            self.target_network.copy_from(self.q_network)
            self.epsilon = data.get("epsilon", self.epsilon_end)
            self.train_steps = data.get("train_steps", 0)
            self.total_episodes = data.get("total_episodes", 0)

            logger.info(
                "✅ RL model loaded from {} ({} episodes, ε={})",
                path, self.total_episodes, round(self.epsilon, 4),
            )
            return True
        except Exception as e:
            logger.error("Failed to load RL model: {}", e)
            return False
