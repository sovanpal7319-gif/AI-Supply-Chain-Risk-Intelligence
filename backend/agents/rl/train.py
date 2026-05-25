"""
RL Training Script — Trains the DQN agent on simulated disruption scenarios.

Usage:
    python -m backend.agents.rl.train [--episodes 5000] [--save-path data/rl_model.json]

The script:
  1. Creates a supply chain disruption environment
  2. Trains a DQN agent over N episodes
  3. Logs reward progression and learned policy
  4. Saves the trained model to JSON
"""

import sys
import os
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.agents.rl.environment import SupplyChainEnv, ACTIONS, STATE_DIM
from backend.agents.rl.dqn_agent import DQNAgent


def print_banner():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   🧠 RL Training — Supply Chain Decision Agent (DQN)    ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║   State dim:   {STATE_DIM:<6}  Actions:  {len(ACTIONS):<6}              ║")
    print(f"║   Network:     8 → 64 → 32 → 4                        ║")
    print(f"║   Algorithm:   DQN + Experience Replay + Target Net    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def evaluate_policy(agent: DQNAgent, env: SupplyChainEnv, n_episodes: int = 200) -> dict:
    """Evaluate the current policy without exploration."""
    rewards = []
    action_counts = {i: 0 for i in range(len(ACTIONS))}
    correct_decisions = 0

    for _ in range(n_episodes):
        state = env.reset()
        action = agent.select_action(state, training=False)
        _, reward, _, info = env.step(action)

        rewards.append(reward)
        action_counts[action] += 1

        # Check if decision aligns with severity
        severity = state[0]
        if severity >= 0.8 and action == 3:
            correct_decisions += 1
        elif 0.3 <= severity < 0.8 and action in (1, 2):
            correct_decisions += 1
        elif severity < 0.3 and action == 0:
            correct_decisions += 1

    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "accuracy": correct_decisions / n_episodes,
        "action_distribution": {ACTIONS[k]: v for k, v in action_counts.items()},
    }


def train(
    num_episodes: int = 5000,
    save_path: str = None,
    seed: int = 42,
    eval_interval: int = 500,
):
    """Train the DQN agent."""

    print_banner()

    env = SupplyChainEnv(seed=seed)
    agent = DQNAgent(
        lr=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.998,
        buffer_size=10000,
        batch_size=64,
        target_sync=50,
    )

    print(f"🏋️  Training for {num_episodes} episodes...")
    print(f"{'─' * 60}")

    window_size = 100
    reward_window = []
    best_avg_reward = -float("inf")

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        action = agent.select_action(state, training=True)
        next_state, reward, done, info = env.step(action)

        agent.store_transition(state, action, reward, next_state, done)
        loss = agent.train_step()

        reward_window.append(reward)
        if len(reward_window) > window_size:
            reward_window.pop(0)
        agent.reward_history.append(reward)
        agent.total_episodes = episode

        # Progress logging
        if episode % eval_interval == 0 or episode == num_episodes:
            avg_reward = np.mean(reward_window)
            eval_result = evaluate_policy(agent, SupplyChainEnv(seed=seed + episode))

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                marker = " ⭐ best"
            else:
                marker = ""

            print(
                f"  Episode {episode:>5d}/{num_episodes} │ "
                f"Avg Reward: {avg_reward:>7.3f} │ "
                f"Accuracy: {eval_result['accuracy']:>5.1%} │ "
                f"ε: {agent.epsilon:.4f} │ "
                f"Loss: {loss:.5f}"
                f"{marker}"
            )

    print(f"{'─' * 60}")
    print()

    # Final evaluation
    print("📊 Final Policy Evaluation (1000 episodes):")
    eval_env = SupplyChainEnv(seed=999)
    final_eval = evaluate_policy(agent, eval_env, n_episodes=1000)

    print(f"   Average Reward:  {final_eval['avg_reward']:.3f} ± {final_eval['std_reward']:.3f}")
    print(f"   Decision Accuracy: {final_eval['accuracy']:.1%}")
    print(f"   Action Distribution:")
    for action_name, count in final_eval['action_distribution'].items():
        bar = "█" * int(count / 1000 * 40)
        print(f"     {action_name:<22s} {count:>4d} ({count/10:.1f}%) {bar}")
    print()

    # Demonstrate learned policy on specific scenarios
    print("🔍 Learned Policy Examples:")
    test_scenarios = [
        ("HIGH severity, many affected", [1.0, 0.6, 0.8, 0.95, 0.75, 0.7, 0.85, 0.7]),
        ("MEDIUM severity, moderate",    [0.5, 0.3, 0.5, 0.65, 0.50, 0.3, 0.50, 0.5]),
        ("LOW severity, few affected",   [0.0, 0.1, 0.2, 0.30, 0.25, 0.1, 0.15, 0.3]),
        ("HIGH severity, cyber attack",  [1.0, 0.4, 0.7, 0.90, 0.50, 0.6, 0.75, 0.4]),
        ("MEDIUM severity, persistent",  [0.5, 0.4, 0.6, 0.75, 0.60, 0.4, 0.60, 0.9]),
    ]

    for desc, state_vals in test_scenarios:
        state = np.array(state_vals, dtype=np.float32)
        q_values = agent.get_q_values(state)
        action = int(np.argmax(q_values))
        print(f"   {desc:<35s} → {ACTIONS[action]:<22s} (Q: {q_values[action]:.3f})")
    print()

    # Save model
    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "data", "rl_model.json",
        )

    agent.save(save_path)
    print(f"✅ Model saved to: {save_path}")
    print(f"   Total training steps: {agent.train_steps}")
    print(f"   Final epsilon: {agent.epsilon:.4f}")

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RL Decision Agent")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save the trained model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        save_path=args.save_path,
        seed=args.seed,
    )
