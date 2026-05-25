"""
Supply Chain Disruption Environment — Gym-style RL environment.

Simulates disruption scenarios and evaluates the agent's
mitigation decisions based on cost-effectiveness and risk reduction.
"""

import random
import numpy as np
from typing import Optional


# ── Action space ─────────────────────────────────────────────────────────────
ACTIONS = {
    0: "NO_ACTION",
    1: "INCREASE_MONITORING",
    2: "DIVERSIFY_SUPPLY",
    3: "REPLACE_SUPPLIER",
}
NUM_ACTIONS = len(ACTIONS)

# Action costs (normalized 0-1): more aggressive = more expensive
ACTION_COSTS = {
    0: 0.0,    # No action — free
    1: 0.15,   # Monitoring — cheap
    2: 0.45,   # Diversify — moderate
    3: 0.80,   # Replace — expensive
}

# ── Disruption types ─────────────────────────────────────────────────────────
DISRUPTION_TYPES = [
    "natural_disaster", "geopolitical", "labor", "pandemic",
    "operational", "financial", "logistics", "supply", "cyber_attack",
]

# How quickly each disruption type resolves (lower = faster)
DISRUPTION_PERSISTENCE = {
    "natural_disaster": 0.7,
    "geopolitical": 0.9,
    "labor": 0.4,
    "pandemic": 0.85,
    "operational": 0.5,
    "financial": 0.6,
    "logistics": 0.3,
    "supply": 0.5,
    "cyber_attack": 0.4,
}

# ── State vector dimensions ──────────────────────────────────────────────────
STATE_DIM = 8
"""
State vector:
  [0] severity            — 0.0 (low), 0.5 (medium), 1.0 (high)
  [1] num_affected_norm   — number of affected companies / 50
  [2] avg_risk_score      — average risk score (0–1)
  [3] max_risk_score      — maximum risk score (0–1)
  [4] avg_depth_norm      — average disruption depth / 4
  [5] high_risk_ratio     — fraction of companies at HIGH risk
  [6] criticality         — supply chain criticality (0–1)
  [7] persistence         — disruption persistence factor (0–1)
"""


class SupplyChainEnv:
    """
    Gym-style environment simulating supply chain disruption scenarios.

    Each episode:
      1. A random disruption is generated
      2. The agent observes the state
      3. The agent picks an action (mitigation strategy)
      4. The environment computes a reward based on risk reduction vs cost
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.state = None
        self.disruption_type = None
        self.episode_count = 0

    def reset(self) -> np.ndarray:
        """Generate a new random disruption scenario."""
        self.episode_count += 1

        # Random disruption parameters
        self.disruption_type = self.rng.choice(DISRUPTION_TYPES)
        severity_val = self.rng.choice([0.0, 0.5, 1.0], )  # low/med/high
        if self.disruption_type in ("natural_disaster", "geopolitical", "pandemic"):
            severity_val = self.rng.choice([0.5, 1.0])  # Tend to be worse

        num_affected = self.rng.randint(1, 40)
        num_affected_norm = min(1.0, num_affected / 50.0)

        # Risk scores depend on severity
        base_risk = 0.2 + severity_val * 0.5 + self.np_rng.uniform(-0.1, 0.1)
        avg_risk = np.clip(base_risk + self.np_rng.uniform(-0.15, 0.15), 0, 1)
        max_risk = np.clip(avg_risk + self.np_rng.uniform(0.05, 0.3), 0, 1)

        avg_depth = self.np_rng.uniform(1, 4) / 4.0
        high_risk_ratio = np.clip(severity_val * 0.6 + self.np_rng.uniform(-0.1, 0.2), 0, 1)

        # Criticality: how important the affected companies are
        criticality = np.clip(
            0.3 + severity_val * 0.4 + num_affected_norm * 0.3 + self.np_rng.uniform(-0.1, 0.1),
            0, 1,
        )

        persistence = DISRUPTION_PERSISTENCE.get(self.disruption_type, 0.5)

        self.state = np.array([
            severity_val,
            num_affected_norm,
            avg_risk,
            max_risk,
            avg_depth,
            high_risk_ratio,
            criticality,
            persistence,
        ], dtype=np.float32)

        return self.state.copy()

    def step(self, action: int) -> tuple:
        """
        Execute the agent's action and compute reward.

        Returns: (next_state, reward, done, info)
        """
        assert 0 <= action < NUM_ACTIONS, f"Invalid action: {action}"
        assert self.state is not None, "Call reset() before step()"

        severity = self.state[0]
        avg_risk = self.state[2]
        max_risk = self.state[3]
        high_ratio = self.state[5]
        criticality = self.state[6]
        persistence = self.state[7]

        # ── Compute reward ───────────────────────────────────────────────
        # 1. Risk-action alignment reward
        #    Ideal mapping: HIGH risk → REPLACE, MEDIUM → DIVERSIFY/MONITOR, LOW → NO_ACTION
        alignment_reward = self._compute_alignment_reward(action, severity, max_risk, high_ratio)

        # 2. Cost penalty — aggressive actions cost more
        cost_penalty = -ACTION_COSTS[action]

        # 3. Risk reduction effectiveness
        risk_reduction = self._compute_risk_reduction(action, avg_risk, persistence)

        # 4. Criticality bonus — correct action on critical chains worth more
        criticality_bonus = alignment_reward * criticality * 0.5

        # Total reward
        reward = (
            alignment_reward * 3.0     # Primary signal
            + cost_penalty * 1.5       # Cost awareness
            + risk_reduction * 2.0     # Effectiveness
            + criticality_bonus        # Importance weighting
        )

        # Episode is single-step (one decision per disruption)
        done = True
        info = {
            "action_name": ACTIONS[action],
            "alignment": alignment_reward,
            "cost": cost_penalty,
            "risk_reduction": risk_reduction,
            "criticality_bonus": criticality_bonus,
        }

        return self.state.copy(), reward, done, info

    def _compute_alignment_reward(
        self, action: int, severity: float, max_risk: float, high_ratio: float
    ) -> float:
        """
        Reward based on how well the action matches the situation severity.

        Optimal policy:
            severity=1.0 (high)   → action=3 (replace)      +1.0
            severity=0.5 (medium) → action=1 or 2            +1.0
            severity=0.0 (low)    → action=0 (no action)     +1.0
        """
        # Ideal action mapping
        if severity >= 0.8:  # HIGH severity
            ideal_actions = {3: 1.0, 2: 0.5, 1: -0.2, 0: -1.0}
        elif severity >= 0.4:  # MEDIUM severity
            ideal_actions = {2: 1.0, 1: 0.8, 3: 0.0, 0: -0.5}
        else:  # LOW severity
            ideal_actions = {0: 1.0, 1: 0.5, 2: -0.3, 3: -0.8}

        base = ideal_actions.get(action, 0.0)

        # Adjust for high-risk ratio — even in medium severity,
        # if lots of companies are at high risk, aggressive action is better
        if high_ratio > 0.5 and action >= 2:
            base += 0.3

        # Adjust for max risk — if any single company has extreme risk
        if max_risk > 0.85 and action == 3:
            base += 0.2

        return np.clip(base, -1.0, 1.0)

    def _compute_risk_reduction(self, action: int, avg_risk: float, persistence: float) -> float:
        """
        Simulate how much the action reduces the disruption risk.
        More aggressive actions reduce persistent disruptions better.
        """
        # Action effectiveness against disruption persistence
        effectiveness = {
            0: 0.0,                             # No action
            1: 0.2 * (1 - persistence * 0.5),   # Monitoring helps a bit
            2: 0.5 * (1 - persistence * 0.3),   # Diversification is solid
            3: 0.8 * (1 - persistence * 0.1),   # Replacement is most effective
        }

        reduction = effectiveness[action] * avg_risk
        return np.clip(reduction, 0, 1)

    @staticmethod
    def state_from_risk_assessments(disruption_data: dict, risk_assessments: list) -> np.ndarray:
        """
        Convert real analysis data into the RL state vector.
        Used by the Decision Agent to query the trained RL model.
        """
        severity_map = {"high": 1.0, "medium": 0.5, "low": 0.0}
        severity = severity_map.get(disruption_data.get("severity", "medium"), 0.5)

        if not risk_assessments:
            return np.array([severity, 0, 0, 0, 0, 0, 0, 0.5], dtype=np.float32)

        num_affected = len(risk_assessments)
        scores = [r.get("risk_score", 0) for r in risk_assessments]
        depths = [r.get("depth", 1) for r in risk_assessments]
        levels = [r.get("risk_level", "LOW") for r in risk_assessments]

        avg_risk = float(np.mean(scores)) if scores else 0.0
        max_risk = float(np.max(scores)) if scores else 0.0
        avg_depth = float(np.mean(depths)) / 4.0 if depths else 0.0
        high_ratio = sum(1 for l in levels if l == "HIGH") / max(len(levels), 1)

        # Criticality estimate
        criticality = min(1.0, avg_risk * 0.5 + (num_affected / 50.0) * 0.3 + severity * 0.2)

        # Persistence (default medium)
        dtype = disruption_data.get("disruption_type", "unknown")
        persistence = DISRUPTION_PERSISTENCE.get(dtype, 0.5)

        state = np.array([
            severity,
            min(1.0, num_affected / 50.0),
            avg_risk,
            max_risk,
            avg_depth,
            high_ratio,
            criticality,
            persistence,
        ], dtype=np.float32)

        return state
