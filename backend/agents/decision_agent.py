"""
Agent 4 — Decision Agent (Rule-Based + RL Hybrid)

Supports two modes:
  1. RULE-BASED (default) — deterministic rules mapping risk levels to actions
  2. RL (DQN)            — trained Deep Q-Network makes decisions

Toggle via config: USE_RL_DECISION=true
"""

import numpy as np
from loguru import logger
from backend.config import settings


# ── Rule-based decision rules ────────────────────────────────────────────────
_DECISION_RULES = {
    "HIGH": {
        "action": "REPLACE_SUPPLIER",
        "icon": "🔴",
        "recommendation": (
            "CRITICAL: Immediately identify and onboard alternative suppliers. "
            "Activate contingency inventory reserves. Notify downstream partners "
            "of potential delivery delays. Escalate to supply chain leadership."
        ),
    },
    "MEDIUM": {
        "action": "INCREASE_MONITORING",
        "icon": "🟡",
        "recommendation": (
            "WARNING: Increase monitoring frequency to daily updates. "
            "Begin evaluating backup supplier options. Review safety stock levels "
            "and assess buffer inventory adequacy. Prepare contingency plans."
        ),
    },
    "LOW": {
        "action": "NO_ACTION",
        "icon": "🟢",
        "recommendation": (
            "STABLE: No immediate action required. Continue standard monitoring. "
            "Maintain awareness of the disruption and watch for escalation signals."
        ),
    },
}

# ── RL action → recommendation mapping ──────────────────────────────────────
_RL_ACTION_MAP = {
    "NO_ACTION": {
        "icon": "🟢",
        "recommendation": (
            "RL DECISION — STABLE: The trained agent determined no immediate action "
            "is needed. Continue standard monitoring protocols."
        ),
    },
    "INCREASE_MONITORING": {
        "icon": "🟡",
        "recommendation": (
            "RL DECISION — MONITOR: The trained agent recommends increased monitoring "
            "frequency. Set up daily alerts and track disruption developments closely."
        ),
    },
    "DIVERSIFY_SUPPLY": {
        "icon": "🟠",
        "recommendation": (
            "RL DECISION — DIVERSIFY: The trained agent recommends diversifying the "
            "supply base. Identify 2-3 secondary suppliers to reduce single-source risk. "
            "Begin preliminary qualification processes."
        ),
    },
    "REPLACE_SUPPLIER": {
        "icon": "🔴",
        "recommendation": (
            "RL DECISION — REPLACE: The trained agent determined immediate supplier "
            "replacement is optimal. Activate emergency procurement protocols and "
            "begin onboarding alternative suppliers."
        ),
    },
}


class DecisionAgent:
    """
    Generates mitigation decisions from risk assessments.

    Supports:
      - Rule-based mode (deterministic, always available)
      - RL mode (DQN-based, requires trained model)
    """

    def __init__(self):
        self.use_rl = getattr(settings, "use_rl_decision", False)
        self.rl_agent = None

        if self.use_rl:
            self._init_rl()

        mode = "RL (DQN)" if (self.use_rl and self.rl_agent) else "Rule-Based"
        logger.info("🟢 Decision Agent initialized [mode: {}]", mode)

    def _init_rl(self):
        """Load the trained RL model, or auto-train if not found."""
        try:
            from backend.agents.rl.dqn_agent import DQNAgent
            self.rl_agent = DQNAgent()
            loaded = self.rl_agent.load()
            if not loaded:
                logger.info("🧠 No trained RL model found — auto-training DQN (5000 episodes)...")
                self._auto_train()
        except Exception as e:
            logger.error("Failed to initialize RL agent: {} — using rule-based", e)
            self.rl_agent = None
            self.use_rl = False

    def _auto_train(self):
        """Auto-train the DQN agent on simulated disruption scenarios."""
        try:
            from backend.agents.rl.environment import SupplyChainEnv
            from backend.agents.rl.dqn_agent import DQNAgent

            env = SupplyChainEnv(seed=42)
            agent = DQNAgent(
                lr=0.001, gamma=0.95,
                epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.998,
                buffer_size=10000, batch_size=64, target_sync=50,
            )

            num_episodes = 5000
            for episode in range(1, num_episodes + 1):
                state = env.reset()
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step()
                agent.total_episodes = episode

                if episode % 1000 == 0:
                    logger.info(
                        "   Training progress: {}/{} episodes (ε={:.4f})",
                        episode, num_episodes, agent.epsilon,
                    )

            agent.save()
            logger.info("✅ RL model auto-trained and saved ({} episodes)", num_episodes)

            # Load the freshly trained model into self
            self.rl_agent = DQNAgent()
            self.rl_agent.load()

        except Exception as e:
            logger.error("Auto-training failed: {} — falling back to rule-based", e)
            self.rl_agent = None
            self.use_rl = False

    def run(self, risk_assessments: list[dict], disruption_data: dict = None) -> list[dict]:
        """
        Generate recommendations for each risk-assessed company.

        Parameters
        ----------
        risk_assessments : list[dict]
            Output from RiskAssessmentAgent.
        disruption_data : dict, optional
            Output from DisruptionAgent (needed for RL mode)

        Returns
        -------
        list[dict] with keys:
            company, risk_level, risk_score, action, recommendation, icon,
            decision_mode, rl_q_values (if RL mode)
        """
        if self.use_rl and self.rl_agent and disruption_data:
            return self._run_rl(risk_assessments, disruption_data)
        return self._run_rules(risk_assessments)

    def _run_rules(self, risk_assessments: list[dict]) -> list[dict]:
        """Rule-based decisions."""
        logger.info("Agent 4 ▶ Generating rule-based decisions for {} companies", len(risk_assessments))

        decisions = []
        for assessment in risk_assessments:
            level = assessment.get("risk_level", "LOW")
            rule = _DECISION_RULES.get(level, _DECISION_RULES["LOW"])

            decisions.append({
                "company": assessment["company"],
                "country": assessment.get("country", "Unknown"),
                "industry": assessment.get("industry", "Unknown"),
                "risk_score": assessment["risk_score"],
                "risk_level": level,
                "action": rule["action"],
                "recommendation": rule["recommendation"],
                "icon": rule["icon"],
                "depth": assessment.get("depth", 0),
                "path": assessment.get("path", []),
                "decision_mode": "rule-based",
            })

        high_count = sum(1 for d in decisions if d["risk_level"] == "HIGH")
        logger.info(
            "Agent 4 ✅ {} rule-based decisions ({} require immediate action)",
            len(decisions), high_count,
        )
        return decisions

    def _run_rl(self, risk_assessments: list[dict], disruption_data: dict) -> list[dict]:
        """RL-based decisions using the trained DQN."""
        from backend.agents.rl.environment import SupplyChainEnv, ACTIONS

        logger.info("Agent 4 ▶ Generating RL (DQN) decisions for {} companies", len(risk_assessments))

        # Convert analysis data to RL state vector
        state = SupplyChainEnv.state_from_risk_assessments(disruption_data, risk_assessments)

        # Get Q-values and best action from the trained network
        q_values = self.rl_agent.get_q_values(state)
        best_action_idx = int(np.argmax(q_values))
        best_action_name = ACTIONS[best_action_idx]

        # Get action info
        action_info = _RL_ACTION_MAP.get(best_action_name, _RL_ACTION_MAP["NO_ACTION"])

        logger.info(
            "Agent 4 🧠 RL decided: {} (Q-values: {})",
            best_action_name, {ACTIONS[i]: round(float(q_values[i]), 3) for i in range(len(ACTIONS))},
        )

        # Apply the RL decision to each company, but blend with individual risk levels
        decisions = []
        for assessment in risk_assessments:
            level = assessment.get("risk_level", "LOW")

            # For individual companies, the global RL decision is the baseline,
            # but very high-risk companies might escalate
            individual_action = best_action_name
            individual_info = action_info

            if level == "HIGH" and best_action_idx < 3:
                # RL says moderate action, but this company is HIGH risk → escalate
                individual_action = "REPLACE_SUPPLIER"
                individual_info = _RL_ACTION_MAP["REPLACE_SUPPLIER"]
            elif level == "LOW" and best_action_idx > 1:
                # RL says aggressive action, but this company is LOW risk → de-escalate
                individual_action = "INCREASE_MONITORING"
                individual_info = _RL_ACTION_MAP["INCREASE_MONITORING"]

            decisions.append({
                "company": assessment["company"],
                "country": assessment.get("country", "Unknown"),
                "industry": assessment.get("industry", "Unknown"),
                "risk_score": assessment["risk_score"],
                "risk_level": level,
                "action": individual_action,
                "recommendation": individual_info["recommendation"],
                "icon": individual_info["icon"],
                "depth": assessment.get("depth", 0),
                "path": assessment.get("path", []),
                "decision_mode": "rl-dqn",
                "rl_q_values": {
                    ACTIONS[i]: round(float(q_values[i]), 3)
                    for i in range(len(ACTIONS))
                },
                "rl_global_action": best_action_name,
            })

        replace_count = sum(1 for d in decisions if d["action"] == "REPLACE_SUPPLIER")
        logger.info(
            "Agent 4 ✅ {} RL decisions ({} replace, global: {})",
            len(decisions), replace_count, best_action_name,
        )
        return decisions
