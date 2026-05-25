"""
Agent 3 Upgrade — GraphSAGE Hybrid Risk Assessment Agent

Blends GraphSAGE-predicted risk scores with existing rule-based scores
for a more accurate risk assessment.

Hybrid formula:
  final_risk = BLEND_WEIGHT × graphsage_score + (1 - BLEND_WEIGHT) × rule_based_score

Fallback:
  If GraphSAGE model is unavailable → 100% rule-based (existing Agent 3).

Output is backward-compatible with Agent 4 (Decision) and Agent 5 (Alt Suppliers).
"""

from loguru import logger

from backend.config import settings
from backend.agents.risk_agent import RiskAssessmentAgent


class GraphSAGERiskAgent:
    """
    Hybrid risk agent: GraphSAGE predictions + rule-based scoring.

    Composes (not replaces) the existing RiskAssessmentAgent.
    """

    def __init__(self):
        self.rule_agent = RiskAssessmentAgent()
        self.inference = None
        self.blend_weight = getattr(settings, "graphsage_blend_weight", 0.70)
        self.use_graphsage = getattr(settings, "use_graphsage", True)

        if self.use_graphsage:
            self._init_inference()

        mode = "hybrid" if (self.inference and self.inference.is_available) else "rule-based (fallback)"
        logger.info(
            "🟡 GraphSAGE Risk Agent initialized [mode: {}, blend: {:.0%}/{:.0%}]",
            mode, self.blend_weight, 1 - self.blend_weight,
        )

    def _init_inference(self):
        """Load GraphSAGE inference service. Fail gracefully."""
        try:
            from backend.services.graphsage_inference_service import GraphSAGEInferenceService
            self.inference = GraphSAGEInferenceService()
            if not self.inference.is_available:
                logger.warning("GraphSAGE model not loaded — will use rule-based only")
        except Exception as exc:
            logger.warning("GraphSAGE init failed: {} — using rule-based", exc)
            self.inference = None

    def run(
        self,
        disruption_data: dict,
        supply_chain_data: dict,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Compute hybrid risk scores.

        Returns
        -------
        tuple of 3 lists:
            hybrid_risks     : list[dict] — final blended scores (used by Agent 4/5)
            rule_based_risks : list[dict] — pure rule-based scores
            graphsage_risks  : list[dict] — pure GraphSAGE scores (empty if unavailable)
        """
        # ── Step 1: Run existing rule-based Agent 3 ──────────────────────
        rule_based = self.rule_agent.run(disruption_data, supply_chain_data)

        # ── Step 2: Try GraphSAGE predictions ────────────────────────────
        gs_predictions = {}
        gs_embeddings = {}
        gs_available = False

        if self.inference and self.inference.is_available:
            companies = disruption_data.get("affected_companies", [])
            disrupted = companies[0] if companies else None
            severity = disruption_data.get("severity", "high")

            if disrupted:
                result = self.inference.predict(
                    disrupted_company=disrupted,
                    severity=severity,
                )
                gs_predictions = result.get("predictions", {})
                gs_embeddings = result.get("embeddings", {})
                gs_available = result.get("available", False)

        # ── Step 3: Build hybrid scores ──────────────────────────────────
        if gs_available and gs_predictions:
            hybrid, graphsage_list = self._blend_scores(
                rule_based, gs_predictions, gs_embeddings,
            )
            logger.info(
                "Agent 3 ✅ Hybrid risk: {} companies (GraphSAGE={:.0%} + Rules={:.0%})",
                len(hybrid), self.blend_weight, 1 - self.blend_weight,
            )
            return hybrid, rule_based, graphsage_list
        else:
            # Fallback: annotate rule-based results with mode info
            for r in rule_based:
                r["risk_mode"] = "rule_based"
                r["graphsage_risk"] = None
                r["rule_based_risk"] = r["risk_score"]
                r["embedding_vector"] = []

            logger.info("Agent 3 ✅ Rule-based only: {} companies", len(rule_based))
            return rule_based, rule_based, []

    def _blend_scores(
        self,
        rule_based: list[dict],
        gs_predictions: dict[str, float],
        gs_embeddings: dict[str, list],
    ) -> tuple[list[dict], list[dict]]:
        """Blend GraphSAGE and rule-based scores."""
        hybrid_results = []
        graphsage_list = []
        w = self.blend_weight

        for rb in rule_based:
            company = rb["company"]
            rb_score = rb["risk_score"]

            gs_score = gs_predictions.get(company)
            embedding = gs_embeddings.get(company, [])

            if gs_score is not None:
                # Hybrid blend
                final_score = round(w * gs_score + (1 - w) * rb_score, 4)
                mode = "hybrid"
            else:
                # Company not in GraphSAGE predictions — use rule-based
                final_score = rb_score
                gs_score = None
                mode = "rule_based"

            # Classify risk level from final score
            if final_score >= 0.65:
                risk_level = "HIGH"
            elif final_score >= 0.35:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            # Build hybrid result (backward-compatible with Agent 4/5)
            hybrid_entry = {
                # Core fields (Agent 4/5 compatible)
                "company": company,
                "country": rb.get("country", "Unknown"),
                "industry": rb.get("industry", "Unknown"),
                "risk_score": final_score,
                "risk_level": risk_level,
                "depth": rb.get("depth", 0),
                "downstream_count": rb.get("downstream_count", 0),
                "disrupted_source": rb.get("disrupted_source", ""),
                "path": rb.get("path", []),
                # Sub-scores
                "supply_risk": rb.get("supply_risk", 0),
                "financial_risk": rb.get("financial_risk", 0),
                "operational_risk": rb.get("operational_risk", 0),
                "critical_velocity": rb.get("critical_velocity", False),
                "concentration_flag": rb.get("concentration_flag", False),
                # GraphSAGE enrichment
                "risk_mode": mode,
                "rule_based_risk": rb_score,
                "graphsage_risk": gs_score,
                "embedding_vector": embedding,
            }
            hybrid_results.append(hybrid_entry)

            # Build GraphSAGE-only entry
            if gs_score is not None:
                graphsage_list.append({
                    "company": company,
                    "graphsage_risk": gs_score,
                    "embedding_vector": embedding,
                })

        # Sort by final risk score descending
        hybrid_results.sort(key=lambda x: x["risk_score"], reverse=True)

        return hybrid_results, graphsage_list
