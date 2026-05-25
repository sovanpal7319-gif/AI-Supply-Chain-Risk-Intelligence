"""
Agent 5 — Alternative Supplier Agent

For HIGH-risk companies, suggests alternative suppliers from
the same industry that are not in the disrupted path.
"""

from loguru import logger
from backend.services.neo4j_service import Neo4jService


class AlternativeSupplierAgent:
    """Suggests alternative suppliers for high-risk disrupted companies."""

    def __init__(self):
        self.neo4j = Neo4jService()
        logger.info("🟣 Alternative Supplier Agent initialized (Neo4j)")

    def run(self, decisions: list[dict], all_affected_companies: list[str]) -> list[dict]:
        """
        For each HIGH / MEDIUM risk company, find alternative suppliers.

        Parameters
        ----------
        decisions : list[dict]
            Output from DecisionAgent.
        all_affected_companies : list[str]
            Full list of companies in the disrupted path (to exclude).

        Returns
        -------
        list[dict] with keys:
            company, risk_level, alternatives (list of alternative company dicts)
        """
        logger.info(
            "Agent 5 ▶ Finding alternatives for high-risk companies (excluding {} disrupted)",
            len(all_affected_companies),
        )

        suggestions = []
        for decision in decisions:
            if decision["risk_level"] in ("HIGH", "MEDIUM"):
                alternatives = self.neo4j.find_alternatives(
                    company_name=decision["company"],
                    disrupted_companies=all_affected_companies,
                )
                if alternatives:
                    suggestions.append({
                        "company": decision["company"],
                        "risk_level": decision["risk_level"],
                        "alternatives": alternatives,
                    })
                    logger.debug(
                        "  {} → {} alternatives found",
                        decision["company"],
                        len(alternatives),
                    )

        logger.info("Agent 5 ✅ Found alternatives for {} companies", len(suggestions))
        return suggestions
