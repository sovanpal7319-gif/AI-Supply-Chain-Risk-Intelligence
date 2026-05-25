"""
Agent 2 — Knowledge Graph Query Agent

Takes entities from Agent 1 and traverses the Neo4j supply chain
graph up to 4 levels deep to find affected downstream companies.
"""

from loguru import logger
from backend.services.neo4j_service import Neo4jService


class KGQueryAgent:
    """Queries the Neo4j knowledge graph for disrupted supply chain paths."""

    def __init__(self):
        self.neo4j = Neo4jService()
        logger.info("🔵 Knowledge Graph Query Agent initialized (Neo4j)")

    def run(self, disruption_data: dict) -> dict:
        """
        Traverse the supply chain graph for each affected company.

        Parameters
        ----------
        disruption_data : dict
            Output from DisruptionAgent (must contain 'affected_companies')

        Returns
        -------
        dict with keys:
            affected_entities: list of matched company nodes
            supply_chain_paths: list of disrupted downstream paths
            total_downstream_affected: int
        """
        companies = disruption_data.get("affected_companies", [])
        logger.info("Agent 2 ▶ Querying graph for {} entit(ies)", len(companies))

        all_paths = []
        matched_entities = []
        all_affected = set()

        for company_name in companies:
            # Look up company in graph
            company = self.neo4j.find_company(company_name)
            if company:
                matched_entities.append(company)
                logger.debug("  Found: {} ({})", company["name"], company.get("country"))

                # Traverse downstream supply chain (up to 4 levels)
                paths = self.neo4j.get_supply_chain(company["name"], depth=4)
                for p in paths:
                    p["disrupted_source"] = company["name"]
                    all_affected.update(p.get("path", []))

                all_paths.extend(paths)
            else:
                logger.debug("  Not found in graph: {}", company_name)

        result = {
            "affected_entities": matched_entities,
            "supply_chain_paths": all_paths,
            "total_downstream_affected": len(all_affected),
            "all_affected_companies": list(all_affected),
        }

        logger.info(
            "Agent 2 ✅ Found {} paths, {} total affected companies",
            len(all_paths),
            len(all_affected),
        )
        return result
