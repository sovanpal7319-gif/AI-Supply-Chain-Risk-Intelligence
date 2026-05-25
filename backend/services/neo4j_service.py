"""
Neo4j Service — graph queries against the Neo4j knowledge graph.
"""

from typing import Optional
from loguru import logger
from backend.db.connection import Neo4jConnection


class Neo4jService:
    """Queries the supply chain knowledge graph via Neo4j."""

    def __init__(self):
        driver = Neo4jConnection.get_driver()
        if driver is None:
            raise RuntimeError(
                "Neo4j connection is required but unavailable. "
                "Make sure Neo4j is running and connection details in .env are correct."
            )
        logger.info("📊 Neo4j Service initialized (connected)")

    # ── Public API ───────────────────────────────────────────────────────────

    def find_company(self, name: str) -> Optional[dict]:
        """Look up a company by name (case-insensitive fuzzy match)."""
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            result = session.run(
                "MATCH (c:Company) WHERE toLower(c.name) CONTAINS toLower($name) "
                "RETURN c.name AS name, c.country AS country, c.industry AS industry LIMIT 1",
                name=name,
            )
            record = result.single()
            if record:
                return dict(record)
            return None

    def get_supply_chain(self, company_name: str, depth: int = 4) -> list[dict]:
        """
        Get the downstream supply chain for a company.
        Returns paths of companies affected if this company is disrupted.
        """
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            query = """
            MATCH path = (c:Company)-[:SUPPLIES_TO*1..{}]->(downstream:Company)
            WHERE toLower(c.name) CONTAINS toLower($name)
            RETURN [node IN nodes(path) | node.name] AS chain,
                   length(path) AS depth,
                   downstream.name AS end_company,
                   downstream.country AS end_country,
                   downstream.industry AS end_industry
            ORDER BY depth
            LIMIT 100
            """.format(min(depth, 4))

            result = session.run(query, name=company_name)
            paths = []
            for record in result:
                paths.append({
                    "path": record["chain"],
                    "depth": record["depth"],
                    "end_company": record["end_company"],
                    "end_country": record["end_country"],
                    "end_industry": record["end_industry"],
                })
            return paths

    def find_alternatives(self, company_name: str, disrupted_companies: list[str]) -> list[dict]:
        """
        Find alternative suppliers in the same industry
        that are NOT in the disrupted path.
        """
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Company)
                WHERE toLower(c.name) CONTAINS toLower($name)
                WITH c.industry AS target_industry
                MATCH (alt:Company)
                WHERE alt.industry = target_industry
                  AND NOT alt.name IN $disrupted
                RETURN alt.name AS name, alt.country AS country, alt.industry AS industry
                LIMIT 5
                """,
                name=company_name,
                disrupted=disrupted_companies,
            )
            return [dict(record) for record in result]

    def get_all_graph_data(self) -> dict:
        """Return all nodes and edges for visualization."""
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            nodes_result = session.run(
                "MATCH (c:Company) RETURN c.name AS id, c.country AS country, c.industry AS industry"
            )
            nodes = [dict(r) for r in nodes_result]

            edges_result = session.run(
                "MATCH (a:Company)-[:SUPPLIES_TO]->(b:Company) RETURN a.name AS source, b.name AS target"
            )
            edges = [dict(r) for r in edges_result]

        return {"nodes": nodes, "edges": edges}

    def get_dynamic_graph_data(self, center_company: str) -> dict:
        """
        Return only Company nodes and SUPPLIES_TO edges for a specific
        company's neighborhood (2-hop radius).

        Used by the /dynamic-analyze endpoint for visualization of the
        dynamically constructed supply chain graph.
        """
        # Guard: if no valid company, return full graph instead of crashing
        if not center_company or center_company.strip().lower() in ('unknown', 'none', 'n/a', ''):
            logger.warning("get_dynamic_graph_data: no valid center company — returning full graph")
            return self.get_all_graph_data()

        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            # Get only Company nodes within 2 hops via SUPPLIES_TO
            nodes_result = session.run(
                """
                MATCH (center:Company)
                WHERE toLower(center.name) CONTAINS toLower($name)
                  AND NOT center.name IN ['Unknown', 'unknown', 'N/A', 'None']
                WITH center
                OPTIONAL MATCH (center)-[:SUPPLIES_TO*0..2]-(company:Company)
                WHERE NOT company.name IN ['Unknown', 'unknown', 'N/A', 'None']
                WITH collect(DISTINCT company) AS companies
                UNWIND companies AS c
                RETURN DISTINCT
                    c.name AS id,
                    c.country AS country,
                    c.industry AS industry
                """,
                name=center_company,
            )
            nodes = [dict(r) for r in nodes_result if r["id"] and r["id"] not in ("Unknown", "unknown", "N/A", "None")]

            # Get only SUPPLIES_TO edges between Company nodes in the neighborhood
            edges_result = session.run(
                """
                MATCH (center:Company)
                WHERE toLower(center.name) CONTAINS toLower($name)
                WITH center
                OPTIONAL MATCH (center)-[:SUPPLIES_TO*0..2]-(company:Company)
                WITH collect(DISTINCT company) AS companies
                UNWIND companies AS c
                WITH collect(c.name) AS company_names
                MATCH (a:Company)-[:SUPPLIES_TO]->(b:Company)
                WHERE a.name IN company_names AND b.name IN company_names
                RETURN DISTINCT a.name AS source, b.name AS target
                """,
                name=center_company,
            )
            edges = [dict(r) for r in edges_result if r["source"] and r["target"]]

        return {"nodes": nodes, "edges": edges}

    # ── Supply Chain Intelligence Queries ─────────────────────────────

    def find_critical_suppliers(self, limit: int = 10) -> list[dict]:
        """Find the most depended-upon companies (highest in-degree via SUPPLIES_TO)."""
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            result = session.run(
                """
                MATCH (supplier:Company)-[:SUPPLIES_TO]->(customer:Company)
                WITH supplier, count(customer) AS customer_count
                ORDER BY customer_count DESC
                LIMIT $limit
                RETURN supplier.name AS company,
                       supplier.industry AS industry,
                       supplier.country AS country,
                       customer_count AS dependents
                """,
                limit=limit,
            )
            return [dict(r) for r in result]

    def get_upstream_chain(self, company_name: str, depth: int = 4) -> list[dict]:
        """Trace upstream suppliers (who supplies to this company, recursively)."""
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            result = session.run(
                """
                MATCH path = (supplier:Company)-[:SUPPLIES_TO*1..$depth]->(target:Company)
                WHERE toLower(target.name) CONTAINS toLower($name)
                RETURN [n IN nodes(path) | n.name] AS chain,
                       length(path) AS depth
                ORDER BY depth
                """,
                name=company_name,
                depth=depth,
            )
            return [dict(r) for r in result]

    def get_disruption_propagation(self, company_name: str) -> list[dict]:
        """Simulate cascading disruption impact with depth-decayed severity."""
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            result = session.run(
                """
                MATCH path = (source:Company)-[:SUPPLIES_TO*1..4]->(affected:Company)
                WHERE toLower(source.name) CONTAINS toLower($name)
                WITH affected, length(path) AS depth,
                     source.name AS source_name
                RETURN affected.name AS company,
                       affected.industry AS industry,
                       affected.country AS country,
                       depth,
                       CASE depth
                         WHEN 1 THEN 0.9
                         WHEN 2 THEN 0.6
                         WHEN 3 THEN 0.3
                         WHEN 4 THEN 0.1
                         ELSE 0.05
                       END AS propagated_severity,
                       source_name
                ORDER BY depth, affected.name
                """,
                name=company_name,
            )
            return [dict(r) for r in result]

    def get_region_risk_summary(self) -> list[dict]:
        """Aggregate company counts and supply chain density by country."""
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Company)
                WHERE c.country IS NOT NULL AND c.country <> 'Unknown'
                WITH c.country AS country, collect(c.name) AS companies
                RETURN country,
                       size(companies) AS company_count,
                       companies[0..5] AS sample_companies
                ORDER BY company_count DESC
                """
            )
            return [dict(r) for r in result]

    def find_bottleneck_nodes(self, limit: int = 5) -> list[dict]:
        """Find companies that are critical bottlenecks (high betweenness)."""
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Company)
                WITH c,
                     size([(c)<-[:SUPPLIES_TO]-() | 1]) AS in_deg,
                     size([(c)-[:SUPPLIES_TO]->() | 1]) AS out_deg
                WHERE in_deg > 0 AND out_deg > 0
                RETURN c.name AS company,
                       c.industry AS industry,
                       in_deg AS suppliers_count,
                       out_deg AS customers_count,
                       in_deg * out_deg AS bottleneck_score
                ORDER BY bottleneck_score DESC
                LIMIT $limit
                """,
                limit=limit,
            )
            return [dict(r) for r in result]