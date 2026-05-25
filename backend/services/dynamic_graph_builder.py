"""
Dynamic Graph Builder — On-the-fly Neo4j Graph Construction

Dynamically creates/updates Neo4j graph neighborhoods from structured
disruption data. Uses MERGE operations throughout for idempotency.

Node types:  Company, Event, Region
Relationships: SUPPLIES_TO, IMPACTS, LOCATED_IN

Example graph structure:
  (Earthquake)-[:IMPACTS]->(TSMC)
  (TSMC)-[:SUPPLIES_TO]->(Apple)
  (TSMC)-[:LOCATED_IN]->(Taiwan)
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from backend.db.connection import Neo4jConnection
from backend.agents.enhanced_disruption_agent import DisruptionExtraction
from backend.services.company_enrichment_service import CompanyEcosystem


class GraphBuildSummary(BaseModel):
    """Summary of what the dynamic graph builder created."""
    event_node_id: str = ""
    center_company: str = ""
    companies_created: int = 0
    regions_created: int = 0
    events_created: int = 0
    supplies_to_created: int = 0
    impacts_created: int = 0
    located_in_created: int = 0
    total_nodes: int = 0
    total_relationships: int = 0



class DynamicGraphBuilder:
    """
    Builds Neo4j graph neighborhoods on-the-fly from disruption data.

    All operations use MERGE to avoid duplicates and support
    incremental graph updates across multiple disruption events.
    Uses EntityCanonicalizationService for name normalization and
    GraphQualityEngine for post-build cleanup.
    """

    def __init__(self):
        driver = Neo4jConnection.get_driver()
        if driver is None:
            raise RuntimeError("Neo4j connection required for DynamicGraphBuilder")

        from backend.services.entity_canonicalization import EntityCanonicalizationService
        from backend.services.graph_quality_engine import GraphQualityEngine
        self._canon = EntityCanonicalizationService()
        self._quality = GraphQualityEngine()
        logger.info("🔧 Dynamic Graph Builder initialized (with canonicalization + quality engine)")

    def _is_valid_company_name(self, name: str) -> bool:
        """Check if a company name is valid using the canonicalization service."""
        return self._canon.is_valid_entity(name)

    def _canonicalize(self, name: str, entity_type: str = "COMPANY") -> Optional[str]:
        """Canonicalize a name through the entity canonicalization service."""
        return self._canon.canonicalize(name, entity_type)

    # ── High-level orchestration ─────────────────────────────────────────

    def build_dynamic_graph(
        self,
        extraction: DisruptionExtraction,
        ecosystem: CompanyEcosystem,
    ) -> GraphBuildSummary:
        """
        Build a complete graph neighborhood from extraction + ecosystem.

        Steps:
          1. Create/update center Company node
          2. Create Region node + LOCATED_IN
          3. Create Event node + IMPACTS
          4. Create supplier Company nodes + SUPPLIES_TO edges (inbound)
          5. Create customer Company nodes + SUPPLIES_TO edges (outbound)

        Returns GraphBuildSummary with creation counts.
        """
        # Canonicalize center company name
        center_name = self._canonicalize(extraction.company, "COMPANY")
        if not center_name:
            center_name = extraction.company  # fallback

        logger.info(
            "GraphBuilder ▶ Building graph for {} (event={}, location={})",
            center_name, extraction.event_type, extraction.location,
        )

        summary = GraphBuildSummary(center_company=center_name)

        # 1. Center company (skip if name is a placeholder like "Unknown")
        if self._is_valid_company_name(center_name):
            self.merge_company(
                name=center_name,
                industry=ecosystem.industry,
                country=ecosystem.country,
            )
            summary.companies_created += 1
        else:
            logger.warning(
                "GraphBuilder ⚠ Skipping invalid center company name: '{}'",
                center_name,
            )

        # 2. Region + LOCATED_IN
        if extraction.location and extraction.location != "Unknown":
            loc = self._canonicalize(extraction.location, "COUNTRY") or extraction.location
            self.merge_region(loc)
            self.link_company_to_region(center_name, loc)
            summary.regions_created += 1
            summary.located_in_created += 1

        # Also create region from ecosystem country if different
        if ecosystem.country and ecosystem.country != "Unknown":
            country = self._canonicalize(ecosystem.country, "COUNTRY") or ecosystem.country
            if country != extraction.location:
                self.merge_region(country)
                self.link_company_to_region(center_name, country)
                summary.regions_created += 1
                summary.located_in_created += 1

        # 3. Event + IMPACTS
        event_id = self.merge_event(
            event_type=extraction.event_type,
            severity=extraction.severity,
            description=extraction.summary or f"{extraction.event_type} affecting {center_name}",
            company_name=center_name,
        )
        summary.event_node_id = event_id
        summary.events_created += 1

        self.link_event_to_company(event_id, center_name)
        summary.impacts_created += 1

        # 4. Suppliers → center company (with canonicalization)
        for supplier in ecosystem.suppliers:
            canon_sup = self._canonicalize(supplier, "COMPANY") or supplier
            if not self._is_valid_company_name(canon_sup):
                continue
            self.merge_company(name=canon_sup)
            if self._is_valid_company_name(center_name):
                self.merge_supplies_to(canon_sup, center_name)
                summary.supplies_to_created += 1
            summary.companies_created += 1

        # 5. Center company → customers (with canonicalization)
        for customer in ecosystem.customers:
            canon_cust = self._canonicalize(customer, "COMPANY") or customer
            if not self._is_valid_company_name(canon_cust):
                continue
            self.merge_company(name=canon_cust)
            if self._is_valid_company_name(center_name):
                self.merge_supplies_to(center_name, canon_cust)
                summary.supplies_to_created += 1
            summary.companies_created += 1

        # Totals
        summary.total_nodes = (
            summary.companies_created + summary.regions_created + summary.events_created
        )
        summary.total_relationships = (
            summary.supplies_to_created + summary.impacts_created + summary.located_in_created
        )

        # 6. Post-build graph cleanup
        health = self._quality.cleanup_all(center_company=center_name)

        logger.info(
            "GraphBuilder ✅ Built graph: {} nodes, {} rels | cleanup: {} orphans, {} dupes removed",
            summary.total_nodes, summary.total_relationships,
            health.orphans_removed, health.duplicates_merged,
        )
        return summary

    # ── Node operations ──────────────────────────────────────────────────

    def merge_company(
        self,
        name: str,
        industry: Optional[str] = None,
        country: Optional[str] = None,
    ) -> None:
        """Create or update a Company node. Rejects placeholder names."""
        if not self._is_valid_company_name(name):
            logger.debug("  SKIP Company (invalid name): '{}'", name)
            return
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            if industry and country:
                session.run(
                    "MERGE (c:Company {name: $name}) "
                    "ON CREATE SET c.industry = $industry, c.country = $country, "
                    "c.created_at = datetime() "
                    "ON MATCH SET c.industry = COALESCE($industry, c.industry), "
                    "c.country = COALESCE($country, c.country)",
                    name=name, industry=industry, country=country,
                )
            else:
                session.run(
                    "MERGE (c:Company {name: $name}) "
                    "ON CREATE SET c.created_at = datetime()",
                    name=name,
                )
        logger.debug("  MERGE Company: {}", name)

    def merge_region(self, name: str) -> None:
        """Create or update a Region node."""
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            session.run(
                "MERGE (r:Region {name: $name}) "
                "ON CREATE SET r.created_at = datetime()",
                name=name,
            )
        logger.debug("  MERGE Region: {}", name)

    def merge_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        company_name: str,
    ) -> str:
        """
        Create an Event node. Returns the generated event_id.

        Each disruption event gets a unique ID so multiple events
        can coexist in the graph.
        """
        event_id = f"{event_type}_{company_name}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"

        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            session.run(
                "MERGE (e:Event {event_id: $event_id}) "
                "ON CREATE SET e.event_type = $event_type, "
                "e.severity = $severity, "
                "e.description = $description, "
                "e.created_at = datetime()",
                event_id=event_id,
                event_type=event_type,
                severity=severity,
                description=description,
            )
        logger.debug("  MERGE Event: {} ({})", event_id, event_type)
        return event_id

    # ── Relationship operations ──────────────────────────────────────────

    def link_event_to_company(self, event_id: str, company_name: str) -> None:
        """Create IMPACTS relationship: (Event)-[:IMPACTS]->(Company)."""
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            session.run(
                "MATCH (e:Event {event_id: $event_id}), (c:Company {name: $company}) "
                "MERGE (e)-[:IMPACTS]->(c)",
                event_id=event_id, company=company_name,
            )
        logger.debug("  LINK Event {} -[:IMPACTS]-> {}", event_id[:30], company_name)

    def link_company_to_region(self, company_name: str, region_name: str) -> None:
        """Create LOCATED_IN relationship: (Company)-[:LOCATED_IN]->(Region)."""
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            session.run(
                "MATCH (c:Company {name: $company}), (r:Region {name: $region}) "
                "MERGE (c)-[:LOCATED_IN]->(r)",
                company=company_name, region=region_name,
            )
        logger.debug("  LINK {} -[:LOCATED_IN]-> {}", company_name, region_name)

    def merge_supplies_to(self, supplier: str, customer: str) -> None:
        """Create SUPPLIES_TO relationship: (Supplier)-[:SUPPLIES_TO]->(Customer)."""
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            session.run(
                "MATCH (s:Company {name: $supplier}), (c:Company {name: $customer}) "
                "MERGE (s)-[:SUPPLIES_TO]->(c)",
                supplier=supplier, customer=customer,
            )
        logger.debug("  LINK {} -[:SUPPLIES_TO]-> {}", supplier, customer)

    # ── SPERT-specific node operations ───────────────────────────────────

    def merge_product(self, name: str) -> None:
        """Create or update a Product node."""
        if not name or not name.strip():
            return
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            session.run(
                "MERGE (p:Product {name: $name}) "
                "ON CREATE SET p.created_at = datetime()",
                name=name.strip(),
            )
        logger.debug("  MERGE Product: {}", name)

    def merge_generic_node(self, name: str, label: str, **props) -> None:
        """Create or update a node with an arbitrary label."""
        if not name or not name.strip() or not label:
            return
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            prop_set = ", ".join(
                f"n.{k} = ${k}" for k in props.keys()
            )
            on_create = f", {prop_set}" if prop_set else ""
            session.run(
                f"MERGE (n:{label} {{name: $name}}) "
                f"ON CREATE SET n.created_at = datetime(){on_create}",
                name=name.strip(), **props,
            )
        logger.debug("  MERGE {}:{}", label, name)

    # ── SPERT-specific edge operations ───────────────────────────────────

    def merge_affects(self, source: str, target: str) -> None:
        """Create AFFECTS relationship: (source)-[:AFFECTS]->(target)."""
        if not source or not target:
            return
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            session.run(
                "MATCH (a {name: $source}), (b {name: $target}) "
                "MERGE (a)-[:AFFECTS]->(b)",
                source=source.strip(), target=target.strip(),
            )
        logger.debug("  LINK {} -[:AFFECTS]-> {}", source, target)

    def merge_delays(self, source: str, target: str) -> None:
        """Create DELAYS relationship: (source)-[:DELAYS]->(target)."""
        if not source or not target:
            return
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            session.run(
                "MATCH (a {name: $source}), (b {name: $target}) "
                "MERGE (a)-[:DELAYS]->(b)",
                source=source.strip(), target=target.strip(),
            )
        logger.debug("  LINK {} -[:DELAYS]-> {}", source, target)

    def merge_depends_on(self, source: str, target: str) -> None:
        """Create DEPENDS_ON relationship: (source)-[:DEPENDS_ON]->(target)."""
        if not source or not target:
            return
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            session.run(
                "MATCH (a {name: $source}), (b {name: $target}) "
                "MERGE (a)-[:DEPENDS_ON]->(b)",
                source=source.strip(), target=target.strip(),
            )
        logger.debug("  LINK {} -[:DEPENDS_ON]-> {}", source, target)

    def merge_occurs_in(self, source: str, target: str) -> None:
        """Create OCCURS_IN / LOCATED_IN relationship: (source)-[:LOCATED_IN]->(target)."""
        if not source or not target:
            return
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            session.run(
                "MATCH (a {name: $source}), (b {name: $target}) "
                "MERGE (a)-[:LOCATED_IN]->(b)",
                source=source.strip(), target=target.strip(),
            )
        logger.debug("  LINK {} -[:LOCATED_IN]-> {}", source, target)

    # ── SPERT → Neo4j graph construction ─────────────────────────────────

    def build_from_spert(
        self,
        spert_data: dict,
        source_text: str = "",
    ) -> GraphBuildSummary:
        """
        Build Neo4j graph nodes and edges from validated SPERT output.

        Entity type → Neo4j node label mapping:
          COMPANY / SUPPLIER  → :Company
          COUNTRY / REGION    → :Region
          PORT                → :Region (type='port')
          EVENT               → :Event
          PRODUCT             → :Product

        Relation type → Neo4j edge mapping:
          AFFECTS      → [:AFFECTS]
          OCCURS_IN    → [:LOCATED_IN]
          DEPENDS_ON   → [:DEPENDS_ON]
          SUPPLIES_TO  → [:SUPPLIES_TO]
          DELAYS       → [:DELAYS]
          IMPACTS      → [:IMPACTS]

        Parameters
        ----------
        spert_data : dict
            Validated SPERT output with ``entities`` and ``relations`` keys.
        source_text : str, optional
            Original source text for logging.

        Returns
        -------
        GraphBuildSummary
            Counts of created nodes and relationships.
        """
        entities = spert_data.get("entities", [])
        relations = spert_data.get("relations", [])

        logger.info(
            "GraphBuilder (SPERT) ▶ Building graph from {} entities, {} relations",
            len(entities), len(relations),
        )

        summary = GraphBuildSummary(center_company=source_text[:50] if source_text else "spert")

        # ── Create nodes from entities ───────────────────────────────
        for ent in entities:
            text = ent.get("text", "").strip()
            etype = ent.get("type", "").upper()

            if not text:
                continue

            if etype in ("COMPANY", "SUPPLIER"):
                self.merge_company(name=text)
                summary.companies_created += 1
            elif etype == "COUNTRY":
                self.merge_region(text)
                summary.regions_created += 1
            elif etype == "REGION":
                self.merge_region(text)
                summary.regions_created += 1
            elif etype == "PORT":
                self.merge_region(text)
                summary.regions_created += 1
            elif etype == "EVENT":
                event_id = self.merge_event(
                    event_type=text,
                    severity="medium",
                    description=source_text or text,
                    company_name=text,
                )
                # Store event_id for relation linking
                ent["_event_id"] = event_id
                summary.events_created += 1
            elif etype == "PRODUCT":
                self.merge_product(text)
                # Count as a generic node
                summary.companies_created += 1
            else:
                self.merge_generic_node(text, etype.title())

        # ── Create edges from relations ──────────────────────────────
        for rel in relations:
            head = rel.get("head", "").strip()
            tail = rel.get("tail", "").strip()
            rtype = rel.get("type", "").upper()

            if not head or not tail:
                continue

            if rtype == "AFFECTS":
                self.merge_affects(head, tail)
                summary.impacts_created += 1
            elif rtype == "OCCURS_IN":
                self.merge_occurs_in(head, tail)
                summary.located_in_created += 1
            elif rtype == "DEPENDS_ON":
                self.merge_depends_on(head, tail)
                summary.supplies_to_created += 1
            elif rtype == "SUPPLIES_TO":
                self.merge_supplies_to(head, tail)
                summary.supplies_to_created += 1
            elif rtype == "DELAYS":
                self.merge_delays(head, tail)
                summary.impacts_created += 1
            elif rtype == "IMPACTS":
                self.link_event_to_company(head, tail)
                summary.impacts_created += 1

        # Update totals
        summary.total_nodes = (
            summary.companies_created + summary.regions_created + summary.events_created
        )
        summary.total_relationships = (
            summary.supplies_to_created + summary.impacts_created + summary.located_in_created
        )

        logger.info(
            "GraphBuilder (SPERT) ✅ Built: {} nodes, {} relationships",
            summary.total_nodes, summary.total_relationships,
        )
        return summary

    # ── Query helpers ────────────────────────────────────────────────────

    def get_dynamic_subgraph(self, center_company: str) -> dict:
        """
        Return the dynamically-built neighborhood around a company.

        Includes Company, Event, Region nodes and all relationship types
        within 2 hops of the center company.
        """
        driver = Neo4jConnection.get_driver()
        with driver.session() as session:
            # Nodes: center company + neighbors
            nodes_result = session.run(
                """
                MATCH (center:Company)
                WHERE toLower(center.name) CONTAINS toLower($name)
                OPTIONAL MATCH (center)-[:SUPPLIES_TO|LOCATED_IN*0..2]-(neighbor)
                OPTIONAL MATCH (event:Event)-[:IMPACTS]->(center)
                WITH collect(DISTINCT center) + collect(DISTINCT neighbor)
                     + collect(DISTINCT event) AS all_nodes
                UNWIND all_nodes AS n
                RETURN DISTINCT
                    labels(n)[0] AS label,
                    COALESCE(n.name, n.event_id) AS id,
                    n.country AS country,
                    n.industry AS industry,
                    n.event_type AS event_type,
                    n.severity AS severity
                """,
                name=center_company,
            )
            nodes = [dict(r) for r in nodes_result]

            # Edges
            edges_result = session.run(
                """
                MATCH (center:Company)
                WHERE toLower(center.name) CONTAINS toLower($name)
                OPTIONAL MATCH (a)-[r:SUPPLIES_TO|IMPACTS|LOCATED_IN|AFFECTS|DELAYS|DEPENDS_ON]-(b)
                WHERE a = center OR b = center
                   OR (a)-[:SUPPLIES_TO*1..2]-(center)
                RETURN DISTINCT
                    type(r) AS relationship,
                    COALESCE(startNode(r).name, startNode(r).event_id) AS source,
                    COALESCE(endNode(r).name, endNode(r).event_id) AS target
                """,
                name=center_company,
            )
            edges = [dict(r) for r in edges_result if r["source"] and r["target"]]

        return {"nodes": nodes, "edges": edges}