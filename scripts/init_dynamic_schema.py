"""
Neo4j Dynamic Schema Initialization Script

Creates constraints and indexes for the dynamic graph schema
(Event and Region nodes) alongside the existing Company schema.

Does NOT clear or modify existing data — safe to run on a
populated database.

Usage:
    python scripts/init_dynamic_schema.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def init_dynamic_schema():
    """Create constraints and indexes for the dynamic graph schema."""
    print(f"Connecting to Neo4j at {NEO4J_URI}...")

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("✅ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        sys.exit(1)

    with driver.session() as session:
        # ── Constraints ──────────────────────────────────────────────────
        constraints = [
            (
                "Event unique event_id",
                "CREATE CONSTRAINT event_id_unique IF NOT EXISTS "
                "FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
            ),
            (
                "Region unique name",
                "CREATE CONSTRAINT region_name_unique IF NOT EXISTS "
                "FOR (r:Region) REQUIRE r.name IS UNIQUE",
            ),
            (
                "Company unique name (ensure exists)",
                "CREATE CONSTRAINT company_name IF NOT EXISTS "
                "FOR (c:Company) REQUIRE c.name IS UNIQUE",
            ),
            (
                "Product unique name",
                "CREATE CONSTRAINT product_name_unique IF NOT EXISTS "
                "FOR (p:Product) REQUIRE p.name IS UNIQUE",
            ),
        ]

        print("\n📐 Creating constraints...")
        for desc, cypher in constraints:
            try:
                session.run(cypher)
                print(f"  ✓ {desc}")
            except Exception as e:
                print(f"  ⚠ {desc}: {e}")

        # ── Indexes ──────────────────────────────────────────────────────
        indexes = [
            (
                "Company name index",
                "CREATE INDEX company_name_idx IF NOT EXISTS "
                "FOR (c:Company) ON (c.name)",
            ),
            (
                "Company industry index",
                "CREATE INDEX company_industry_idx IF NOT EXISTS "
                "FOR (c:Company) ON (c.industry)",
            ),
            (
                "Company country index",
                "CREATE INDEX company_country_idx IF NOT EXISTS "
                "FOR (c:Company) ON (c.country)",
            ),
            (
                "Event type index",
                "CREATE INDEX event_type_idx IF NOT EXISTS "
                "FOR (e:Event) ON (e.event_type)",
            ),
            (
                "Region name index",
                "CREATE INDEX region_name_idx IF NOT EXISTS "
                "FOR (r:Region) ON (r.name)",
            ),
            (
                "Product name index",
                "CREATE INDEX product_name_idx IF NOT EXISTS "
                "FOR (p:Product) ON (p.name)",
            ),
        ]

        print("\n📇 Creating indexes...")
        for desc, cypher in indexes:
            try:
                session.run(cypher)
                print(f"  ✓ {desc}")
            except Exception as e:
                print(f"  ⚠ {desc}: {e}")

        # ── Verify ───────────────────────────────────────────────────────
        print("\n📊 Current database statistics:")

        for label in ["Company", "Event", "Region", "Product"]:
            result = session.run(
                f"MATCH (n:{label}) RETURN count(n) AS count"
            )
            count = result.single()["count"]
            print(f"  {label} nodes: {count}")

        for rel_type in ["SUPPLIES_TO", "IMPACTS", "LOCATED_IN", "PRODUCES", "AFFECTS", "DEPENDS_ON"]:
            result = session.run(
                f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
            )
            count = result.single()["count"]
            print(f"  {rel_type} relationships: {count}")

    driver.close()
    print("\n✅ Dynamic schema initialization complete!")
    print("\n📝 Graph visualization queries:")
    print("  -- All events and impacted companies:")
    print("  MATCH (e:Event)-[:IMPACTS]->(c:Company) RETURN e, c")
    print("  -- Company locations:")
    print("  MATCH (c:Company)-[:LOCATED_IN]->(r:Region) RETURN c, r")
    print("  -- Full neighborhood of a company:")
    print("  MATCH (c:Company {name:'TSMC'})-[r]-(n) RETURN c, r, n")
    print("  -- Supply chain + events combined:")
    print("  MATCH p=(e:Event)-[:IMPACTS]->(c)-[:SUPPLIES_TO*1..3]->(d) RETURN p")


if __name__ == "__main__":
    init_dynamic_schema()
