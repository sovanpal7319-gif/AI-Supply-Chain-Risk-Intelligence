"""
Neo4j Database Initialization Script

Creates 55+ Company nodes and 80+ SUPPLIES_TO relationships
to model a realistic multi-tier global supply chain.

Usage:
    python scripts/init_neo4j.py

Environment variables (or .env):
    NEO4J_URI      (default: bolt://localhost:7687)
    NEO4J_USER     (default: neo4j)
    NEO4J_PASSWORD (default: password)
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase

# ── Configuration ────────────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Sovan@2004")


# ── Company data ─────────────────────────────────────────────────────────────
COMPANIES = [
    # Semiconductor
    ("TSMC", "Taiwan", "Semiconductor"),
    ("Samsung Semiconductor", "South Korea", "Semiconductor"),
    ("Intel", "United States", "Semiconductor"),
    ("NVIDIA", "United States", "Semiconductor"),
    ("AMD", "United States", "Semiconductor"),
    ("Qualcomm", "United States", "Semiconductor"),
    ("Broadcom", "United States", "Semiconductor"),
    ("Texas Instruments", "United States", "Semiconductor"),
    ("MediaTek", "Taiwan", "Semiconductor"),
    ("SK Hynix", "South Korea", "Semiconductor"),
    ("Micron", "United States", "Semiconductor"),

    # Semiconductor Equipment
    ("ASML", "Netherlands", "Semiconductor Equipment"),
    ("Tokyo Electron", "Japan", "Semiconductor Equipment"),

    # Electronics / Tech
    ("Apple", "United States", "Electronics"),
    ("Microsoft", "United States", "Electronics"),
    ("Google", "United States", "Electronics"),
    ("Sony", "Japan", "Electronics"),
    ("Panasonic", "Japan", "Electronics"),
    ("LG Electronics", "South Korea", "Electronics"),
    ("Huawei", "China", "Electronics"),
    ("Samsung Electronics", "South Korea", "Electronics"),
    ("HP", "United States", "Electronics"),

    # Contract Manufacturing
    ("Foxconn", "Taiwan", "Contract Manufacturing"),
    ("Pegatron", "Taiwan", "Contract Manufacturing"),
    ("Flex", "Singapore", "Contract Manufacturing"),
    ("Jabil", "United States", "Contract Manufacturing"),

    # Automotive
    ("Toyota", "Japan", "Automotive"),
    ("Honda", "Japan", "Automotive"),
    ("BMW", "Germany", "Automotive"),
    ("Volkswagen", "Germany", "Automotive"),
    ("Ford", "United States", "Automotive"),
    ("GM", "United States", "Automotive"),
    ("Tesla", "United States", "Automotive"),
    ("Hyundai", "South Korea", "Automotive"),

    # Auto Parts
    ("Bosch", "Germany", "Auto Parts"),
    ("Denso", "Japan", "Auto Parts"),
    ("Continental", "Germany", "Auto Parts"),
    ("ZF Friedrichshafen", "Germany", "Auto Parts"),
    ("Aisin", "Japan", "Auto Parts"),

    # Chemicals
    ("BASF", "Germany", "Chemicals"),
    ("Dow Chemical", "United States", "Chemicals"),
    ("LG Chem", "South Korea", "Chemicals"),

    # Pharma
    ("Bayer", "Germany", "Pharma"),
    ("Pfizer", "United States", "Pharma"),
    ("Novartis", "Switzerland", "Pharma"),
    ("Roche", "Switzerland", "Pharma"),
    ("AstraZeneca", "United Kingdom", "Pharma"),
    ("Johnson & Johnson", "United States", "Pharma"),

    # Mining
    ("BHP", "Australia", "Mining"),
    ("Rio Tinto", "Australia", "Mining"),
    ("Vale", "Brazil", "Mining"),
    ("Glencore", "Switzerland", "Mining"),

    # Industrial
    ("Siemens", "Germany", "Industrial"),
    ("Honeywell", "United States", "Industrial"),
    ("3M", "United States", "Industrial"),
    ("Caterpillar", "United States", "Industrial"),

    # Indian
    ("Tata Steel", "India", "Steel"),
    ("Reliance Industries", "India", "Conglomerate"),
    ("Infosys", "India", "IT Services"),

    # Shipping
    ("Maersk", "Denmark", "Shipping"),
    ("COSCO", "China", "Shipping"),
    ("Hapag-Lloyd", "Germany", "Shipping"),
]

# SUPPLIES_TO relationships: (supplier, customer)
RELATIONSHIPS = [
    # Semiconductor equipment → Fabs
    ("ASML", "TSMC"), ("ASML", "Samsung Semiconductor"), ("ASML", "Intel"),
    ("Tokyo Electron", "TSMC"), ("Tokyo Electron", "Samsung Semiconductor"),

    # Raw materials → Semiconductor
    ("BHP", "TSMC"), ("Rio Tinto", "Intel"), ("Vale", "Samsung Semiconductor"),

    # Semiconductor → Electronics
    ("TSMC", "Apple"), ("TSMC", "NVIDIA"), ("TSMC", "AMD"),
    ("TSMC", "Qualcomm"), ("TSMC", "MediaTek"), ("TSMC", "Broadcom"),
    ("Samsung Semiconductor", "Samsung Electronics"), ("Samsung Semiconductor", "Google"),
    ("Intel", "Microsoft"), ("Intel", "Google"),
    ("SK Hynix", "Apple"), ("SK Hynix", "Samsung Electronics"),
    ("Micron", "Apple"), ("Micron", "Microsoft"),
    ("Texas Instruments", "Bosch"), ("Texas Instruments", "Continental"),

    # Semiconductor → Auto parts
    ("TSMC", "Bosch"), ("TSMC", "Denso"), ("TSMC", "Continental"),
    ("Intel", "Bosch"),

    # Auto parts → Automotive OEMs
    ("Bosch", "Toyota"), ("Bosch", "BMW"), ("Bosch", "Volkswagen"), ("Bosch", "Ford"),
    ("Denso", "Toyota"), ("Denso", "Honda"),
    ("Continental", "BMW"), ("Continental", "Volkswagen"), ("Continental", "GM"),
    ("ZF Friedrichshafen", "BMW"), ("ZF Friedrichshafen", "Ford"),
    ("Aisin", "Toyota"), ("Aisin", "Honda"),

    # Contract manufacturers
    ("Foxconn", "Apple"), ("Foxconn", "Sony"), ("Foxconn", "Google"),
    ("Pegatron", "Apple"), ("Pegatron", "Microsoft"),
    ("Flex", "Google"), ("Flex", "Huawei"),
    ("Jabil", "Apple"), ("Jabil", "Samsung Electronics"),

    # Chemicals
    ("BASF", "Bosch"), ("BASF", "Continental"), ("BASF", "BMW"),
    ("Dow Chemical", "Ford"), ("Dow Chemical", "GM"), ("Dow Chemical", "3M"),
    ("LG Chem", "LG Electronics"), ("LG Chem", "Tesla"),
    ("Glencore", "BASF"), ("Glencore", "Dow Chemical"),

    # Pharma supply chain
    ("BASF", "Bayer"), ("BASF", "Pfizer"),
    ("Bayer", "Johnson & Johnson"),

    # Industrial / Steel
    ("Tata Steel", "Bosch"), ("Tata Steel", "Siemens"),
    ("Siemens", "BMW"), ("Siemens", "Volkswagen"),
    ("3M", "Apple"), ("3M", "Ford"),

    # Shipping
    ("Maersk", "Apple"), ("Maersk", "Samsung Electronics"),
    ("COSCO", "Huawei"), ("COSCO", "Foxconn"),
    ("Hapag-Lloyd", "BMW"), ("Hapag-Lloyd", "Volkswagen"),

    # Indian companies
    ("Reliance Industries", "Foxconn"),
    ("Tata Steel", "Toyota"),
    ("Infosys", "Apple"), ("Infosys", "Microsoft"),
]


def init_database():
    """Create all nodes and relationships in Neo4j."""
    print(f"Connecting to Neo4j at {NEO4J_URI}...")

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("✅ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        print("\nMake sure Neo4j is running and the connection details are correct.")
        print("You can download Neo4j Desktop from: https://neo4j.com/download/")
        sys.exit(1)

    with driver.session() as session:
        # Clear existing data (optional)
        print("\n🗑️  Clearing existing data...")
        session.run("MATCH (n) DETACH DELETE n")

        # Create constraint for unique company names
        print("📐 Creating constraints...")
        try:
            session.run(
                "CREATE CONSTRAINT company_name IF NOT EXISTS "
                "FOR (c:Company) REQUIRE c.name IS UNIQUE"
            )
        except Exception:
            pass  # Constraint may already exist

        # Create company nodes
        print(f"\n📦 Creating {len(COMPANIES)} company nodes...")
        for name, country, industry in COMPANIES:
            session.run(
                "MERGE (c:Company {name: $name}) "
                "SET c.country = $country, c.industry = $industry",
                name=name, country=country, industry=industry,
            )
            print(f"  ✓ {name} ({country}, {industry})")

        # Create relationships
        print(f"\n🔗 Creating {len(RELATIONSHIPS)} SUPPLIES_TO relationships...")
        for supplier, customer in RELATIONSHIPS:
            result = session.run(
                "MATCH (s:Company {name: $supplier}), (c:Company {name: $customer}) "
                "MERGE (s)-[:SUPPLIES_TO]->(c) "
                "RETURN s.name, c.name",
                supplier=supplier, customer=customer,
            )
            record = result.single()
            if record:
                print(f"  ✓ {supplier} → {customer}")
            else:
                print(f"  ⚠ Could not link: {supplier} → {customer}")

        # Verify
        count_result = session.run("MATCH (n:Company) RETURN count(n) AS count")
        node_count = count_result.single()["count"]

        rel_result = session.run("MATCH ()-[r:SUPPLIES_TO]->() RETURN count(r) AS count")
        rel_count = rel_result.single()["count"]

        print(f"\n{'='*50}")
        print(f"✅ Database initialized successfully!")
        print(f"   Nodes:         {node_count}")
        print(f"   Relationships: {rel_count}")
        print(f"{'='*50}")

    driver.close()


if __name__ == "__main__":
    init_database()
