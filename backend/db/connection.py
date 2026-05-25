"""
Neo4j driver management — singleton connection.

Requires Neo4j to be running and accessible. Fails fast on startup
if the connection cannot be established.
"""

from loguru import logger
from backend.config import settings

from neo4j import GraphDatabase


class Neo4jConnection:
    """Manages a singleton Neo4j driver instance."""

    _driver = None

    @classmethod
    def get_driver(cls):
        """Return the shared Neo4j driver, creating it on first call."""
        if cls._driver is None:
            try:
                cls._driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_user, settings.neo4j_password),
                )
                cls._driver.verify_connectivity()
                logger.info("✅ Connected to Neo4j at {}", settings.neo4j_uri)
            except Exception as exc:
                logger.error("❌ Neo4j connection failed: {}", exc)
                cls._driver = None
                raise RuntimeError(
                    f"Neo4j connection failed ({exc}). "
                    f"Make sure Neo4j is running at {settings.neo4j_uri}"
                ) from exc
        return cls._driver

    @classmethod
    def close(cls):
        """Cleanly close the driver."""
        if cls._driver:
            cls._driver.close()
            cls._driver = None
            logger.info("Neo4j driver closed.")

    @classmethod
    def health_check(cls) -> bool:
        """Return True if the database is reachable."""
        try:
            driver = cls.get_driver()
            driver.verify_connectivity()
            return True
        except Exception:
            return False
