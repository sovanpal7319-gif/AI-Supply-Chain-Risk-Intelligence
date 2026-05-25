"""
Company Enrichment Service — Dynamic Ecosystem Lookup

Given a company name, returns its supply chain ecosystem:
  - industry, country, upstream suppliers, downstream customers

Modes (ENRICHMENT_MODE in .env):
  - "mock": Curated dataset from data/mock_enrichment_data.py (default)
  - "llm" : Groq LLM fallback for unknown companies
"""

import json
import re
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from backend.config import settings


class CompanyEcosystem(BaseModel):
    """Enriched company ecosystem data."""
    company: str = Field(..., description="Canonical company name")
    industry: str = Field(default="Unknown")
    country: str = Field(default="Unknown")
    suppliers: list[str] = Field(default_factory=list)
    customers: list[str] = Field(default_factory=list)
    source: str = Field(default="mock")


_ENRICHMENT_PROMPT = """You are a supply chain intelligence expert. Given a company name, provide its ecosystem.

Return ONLY valid JSON:
{
    "company": "exact company name",
    "industry": "primary industry",
    "country": "headquarters country",
    "suppliers": ["5-8 major upstream suppliers"],
    "customers": ["5-8 major downstream customers"]
}"""


class CompanyEnrichmentService:
    """Enriches a company name with its supply chain ecosystem."""

    def __init__(self):
        self.mode = getattr(settings, "enrichment_mode", "mock")
        self._mock_data: Optional[dict] = None
        logger.info("🏭 Company Enrichment Service initialized (mode={})", self.mode)

    def _load_mock_data(self) -> dict:
        if self._mock_data is None:
            try:
                from data.mock_enrichment_data import MOCK_COMPANY_DATA
                self._mock_data = MOCK_COMPANY_DATA
                logger.debug("Loaded mock enrichment data: {} companies", len(self._mock_data))
            except ImportError:
                logger.warning("Mock enrichment data not found — empty dataset")
                self._mock_data = {}
        return self._mock_data

    async def enrich(self, company_name: str) -> CompanyEcosystem:
        """Get the supply chain ecosystem for a company."""
        # Guard: handle None or empty company name
        if not company_name or not company_name.strip():
            logger.warning("Enrichment ⚠️ No company name provided — returning empty ecosystem")
            return CompanyEcosystem(company=company_name or "Unknown", source="fallback")

        logger.info("Enrichment ▶ Looking up ecosystem for: '{}'", company_name)

        result = self._lookup_mock(company_name)
        if result:
            logger.info(
                "Enrichment ✅ Mock hit: {} — {} suppliers, {} customers",
                result.company, len(result.suppliers), len(result.customers),
            )
            return result

        if self.mode == "llm":
            result = await self._enrich_with_llm(company_name)
            if result:
                logger.info(
                    "Enrichment ✅ LLM: {} — {} suppliers, {} customers",
                    result.company, len(result.suppliers), len(result.customers),
                )
                return result

        logger.warning("Enrichment ⚠️ No data for '{}' — minimal fallback", company_name)
        return CompanyEcosystem(
            company=company_name, source="fallback",
        )

    def _lookup_mock(self, company_name: str) -> Optional[CompanyEcosystem]:
        mock_data = self._load_mock_data()
        if not company_name:
            return None
        data = mock_data.get(company_name.strip().lower())
        if data:
            return CompanyEcosystem(
                company=company_name,
                industry=data["industry"],
                country=data["country"],
                suppliers=data["suppliers"],
                customers=data["customers"],
                source="mock",
            )
        return None

    async def _enrich_with_llm(self, company_name: str) -> Optional[CompanyEcosystem]:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )

            response = await client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": _ENRICHMENT_PROMPT},
                    {"role": "user", "content": f"Company: {company_name}"},
                ],
                temperature=0.2,
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                return None

            data = json.loads(json_match.group())
            return CompanyEcosystem(
                company=data.get("company", company_name),
                industry=data.get("industry", "Unknown"),
                country=data.get("country", "Unknown"),
                suppliers=data.get("suppliers", []),
                customers=data.get("customers", []),
                source="llm",
            )
        except Exception as exc:
            logger.error("Enrichment LLM error: {}", exc)
            return None
