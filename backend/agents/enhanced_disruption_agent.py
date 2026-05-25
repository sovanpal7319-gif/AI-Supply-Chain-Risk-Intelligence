"""
Enhanced Disruption Agent — Structured Entity Extraction for Dynamic Pipeline

Extracts structured entities from live disruption news optimized for
dynamic graph construction:
  - company: the primary affected company
  - event_type: type of disruption event
  - location: geographic location of the event
  - severity: high / medium / low

Uses Groq LLM with a specialized prompt. Falls back to regex-based
extraction if the LLM call fails.

This agent is used ONLY by the /dynamic-analyze endpoint.
The original DisruptionAgent remains untouched for /analyze.
"""

import json
import re
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from backend.services.llm_service import LLMService


# ── Structured output model ──────────────────────────────────────────────────

class DisruptionExtraction(BaseModel):
    """Structured extraction result from disruption news."""
    company: str = Field(default="Unknown", description="Primary affected company")
    event_type: str = Field(default="unknown", description="Type of disruption event")
    location: str = Field(default="Unknown", description="Geographic location of event")
    severity: str = Field(default="medium", description="Severity: high, medium, or low")
    affected_companies: list[str] = Field(default_factory=list, description="All companies mentioned")
    affected_countries: list[str] = Field(default_factory=list, description="All countries mentioned")
    summary: str = Field(default="", description="Brief summary of the disruption")


# ── LLM prompt for structured extraction ─────────────────────────────────────

_EXTRACTION_PROMPT = """You are a supply chain disruption entity extractor. Given a news headline or article, extract structured entities for knowledge graph construction.

Return ONLY valid JSON in this exact format:
{
    "company": "primary affected company name (e.g., TSMC, Samsung, Intel)",
    "event_type": "one of: earthquake, fire, flood, typhoon, cyber_attack, sanctions, strike, pandemic, explosion, shortage, shutdown, geopolitical, logistics, operational, unknown",
    "location": "geographic location of the event (country or region, e.g., Taiwan, South Korea, Oregon)",
    "severity": "one of: high, medium, low",
    "affected_companies": ["list of all company names mentioned or implied"],
    "affected_countries": ["list of all countries mentioned or implied"],
    "summary": "1-sentence summary of the disruption"
}

Rules:
- For company, extract the EXACT company name (e.g., "TSMC" not "Taiwan Semiconductor")
- CRITICAL: If no specific company is mentioned in the article, you MUST infer the most relevant major company that would be directly affected. Use these mappings:
  * Oil/energy/petroleum disruptions → "Saudi Aramco" (oil producer), also include "ExxonMobil", "Shell", "Chevron", "BP", "Maersk" in affected_companies
  * Semiconductor/chip disruptions → "TSMC" or "Samsung" depending on region
  * Automotive disruptions → "Toyota", "Volkswagen", or "Tesla" depending on region
  * Shipping/logistics disruptions → "Maersk"
  * Chemical/pharma disruptions → "BASF" or "Pfizer"
  * NEVER return "Unknown" as the company name. Always infer the best match.
- For location, prefer country-level (e.g., "Taiwan") but include specific region if mentioned
- Severity: high = production halt/major damage, medium = partial impact, low = minor/precautionary
- Include the primary company in affected_companies list too
- For geopolitical events (embargoes, sanctions, trade wars), list ALL major companies in the affected industry/region in affected_companies
- If information is unclear, use your best judgment based on context"""


# ── Known company aliases for regex fallback ─────────────────────────────────

_COMPANY_ALIASES: dict[str, str] = {
    "tsmc": "TSMC",
    "taiwan semiconductor": "TSMC",
    "samsung": "Samsung",
    "samsung semiconductor": "Samsung Semiconductor",
    "intel": "Intel",
    "apple": "Apple",
    "nvidia": "NVIDIA",
    "tesla": "Tesla",
    "amd": "AMD",
    "qualcomm": "Qualcomm",
    "foxconn": "Foxconn",
    "google": "Google",
    "microsoft": "Microsoft",
    "sony": "Sony",
    "toyota": "Toyota",
    "bmw": "BMW",
    "volkswagen": "Volkswagen",
    "ford": "Ford",
    "bosch": "Bosch",
    "basf": "BASF",
    "asml": "ASML",
    "sk hynix": "SK Hynix",
    "micron": "Micron",
    "broadcom": "Broadcom",
    "mediatek": "MediaTek",
    "panasonic": "Panasonic",
    "lg chem": "LG Chem",
    "maersk": "Maersk",
    "pfizer": "Pfizer",
    "bayer": "Bayer",
}

_EVENT_KEYWORDS: dict[str, str] = {
    "earthquake": "earthquake",
    "fire": "fire",
    "flood": "flood",
    "typhoon": "typhoon",
    "hurricane": "typhoon",
    "cyclone": "typhoon",
    "cyber": "cyber_attack",
    "hack": "cyber_attack",
    "ransomware": "cyber_attack",
    "sanction": "sanctions",
    "ban": "sanctions",
    "strike": "strike",
    "labor": "strike",
    "pandemic": "pandemic",
    "covid": "pandemic",
    "explosion": "explosion",
    "shortage": "shortage",
    "shutdown": "shutdown",
    "war": "geopolitical",
    "conflict": "geopolitical",
    "tariff": "geopolitical",
    "blockade": "logistics",
    "port": "logistics",
    "shipping": "logistics",
}

_SEVERITY_KEYWORDS: dict[str, str] = {
    "devastating": "high",
    "catastrophic": "high",
    "major": "high",
    "massive": "high",
    "critical": "high",
    "severe": "high",
    "halt": "high",
    "destroy": "high",
    "collapse": "high",
    "moderate": "medium",
    "partial": "medium",
    "disrupt": "medium",
    "impact": "medium",
    "minor": "low",
    "small": "low",
    "slight": "low",
    "precautionary": "low",
}

_LOCATION_KEYWORDS: dict[str, str] = {
    "taiwan": "Taiwan",
    "south korea": "South Korea",
    "korea": "South Korea",
    "japan": "Japan",
    "china": "China",
    "united states": "United States",
    "usa": "United States",
    "us": "United States",
    "oregon": "United States",
    "arizona": "United States",
    "texas": "United States",
    "california": "United States",
    "germany": "Germany",
    "netherlands": "Netherlands",
    "india": "India",
    "brazil": "Brazil",
    "australia": "Australia",
    "singapore": "Singapore",
    "vietnam": "Vietnam",
    "thailand": "Thailand",
    "mexico": "Mexico",
    "uk": "United Kingdom",
    "switzerland": "Switzerland",
}


class EnhancedDisruptionAgent:
    """
    Extracts structured disruption entities from news text for dynamic
    graph construction.

    Uses Groq LLM with a specialized extraction prompt.
    Falls back to regex-based keyword extraction if LLM fails.
    """

    def __init__(self):
        self.llm = LLMService()
        logger.info("🔴 Enhanced Disruption Agent initialized (structured extraction)")

    async def run(self, news_text: str) -> DisruptionExtraction:
        """
        Extract structured entities from news text.

        Parameters
        ----------
        news_text : str
            Raw news headline or article text.

        Returns
        -------
        DisruptionExtraction
            Structured extraction with company, event_type, location, severity.
        """
        logger.info("Enhanced Agent ▶ Extracting entities from: '{}'", news_text[:100])

        # Try LLM extraction first
        result = await self._extract_with_llm(news_text)
        if result:
            logger.info(
                "Enhanced Agent ✅ LLM extraction: company={}, event={}, location={}, severity={}",
                result.company, result.event_type, result.location, result.severity,
            )
            return result

        # Fallback to regex
        logger.warning("Enhanced Agent ⚠️ LLM failed — using regex fallback")
        result = self._extract_with_regex(news_text)
        logger.info(
            "Enhanced Agent ✅ Regex extraction: company={}, event={}, location={}, severity={}",
            result.company, result.event_type, result.location, result.severity,
        )
        return result

    async def _extract_with_llm(self, text: str) -> Optional[DisruptionExtraction]:
        """Use Groq LLM for structured entity extraction."""
        try:
            from openai import AsyncOpenAI
            from backend.config import settings

            client = AsyncOpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )

            response = await client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": _EXTRACTION_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                logger.error("Enhanced Agent: LLM returned non-JSON: {}", content[:200])
                return None

            data = json.loads(json_match.group())

            return DisruptionExtraction(
                company=data.get("company", "Unknown"),
                event_type=data.get("event_type", "unknown"),
                location=data.get("location", "Unknown"),
                severity=data.get("severity", "medium"),
                affected_companies=data.get("affected_companies", []),
                affected_countries=data.get("affected_countries", []),
                summary=data.get("summary", ""),
            )

        except json.JSONDecodeError as exc:
            logger.error("Enhanced Agent: JSON parse error: {}", exc)
            return None
        except Exception as exc:
            logger.error("Enhanced Agent: LLM extraction failed: {}", exc)
            return None

    def _extract_with_regex(self, text: str) -> DisruptionExtraction:
        """Regex/keyword-based fallback extraction."""
        text_lower = text.lower()

        # Extract company
        company = "Unknown"
        affected_companies = []
        for alias, canonical in _COMPANY_ALIASES.items():
            if alias in text_lower:
                if company == "Unknown":
                    company = canonical
                if canonical not in affected_companies:
                    affected_companies.append(canonical)

        # Extract event type
        event_type = "unknown"
        for keyword, etype in _EVENT_KEYWORDS.items():
            if keyword in text_lower:
                event_type = etype
                break

        # Extract location
        location = "Unknown"
        affected_countries = []
        for keyword, loc in _LOCATION_KEYWORDS.items():
            if keyword in text_lower:
                if location == "Unknown":
                    location = loc
                if loc not in affected_countries:
                    affected_countries.append(loc)

        # Extract severity
        severity = "medium"  # default
        for keyword, sev in _SEVERITY_KEYWORDS.items():
            if keyword in text_lower:
                severity = sev
                break

        return DisruptionExtraction(
            company=company,
            event_type=event_type,
            location=location,
            severity=severity,
            affected_companies=affected_companies if affected_companies else [company],
            affected_countries=affected_countries if affected_countries else [location],
            summary=f"{event_type.replace('_', ' ').title()} affecting {company}",
        )
