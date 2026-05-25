"""
Entity Canonicalization Service — Production-Grade Name Normalization

Normalizes all entity names (companies, countries, regions, products) before
they reach Neo4j, preventing duplicate nodes, noisy names, and inconsistent
references across the knowledge graph.

Pipeline:
  raw entity text
  → blocklist check (reject "Unknown", "Reuters", etc.)
  → strip corporate suffixes ("Inc.", "Corp.", "Ltd.", etc.)
  → alias lookup (exact canonical match)
  → fuzzy match (difflib, threshold=0.85)
  → title-case normalization
  → return canonical name or None

Usage::

    canon = EntityCanonicalizationService()
    canon.canonicalize("Taiwan Semiconductor", "COMPANY")  → "TSMC"
    canon.canonicalize("Exxon Mobil Corp.", "COMPANY")      → "ExxonMobil"
    canon.canonicalize("Unknown", "COMPANY")                → None
    canon.canonicalize("united states", "COUNTRY")          → "United States"
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional

from loguru import logger


# ── Blocklist: names that should NEVER become graph nodes ────────────────────

_BLOCKLIST: set[str] = {
    # Placeholder / null values
    "unknown", "n/a", "none", "null", "", "unnamed", "unspecified",
    "not specified", "not available", "undisclosed", "tbd", "tba",
    # News source artifacts
    "reuters", "ap news", "bloomberg", "cnbc", "bbc", "cnn",
    "the wall street journal", "financial times", "the guardian",
    "associated press", "afp", "xinhua",
    # Generic words that aren't entities
    "the", "a", "an", "company", "companies", "market", "markets",
    "industry", "global", "world", "international", "report",
    "analysis", "update", "news", "article", "source", "sources",
    "advertisement", "key takeaways", "market snapshot",
    "what to watch", "article body", "market interpretation",
}

# ── Corporate suffixes to strip ──────────────────────────────────────────────

_CORPORATE_SUFFIXES: list[str] = [
    r"\s+Inc\.?$", r"\s+Corp\.?$", r"\s+Corporation$",
    r"\s+Ltd\.?$", r"\s+Limited$", r"\s+LLC$", r"\s+LLP$",
    r"\s+PLC$", r"\s+plc$", r"\s+AG$", r"\s+GmbH$",
    r"\s+S\.?A\.?$", r"\s+N\.?V\.?$", r"\s+B\.?V\.?$",
    r"\s+Co\.?$", r"\s+Company$", r"\s+Group$",
    r"\s+Holdings?$", r"\s+Enterprises?$",
    r"\s+International$", r"\s+Technologies$",
    r"\s+Manufacturing\s+Company$",
    r"\s+Manufacturing\s+Co\.?$",
    r"\s+Semiconductor\s+Manufacturing\s+Company$",
]

_SUFFIX_PATTERN = re.compile(
    "|".join(f"({s})" for s in _CORPORATE_SUFFIXES),
    re.IGNORECASE,
)

# ── Company alias dictionary ────────────────────────────────────────────────
# Maps lowercase variants → canonical name.
# The canonical name is what appears in Neo4j.

_COMPANY_ALIASES: dict[str, str] = {
    # Semiconductors
    "tsmc": "TSMC",
    "taiwan semiconductor": "TSMC",
    "taiwan semiconductor manufacturing": "TSMC",
    "taiwan semiconductor manufacturing company": "TSMC",
    "samsung": "Samsung",
    "samsung electronics": "Samsung Electronics",
    "samsung semiconductor": "Samsung Semiconductor",
    "samsung elec": "Samsung Electronics",
    "samsung semi": "Samsung Semiconductor",
    "intel": "Intel",
    "intel corporation": "Intel",
    "nvidia": "NVIDIA",
    "nvidia corp": "NVIDIA",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "qualcomm": "Qualcomm",
    "broadcom": "Broadcom",
    "mediatek": "MediaTek",
    "sk hynix": "SK Hynix",
    "sk hynix inc": "SK Hynix",
    "micron": "Micron",
    "micron technology": "Micron",
    "asml": "ASML",
    "asml holding": "ASML",
    "tokyo electron": "Tokyo Electron",

    # Tech giants
    "apple": "Apple",
    "apple inc": "Apple",
    "google": "Google",
    "alphabet": "Google",
    "microsoft": "Microsoft",
    "microsoft corp": "Microsoft",
    "meta": "Meta",
    "facebook": "Meta",
    "amazon": "Amazon",
    "amazon.com": "Amazon",

    # Manufacturing
    "foxconn": "Foxconn",
    "hon hai": "Foxconn",
    "hon hai precision": "Foxconn",
    "pegatron": "Pegatron",
    "flex": "Flex",
    "flextronics": "Flex",
    "jabil": "Jabil",

    # Automotive
    "tesla": "Tesla",
    "tesla inc": "Tesla",
    "toyota": "Toyota",
    "toyota motor": "Toyota",
    "bmw": "BMW",
    "volkswagen": "Volkswagen",
    "vw": "Volkswagen",
    "ford": "Ford",
    "ford motor": "Ford",
    "gm": "GM",
    "general motors": "GM",
    "hyundai": "Hyundai",
    "hyundai motor": "Hyundai",
    "honda": "Honda",
    "honda motor": "Honda",

    # Auto parts
    "bosch": "Bosch",
    "robert bosch": "Bosch",
    "denso": "Denso",
    "continental": "Continental",
    "continental ag": "Continental",
    "zf friedrichshafen": "ZF Friedrichshafen",
    "zf": "ZF Friedrichshafen",

    # Chemicals
    "basf": "BASF",
    "basf se": "BASF",
    "dow": "Dow Chemical",
    "dow chemical": "Dow Chemical",
    "dow inc": "Dow Chemical",
    "bayer": "Bayer",
    "bayer ag": "Bayer",
    "lg chem": "LG Chem",

    # Oil & Gas
    "saudi aramco": "Saudi Aramco",
    "aramco": "Saudi Aramco",
    "exxonmobil": "ExxonMobil",
    "exxon mobil": "ExxonMobil",
    "exxon": "ExxonMobil",
    "shell": "Shell",
    "royal dutch shell": "Shell",
    "chevron": "Chevron",
    "chevron corp": "Chevron",
    "bp": "BP",
    "british petroleum": "BP",
    "totalenergies": "TotalEnergies",
    "total": "TotalEnergies",

    # Logistics
    "maersk": "Maersk",
    "a.p. moller-maersk": "Maersk",
    "ap moller maersk": "Maersk",
    "fedex": "FedEx",
    "federal express": "FedEx",
    "ups": "UPS",
    "united parcel service": "UPS",
    "dhl": "DHL",
    "cma cgm": "CMA CGM",
    "msc": "MSC",
    "cosco": "COSCO",
    "cosco shipping": "COSCO",

    # Pharma
    "pfizer": "Pfizer",
    "pfizer inc": "Pfizer",

    # Industrial
    "siemens": "Siemens",
    "siemens ag": "Siemens",
    "caterpillar": "Caterpillar",
    "honeywell": "Honeywell",
    "3m": "3M",
    "ge": "GE",
    "general electric": "GE",
    "abb": "ABB",

    # Mining
    "bhp": "BHP",
    "bhp billiton": "BHP",
    "rio tinto": "Rio Tinto",
    "vale": "Vale",
    "glencore": "Glencore",
    "albemarle": "Albemarle",

    # Electronics
    "sony": "Sony",
    "sony corp": "Sony",
    "panasonic": "Panasonic",
    "lg electronics": "LG Electronics",
    "lg": "LG Electronics",
    "huawei": "Huawei",
    "xiaomi": "Xiaomi",
    "hp": "HP",
    "dell": "Dell",
    "lenovo": "Lenovo",

    # Energy organizations
    "opec": "OPEC",
    "opec+": "OPEC",
}

# ── Country / Region alias dictionary ────────────────────────────────────────

_COUNTRY_ALIASES: dict[str, str] = {
    "us": "United States", "usa": "United States",
    "united states of america": "United States",
    "u.s.": "United States", "u.s.a.": "United States",
    "uk": "United Kingdom", "britain": "United Kingdom",
    "great britain": "United Kingdom", "england": "United Kingdom",
    "korea": "South Korea", "south korea": "South Korea",
    "republic of korea": "South Korea",
    "taiwan": "Taiwan", "republic of china": "Taiwan",
    "china": "China", "prc": "China",
    "japan": "Japan", "germany": "Germany",
    "netherlands": "Netherlands", "holland": "Netherlands",
    "india": "India", "brazil": "Brazil",
    "australia": "Australia", "singapore": "Singapore",
    "vietnam": "Vietnam", "thailand": "Thailand",
    "mexico": "Mexico", "canada": "Canada",
    "france": "France", "italy": "Italy",
    "switzerland": "Switzerland", "sweden": "Sweden",
    "denmark": "Denmark", "norway": "Norway",
    "russia": "Russia", "ukraine": "Ukraine",
    "israel": "Israel", "iran": "Iran",
    "saudi arabia": "Saudi Arabia", "uae": "UAE",
    "united arab emirates": "UAE",
    "indonesia": "Indonesia", "malaysia": "Malaysia",
    "philippines": "Philippines", "cuba": "Cuba",
}

# ── Product / Raw Material aliases ───────────────────────────────────────────

_PRODUCT_ALIASES: dict[str, str] = {
    "semiconductor": "Semiconductors", "semiconductors": "Semiconductors",
    "chip": "Semiconductors", "chips": "Semiconductors",
    "microchip": "Semiconductors", "microchips": "Semiconductors",
    "battery": "Batteries", "batteries": "Batteries",
    "ev battery": "EV Batteries", "ev batteries": "EV Batteries",
    "lithium": "Lithium", "cobalt": "Cobalt",
    "nickel": "Nickel", "palladium": "Palladium",
    "neon gas": "Neon Gas", "neon": "Neon Gas",
    "silicon": "Silicon", "silicon wafer": "Silicon Wafers",
    "rare earth": "Rare Earth Minerals", "rare earths": "Rare Earth Minerals",
    "oil": "Crude Oil", "crude oil": "Crude Oil",
    "petroleum": "Crude Oil", "wti": "Crude Oil",
    "natural gas": "Natural Gas", "gas": "Natural Gas",
    "lng": "LNG",
    "steel": "Steel", "aluminum": "Aluminum", "copper": "Copper",
    "vaccine": "Vaccines", "vaccines": "Vaccines",
    "pharmaceutical": "Pharmaceuticals", "pharmaceuticals": "Pharmaceuticals",
    "auto parts": "Auto Parts", "display": "Displays",
    "gpu": "GPUs", "gpus": "GPUs",
    "cpu": "CPUs", "cpus": "CPUs",
    "memory": "Memory Chips", "dram": "DRAM", "nand": "NAND Flash",
}

# ── Fuzzy match threshold ────────────────────────────────────────────────────

_FUZZY_THRESHOLD: float = 0.85


class EntityCanonicalizationService:
    """
    Normalizes entity names for consistent knowledge graph construction.

    Applies a multi-step pipeline:
      1. Blocklist check → reject noise words
      2. Strip corporate suffixes → "Apple Inc." → "Apple"
      3. Alias lookup → exact match to canonical name
      4. Fuzzy match → catch typos and near-misses
      5. Title-case fallback → if no alias found
    """

    def __init__(self):
        self._company_aliases = dict(_COMPANY_ALIASES)
        self._country_aliases = dict(_COUNTRY_ALIASES)
        self._product_aliases = dict(_PRODUCT_ALIASES)
        self._blocklist = set(_BLOCKLIST)
        logger.info(
            "🔤 Entity Canonicalization Service initialized "
            "({} company aliases, {} country aliases, {} product aliases)",
            len(self._company_aliases),
            len(self._country_aliases),
            len(self._product_aliases),
        )

    # ── Public API ───────────────────────────────────────────────────────

    def canonicalize(
        self,
        text: str,
        entity_type: str = "COMPANY",
    ) -> Optional[str]:
        """
        Normalize an entity name to its canonical form.

        Parameters
        ----------
        text : str
            Raw entity text from extraction.
        entity_type : str
            Entity type hint: COMPANY, COUNTRY, REGION, PRODUCT, etc.

        Returns
        -------
        str or None
            Canonical name, or None if the entity should be rejected.
        """
        if not text or not text.strip():
            return None

        cleaned = text.strip()

        # Step 1: Blocklist check
        if cleaned.lower() in self._blocklist:
            logger.debug("  CANON reject (blocklist): '{}'", cleaned)
            return None

        # Step 2: Strip corporate suffixes
        cleaned = self._strip_suffixes(cleaned)

        # Step 3: Empty after stripping?
        if not cleaned or cleaned.lower() in self._blocklist:
            return None

        # Step 4: Route to type-specific canonicalization
        upper_type = entity_type.upper()

        if upper_type in ("COMPANY", "SUPPLIER", "CUSTOMER", "LOGISTICS_PROVIDER"):
            return self._canonicalize_company(cleaned)
        elif upper_type in ("COUNTRY", "REGION"):
            return self._canonicalize_country(cleaned)
        elif upper_type in ("PRODUCT", "RAW_MATERIAL"):
            return self._canonicalize_product(cleaned)
        elif upper_type == "EVENT":
            # Events don't need alias resolution, just clean up
            return cleaned.title() if not cleaned.isupper() else cleaned
        elif upper_type == "PORT":
            return cleaned.title() if not cleaned.isupper() else cleaned
        elif upper_type == "INDUSTRY":
            return cleaned.title()
        else:
            return cleaned.strip()

    def is_valid_entity(self, text: str) -> bool:
        """Check if an entity name is valid (not blocklisted or empty)."""
        if not text or not text.strip():
            return False
        return text.strip().lower() not in self._blocklist

    def canonicalize_company(self, name: str) -> Optional[str]:
        """Public shortcut for company canonicalization."""
        return self.canonicalize(name, "COMPANY")

    # ── Internal canonicalization methods ─────────────────────────────────

    def _canonicalize_company(self, text: str) -> Optional[str]:
        """Resolve a company name to its canonical form."""
        key = text.lower().strip()

        # Exact alias match
        if key in self._company_aliases:
            canonical = self._company_aliases[key]
            if canonical != text:
                logger.debug("  CANON alias: '{}' → '{}'", text, canonical)
            return canonical

        # Fuzzy match against known aliases
        best = self._fuzzy_match(key, self._company_aliases)
        if best:
            logger.debug("  CANON fuzzy: '{}' → '{}'", text, best)
            return best

        # No alias found — apply title case (preserve all-caps acronyms)
        if text.isupper() and len(text) <= 6:
            return text  # Likely an acronym like "TSMC", "ASML"
        return text.title() if not text.isupper() else text

    def _canonicalize_country(self, text: str) -> Optional[str]:
        """Resolve a country/region name to its canonical form."""
        key = text.lower().strip()

        if key in self._country_aliases:
            return self._country_aliases[key]

        # Fuzzy match for countries
        best = self._fuzzy_match(key, self._country_aliases, threshold=0.80)
        if best:
            return best

        return text.title()

    def _canonicalize_product(self, text: str) -> Optional[str]:
        """Resolve a product/material name to its canonical form."""
        key = text.lower().strip()

        if key in self._product_aliases:
            return self._product_aliases[key]

        return text.title()

    # ── Utility methods ──────────────────────────────────────────────────

    @staticmethod
    def _strip_suffixes(text: str) -> str:
        """Remove corporate suffixes like Inc., Corp., Ltd., etc."""
        result = _SUFFIX_PATTERN.sub("", text).strip()
        # Remove trailing punctuation artifacts
        result = re.sub(r"[,.\s]+$", "", result).strip()
        return result

    @staticmethod
    def _fuzzy_match(
        query: str,
        alias_dict: dict[str, str],
        threshold: float = _FUZZY_THRESHOLD,
    ) -> Optional[str]:
        """
        Find the best fuzzy match for a query in the alias dictionary.

        Uses difflib.SequenceMatcher (stdlib, no external dependencies).
        Returns the canonical value if match ratio >= threshold, else None.
        """
        best_ratio = 0.0
        best_canonical = None

        for alias, canonical in alias_dict.items():
            ratio = SequenceMatcher(None, query, alias).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_canonical = canonical

        if best_ratio >= threshold:
            return best_canonical

        return None
