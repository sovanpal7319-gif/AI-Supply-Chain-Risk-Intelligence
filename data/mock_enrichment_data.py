"""
Mock Company Enrichment Dataset

Curated supplier/customer ecosystem data for major global companies.
Used by CompanyEnrichmentService in mock mode to provide realistic
graph construction data without requiring external API calls.

Each entry maps a company name (case-insensitive key) to:
  - industry: primary industry classification
  - country: headquarters country
  - suppliers: list of known upstream suppliers
  - customers: list of known downstream customers
"""

# ── Enrichment Dataset ───────────────────────────────────────────────────────
# Keys are LOWERCASE for case-insensitive lookup.

MOCK_COMPANY_DATA: dict[str, dict] = {
    # ═══════════════════════════════════════════════════════════════════════
    # TIER 1 — Full ecosystem (primary focus companies)
    # ═══════════════════════════════════════════════════════════════════════

    "tsmc": {
        "industry": "Semiconductor",
        "country": "Taiwan",
        "suppliers": [
            "ASML", "Tokyo Electron", "BHP", "Shin-Etsu Chemical",
            "Sumco", "Air Liquide", "Lam Research",
        ],
        "customers": [
            "Apple", "NVIDIA", "AMD", "Qualcomm", "MediaTek",
            "Broadcom", "Bosch", "Denso", "Continental", "Sony",
        ],
    },

    "samsung": {
        "industry": "Semiconductor",
        "country": "South Korea",
        "suppliers": [
            "ASML", "Tokyo Electron", "Vale", "Lam Research",
            "Applied Materials", "SK Materials",
        ],
        "customers": [
            "Samsung Electronics", "Google", "Tesla", "Qualcomm",
            "Xiaomi", "Hyundai", "LG Electronics",
        ],
    },

    "samsung semiconductor": {
        "industry": "Semiconductor",
        "country": "South Korea",
        "suppliers": [
            "ASML", "Tokyo Electron", "Vale", "Lam Research",
            "Applied Materials", "SK Materials",
        ],
        "customers": [
            "Samsung Electronics", "Google", "Tesla", "Qualcomm",
            "Xiaomi", "Hyundai", "LG Electronics",
        ],
    },

    "intel": {
        "industry": "Semiconductor",
        "country": "United States",
        "suppliers": [
            "ASML", "Rio Tinto", "Lam Research", "Applied Materials",
            "KLA Corporation", "Tokyo Electron",
        ],
        "customers": [
            "Microsoft", "Google", "HP", "Dell", "Lenovo",
            "Bosch", "BMW", "Ford",
        ],
    },

    "apple": {
        "industry": "Electronics",
        "country": "United States",
        "suppliers": [
            "TSMC", "Foxconn", "Pegatron", "SK Hynix", "Micron",
            "Samsung Semiconductor", "Jabil", "3M", "Maersk",
            "Infosys", "Corning", "Texas Instruments",
        ],
        "customers": [
            "Best Buy", "Amazon", "Walmart", "AT&T", "Verizon",
            "T-Mobile", "Softbank",
        ],
    },

    "nvidia": {
        "industry": "Semiconductor",
        "country": "United States",
        "suppliers": [
            "TSMC", "SK Hynix", "Micron", "Samsung Semiconductor",
            "Foxconn", "Amkor Technology",
        ],
        "customers": [
            "Microsoft", "Google", "Amazon", "Meta", "Tesla",
            "Oracle", "Dell", "HP", "Lenovo",
        ],
    },

    "tesla": {
        "industry": "Automotive",
        "country": "United States",
        "suppliers": [
            "LG Chem", "Panasonic", "CATL", "Samsung SDI",
            "NVIDIA", "Intel", "Bosch", "Continental",
            "BHP", "Glencore", "Albemarle",
        ],
        "customers": [
            "Enterprise fleet buyers", "Hertz",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # TIER 2 — Partial ecosystems
    # ═══════════════════════════════════════════════════════════════════════

    "amd": {
        "industry": "Semiconductor",
        "country": "United States",
        "suppliers": ["TSMC", "GlobalFoundries", "ASE Technology"],
        "customers": ["Microsoft", "Sony", "HP", "Dell", "Lenovo", "Google"],
    },

    "qualcomm": {
        "industry": "Semiconductor",
        "country": "United States",
        "suppliers": ["TSMC", "Samsung Semiconductor", "GlobalFoundries"],
        "customers": ["Samsung Electronics", "Xiaomi", "Oppo", "Vivo", "Motorola"],
    },

    "foxconn": {
        "industry": "Contract Manufacturing",
        "country": "Taiwan",
        "suppliers": [
            "TSMC", "Samsung Semiconductor", "Intel", "Corning",
            "Reliance Industries", "COSCO",
        ],
        "customers": ["Apple", "Sony", "Google", "Amazon", "HP", "Dell"],
    },

    "bosch": {
        "industry": "Auto Parts",
        "country": "Germany",
        "suppliers": [
            "TSMC", "Intel", "Texas Instruments", "BASF",
            "Tata Steel", "Infineon",
        ],
        "customers": ["Toyota", "BMW", "Volkswagen", "Ford", "GM", "Hyundai"],
    },

    "toyota": {
        "industry": "Automotive",
        "country": "Japan",
        "suppliers": [
            "Bosch", "Denso", "Aisin", "Tata Steel",
            "Continental", "Panasonic",
        ],
        "customers": [],
    },

    "microsoft": {
        "industry": "Electronics",
        "country": "United States",
        "suppliers": [
            "Intel", "AMD", "NVIDIA", "Pegatron", "Foxconn",
            "Micron", "Infosys",
        ],
        "customers": [],
    },

    "google": {
        "industry": "Electronics",
        "country": "United States",
        "suppliers": [
            "Intel", "Samsung Semiconductor", "NVIDIA", "Foxconn",
            "Flex", "Broadcom",
        ],
        "customers": [],
    },

    "sony": {
        "industry": "Electronics",
        "country": "Japan",
        "suppliers": ["TSMC", "AMD", "Foxconn", "SK Hynix", "Samsung Semiconductor"],
        "customers": [],
    },

    "basf": {
        "industry": "Chemicals",
        "country": "Germany",
        "suppliers": ["Glencore", "BHP", "Linde", "Air Liquide"],
        "customers": [
            "Bosch", "Continental", "BMW", "Bayer", "Pfizer",
            "Dow Chemical", "Volkswagen",
        ],
    },

    "sk hynix": {
        "industry": "Semiconductor",
        "country": "South Korea",
        "suppliers": ["ASML", "Tokyo Electron", "Lam Research"],
        "customers": ["Apple", "Samsung Electronics", "NVIDIA", "Google"],
    },

    "micron": {
        "industry": "Semiconductor",
        "country": "United States",
        "suppliers": ["ASML", "Lam Research", "Applied Materials"],
        "customers": ["Apple", "Microsoft", "NVIDIA", "Dell", "HP"],
    },

    "broadcom": {
        "industry": "Semiconductor",
        "country": "United States",
        "suppliers": ["TSMC", "ASE Technology"],
        "customers": ["Apple", "Google", "Cisco", "HP", "Dell"],
    },

    "mediatek": {
        "industry": "Semiconductor",
        "country": "Taiwan",
        "suppliers": ["TSMC", "ASE Technology"],
        "customers": ["Samsung Electronics", "Xiaomi", "Oppo", "Realme", "Vivo"],
    },

    "lg chem": {
        "industry": "Chemicals",
        "country": "South Korea",
        "suppliers": ["Glencore", "Albemarle", "SQM"],
        "customers": ["Tesla", "GM", "Ford", "Hyundai", "Volkswagen", "LG Electronics"],
    },

    "asml": {
        "industry": "Semiconductor Equipment",
        "country": "Netherlands",
        "suppliers": ["Carl Zeiss", "Trumpf", "BASF"],
        "customers": ["TSMC", "Samsung Semiconductor", "Intel", "SK Hynix", "Micron"],
    },

    "panasonic": {
        "industry": "Electronics",
        "country": "Japan",
        "suppliers": ["TSMC", "Sumitomo Metal", "Murata"],
        "customers": ["Tesla", "Toyota", "Honda"],
    },

    "maersk": {
        "industry": "Shipping",
        "country": "Denmark",
        "suppliers": [],
        "customers": ["Apple", "Samsung Electronics", "Nike", "Walmart", "Amazon"],
    },

    "bmw": {
        "industry": "Automotive",
        "country": "Germany",
        "suppliers": [
            "Bosch", "Continental", "ZF Friedrichshafen", "Siemens",
            "BASF", "Hapag-Lloyd",
        ],
        "customers": [],
    },

    "volkswagen": {
        "industry": "Automotive",
        "country": "Germany",
        "suppliers": [
            "Bosch", "Continental", "Siemens", "BASF",
            "Hapag-Lloyd", "LG Chem",
        ],
        "customers": [],
    },

    "ford": {
        "industry": "Automotive",
        "country": "United States",
        "suppliers": [
            "Bosch", "ZF Friedrichshafen", "Dow Chemical",
            "3M", "LG Chem", "Continental",
        ],
        "customers": [],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # TIER 3 — Oil & Energy companies
    # ═══════════════════════════════════════════════════════════════════════

    "saudi aramco": {
        "industry": "Oil & Gas",
        "country": "Saudi Arabia",
        "suppliers": [
            "Schlumberger", "Halliburton", "Baker Hughes",
            "Siemens", "ABB",
        ],
        "customers": [
            "ExxonMobil", "Shell", "Chevron", "BP",
            "BASF", "Dow Chemical", "Maersk", "Toyota",
            "Samsung Electronics", "Apple",
        ],
    },

    "exxonmobil": {
        "industry": "Oil & Gas",
        "country": "United States",
        "suppliers": [
            "Saudi Aramco", "Schlumberger", "Halliburton",
            "Baker Hughes", "Caterpillar",
        ],
        "customers": [
            "BASF", "Dow Chemical", "Shell", "Chevron",
            "Maersk", "FedEx", "UPS",
        ],
    },

    "shell": {
        "industry": "Oil & Gas",
        "country": "Netherlands",
        "suppliers": [
            "Saudi Aramco", "Schlumberger", "Siemens",
            "ABB", "Baker Hughes",
        ],
        "customers": [
            "BASF", "Maersk", "BMW", "Volkswagen",
            "Lufthansa", "British Airways",
        ],
    },

    "chevron": {
        "industry": "Oil & Gas",
        "country": "United States",
        "suppliers": [
            "Saudi Aramco", "Halliburton", "Schlumberger",
            "Baker Hughes", "Caterpillar",
        ],
        "customers": [
            "BASF", "Dow Chemical", "FedEx",
            "Maersk", "Ford", "GM",
        ],
    },

    "bp": {
        "industry": "Oil & Gas",
        "country": "United Kingdom",
        "suppliers": [
            "Saudi Aramco", "Schlumberger", "Halliburton",
            "Siemens", "ABB",
        ],
        "customers": [
            "BASF", "Maersk", "BMW", "Volkswagen",
            "British Airways", "Shell",
        ],
    },
}


def get_company_data(company_name: str) -> dict | None:
    """
    Look up a company in the mock dataset (case-insensitive).

    Returns the enrichment dict or None if not found.
    """
    return MOCK_COMPANY_DATA.get(company_name.strip().lower())


def list_known_companies() -> list[str]:
    """Return all company names in the mock dataset."""
    return list(MOCK_COMPANY_DATA.keys())
