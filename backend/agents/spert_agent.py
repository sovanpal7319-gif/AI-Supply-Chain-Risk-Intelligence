"""
SPERT Agent — Joint NER + Relation Extraction Middleware

Production-grade agent that uses the SpERT (Span-based Entity and Relation
Transformer) model to jointly extract named entities and relations from
supply-chain disruption text.

Pipeline position:
  Disruption Classification (existing)
  → **SPERT Inference** (this agent)
  → Validation Layer
  → Dynamic Graph Builder
  → Neo4j

If no fine-tuned SPERT checkpoint is available, falls back to a robust
rule-based NER+RE extractor so the pipeline works end-to-end on day one.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

# ── Resolve SPERT imports ────────────────────────────────────────────────────
# The upstream SPERT code uses bare `from spert import ...` imports.
# We add the spert/ repo root to sys.path so those resolve without edits.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # ET/
_SPERT_ROOT = _PROJECT_ROOT / "spert"
if str(_SPERT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPERT_ROOT))


# ── Known supply-chain entity dictionaries (rule-based fallback) ─────────────

_COMPANY_PATTERNS: dict[str, str] = {
    # Semiconductors
    "tsmc": "TSMC", "taiwan semiconductor": "TSMC",
    "samsung": "Samsung", "intel": "Intel", "apple": "Apple",
    "nvidia": "NVIDIA", "tesla": "Tesla", "amd": "AMD",
    "qualcomm": "Qualcomm", "foxconn": "Foxconn",
    "google": "Google", "microsoft": "Microsoft", "sony": "Sony",
    "toyota": "Toyota", "bmw": "BMW", "volkswagen": "Volkswagen",
    "ford": "Ford", "bosch": "Bosch", "basf": "BASF",
    "asml": "ASML", "sk hynix": "SK Hynix", "micron": "Micron",
    "broadcom": "Broadcom", "mediatek": "MediaTek",
    "panasonic": "Panasonic", "lg chem": "LG Chem",
    "pfizer": "Pfizer", "bayer": "Bayer",
    "caterpillar": "Caterpillar", "3m": "3M", "honeywell": "Honeywell",
    "siemens": "Siemens", "ge": "GE", "general electric": "GE",
    "rio tinto": "Rio Tinto", "bhp": "BHP", "vale": "Vale",
    "dow": "Dow Chemical", "glencore": "Glencore", "albemarle": "Albemarle",
    "lg electronics": "LG Electronics", "huawei": "Huawei",
    "honda": "Honda", "hyundai": "Hyundai", "denso": "Denso",
    "continental": "Continental", "hp": "HP", "dell": "Dell", "lenovo": "Lenovo",
    # Oil & Gas
    "saudi aramco": "Saudi Aramco", "aramco": "Saudi Aramco",
    "exxonmobil": "ExxonMobil", "exxon mobil": "ExxonMobil", "exxon": "ExxonMobil",
    "shell": "Shell", "royal dutch shell": "Shell",
    "bp": "BP", "british petroleum": "BP",
    "chevron": "Chevron", "totalenergies": "TotalEnergies",
    "opec": "OPEC", "opec+": "OPEC",
}

_LOGISTICS_PATTERNS: dict[str, str] = {
    "maersk": "Maersk", "a.p. moller-maersk": "Maersk",
    "fedex": "FedEx", "federal express": "FedEx",
    "ups": "UPS", "united parcel service": "UPS",
    "dhl": "DHL", "cma cgm": "CMA CGM",
    "msc": "MSC", "cosco": "COSCO", "cosco shipping": "COSCO",
    "hapag-lloyd": "Hapag-Lloyd", "hapag lloyd": "Hapag-Lloyd",
}

_INDUSTRY_PATTERNS: dict[str, str] = {
    "semiconductor industry": "Semiconductor", "semiconductor sector": "Semiconductor",
    "automotive industry": "Automotive", "auto industry": "Automotive",
    "oil and gas": "Oil & Gas", "oil & gas": "Oil & Gas",
    "energy sector": "Energy", "energy industry": "Energy",
    "pharmaceutical industry": "Pharmaceutical",
    "shipping industry": "Shipping", "logistics industry": "Logistics",
    "chemical industry": "Chemicals", "mining industry": "Mining",
    "electronics industry": "Electronics",
}

_RAW_MATERIAL_PATTERNS: dict[str, str] = {
    "lithium": "Lithium", "cobalt": "Cobalt", "nickel": "Nickel",
    "palladium": "Palladium", "neon gas": "Neon Gas", "neon": "Neon Gas",
    "silicon": "Silicon", "silicon wafer": "Silicon Wafers",
    "rare earth": "Rare Earth Minerals", "rare earths": "Rare Earth Minerals",
    "crude oil": "Crude Oil", "petroleum": "Crude Oil",
    "natural gas": "Natural Gas", "lng": "LNG",
    "copper": "Copper", "aluminum": "Aluminum", "steel": "Steel",
}

_SUPPLIER_KEYWORDS = {
    "supplier", "suppliers", "vendor", "vendors",
    "manufacturer", "manufacturers", "producer", "producers",
}

_EVENT_PATTERNS: dict[str, str] = {
    "earthquake": "Earthquake", "fire": "Fire", "flood": "Flooding",
    "flooding": "Flooding", "typhoon": "Typhoon", "hurricane": "Hurricane",
    "cyclone": "Cyclone", "cyber attack": "Cyber Attack",
    "cyberattack": "Cyber Attack", "hack": "Cyber Attack",
    "ransomware": "Ransomware", "sanction": "Sanctions",
    "sanctions": "Sanctions", "embargo": "Embargo",
    "strike": "Strike", "labor strike": "Labor Strike",
    "pandemic": "Pandemic", "covid": "COVID-19",
    "explosion": "Explosion", "shortage": "Shortage",
    "shutdown": "Shutdown", "war": "War", "conflict": "Conflict",
    "tariff": "Tariffs", "blockade": "Blockade",
    "disruption": "Disruption", "disrupted": "Disruption",
    "drought": "Drought", "volcano": "Volcanic Eruption",
}

_COUNTRY_PATTERNS: dict[str, str] = {
    "taiwan": "Taiwan", "china": "China", "japan": "Japan",
    "south korea": "South Korea", "korea": "South Korea",
    "united states": "United States", "usa": "United States",
    "us": "United States", "germany": "Germany",
    "netherlands": "Netherlands", "india": "India",
    "brazil": "Brazil", "australia": "Australia",
    "singapore": "Singapore", "vietnam": "Vietnam",
    "thailand": "Thailand", "mexico": "Mexico",
    "united kingdom": "United Kingdom", "uk": "United Kingdom",
    "switzerland": "Switzerland", "france": "France",
    "italy": "Italy", "canada": "Canada", "indonesia": "Indonesia",
    "malaysia": "Malaysia", "philippines": "Philippines",
    "russia": "Russia", "ukraine": "Ukraine", "israel": "Israel",
    "saudi arabia": "Saudi Arabia", "uae": "UAE",
}

_PORT_PATTERNS: dict[str, str] = {
    "port of shanghai": "Port of Shanghai",
    "port of singapore": "Port of Singapore",
    "port of rotterdam": "Port of Rotterdam",
    "port of los angeles": "Port of Los Angeles",
    "port of long beach": "Port of Long Beach",
    "suez canal": "Suez Canal", "panama canal": "Panama Canal",
    "strait of malacca": "Strait of Malacca",
    "strait of hormuz": "Strait of Hormuz",
    "bab el-mandeb": "Bab el-Mandeb Strait",
    "port of busan": "Port of Busan",
    "port of shenzhen": "Port of Shenzhen",
    "port of hamburg": "Port of Hamburg",
}

_PRODUCT_PATTERNS: dict[str, str] = {
    "semiconductor": "Semiconductors", "semiconductors": "Semiconductors",
    "chip": "Semiconductors", "chips": "Semiconductors", "microchip": "Semiconductors",
    "battery": "Batteries", "batteries": "Batteries",
    "ev battery": "EV Batteries", "ev batteries": "EV Batteries",
    "oil": "Crude Oil", "wti": "Crude Oil", "wti crude": "Crude Oil",
    "vaccine": "Vaccines", "pharmaceutical": "Pharmaceuticals",
    "auto parts": "Auto Parts", "display": "Displays",
    "memory": "Memory Chips", "dram": "DRAM", "nand": "NAND Flash",
    "gpu": "GPUs", "cpu": "CPUs",
}

# ── Heuristic relation patterns ──────────────────────────────────────────────

_RELATION_TRIGGERS: dict[str, list[str]] = {
    "AFFECTS": ["affected", "affects", "disrupted", "disrupts", "hit", "hits",
                "damaged", "damages", "impacted", "impacts"],
    "OCCURS_IN": ["in", "at", "across", "throughout", "near"],
    "DEPENDS_ON": ["depends on", "dependent on", "relies on", "reliant on",
                   "sourced from", "supplied by"],
    "SUPPLIES_TO": ["supplies", "supplies to", "ships to", "exports to",
                    "provides", "delivers"],
    "DELAYS": ["delayed", "delays", "slowed", "postponed", "halted"],
    "IMPACTS": ["impacts", "impacted", "hurts", "threatens", "endangered"],
    "PRODUCES": ["produces", "manufactures", "makes", "fabricates", "builds"],
    "SHIPS_TO": ["ships to", "transports", "delivers to", "exports to"],
    "LOCATED_IN": ["located in", "based in", "headquartered in", "operates in"],
}

_CUSTOMER_KEYWORDS = {
    "customer", "customers", "buyer", "buyers",
    "client", "clients", "consumer", "consumers",
}


class SpertAgent:
    """
    Joint NER + RE extraction agent using SpERT.

    Loads a pretrained SpERT model for inference.  If no checkpoint exists,
    falls back to rule-based extraction using domain dictionaries.

    Usage::

        agent = SpertAgent()
        result = agent.extract("Flooding in China disrupted Tesla battery suppliers")
        # → {"entities": [...], "relations": [...]}
    """

    def __init__(self):
        from backend.config import settings
        from backend.services.entity_canonicalization import EntityCanonicalizationService
        self._canon = EntityCanonicalizationService()

        self._model = None
        self._tokenizer = None
        self._input_reader = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_spert_model = False

        # Configuration
        self._types_path = str(_PROJECT_ROOT / settings.spert_types_path)
        self._model_path = str(_PROJECT_ROOT / settings.spert_model_path)
        self._confidence_threshold = settings.spert_confidence_threshold
        self._max_span_size = settings.spert_max_span_size
        self._rel_filter_threshold = settings.spert_rel_filter_threshold

        # Attempt to load fine-tuned SPERT model
        if settings.use_spert and os.path.isdir(self._model_path):
            try:
                self._load_spert_model()
                self._use_spert_model = True
                logger.info(
                    "🧠 SPERT Agent initialized (model loaded from {})",
                    self._model_path,
                )
            except Exception as exc:
                logger.warning("⚠️ SPERT model failed to load — using rule-based fallback: {}", exc)
                logger.info("🧠 SPERT Agent initialized (rule-based mode)")
        else:
            if not settings.use_spert:
                logger.info("🧠 SPERT Agent initialized (disabled via config)")
            else:
                logger.info(
                    "🧠 SPERT Agent initialized (no checkpoint at {} — rule-based mode)",
                    self._model_path,
                )

    # ── Public API ───────────────────────────────────────────────────────

    def extract(self, text: str) -> dict:
        """
        Extract entities and relations from text.

        Parameters
        ----------
        text : str
            Raw sentence or short paragraph of disruption news.

        Returns
        -------
        dict
            Standardized extraction result::

                {
                    "entities": [
                        {"text": "...", "type": "...", "start": int, "end": int, "confidence": float},
                    ],
                    "relations": [
                        {"head": "...", "type": "...", "tail": "...", "confidence": float},
                    ],
                }
        """
        if not text or not text.strip():
            return {"entities": [], "relations": []}

        if self._use_spert_model:
            try:
                return self._extract_with_spert(text)
            except Exception as exc:
                logger.error("SPERT inference error — falling back to rules: {}", exc)

        return self._rule_based_extract(text)

    # ── SPERT model inference ────────────────────────────────────────────

    def _load_spert_model(self) -> None:
        """Load SpERT model, tokenizer, and input reader."""
        from transformers import BertTokenizer, BertConfig
        from spert.models import SpERT
        from spert.input_reader import JsonPredictionInputReader

        self._tokenizer = BertTokenizer.from_pretrained(self._model_path)

        # Build input reader to get type mappings
        self._input_reader = JsonPredictionInputReader(
            types_path=self._types_path,
            tokenizer=self._tokenizer,
            max_span_size=self._max_span_size,
        )

        # Load model
        config = BertConfig.from_pretrained(self._model_path)
        config.spert_version = SpERT.VERSION

        self._model = SpERT.from_pretrained(
            self._model_path,
            config=config,
            cls_token=self._tokenizer.convert_tokens_to_ids("[CLS]"),
            relation_types=self._input_reader.relation_type_count - 1,
            entity_types=self._input_reader.entity_type_count,
            max_pairs=1000,
            prop_drop=0.1,
            size_embedding=25,
            freeze_transformer=False,
        )
        self._model.to(self._device)
        self._model.eval()

    def _extract_with_spert(self, text: str) -> dict:
        """Run full SPERT inference pipeline on text."""
        from spert import prediction as spert_prediction
        from spert import sampling as spert_sampling
        from spert import util as spert_util
        from spert.input_reader import JsonPredictionInputReader
        from torch.utils.data import DataLoader

        # Tokenize at word level
        tokens = text.split()

        # Write temp prediction JSON
        pred_input = [{"tokens": tokens}]
        tmp_path = os.path.join(tempfile.gettempdir(), "spert_pred_input.json")
        with open(tmp_path, "w") as f:
            json.dump(pred_input, f)

        # Build fresh input reader and read dataset
        input_reader = JsonPredictionInputReader(
            types_path=self._types_path,
            tokenizer=self._tokenizer,
            max_span_size=self._max_span_size,
        )
        dataset = input_reader.read(tmp_path, "predict")

        # Create data loader
        from spert.entities import Dataset
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            drop_last=False, num_workers=0,
            collate_fn=spert_sampling.collate_fn_padding,
        )

        pred_entities_all = []
        pred_relations_all = []

        with torch.no_grad():
            for batch in data_loader:
                batch = spert_util.to_device(batch, self._device)

                entity_clf, rel_clf, rels = self._model(
                    encodings=batch["encodings"],
                    context_masks=batch["context_masks"],
                    entity_masks=batch["entity_masks"],
                    entity_sizes=batch["entity_sizes"],
                    entity_spans=batch["entity_spans"],
                    entity_sample_masks=batch["entity_sample_masks"],
                    inference=True,
                )

                batch_pred_entities, batch_pred_relations = spert_prediction.convert_predictions(
                    entity_clf, rel_clf, rels, batch,
                    self._rel_filter_threshold, input_reader,
                )
                pred_entities_all.extend(batch_pred_entities)
                pred_relations_all.extend(batch_pred_relations)

        # Convert to standardized format
        return self._convert_spert_output(
            tokens, pred_entities_all, pred_relations_all, input_reader,
        )

    def _convert_spert_output(
        self,
        tokens: list[str],
        pred_entities: list,
        pred_relations: list,
        input_reader,
    ) -> dict:
        """Convert raw SPERT predictions to standardized dict."""
        entities = []
        entity_text_map: dict[tuple[int, int], str] = {}

        for sample_entities in pred_entities:
            for entity in sample_entities:
                start, end = entity[0], entity[1]
                entity_type = entity[2]
                score = entity[3] if len(entity) > 3 else 1.0

                # Reconstruct text from token indices
                # SPERT uses sub-token spans; map back to word tokens
                entity_tokens = tokens[start:end]
                entity_text = " ".join(entity_tokens) if entity_tokens else ""
                type_name = entity_type.identifier if hasattr(entity_type, "identifier") else str(entity_type)

                if entity_text and type_name != "None":
                    entities.append({
                        "text": entity_text,
                        "type": type_name,
                        "start": start,
                        "end": end,
                        "confidence": round(score, 4),
                    })
                    entity_text_map[(start, end)] = entity_text

        relations = []
        for sample_relations in pred_relations:
            for rel in sample_relations:
                head_info, tail_info = rel[0], rel[1]
                rel_type = rel[2]
                score = rel[3] if len(rel) > 3 else 1.0

                head_start, head_end = head_info[0], head_info[1]
                tail_start, tail_end = tail_info[0], tail_info[1]

                head_text = entity_text_map.get(
                    (head_start, head_end),
                    " ".join(tokens[head_start:head_end]),
                )
                tail_text = entity_text_map.get(
                    (tail_start, tail_end),
                    " ".join(tokens[tail_start:tail_end]),
                )

                type_name = rel_type.identifier if hasattr(rel_type, "identifier") else str(rel_type)

                if head_text and tail_text and type_name != "None":
                    relations.append({
                        "head": head_text,
                        "type": type_name,
                        "tail": tail_text,
                        "confidence": round(score, 4),
                    })

        return {"entities": entities, "relations": relations}

    # ── Rule-based fallback ──────────────────────────────────────────────

    def _rule_based_extract(self, text: str) -> dict:
        """
        Domain-aware rule-based NER + RE fallback.

        Uses pattern dictionaries and positional heuristics to extract
        entities and infer relations when no SPERT model is available.
        """
        text_lower = text.lower()
        tokens = text.split()

        entities: list[dict] = []
        found_entities: dict[str, dict] = {}  # text → entity dict

        # ── NER: Extract entities by pattern matching ────────────────

        # Companies
        for pattern, canonical in _COMPANY_PATTERNS.items():
            if pattern in text_lower and canonical not in found_entities:
                start = self._find_token_index(tokens, pattern)
                entities.append({
                    "text": canonical,
                    "type": "COMPANY",
                    "start": start,
                    "end": start + len(pattern.split()),
                    "confidence": 0.85,
                })
                found_entities[canonical] = entities[-1]

        # Countries
        for pattern, canonical in _COUNTRY_PATTERNS.items():
            if pattern in text_lower and canonical not in found_entities:
                # Avoid false positives: "us" should only match as standalone
                if pattern == "us" and not re.search(r'\bus\b', text_lower):
                    continue
                start = self._find_token_index(tokens, pattern)
                entities.append({
                    "text": canonical,
                    "type": "COUNTRY",
                    "start": start,
                    "end": start + len(pattern.split()),
                    "confidence": 0.85,
                })
                found_entities[canonical] = entities[-1]

        # Events
        for pattern, canonical in _EVENT_PATTERNS.items():
            if pattern in text_lower and canonical not in found_entities:
                start = self._find_token_index(tokens, pattern)
                entities.append({
                    "text": canonical,
                    "type": "EVENT",
                    "start": start,
                    "end": start + len(pattern.split()),
                    "confidence": 0.80,
                })
                found_entities[canonical] = entities[-1]

        # Ports
        for pattern, canonical in _PORT_PATTERNS.items():
            if pattern in text_lower and canonical not in found_entities:
                start = self._find_token_index(tokens, pattern)
                entities.append({
                    "text": canonical,
                    "type": "PORT",
                    "start": start,
                    "end": start + len(pattern.split()),
                    "confidence": 0.80,
                })
                found_entities[canonical] = entities[-1]

        # Products / Commodities
        for pattern, canonical in _PRODUCT_PATTERNS.items():
            if pattern in text_lower and canonical not in found_entities:
                start = self._find_token_index(tokens, pattern)
                entities.append({
                    "text": canonical,
                    "type": "PRODUCT",
                    "start": start,
                    "end": start + len(pattern.split()),
                    "confidence": 0.75,
                })
                found_entities[canonical] = entities[-1]

        # Supplier (contextual: if "supplier" keyword near a company)
        for kw in _SUPPLIER_KEYWORDS:
            if kw in text_lower:
                # Look for companies near the keyword → mark them as SUPPLIER
                for ent in list(entities):
                    if ent["type"] == "COMPANY":
                        # Check proximity (within 5 tokens)
                        kw_idx = self._find_token_index(tokens, kw)
                        if abs(ent["start"] - kw_idx) <= 5:
                            # Add a SUPPLIER entity variant
                            sup_text = ent["text"] + " (Supplier)"
                            if sup_text not in found_entities:
                                entities.append({
                                    "text": ent["text"],
                                    "type": "SUPPLIER",
                                    "start": ent["start"],
                                    "end": ent["end"],
                                    "confidence": 0.70,
                                })
                                found_entities[sup_text] = entities[-1]
                break  # one supplier pass is enough

        # Customers (contextual: if "customer" keyword near a company)
        for kw in _CUSTOMER_KEYWORDS:
            if kw in text_lower:
                for ent in list(entities):
                    if ent["type"] == "COMPANY":
                        kw_idx = self._find_token_index(tokens, kw)
                        if abs(ent["start"] - kw_idx) <= 5:
                            cust_text = ent["text"] + " (Customer)"
                            if cust_text not in found_entities:
                                entities.append({
                                    "text": ent["text"], "type": "CUSTOMER",
                                    "start": ent["start"], "end": ent["end"],
                                    "confidence": 0.70,
                                })
                                found_entities[cust_text] = entities[-1]
                break

        # Logistics Providers
        for pattern, canonical in _LOGISTICS_PATTERNS.items():
            if pattern in text_lower and canonical not in found_entities:
                start = self._find_token_index(tokens, pattern)
                entities.append({
                    "text": canonical, "type": "LOGISTICS_PROVIDER",
                    "start": start, "end": start + len(pattern.split()),
                    "confidence": 0.80,
                })
                found_entities[canonical] = entities[-1]

        # Industries
        for pattern, canonical in _INDUSTRY_PATTERNS.items():
            if pattern in text_lower and canonical not in found_entities:
                start = self._find_token_index(tokens, pattern)
                entities.append({
                    "text": canonical, "type": "INDUSTRY",
                    "start": start, "end": start + len(pattern.split()),
                    "confidence": 0.75,
                })
                found_entities[canonical] = entities[-1]

        # Raw Materials
        for pattern, canonical in _RAW_MATERIAL_PATTERNS.items():
            if pattern in text_lower and canonical not in found_entities:
                start = self._find_token_index(tokens, pattern)
                entities.append({
                    "text": canonical, "type": "RAW_MATERIAL",
                    "start": start, "end": start + len(pattern.split()),
                    "confidence": 0.75,
                })
                found_entities[canonical] = entities[-1]

        # ── RE: Infer relations from entity co-occurrence ────────────

        relations: list[dict] = []
        events = [e for e in entities if e["type"] == "EVENT"]
        companies = [e for e in entities if e["type"] in ("COMPANY", "SUPPLIER", "CUSTOMER")]
        countries = [e for e in entities if e["type"] == "COUNTRY"]
        regions = [e for e in entities if e["type"] in ("REGION", "PORT")]
        products = [e for e in entities if e["type"] in ("PRODUCT", "RAW_MATERIAL")]
        logistics = [e for e in entities if e["type"] == "LOGISTICS_PROVIDER"]

        # EVENT → AFFECTS → COMPANY
        for evt in events:
            for comp in companies:
                if self._has_trigger(text_lower, "AFFECTS"):
                    relations.append({
                        "head": evt["text"], "type": "AFFECTS",
                        "tail": comp["text"], "confidence": 0.75,
                    })
                elif self._has_trigger(text_lower, "IMPACTS"):
                    relations.append({
                        "head": evt["text"], "type": "IMPACTS",
                        "tail": comp["text"], "confidence": 0.70,
                    })
                elif self._has_trigger(text_lower, "DELAYS"):
                    relations.append({
                        "head": evt["text"], "type": "DELAYS",
                        "tail": comp["text"], "confidence": 0.70,
                    })

        # EVENT → OCCURS_IN → COUNTRY/REGION
        for evt in events:
            for loc in countries + regions:
                relations.append({
                    "head": evt["text"], "type": "OCCURS_IN",
                    "tail": loc["text"], "confidence": 0.80,
                })

        # COMPANY → DEPENDS_ON → PRODUCT/RAW_MATERIAL
        for comp in companies:
            for prod in products:
                relations.append({
                    "head": comp["text"], "type": "DEPENDS_ON",
                    "tail": prod["text"], "confidence": 0.65,
                })

        # COMPANY → LOCATED_IN → COUNTRY
        if self._has_trigger(text_lower, "LOCATED_IN"):
            for comp in companies:
                for ctry in countries:
                    relations.append({
                        "head": comp["text"], "type": "LOCATED_IN",
                        "tail": ctry["text"], "confidence": 0.70,
                    })

        # COMPANY → PRODUCES → PRODUCT (if trigger present)
        if self._has_trigger(text_lower, "PRODUCES"):
            for comp in companies:
                for prod in products:
                    relations.append({
                        "head": comp["text"], "type": "PRODUCES",
                        "tail": prod["text"], "confidence": 0.65,
                    })

        # LOGISTICS_PROVIDER → SHIPS_TO → COUNTRY/PORT
        for lp in logistics:
            for loc in countries + regions:
                if self._has_trigger(text_lower, "SHIPS_TO"):
                    relations.append({
                        "head": lp["text"], "type": "SHIPS_TO",
                        "tail": loc["text"], "confidence": 0.65,
                    })

        # COMPANY (supplier) → SUPPLIES_TO → COMPANY
        suppliers = [e for e in entities if e["type"] == "SUPPLIER"]
        non_supplier_companies = [e for e in companies if e["type"] == "COMPANY"]
        for sup in suppliers:
            for comp in non_supplier_companies:
                if sup["text"] != comp["text"]:
                    relations.append({
                        "head": sup["text"], "type": "SUPPLIES_TO",
                        "tail": comp["text"], "confidence": 0.65,
                    })

        # ── Canonicalize all entity names ────────────────────────────
        for ent in entities:
            canonical = self._canon.canonicalize(ent["text"], ent["type"])
            if canonical:
                ent["text"] = canonical
        # Update relation head/tail to use canonical names
        for rel in relations:
            h = self._canon.canonicalize(rel["head"], "COMPANY")
            t = self._canon.canonicalize(rel["tail"], "COMPANY")
            if h:
                rel["head"] = h
            if t:
                rel["tail"] = t

        logger.info(
            "SPERT Agent (rules) ✅ Extracted {} entities, {} relations",
            len(entities), len(relations),
        )

        return {"entities": entities, "relations": relations}

    # ── Utility methods ──────────────────────────────────────────────────

    @staticmethod
    def _find_token_index(tokens: list[str], pattern: str) -> int:
        """Find the approximate token index for a pattern in the token list."""
        pattern_lower = pattern.lower()
        pattern_tokens = pattern_lower.split()

        for i in range(len(tokens)):
            if tokens[i].lower().startswith(pattern_tokens[0]):
                # Check multi-word match
                match = True
                for j, pt in enumerate(pattern_tokens):
                    if i + j >= len(tokens) or not tokens[i + j].lower().startswith(pt):
                        match = False
                        break
                if match:
                    return i

        # Fallback: return 0
        return 0

    @staticmethod
    def _has_trigger(text_lower: str, relation_type: str) -> bool:
        """Check if text contains trigger words for a relation type."""
        triggers = _RELATION_TRIGGERS.get(relation_type, [])
        return any(t in text_lower for t in triggers)