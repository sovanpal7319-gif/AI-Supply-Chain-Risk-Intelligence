"""
SPERT Validation Layer — Post-processing for NER+RE Extraction

Cleans, deduplicates, and normalizes SPERT output before it reaches
the dynamic graph builder.  Every entity and relation is validated
to prevent null nodes, empty spans, and duplicate edges in Neo4j.

Pipeline:
  raw SPERT output
  → remove None / unknown entities
  → remove empty spans
  → normalize entity labels
  → merge duplicate entities
  → confidence threshold filter
  → deduplicate relations
  → return clean dict
"""

from __future__ import annotations

from typing import Optional

from loguru import logger


# ── Valid domain types (must match supply_chain_types.json) ───────────────────

_VALID_ENTITY_TYPES = {
    "COMPANY", "COUNTRY", "EVENT", "SUPPLIER", "CUSTOMER",
    "PORT", "REGION", "PRODUCT", "INDUSTRY", "RAW_MATERIAL",
    "LOGISTICS_PROVIDER",
}

_VALID_RELATION_TYPES = {
    "AFFECTS", "OCCURS_IN", "DEPENDS_ON", "SUPPLIES_TO",
    "DELAYS", "IMPACTS", "PRODUCES", "SHIPS_TO",
    "MANUFACTURES", "LOCATED_IN", "ALTERNATIVE_TO",
}

# Semantic constraints: which entity types can be head/tail for each relation
_RELATION_SEMANTIC_RULES: dict[str, dict] = {
    "SUPPLIES_TO":  {"head": {"COMPANY", "SUPPLIER"}, "tail": {"COMPANY", "CUSTOMER"}},
    "AFFECTS":      {"head": {"EVENT"}, "tail": {"COMPANY", "SUPPLIER", "CUSTOMER"}},
    "IMPACTS":      {"head": {"EVENT"}, "tail": {"COMPANY", "SUPPLIER", "CUSTOMER"}},
    "OCCURS_IN":    {"head": {"EVENT"}, "tail": {"COUNTRY", "REGION", "PORT"}},
    "DEPENDS_ON":   {"head": {"COMPANY"}, "tail": {"PRODUCT", "RAW_MATERIAL", "COMPANY"}},
    "LOCATED_IN":   {"head": {"COMPANY"}, "tail": {"COUNTRY", "REGION"}},
    "PRODUCES":     {"head": {"COMPANY"}, "tail": {"PRODUCT"}},
    "SHIPS_TO":     {"head": {"LOGISTICS_PROVIDER", "COMPANY"}, "tail": {"COUNTRY", "PORT", "COMPANY"}},
}


class SpertValidationService:
    """
    Validates and cleans SPERT extraction output.

    All methods are pure functions (no side effects) and can be called
    independently or via the convenience ``validate()`` orchestrator.
    """

    def __init__(self, confidence_threshold: float = 0.4):
        self._confidence_threshold = confidence_threshold
        from backend.services.entity_canonicalization import EntityCanonicalizationService
        self._canon = EntityCanonicalizationService()
        logger.info(
            "🔍 SPERT Validation Service initialized (threshold={})",
            self._confidence_threshold,
        )

    # ── Public orchestrator ──────────────────────────────────────────────

    def validate(self, spert_output: dict) -> dict:
        """
        Run the full validation pipeline on raw SPERT output.

        Parameters
        ----------
        spert_output : dict
            Raw output from ``SpertAgent.extract()`` with keys
            ``entities`` and ``relations``.

        Returns
        -------
        dict
            Cleaned output with the same structure.
        """
        entities = list(spert_output.get("entities", []))
        relations = list(spert_output.get("relations", []))

        original_e = len(entities)
        original_r = len(relations)

        # 1. Remove None / unknown entities
        entities = self._remove_none_entities(entities)

        # 2. Remove empty spans
        entities = self._remove_empty_spans(entities)

        # 3. Normalize entity labels
        entities = self._normalize_entity_labels(entities)

        # 4. Merge duplicate entities
        entities = self._merge_duplicate_entities(entities)

        # 5. Confidence threshold filter
        entities = self._filter_by_confidence(entities)

        # 6. Canonicalize entity names
        entities = self._canonicalize_entities(entities)

        # 7. Cross-entity deduplication (same text, different types → keep primary)
        entities = self._cross_dedup_entities(entities)

        # 8. Rebuild valid entity text set for relation filtering
        valid_texts = {e["text"] for e in entities}

        # 9. Remove relations referencing removed entities
        relations = [
            r for r in relations
            if r.get("head") in valid_texts and r.get("tail") in valid_texts
        ]

        # 10. Remove self-loop relations (head == tail)
        relations = [r for r in relations if r.get("head") != r.get("tail")]

        # 11. Remove relations with invalid types
        relations = self._filter_valid_relations(relations)

        # 12. Semantic validation (entity type constraints per relation)
        relations = self._validate_relation_semantics(relations, entities)

        # 13. Deduplicate relations
        relations = self._deduplicate_relations(relations)

        # 14. Confidence filter on relations
        relations = self._filter_relations_by_confidence(relations)

        logger.info(
            "Validation ✅ entities: {} → {}, relations: {} → {}",
            original_e, len(entities), original_r, len(relations),
        )

        return {"entities": entities, "relations": relations}

    # ── Entity validators ────────────────────────────────────────────────

    @staticmethod
    def _remove_none_entities(entities: list[dict]) -> list[dict]:
        """Remove entities with type 'None' or 'unknown'."""
        return [
            e for e in entities
            if e.get("type", "None") not in ("None", "none", "unknown", "Unknown", "")
            and e.get("type", "") in _VALID_ENTITY_TYPES
        ]

    @staticmethod
    def _remove_empty_spans(entities: list[dict]) -> list[dict]:
        """Remove entities with empty or whitespace-only text."""
        return [
            e for e in entities
            if e.get("text", "").strip()
        ]

    @staticmethod
    def _normalize_entity_labels(entities: list[dict]) -> list[dict]:
        """Strip whitespace and apply consistent casing to entity text."""
        normalized = []
        for e in entities:
            e = dict(e)  # shallow copy
            text = e.get("text", "").strip()
            # Title-case for proper nouns; keep all-caps acronyms
            if text and not text.isupper():
                text = text.title()
            e["text"] = text
            e["type"] = e.get("type", "").upper().strip()
            normalized.append(e)
        return normalized

    @staticmethod
    def _merge_duplicate_entities(entities: list[dict]) -> list[dict]:
        """
        Merge duplicate entities (case-insensitive text match).
        Keeps the entry with the highest confidence score.
        """
        seen: dict[tuple[str, str], dict] = {}  # (lower_text, type) → entity

        for e in entities:
            key = (e["text"].lower(), e["type"])
            existing = seen.get(key)
            if existing is None:
                seen[key] = e
            else:
                # Keep higher confidence
                if e.get("confidence", 0.0) > existing.get("confidence", 0.0):
                    seen[key] = e

        return list(seen.values())

    def _filter_by_confidence(self, entities: list[dict]) -> list[dict]:
        """Remove entities below the confidence threshold."""
        return [
            e for e in entities
            if e.get("confidence", 1.0) >= self._confidence_threshold
        ]

    # ── Relation validators ──────────────────────────────────────────────

    @staticmethod
    def _filter_valid_relations(relations: list[dict]) -> list[dict]:
        """Remove relations with invalid or empty types."""
        return [
            r for r in relations
            if r.get("type", "") in _VALID_RELATION_TYPES
            and r.get("head", "").strip()
            and r.get("tail", "").strip()
        ]

    @staticmethod
    def _deduplicate_relations(relations: list[dict]) -> list[dict]:
        """Remove exact duplicate relations (head, type, tail)."""
        seen: set[tuple[str, str, str]] = set()
        unique = []

        for r in relations:
            key = (r["head"], r["type"], r["tail"])
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique

    def _filter_relations_by_confidence(self, relations: list[dict]) -> list[dict]:
        """Remove relations below the confidence threshold."""
        return [
            r for r in relations
            if r.get("confidence", 1.0) >= self._confidence_threshold
        ]

    # ── Canonicalization & cross-dedup ────────────────────────────────────

    def _canonicalize_entities(self, entities: list[dict]) -> list[dict]:
        """Canonicalize all entity names through the canonicalization service."""
        result = []
        for e in entities:
            canonical = self._canon.canonicalize(e["text"], e.get("type", "COMPANY"))
            if canonical:
                e = dict(e)
                e["text"] = canonical
                result.append(e)
            # else: rejected by canonicalization (blocklisted)
        return result

    @staticmethod
    def _cross_dedup_entities(entities: list[dict]) -> list[dict]:
        """
        Cross-type deduplication: if same text appears as COMPANY and SUPPLIER,
        keep only the primary type (priority: COMPANY > SUPPLIER > CUSTOMER).
        """
        type_priority = {
            "COMPANY": 10, "SUPPLIER": 5, "CUSTOMER": 5,
            "LOGISTICS_PROVIDER": 8, "EVENT": 9, "COUNTRY": 9,
            "REGION": 7, "PORT": 7, "PRODUCT": 6,
            "RAW_MATERIAL": 6, "INDUSTRY": 4,
        }
        seen: dict[str, dict] = {}  # lower_text → best entity
        for e in entities:
            key = e["text"].lower()
            existing = seen.get(key)
            if existing is None:
                seen[key] = e
            else:
                # Keep higher priority type
                if type_priority.get(e["type"], 0) > type_priority.get(existing["type"], 0):
                    seen[key] = e
        return list(seen.values())

    @staticmethod
    def _validate_relation_semantics(
        relations: list[dict],
        entities: list[dict],
    ) -> list[dict]:
        """Validate that relation head/tail types match semantic constraints."""
        # Build text → type mapping
        text_type: dict[str, str] = {}
        for e in entities:
            text_type[e["text"]] = e["type"]

        valid = []
        for r in relations:
            rel_type = r.get("type", "")
            rules = _RELATION_SEMANTIC_RULES.get(rel_type)
            if rules is None:
                # No semantic rules defined — allow
                valid.append(r)
                continue

            head_type = text_type.get(r.get("head", ""), "")
            tail_type = text_type.get(r.get("tail", ""), "")

            if head_type in rules["head"] and tail_type in rules["tail"]:
                valid.append(r)
            else:
                logger.debug(
                    "  Validation rejected relation: {}({}) -[{}]-> {}({})",
                    r.get("head"), head_type, rel_type, r.get("tail"), tail_type,
                )

        return valid