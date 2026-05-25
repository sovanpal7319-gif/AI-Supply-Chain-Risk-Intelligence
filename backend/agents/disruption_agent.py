"""
Agent 1 — Disruption Monitoring Agent

Ingests raw news text and uses a fine-tuned BERT model for fast,
high-confidence disruption detection.  When BERT confidence is below
the configured threshold, falls back to Groq (LLaMA-3) for analysis.

Pipeline:
  1. BERT predicts disruption_type + severity (fast, offline)
  2. If confidence ≥ threshold → use BERT result, call Groq only for
     supplementary fields (companies, countries, summary)
  3. If confidence < threshold → delegate entirely to Groq
  4. If both fail → return safe "unknown" defaults
"""

from loguru import logger
from backend.config import settings
from backend.services.llm_service import LLMService


class DisruptionAgent:
    """Detects supply chain disruptions from news articles."""

    def __init__(self):
        self.llm = LLMService()
        self.bert = None
        self.confidence_threshold = settings.bert_confidence_threshold

        # Attempt to load fine-tuned BERT model
        try:
            from backend.services.bert_service import BERTService
            self.bert = BERTService()
            logger.info(
                "🔴 Disruption Agent initialized (BERT + Groq fallback, threshold={})",
                self.confidence_threshold,
            )
        except Exception as exc:
            logger.warning("⚠️ BERT model failed to load — using Groq only: {}", exc)
            logger.info("🔴 Disruption Agent initialized (Groq LLM only)")

    async def run(self, news_text: str) -> dict:
        """
        Analyze news text and return structured disruption data.

        Returns
        -------
        dict with keys:
            disruption_type, severity, confidence, source,
            affected_companies, affected_countries, summary
        """
        logger.info("Agent 1 ▶ Analyzing news article ({} chars)", len(news_text))

        # ── Step 1: Try BERT prediction ──────────────────────────────────
        bert_result = self._predict_with_bert(news_text)

        if bert_result and bert_result["confidence"] >= self.confidence_threshold:
            # BERT is confident — use its type/severity, get extras from Groq
            logger.info(
                "Agent 1 🧠 BERT confident ({:.1%}) — using BERT for type/severity",
                bert_result["confidence"],
            )
            result = await self._enrich_with_groq(news_text, bert_result)
            result["source"] = "bert"
            return self._finalize(result)

        # ── Step 2: Fallback to Groq ─────────────────────────────────────
        if bert_result:
            logger.info(
                "Agent 1 ⚠️ BERT low confidence ({:.1%}) — falling back to Groq",
                bert_result["confidence"],
            )
        else:
            logger.info("Agent 1 ⚠️ BERT unavailable — using Groq")

        groq_result = await self._call_groq(news_text)
        if groq_result:
            groq_result["source"] = "groq"
            groq_result.setdefault("confidence", 0.0)
            return self._finalize(groq_result)

        # ── Step 3: Both failed — safe fallback ──────────────────────────
        logger.error("Agent 1 ❌ Both BERT and Groq failed — returning unknown")
        return self._unknown_result()

    # ── Private helpers ──────────────────────────────────────────────────

    def _predict_with_bert(self, text: str) -> dict | None:
        """Run BERT inference. Returns None on any failure."""
        if self.bert is None:
            return None
        try:
            return self.bert.predict(text)
        except Exception as exc:
            logger.error("BERT inference error: {}", exc)
            return None

    async def _call_groq(self, text: str) -> dict | None:
        """Call Groq API for full disruption analysis. Returns None on failure."""
        try:
            return await self.llm.analyze_disruption(text)
        except Exception as exc:
            logger.error("Groq API error: {}", exc)
            return None

    async def _enrich_with_groq(self, text: str, bert_result: dict) -> dict:
        """
        Use BERT's type/severity but get companies, countries, summary from Groq.
        If Groq fails, return BERT result with empty supplementary fields.
        """
        groq_result = await self._call_groq(text)

        result = {
            "disruption_type": bert_result["disruption_type"],
            "severity": bert_result["severity"],
            "confidence": bert_result["confidence"],
        }

        if groq_result:
            result["affected_companies"] = groq_result.get("affected_companies", [])
            result["affected_countries"] = groq_result.get("affected_countries", [])
            result["summary"] = groq_result.get("summary", "No summary available.")
        else:
            logger.warning("Groq enrichment failed — returning BERT result without extras")
            result["affected_companies"] = []
            result["affected_countries"] = []
            result["summary"] = "BERT detected disruption but supplementary details unavailable."

        return result

    @staticmethod
    def _finalize(result: dict) -> dict:
        """Ensure all required keys exist with sensible defaults."""
        result.setdefault("disruption_type", "unknown")
        result.setdefault("severity", "medium")
        result.setdefault("confidence", 0.0)
        result.setdefault("source", "unknown")
        result.setdefault("affected_companies", [])
        result.setdefault("affected_countries", [])
        result.setdefault("summary", "No summary available.")

        logger.info(
            "Agent 1 ✅ Detected: type={}, severity={}, confidence={:.1%}, source={}, companies={}, countries={}",
            result["disruption_type"],
            result["severity"],
            result["confidence"],
            result["source"],
            len(result["affected_companies"]),
            len(result["affected_countries"]),
        )
        return result

    @staticmethod
    def _unknown_result() -> dict:
        """Return a safe fallback result when all sources fail."""
        return {
            "disruption_type": "unknown",
            "severity": "unknown",
            "confidence": 0.0,
            "source": "fallback",
            "affected_companies": [],
            "affected_countries": [],
            "summary": "Analysis failed — both BERT and Groq were unavailable.",
        }