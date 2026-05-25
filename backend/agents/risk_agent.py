"""
Agent 3 — Risk Assessment Agent (Enhanced)

Computes multi-dimensional risk scores for each company in the
disrupted supply chain.  The formula accounts for:

  • Disruption type severity multiplier
  • Industry criticality (harder-to-replace sectors score higher)
  • BERT confidence scaling (uncertain predictions are dampened)
  • Depth-aware impact weighting (direct vs. systemic)
  • Concentration risk (single-point-of-failure detection)

Each company receives three sub-scores:
  supply_risk, financial_risk, operational_risk

A composite risk_score drives the risk_level used by Agent 4.

Formula overview (per company):
──────────────────────────────────────────────────────────────
  base       = min(1.0, downstream_count / MAX_DOWNSTREAM)
  depth_w    = 1 / max(depth, 1)
  composite  = (  base        × W_BASE
                + depth_w     × W_DEPTH
                + sev_weight  × W_SEVERITY
                + dt_mult     × W_DISRUPTION_TYPE
                + ind_boost   × W_INDUSTRY  )
               × confidence_scale
               × concentration_boost

  supply_risk      = composite × depth_w
  financial_risk   = composite × sev_weight × ind_boost
  operational_risk = composite × dt_mult
  risk_score       = clamp(composite, 0, 1)
──────────────────────────────────────────────────────────────
"""

from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
# TUNABLE CONSTANTS  (edit here to recalibrate without touching logic)
# ══════════════════════════════════════════════════════════════════════════════

# ── Composite weights (must sum to 1.0) ──────────────────────────────────────
W_BASE: float = 0.20
W_DEPTH: float = 0.20
W_SEVERITY: float = 0.20
W_DISRUPTION_TYPE: float = 0.20
W_INDUSTRY: float = 0.20

# ── Severity multipliers ─────────────────────────────────────────────────────
SEVERITY_WEIGHTS: dict[str, float] = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
    "unknown": 0.5,   # BERT fallback / missing data
}

# ── Disruption type risk multipliers ─────────────────────────────────────────
# Higher = more damaging to supply chains on average
DISRUPTION_TYPE_MULTIPLIERS: dict[str, float] = {
    "natural_disaster": 1.0,
    "cyber_attack": 0.95,
    "geopolitical": 0.85,
    "logistics": 0.70,
    "operational": 0.60,
    "labor": 0.50,
    "none": 0.10,
    "unknown": 0.50,
}

# ── Industry criticality boosts ──────────────────────────────────────────────
# Harder-to-replace / longer-lead-time industries score higher
INDUSTRY_CRITICALITY: dict[str, float] = {
    "semiconductor": 1.0,
    "semiconductor equipment": 0.95,
    "pharma": 0.90,
    "chemicals": 0.80,
    "mining": 0.75,
    "electronics": 0.70,
    "auto parts": 0.65,
    "automotive": 0.60,
    "contract manufacturing": 0.55,
    "steel": 0.55,
    "industrial": 0.50,
    "shipping": 0.50,
    "it": 0.45,
    "conglomerate": 0.40,
}
DEFAULT_INDUSTRY_CRITICALITY: float = 0.50

# ── Concentration risk (single-point-of-failure) ─────────────────────────────
CONCENTRATION_THRESHOLD: int = 3   # ≥ this many path appearances → boost
CONCENTRATION_BOOST: float = 1.25  # multiplier when threshold met

# ── Downstream normalization ─────────────────────────────────────────────────
MAX_DOWNSTREAM: float = 10.0

# ── Risk-level thresholds ────────────────────────────────────────────────────
THRESHOLD_HIGH: float = 0.65
THRESHOLD_MEDIUM: float = 0.35


# ══════════════════════════════════════════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════════════════════════════════════════

class RiskAssessmentAgent:
    """Calculates multi-dimensional risk scores for companies in disrupted supply chains."""

    def __init__(self) -> None:
        logger.info("🟡 Risk Assessment Agent initialized (enhanced formula)")

    def run(
        self,
        disruption_data: dict,
        supply_chain_data: dict,
    ) -> list[dict]:
        """
        Compute risk scores for affected companies.

        Parameters
        ----------
        disruption_data : dict
            Output from DisruptionAgent.
            Required keys: severity
            Optional keys: disruption_type, confidence, source
        supply_chain_data : dict
            Output from KGQueryAgent.
            Required keys: supply_chain_paths (list[dict])

        Returns
        -------
        list[dict] sorted by risk_score descending.  Each dict has:
            company, country, industry, risk_score, risk_level, depth,
            downstream_count, disrupted_source, path,
            supply_risk, financial_risk, operational_risk,
            critical_velocity, concentration_flag
        """
        paths: list[dict] = supply_chain_data.get("supply_chain_paths", [])
        if not paths:
            logger.info("Agent 3 ▶ No supply chain paths — returning empty list")
            return []

        # ── Extract disruption-level parameters ──────────────────────────
        severity: str = disruption_data.get("severity", "medium")
        disruption_type: str = disruption_data.get("disruption_type", "unknown")
        source: str = disruption_data.get("source", "groq")
        confidence: float = float(disruption_data.get("confidence", 1.0))

        severity_weight: float = SEVERITY_WEIGHTS.get(severity, SEVERITY_WEIGHTS["unknown"])
        dt_mult: float = DISRUPTION_TYPE_MULTIPLIERS.get(
            disruption_type, DISRUPTION_TYPE_MULTIPLIERS["unknown"]
        )

        # BERT confidence scaling: dampen score for uncertain BERT predictions
        confidence_scale: float = self._compute_confidence_scale(source, confidence)

        logger.info(
            "Agent 3 ▶ Computing risk for {} paths (severity={}, type={}, source={}, conf={:.2f})",
            len(paths), severity, disruption_type, source, confidence,
        )

        # ── Count downstream appearances per company ─────────────────────
        downstream_counts: dict[str, int] = {}
        for path in paths:
            for company in path.get("path", []):
                downstream_counts[company] = downstream_counts.get(company, 0) + 1

        # ── Compute per-company risk ─────────────────────────────────────
        risk_assessments: list[dict] = []
        seen: set[str] = set()

        for path_data in paths:
            company: str = path_data.get("end_company", "Unknown")
            if company in seen:
                continue
            seen.add(company)

            depth: int = max(path_data.get("depth", 1), 1)  # avoid depth=0
            downstream_count: int = downstream_counts.get(company, 1)
            industry: str = (path_data.get("end_industry") or "Unknown").lower()

            # Industry criticality
            ind_boost: float = INDUSTRY_CRITICALITY.get(
                industry, DEFAULT_INDUSTRY_CRITICALITY
            )

            # Concentration risk
            conc_boost: float = (
                CONCENTRATION_BOOST if downstream_count >= CONCENTRATION_THRESHOLD else 1.0
            )

            # Core formula components
            base_risk: float = min(1.0, downstream_count / MAX_DOWNSTREAM)
            depth_weight: float = 1.0 / depth

            composite: float = (
                base_risk * W_BASE
                + depth_weight * W_DEPTH
                + severity_weight * W_SEVERITY
                + dt_mult * W_DISRUPTION_TYPE
                + ind_boost * W_INDUSTRY
            ) * confidence_scale * conc_boost

            risk_score: float = round(min(1.0, max(0.0, composite)), 4)

            # ── Multi-dimensional sub-scores ─────────────────────────────
            supply_risk: float = round(min(1.0, composite * depth_weight), 4)
            financial_risk: float = round(
                min(1.0, composite * severity_weight * ind_boost), 4
            )
            operational_risk: float = round(min(1.0, composite * dt_mult), 4)

            # ── Risk level classification ────────────────────────────────
            if risk_score >= THRESHOLD_HIGH:
                risk_level = "HIGH"
            elif risk_score >= THRESHOLD_MEDIUM:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            # ── Critical velocity flag ───────────────────────────────────
            critical_velocity: bool = risk_level == "HIGH" and depth == 1

            risk_assessments.append({
                # Backward-compatible keys (Agent 4 reads these)
                "company": company,
                "country": path_data.get("end_country", "Unknown"),
                "industry": path_data.get("end_industry", "Unknown"),
                "risk_score": risk_score,
                "risk_level": risk_level,
                "depth": depth,
                "downstream_count": downstream_count,
                "disrupted_source": path_data.get("disrupted_source", ""),
                "path": path_data.get("path", []),
                # New enrichment fields
                "supply_risk": supply_risk,
                "financial_risk": financial_risk,
                "operational_risk": operational_risk,
                "critical_velocity": critical_velocity,
                "concentration_flag": downstream_count >= CONCENTRATION_THRESHOLD,
            })

        # Sort by composite risk score descending
        risk_assessments.sort(key=lambda x: x["risk_score"], reverse=True)

        # ── Summary logging ──────────────────────────────────────────────
        high = sum(1 for r in risk_assessments if r["risk_level"] == "HIGH")
        medium = sum(1 for r in risk_assessments if r["risk_level"] == "MEDIUM")
        low = sum(1 for r in risk_assessments if r["risk_level"] == "LOW")
        crit_v = sum(1 for r in risk_assessments if r["critical_velocity"])

        logger.info(
            "Agent 3 ✅ {} companies assessed — HIGH:{}, MEDIUM:{}, LOW:{} | 🚨 critical_velocity:{}",
            len(risk_assessments), high, medium, low, crit_v,
        )
        return risk_assessments

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence_scale(source: str, confidence: float) -> float:
        """
        Scale the overall risk based on prediction confidence.

        - Groq (LLM) responses are treated as authoritative → scale = 1.0
        - BERT predictions with confidence < 1.0 get linearly scaled down
          (floor = 0.5 to avoid zeroing out risk entirely)
        """
        if source != "bert":
            return 1.0
        # Linear scale: conf=1.0 → 1.0, conf=0.5 → 0.75, conf=0.0 → 0.5
        return max(0.5, 0.5 + 0.5 * confidence)
