"""
Application configuration — loads from .env file.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""

    # ── Groq LLM (required) ─────────────────────────────
    groq_api_key: str = Field(..., description="Groq API key (free at console.groq.com)")
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Groq model name")

    # ── NewsAPI (required) ──────────────────────────────
    news_api_key: str = Field(..., description="NewsAPI.org API key (free at newsapi.org/register)")

    # ── Neo4j (required) ────────────────────────────────
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password", description="Neo4j password")

    # ── RL Decision Agent ────────────────────────────────
    use_rl_decision: bool = Field(default=False, description="Use RL (DQN) for decisions instead of rules")

    # ── Dynamic Graph Pipeline ───────────────────────────────
    enrichment_mode: str = Field(
        default="mock",
        description="Company enrichment source: 'mock' (curated dataset) or 'llm' (Groq fallback)",
    )

    # ── GraphSAGE Risk Prediction ────────────────────────────
    use_graphsage: bool = Field(
        default=True,
        description="Use GraphSAGE model for hybrid risk scoring",
    )
    graphsage_model_path: str = Field(
        default="models/graphsage_risk.pt",
        description="Path to trained GraphSAGE model weights",
    )
    graphsage_hidden_dim: int = Field(
        default=64,
        description="GraphSAGE hidden layer dimension",
    )
    graphsage_blend_weight: float = Field(
        default=0.70,
        description="GraphSAGE weight in hybrid scoring (0.70 = 70% GS + 30% rules)",
    )

    # ── Fine-tuned BERT Model ────────────────────────────────
    bert_model_path: str = Field(
        default="models/bert_supply_chain_final.pt",
        description="Path to fine-tuned BERT model weights (.pt file)",
    )
    bert_tokenizer_path: str = Field(
        default="models/bert_supply_chain_tokenizer",
        description="Path to fine-tuned BERT tokenizer directory",
    )
    bert_confidence_threshold: float = Field(
        default=0.8,
        description="Minimum BERT confidence to use its prediction (else fallback to Groq)",
    )

    # ── SPERT NER+RE ─────────────────────────────────────────
    use_spert: bool = Field(
        default=True,
        description="Enable SPERT joint NER+RE extraction middleware",
    )
    spert_model_path: str = Field(
        default="models/spert_supply_chain",
        description="Path to fine-tuned SPERT model directory",
    )
    spert_types_path: str = Field(
        default="data/spert/supply_chain_types.json",
        description="Path to SPERT entity/relation type definitions",
    )
    spert_confidence_threshold: float = Field(
        default=0.4,
        description="Minimum confidence for SPERT entity/relation predictions",
    )
    spert_max_span_size: int = Field(
        default=10,
        description="Maximum entity span size for SPERT",
    )
    spert_rel_filter_threshold: float = Field(
        default=0.4,
        description="Relation filter threshold for SPERT predictions",
    )

    # ── Server ───────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton settings instance
settings = Settings()
