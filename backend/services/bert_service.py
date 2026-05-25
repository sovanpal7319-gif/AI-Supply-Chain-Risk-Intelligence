"""
BERT Inference Service — Fine-tuned multi-task BERT for supply chain disruption detection.

Loads a fine-tuned BertMultiTask model that predicts:
  • disruption_type  (7 classes)
  • severity         (4 classes)

Falls back gracefully if model files are missing or inference fails.
"""

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import BertModel, BertTokenizer

from backend.config import settings


# ── Label Mappings (must match training LabelEncoder order) ──────────────────
DISRUPTION_TYPE_LABELS = [
    "cyber_attack",
    "geopolitical",
    "labor",
    "logistics",
    "natural_disaster",
    "none",
    "operational",
]

SEVERITY_LABELS = [
    "high",
    "low",
    "medium",
    "none",
]


# ── Model Architecture (identical to training notebook) ──────────────────────

class BertMultiTask(nn.Module):
    """
    Shared BERT backbone → two classification heads:
      • head 1 : disruption_type  (7 classes)
      • head 2 : severity         (4 classes)
    """

    def __init__(self, model_name: str, num_dt: int, num_sev: int, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size  # 768

        self.dropout = nn.Dropout(dropout)
        self.dt_head = nn.Linear(hidden, num_dt)
        self.sev_head = nn.Linear(hidden, num_sev)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = self.dropout(out.pooler_output)  # [B, 768]
        return self.dt_head(pooled), self.sev_head(pooled)


# ── Service Class ────────────────────────────────────────────────────────────

class BERTService:
    """Loads and runs inference on the fine-tuned BertMultiTask model."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[BertMultiTask] = None
        self.tokenizer: Optional[BertTokenizer] = None

        model_path = model_path or settings.bert_model_path
        tokenizer_path = tokenizer_path or settings.bert_tokenizer_path

        self._load_model(model_path, tokenizer_path)

    def _load_model(self, model_path: str, tokenizer_path: str) -> None:
        """Load tokenizer and model weights. Raises on failure."""
        # Resolve paths relative to project root if not absolute
        project_root = Path(__file__).resolve().parent.parent.parent
        model_path = self._resolve_path(model_path, project_root)
        tokenizer_path = self._resolve_path(tokenizer_path, project_root)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BERT model weights not found: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"BERT tokenizer not found: {tokenizer_path}")

        logger.info("🧠 Loading BERT tokenizer from: {}", tokenizer_path)
        self.tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))

        logger.info("🧠 Loading BERT model from: {}", model_path)
        self.model = BertMultiTask(
            model_name="bert-base-uncased",
            num_dt=len(DISRUPTION_TYPE_LABELS),
            num_sev=len(SEVERITY_LABELS),
        )

        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "🧠 BERT model loaded successfully (device={}, dt_classes={}, sev_classes={})",
            self.device,
            len(DISRUPTION_TYPE_LABELS),
            len(SEVERITY_LABELS),
        )

    @staticmethod
    def _resolve_path(path: str, project_root: Path) -> Path:
        """Resolve a path — use as-is if absolute, otherwise relative to project root."""
        p = Path(path)
        return p if p.is_absolute() else project_root / p

    def predict(self, text: str) -> dict:
        """
        Run BERT inference on a single text input.

        Returns
        -------
        dict with keys:
            disruption_type : str
            severity        : str
            confidence      : float (0–1), min of the two head max-probabilities
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("BERT model is not loaded")

        # Tokenize
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Inference
        with torch.no_grad():
            dt_logits, sev_logits = self.model(input_ids, attention_mask)

        # Softmax probabilities
        dt_probs = F.softmax(dt_logits, dim=1).cpu().numpy()[0]
        sev_probs = F.softmax(sev_logits, dim=1).cpu().numpy()[0]

        dt_idx = int(dt_probs.argmax())
        sev_idx = int(sev_probs.argmax())

        dt_confidence = float(dt_probs[dt_idx])
        sev_confidence = float(sev_probs[sev_idx])

        # Overall confidence = minimum of the two heads
        confidence = min(dt_confidence, sev_confidence)

        return {
            "disruption_type": DISRUPTION_TYPE_LABELS[dt_idx],
            "severity": SEVERITY_LABELS[sev_idx],
            "confidence": round(confidence, 4),
        }


# ── Module-level factory ─────────────────────────────────────────────────────

def predict_with_bert(text: str, service: BERTService) -> dict:
    """Convenience wrapper around BERTService.predict()."""
    return service.predict(text)
