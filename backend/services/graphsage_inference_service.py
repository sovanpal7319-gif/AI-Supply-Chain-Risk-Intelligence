"""
GraphSAGE Inference Service — Model Loading & Risk Prediction

Loads the trained GraphSAGE model and runs inference on the live
Neo4j graph to predict downstream disruption risk scores.

Thread-safe: model stays in eval() mode, inference uses torch.no_grad().
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from backend.config import settings
from backend.agents.graphsage.graphsage_model import GraphSAGEModel
from backend.services.graph_data_adapter import GraphDataAdapter


class GraphSAGEInferenceService:
    """
    Loads a trained GraphSAGE model and predicts node-level risk scores
    on the live Neo4j supply chain graph.
    """

    def __init__(self):
        self.model: Optional[GraphSAGEModel] = None
        self.adapter = GraphDataAdapter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._needs_padding = False  # True if model trained on 7 features

        model_path = getattr(settings, "graphsage_model_path", "models/graphsage_risk.pt")
        hidden_dim = getattr(settings, "graphsage_hidden_dim", 64)

        self._load_model(model_path, hidden_dim)

    def _load_model(self, model_path: str, hidden_dim: int) -> None:
        """Load trained GraphSAGE weights. Fails gracefully if missing."""
        project_root = Path(__file__).resolve().parent.parent.parent
        resolved = Path(model_path)
        if not resolved.is_absolute():
            resolved = project_root / model_path

        if not resolved.exists():
            logger.warning(
                "⚠️ GraphSAGE model not found at {} — inference disabled. "
                "Run 'python scripts/train_graphsage.py' to train.",
                resolved,
            )
            return

        try:
            # Try loading with 12 features (new format)
            self.model = GraphSAGEModel(
                in_channels=12,
                hidden_channels=hidden_dim,
                dropout=0.0,  # No dropout at inference
            )

            checkpoint = torch.load(str(resolved), map_location=self.device, weights_only=False)
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError:
                # Backward compat: model was trained with 7 features
                logger.info("GraphSAGE: checkpoint has 7 features — loading with compat mode")
                self.model = GraphSAGEModel(
                    in_channels=7,
                    hidden_channels=hidden_dim,
                    dropout=0.0,
                )
                self.model.load_state_dict(state_dict)
                self._needs_padding = True

            self.model.to(self.device)
            self.model.eval()

            logger.info(
                "🧠 GraphSAGE model loaded (hidden={}, device={})",
                hidden_dim, self.device,
            )
        except Exception as exc:
            logger.error("❌ Failed to load GraphSAGE model: {}", exc)
            self.model = None

    @property
    def is_available(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self.model is not None

    def predict(
        self,
        disrupted_company: str,
        severity: str = "high",
    ) -> dict:
        """
        Predict risk scores for all companies in the Neo4j graph.

        Parameters
        ----------
        disrupted_company : str
            Name of the disrupted company (epicenter).
        severity : str
            Severity of the disruption event.

        Returns
        -------
        dict with keys:
            predictions : dict[str, float]  — company_name → risk_score
            embeddings  : dict[str, list]   — company_name → embedding vector
            available   : bool
        """
        if self.model is None:
            logger.warning("GraphSAGE predict called but model not available")
            return {"predictions": {}, "embeddings": {}, "available": False}

        # Build graph data from Neo4j
        graph_data = self.adapter.build_from_neo4j(
            disrupted_company=disrupted_company,
            severity=severity,
        )

        if graph_data["num_nodes"] == 0:
            logger.warning("GraphSAGE: empty graph — no predictions")
            return {"predictions": {}, "embeddings": {}, "available": True}

        x = graph_data["x"].to(self.device)
        edge_index = graph_data["edge_index"].to(self.device)
        node_names = graph_data["node_names"]

        # If model trained on 7 features, truncate 12-feature input
        if self._needs_padding and x.size(1) > 7:
            x = x[:, :7]

        # Run inference
        with torch.no_grad():
            risk_scores = self.model(x, edge_index).cpu()
            embeddings = self.model.get_embeddings(x, edge_index).cpu()

        # Build result dicts
        predictions = {}
        embedding_dict = {}
        for i, name in enumerate(node_names):
            predictions[name] = round(float(risk_scores[i]), 4)
            embedding_dict[name] = embeddings[i].tolist()

        logger.info(
            "🧠 GraphSAGE predicted risk for {} companies (disrupted={})",
            len(predictions), disrupted_company,
        )

        return {
            "predictions": predictions,
            "embeddings": embedding_dict,
            "available": True,
        }
