"""
GraphSAGE Model — 2-Layer SAGEConv for Supply Chain Risk Prediction

Predicts downstream disruption risk scores (0–1) for each company node
in the supply chain graph using learned neighborhood aggregation.

Architecture:
  Input (12 features) → SAGEConv(12→hidden) → ReLU → Dropout
                       → SAGEConv(hidden→hidden) → ReLU → Dropout
                       → Linear(hidden→1) → Sigmoid

The hidden layer output serves as the node embedding vector.
Note: Backward compatible with 7-feature inputs via zero-padding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
    _HAS_PYG = True
except ImportError:
    SAGEConv = None
    _HAS_PYG = False


class GraphSAGEModel(nn.Module):
    """
    2-layer GraphSAGE for node-level risk score prediction.

    Parameters
    ----------
    in_channels : int
        Number of input node features (default: 12).
    hidden_channels : int
        Hidden dimension / embedding size (default: 64).
    dropout : float
        Dropout probability between layers (default: 0.3).
    """

    def __init__(
        self,
        in_channels: int = 12,
        hidden_channels: int = 64,
        dropout: float = 0.3,
    ):
        if not _HAS_PYG:
            raise ImportError(
                "torch-geometric is required for GraphSAGE. "
                "Install with: pip install torch-geometric torch-scatter torch-sparse "
                "-f https://data.pyg.org/whl/torch-2.4.0+cpu.html"
            )
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # GraphSAGE convolution layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # Prediction head
        self.predictor = nn.Linear(hidden_channels, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass — predict risk scores for all nodes.

        Parameters
        ----------
        x : Tensor [N, in_channels]
            Node feature matrix.
        edge_index : Tensor [2, E]
            Edge index in COO format.

        Returns
        -------
        Tensor [N] — predicted risk scores in [0, 1].
        """
        # Layer 1
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.dropout(h)

        # Prediction
        out = self.predictor(h)
        out = torch.sigmoid(out).squeeze(-1)  # [N]

        return out

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract node embeddings (hidden-layer output) without prediction head.

        Returns
        -------
        Tensor [N, hidden_channels] — node embedding vectors.
        """
        with torch.no_grad():
            h = self.conv1(x, edge_index)
            h = F.relu(h)

            h = self.conv2(h, edge_index)
            h = F.relu(h)

        return h
