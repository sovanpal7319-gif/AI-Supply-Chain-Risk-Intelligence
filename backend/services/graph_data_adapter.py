"""
Graph Data Adapter — Neo4j → PyTorch Geometric Conversion

Pulls Company nodes and SUPPLIES_TO edges from Neo4j and converts
them into a PyTorch Geometric Data object with engineered node features.

Node Features (12 per node):
  0.  country_risk          — geopolitical risk encoding [0–1]
  1.  industry_crit         — industry criticality [0–1]
  2.  in_degree             — supplier count (normalized)
  3.  out_degree            — customer count (normalized)
  4.  severity_score        — disruption severity (0 if unaffected)
  5.  depth_normalized      — distance from disrupted node (normalized)
  6.  is_disrupted          — binary flag (1 = epicenter)
  7.  betweenness_centrality— bridge importance (approx BFS) [0–1]
  8.  supplier_importance   — ratio of downstream dependents [0–1]
  9.  industry_embedding    — hashed industry category [0–1]
  10. region_cluster        — geographic region encoding [0–1]
  11. edge_density          — local connectivity ratio [0–1]

New features (7-11) default to 0.0 for backward compatibility
with existing GraphSAGE model trained on 7 features.
"""

from collections import deque
from typing import Optional

import torch
from loguru import logger

from backend.db.connection import Neo4jConnection


# ── Country geopolitical risk encoding ───────────────────────────────────────
# Higher = more geopolitical risk to supply chains
COUNTRY_RISK: dict[str, float] = {
    "taiwan": 0.85,
    "south korea": 0.55,
    "china": 0.80,
    "japan": 0.30,
    "united states": 0.25,
    "germany": 0.20,
    "netherlands": 0.15,
    "india": 0.50,
    "brazil": 0.45,
    "australia": 0.20,
    "singapore": 0.15,
    "united kingdom": 0.20,
    "switzerland": 0.10,
    "denmark": 0.10,
    "vietnam": 0.45,
    "thailand": 0.40,
    "mexico": 0.50,
    # Oil & Geopolitical hotspots
    "saudi arabia": 0.60,
    "uae": 0.45,
    "iran": 0.90,
    "russia": 0.85,
    "ukraine": 0.90,
    "cuba": 0.70,
    "israel": 0.65,
    "indonesia": 0.40,
    "malaysia": 0.35,
    "philippines": 0.45,
    "canada": 0.15,
    "france": 0.20,
    "italy": 0.25,
    "sweden": 0.10,
    "norway": 0.10,
}
DEFAULT_COUNTRY_RISK: float = 0.40

# ── Region cluster encoding (for feature 10) ────────────────────────────────
# Groups countries into broad geographic regions for embedding
REGION_CLUSTER: dict[str, float] = {
    "taiwan": 0.9, "south korea": 0.85, "china": 0.9, "japan": 0.8,
    "vietnam": 0.75, "thailand": 0.75, "singapore": 0.7, "indonesia": 0.7,
    "malaysia": 0.7, "philippines": 0.7, "india": 0.6,
    "united states": 0.3, "canada": 0.3, "mexico": 0.35, "brazil": 0.4, "cuba": 0.35,
    "germany": 0.5, "netherlands": 0.5, "united kingdom": 0.5, "france": 0.5,
    "italy": 0.5, "switzerland": 0.5, "sweden": 0.5, "denmark": 0.5, "norway": 0.5,
    "saudi arabia": 0.65, "uae": 0.65, "iran": 0.7, "israel": 0.6,
    "russia": 0.75, "ukraine": 0.7, "australia": 0.4,
}
DEFAULT_REGION_CLUSTER: float = 0.5

# ── Industry criticality encoding ────────────────────────────────────────────
INDUSTRY_CRITICALITY: dict[str, float] = {
    "semiconductor": 1.0,
    "semiconductor equipment": 0.95,
    "pharma": 0.90,
    "pharmaceutical": 0.90,
    "chemicals": 0.80,
    "mining": 0.75,
    "electronics": 0.70,
    "auto parts": 0.65,
    "automotive": 0.60,
    "contract manufacturing": 0.55,
    "steel": 0.55,
    "industrial": 0.50,
    "shipping": 0.50,
    "logistics": 0.55,
    "oil & gas": 0.85,
    "energy": 0.80,
    "oil": 0.85,
    "it services": 0.45,
    "it": 0.45,
    "conglomerate": 0.40,
    "consumer electronics": 0.45,
    "telecommunications": 0.50,
}
DEFAULT_INDUSTRY_CRIT: float = 0.50

# ── Industry embedding hash (for feature 9) ─────────────────────────────────
# Simple deterministic hash of industry name to [0, 1] range
def _industry_hash(industry: str) -> float:
    """Deterministic hash of industry name normalized to [0, 1]."""
    if not industry:
        return 0.5
    h = sum(ord(c) for c in industry.lower()) % 100
    return h / 100.0

# ── Severity encoding ────────────────────────────────────────────────────────
SEVERITY_ENCODING: dict[str, float] = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
    "unknown": 0.5,
}


class GraphDataAdapter:
    """
    Converts a Neo4j supply chain graph into a PyTorch Geometric Data object
    with engineered node features for GraphSAGE training/inference.
    """

    def __init__(self):
        logger.debug("GraphDataAdapter initialized")

    def build_from_neo4j(
        self,
        disrupted_company: Optional[str] = None,
        severity: str = "high",
    ) -> dict:
        """
        Pull the full graph from Neo4j and convert to PyG-compatible tensors.

        Parameters
        ----------
        disrupted_company : str or None
            Name of the disrupted company (epicenter). If None, no disruption
            features are set.
        severity : str
            Severity of the disruption event.

        Returns
        -------
        dict with keys:
            x            : Tensor [N, 12] — node features
            edge_index   : Tensor [2, E] — edge index (COO)
            node_names   : list[str] — company name for each node index
            node_map     : dict[str, int] — name → index mapping
            num_nodes    : int
            num_edges    : int
        """
        driver = Neo4jConnection.get_driver()

        # ── Pull nodes ───────────────────────────────────────────────────
        with driver.session() as session:
            nodes_result = session.run(
                "MATCH (c:Company) "
                "RETURN c.name AS name, c.country AS country, c.industry AS industry"
            )
            nodes = [dict(r) for r in nodes_result]

            edges_result = session.run(
                "MATCH (a:Company)-[:SUPPLIES_TO]->(b:Company) "
                "RETURN a.name AS source, b.name AS target"
            )
            edges = [dict(r) for r in edges_result]

        if not nodes:
            logger.warning("GraphDataAdapter: No nodes found in Neo4j")
            return self._empty_result()

        # ── Build name ↔ index mapping ───────────────────────────────────
        node_names: list[str] = [n["name"] for n in nodes]
        node_map: dict[str, int] = {name: idx for idx, name in enumerate(node_names)}
        num_nodes = len(node_names)

        # ── Build edge_index ─────────────────────────────────────────────
        src_indices = []
        tgt_indices = []
        for edge in edges:
            s = node_map.get(edge["source"])
            t = node_map.get(edge["target"])
            if s is not None and t is not None:
                src_indices.append(s)
                tgt_indices.append(t)

        if src_indices:
            edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # ── Compute degree ───────────────────────────────────────────────
        in_degree = [0] * num_nodes
        out_degree = [0] * num_nodes
        for s, t in zip(src_indices, tgt_indices):
            out_degree[s] += 1
            in_degree[t] += 1

        max_in = max(max(in_degree), 1)
        max_out = max(max(out_degree), 1)

        # ── Compute BFS depth from disrupted node ────────────────────────
        depths = [999] * num_nodes
        if disrupted_company and disrupted_company in node_map:
            disrupted_idx = node_map[disrupted_company]
            depths = self._bfs_depth(disrupted_idx, src_indices, tgt_indices, num_nodes)

        max_depth = max(d for d in depths if d < 999) if any(d < 999 for d in depths) else 1
        max_depth = max(max_depth, 1)  # Guard against depth=0 (epicenter only)

        # ── Build feature matrix [N, 12] ─────────────────────────────────
        sev_score = SEVERITY_ENCODING.get(severity, 0.5)
        total_edges = len(src_indices)
        max_possible_edges = max(num_nodes * (num_nodes - 1), 1)  # for edge density
        features = []

        for i, node in enumerate(nodes):
            country = (node.get("country") or "").lower()
            industry = (node.get("industry") or "").lower()

            country_risk = COUNTRY_RISK.get(country, DEFAULT_COUNTRY_RISK)
            industry_crit = INDUSTRY_CRITICALITY.get(industry, DEFAULT_INDUSTRY_CRIT)
            in_deg_norm = in_degree[i] / max_in
            out_deg_norm = out_degree[i] / max_out

            # Disruption features
            is_disrupted = 1.0 if (disrupted_company and node_names[i] == disrupted_company) else 0.0
            node_severity = sev_score if is_disrupted else 0.0
            depth_norm = min(depths[i] / max_depth, 1.0) if depths[i] < 999 else 1.0

            # NEW Feature 7: Betweenness centrality (approximation)
            # Nodes with both in and out edges are bridges
            betweenness = 0.0
            if in_degree[i] > 0 and out_degree[i] > 0:
                betweenness = min((in_degree[i] * out_degree[i]) / max(total_edges, 1), 1.0)

            # NEW Feature 8: Supplier importance (downstream dependents ratio)
            supplier_importance = out_degree[i] / max(num_nodes - 1, 1)

            # NEW Feature 9: Industry embedding (hashed)
            ind_embed = _industry_hash(industry)

            # NEW Feature 10: Region cluster
            region_clust = REGION_CLUSTER.get(country, DEFAULT_REGION_CLUSTER)

            # NEW Feature 11: Edge density (local connectivity)
            local_edges = in_degree[i] + out_degree[i]
            edge_density = local_edges / max(max_in + max_out, 1)

            features.append([
                country_risk,           # 0
                industry_crit,          # 1
                in_deg_norm,            # 2
                out_deg_norm,           # 3
                node_severity,          # 4
                depth_norm,             # 5
                is_disrupted,           # 6
                betweenness,            # 7  (NEW)
                supplier_importance,    # 8  (NEW)
                ind_embed,              # 9  (NEW)
                region_clust,           # 10 (NEW)
                edge_density,           # 11 (NEW)
            ])

        x = torch.tensor(features, dtype=torch.float32)

        logger.debug(
            "GraphDataAdapter: {} nodes, {} edges, disrupted={}",
            num_nodes, edge_index.size(1), disrupted_company,
        )

        return {
            "x": x,
            "edge_index": edge_index,
            "node_names": node_names,
            "node_map": node_map,
            "num_nodes": num_nodes,
            "num_edges": edge_index.size(1),
        }

    def generate_labels(
        self,
        node_names: list[str],
        node_map: dict[str, int],
        disrupted_company: str,
        src_indices: list[int],
        tgt_indices: list[int],
        num_nodes: int,
        decay_rate: float = 0.25,
    ) -> torch.Tensor:
        """
        Generate ground-truth risk labels based on BFS depth from epicenter.

        Label formula: label = max(0, 1.0 - decay_rate * depth)
        Epicenter gets 1.0, direct customers get 0.75, etc.

        Returns
        -------
        Tensor [N] — risk labels in [0, 1].
        """
        if disrupted_company not in node_map:
            return torch.zeros(num_nodes, dtype=torch.float32)

        disrupted_idx = node_map[disrupted_company]
        depths = self._bfs_depth(disrupted_idx, src_indices, tgt_indices, num_nodes)

        labels = []
        for d in depths:
            if d >= 999:
                labels.append(0.0)
            else:
                labels.append(max(0.0, 1.0 - decay_rate * d))

        return torch.tensor(labels, dtype=torch.float32)

    @staticmethod
    def _bfs_depth(
        start: int,
        src_indices: list[int],
        tgt_indices: list[int],
        num_nodes: int,
    ) -> list[int]:
        """BFS from start node following SUPPLIES_TO direction."""
        adj: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
        for s, t in zip(src_indices, tgt_indices):
            adj[s].append(t)

        depths = [999] * num_nodes
        depths[start] = 0
        queue = deque([start])

        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if depths[neighbor] > depths[node] + 1:
                    depths[neighbor] = depths[node] + 1
                    queue.append(neighbor)

        return depths

    @staticmethod
    def _empty_result() -> dict:
        return {
            "x": torch.zeros((0, 7), dtype=torch.float32),
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "node_names": [],
            "node_map": {},
            "num_nodes": 0,
            "num_edges": 0,
        }
