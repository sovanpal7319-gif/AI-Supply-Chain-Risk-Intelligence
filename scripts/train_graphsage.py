"""
GraphSAGE Training Pipeline — Mock Disruption Scenarios

Trains the GraphSAGE model on simulated disruption propagation scenarios
using the existing Neo4j supply chain graph.

Scenarios:
  1. Earthquake → TSMC (semiconductor cascade)
  2. Fire → Samsung Semiconductor (electronics cascade)
  3. Cyber attack → Intel (computing cascade)
  4. Flood → BASF (chemicals cascade)
  5. Strike → Maersk (logistics cascade)
  6. Sanctions → NVIDIA (AI chip cascade)

Label generation:
  label = max(0, 1.0 - 0.25 × depth_from_epicenter)
  Epicenter=1.0, depth-1=0.75, depth-2=0.50, depth-3=0.25, depth-4+=0.0

Usage:
  python scripts/train_graphsage.py

Saves to: models/graphsage_risk.pt
"""

import os
import sys
from pathlib import Path
from collections import deque

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
import torch.optim as optim

from backend.agents.graphsage.graphsage_model import GraphSAGEModel
from backend.db.connection import Neo4jConnection
from backend.services.graph_data_adapter import (
    GraphDataAdapter,
    SEVERITY_ENCODING,
)


# ── Training scenarios ───────────────────────────────────────────────────────

SCENARIOS = [
    {"company": "TSMC",                 "event": "earthquake",  "severity": "high"},
    {"company": "Samsung Semiconductor","event": "fire",        "severity": "high"},
    {"company": "Intel",                "event": "cyber_attack","severity": "high"},
    {"company": "BASF",                 "event": "flood",       "severity": "medium"},
    {"company": "Maersk",               "event": "strike",      "severity": "medium"},
    {"company": "NVIDIA",               "event": "sanctions",   "severity": "high"},
]

# ── Training hyperparameters ─────────────────────────────────────────────────

HIDDEN_DIM = 64
LEARNING_RATE = 0.01
EPOCHS = 200
DECAY_RATE = 0.25       # Label decay per hop
DROPOUT = 0.3
SAVE_PATH = "models/graphsage_risk.pt"


def build_training_data(adapter: GraphDataAdapter) -> list[dict]:
    """
    Build PyG-compatible training data for each disruption scenario.

    For each scenario, pulls the full graph from Neo4j, marks the disrupted
    company, and generates ground-truth labels based on BFS depth.
    """
    training_data = []

    driver = Neo4jConnection.get_driver()

    # Pull graph structure once (it's the same for all scenarios)
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
        print("❌ No nodes found in Neo4j. Run 'python scripts/init_neo4j.py' first.")
        sys.exit(1)

    node_names = [n["name"] for n in nodes]
    node_map = {name: idx for idx, name in enumerate(node_names)}
    num_nodes = len(node_names)

    # Build edge index
    src_indices = []
    tgt_indices = []
    for edge in edges:
        s = node_map.get(edge["source"])
        t = node_map.get(edge["target"])
        if s is not None and t is not None:
            src_indices.append(s)
            tgt_indices.append(t)

    edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long)

    print(f"📊 Graph: {num_nodes} nodes, {len(src_indices)} edges")

    for scenario in SCENARIOS:
        company = scenario["company"]
        severity = scenario["severity"]

        if company not in node_map:
            print(f"  ⚠ {company} not in graph — skipping")
            continue

        # Build node features for this scenario
        graph_data = adapter.build_from_neo4j(
            disrupted_company=company,
            severity=severity,
        )

        # Generate labels via BFS depth
        labels = adapter.generate_labels(
            node_names=node_names,
            node_map=node_map,
            disrupted_company=company,
            src_indices=src_indices,
            tgt_indices=tgt_indices,
            num_nodes=num_nodes,
            decay_rate=DECAY_RATE,
        )

        affected_count = int((labels > 0).sum())
        print(f"  ✓ Scenario: {scenario['event']} → {company} | affected: {affected_count}/{num_nodes}")

        training_data.append({
            "scenario": scenario,
            "x": graph_data["x"],
            "edge_index": graph_data["edge_index"],
            "labels": labels,
            "node_names": node_names,
        })

    return training_data


def train():
    """Train GraphSAGE model on all scenarios."""
    print("=" * 60)
    print("🧠 GraphSAGE Training Pipeline")
    print("=" * 60)

    adapter = GraphDataAdapter()
    training_data = build_training_data(adapter)

    if not training_data:
        print("❌ No training data generated. Exiting.")
        sys.exit(1)

    print(f"\n📦 {len(training_data)} training scenarios loaded")

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGEModel(
        in_channels=12,
        hidden_channels=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"🔧 Model: GraphSAGE (12→{HIDDEN_DIM}→1)")
    print(f"🔧 Device: {device}")
    print(f"🔧 Epochs: {EPOCHS}")
    print(f"🔧 LR: {LEARNING_RATE}")
    print()

    # Training loop
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

        for data in training_data:
            x = data["x"].to(device)
            edge_index = data["edge_index"].to(device)
            labels = data["labels"].to(device)

            optimizer.zero_grad()
            predictions = model(x, edge_index)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(training_data)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{EPOCHS} | Loss: {avg_loss:.6f}")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 Evaluation Results")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        for data in training_data:
            scenario = data["scenario"]
            x = data["x"].to(device)
            edge_index = data["edge_index"].to(device)
            labels = data["labels"]

            predictions = model(x, edge_index).cpu()

            # MSE
            mse = float(nn.MSELoss()(predictions, labels))

            # Show top-5 predicted risks
            sorted_idx = torch.argsort(predictions, descending=True)
            top5 = sorted_idx[:5]

            print(f"\n  {scenario['event'].upper()} → {scenario['company']}")
            print(f"  MSE: {mse:.6f}")
            print(f"  Top-5 predicted at-risk companies:")
            for rank, idx in enumerate(top5, 1):
                name = data["node_names"][idx]
                pred = float(predictions[idx])
                actual = float(labels[idx])
                print(f"    {rank}. {name:30s} | pred={pred:.4f} | actual={actual:.4f}")

    # ── Save model ───────────────────────────────────────────────────────
    save_path = Path(SAVE_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": 12,
            "hidden_channels": HIDDEN_DIM,
            "epochs": EPOCHS,
            "scenarios": len(training_data),
        },
        str(save_path),
    )

    print(f"\n✅ Model saved to: {save_path}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)


if __name__ == "__main__":
    train()
