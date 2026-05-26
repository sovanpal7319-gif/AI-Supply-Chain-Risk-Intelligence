# 🔗 Agentic Supply Chain Disruption Monitoring System

A **production-grade** AI-powered multi-agent system that ingests live news articles, detects supply chain disruptions using **BERT + Groq LLM**, extracts structured entities and relations with **SpERT (joint NER+RE)**, maps affected companies through a **Neo4j knowledge graph** with **entity canonicalization and automated graph quality enforcement**, computes hybrid risk scores with **GraphSAGE (12-feature) + rule-based scoring**, and generates actionable mitigation strategies — powered by a **DQN reinforcement learning** decision engine.

![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent-blueviolet)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688)
![Frontend](https://img.shields.io/badge/Frontend-React%2018-61DAFB)
![Database](https://img.shields.io/badge/Database-Neo4j-008CC1)
![LLM](https://img.shields.io/badge/LLM-Groq%20(Free)-412991)
![ML](https://img.shields.io/badge/ML-BERT%20%2B%20GraphSAGE-FF6F00)
![NER+RE](https://img.shields.io/badge/NER%2BRE-SpERT-E91E63)
![RL](https://img.shields.io/badge/RL-DQN%20(NumPy)-FF6F00)

---

## 🧠 System Architecture

The system implements **two analysis pipelines** orchestrated by a FastAPI backend:

### Pipeline 1: `/analyze` — Standard 5-Agent Pipeline

```
📰 News Article
        │
        ▼
┌─────────────────────────────────────────────┐
│ Agent 1: Disruption Detection               │
│  ① BERT (fine-tuned) → type + severity      │
│  ② If confidence < threshold → Groq LLM     │
│  ③ Groq enriches: companies, countries      │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ 🆕 SPERT: Joint NER + Relation Extraction   │
│  Entities: Company, Country, Event, etc.    │
│  Relations: AFFECTS, OCCURS_IN, DELAYS, etc.│
│  → Validated → build_from_spert() → Neo4j   │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ Agent 2: Knowledge Graph Query (Neo4j)      │
│  BFS traversal of SUPPLIES_TO (4 levels)    │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ Agent 3: Risk Assessment (Enhanced)         │
│  Multi-dimensional: supply, financial, ops  │
│  BERT confidence scaling + concentration    │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ Agent 4: Decision Engine (Rule + RL DQN)    │
│  8-dim state → 4 actions                    │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ Agent 5: Alternative Supplier Finder        │
│  Same-industry, not-in-disrupted-path       │
└─────────────────────────────────────────────┘
```

### Pipeline 2: `/dynamic-analyze` — Dynamic KG Pipeline (9 Steps)

```
📰 News Text
        │
        ▼
Step 1  : Enhanced Disruption Extraction (Groq LLM → regex fallback)
        ▼
Step 1.5: SPERT NER+RE (11 entity types + 11 relation types → validated)
        ▼
Step 1.6: 🆕 Entity Canonicalization (alias resolution, fuzzy match, blocklist)
        ▼
Step 2  : Company Ecosystem Enrichment (suppliers, customers, industry)
        ▼
Step 3  : Dynamic Neo4j Graph Construction (MERGE + canonicalization)
        ▼
Step 3.5: SPERT → Neo4j (build_from_spert → nodes + edges)
        ▼
Step 3.6: 🆕 Graph Quality Cleanup (orphans, dupes, self-loops, low-confidence)
        ▼
Step 4  : KG Traversal (Agent 2 — downstream impact)
        ▼
Step 5  : Hybrid Risk (GraphSAGE 12-feature 70% + Rule-based 30%)
        ▼
Step 6  : Decision Engine (Agent 4 — RL DQN)
        ▼
Step 7  : Alternative Suppliers (Agent 5)
```

---

## 📁 Project Structure

```
Project/
├── backend/
│   ├── __init__.py
│   ├── main.py                         # FastAPI app & pipeline orchestrator
│   ├── config.py                       # Pydantic settings (.env loader)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── disruption_agent.py         # Agent 1: BERT + Groq disruption detection
│   │   ├── enhanced_disruption_agent.py # Enhanced extraction for dynamic pipeline
│   │   ├── spert_agent.py             # SPERT NER+RE middleware (11 entity types)
│   │   ├── kg_query_agent.py           # Agent 2: Graph traversal (BFS)
│   │   ├── risk_agent.py              # Agent 3: Multi-dimensional risk scoring
│   │   ├── graphsage_risk_agent.py    # Agent 3+: GraphSAGE hybrid risk
│   │   ├── decision_agent.py          # Agent 4: Rule-based + RL hybrid
│   │   ├── alt_supplier_agent.py      # Agent 5: Alternative suppliers
│   │   ├── graphsage/
│   │   │   ├── graphsage_model.py     # 2-layer SAGEConv model (12→64→1)
│   │   │   └── __init__.py
│   │   └── rl/
│   │       ├── environment.py          # Gym-style RL environment
│   │       ├── dqn_agent.py            # DQN with NumPy neural network
│   │       └── train.py                # Training script (CLI)
│   ├── services/
│   │   ├── llm_service.py             # Groq API (OpenAI-compatible)
│   │   ├── bert_service.py            # Fine-tuned BERT inference
│   │   ├── neo4j_service.py           # Neo4j driver + graph queries + intelligence
│   │   ├── news_service.py            # NewsAPI.org integration
│   │   ├── company_enrichment_service.py  # Company ecosystem lookup
│   │   ├── dynamic_graph_builder.py   # On-the-fly Neo4j graph (canonicalized)
│   │   ├── entity_canonicalization.py  # 🆕 Alias resolution, fuzzy match, blocklist
│   │   ├── graph_quality_engine.py    # 🆕 Automated graph cleanup & health
│   │   ├── spert_validation.py        # SPERT validation + semantic constraints
│   │   ├── graph_data_adapter.py      # Neo4j → PyG conversion (12 features)
│   │   └── graphsage_inference_service.py # GraphSAGE model loading & prediction
│   └── db/
│       └── connection.py               # Neo4j driver singleton
├── spert/                              # Upstream SpERT framework (unchanged)
│   └── spert/                          # Core: models.py, prediction.py, etc.
├── frontend/
│   ├── index.html                      # SPA shell (React 18 via CDN)
│   ├── styles.css                      # Dark glassmorphic UI + graph toolbar
│   └── app.js                          # React components + D3 risk graph viz
├── data/
│   ├── sample_news.json                # Sample disruption scenarios
│   ├── mock_enrichment_data.py         # Curated company ecosystem dataset
│   ├── rl_model.json                   # Trained DQN weights
│   └── spert/                          # SPERT domain config & training data
│       ├── supply_chain_types.json     # 11 entity + 11 relation type definitions
│       ├── supply_chain_train.json     # 12 annotated training examples
│       └── supply_chain_train.conf     # Fine-tuning configuration
├── models/
│   ├── graphsage_risk.pt               # Trained GraphSAGE weights
│   └── spert_supply_chain/             # Fine-tuned SPERT checkpoint (after training)
├── scripts/
│   ├── init_neo4j.py                   # Seed Neo4j with 57+ companies
│   ├── init_dynamic_schema.py         # 🆕 Product constraints, indexes, verification
│   ├── train_graphsage.py             # GraphSAGE training pipeline
│   └── train_spert.py                 # SPERT fine-tuning + prediction demo
├── requirements.txt
├── .env                                # Environment variables (not committed)
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Neo4j** — running at `bolt://localhost:7687`
- **Groq API Key** — FREE at [console.groq.com](https://console.groq.com)
- **NewsAPI Key** — FREE at [newsapi.org/register](https://newsapi.org/register)

### 1. Clone & Setup

```bash
cd Project

# Copy environment config
cp .env.example .env    # then edit with your API keys

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate    # Linux/macOS

# Install core dependencies
pip install -r requirements.txt
```

> **Note**: For `torch-geometric` (GraphSAGE), use pre-built wheels to avoid slow compilation:
> ```bash
> pip install torch-geometric torch-scatter torch-sparse \
>     -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
> ```
> Replace `torch-2.4.0+cpu` with your PyTorch version (check with `python -c "import torch; print(torch.__version__)"`).

### 2. Set API Keys

Edit `.env` with your keys:

```env
GROQ_API_KEY=gsk_your-key-here
NEWS_API_KEY=your-newsapi-key-here
NEO4J_PASSWORD=your-neo4j-password
```

### 3. Initialize Neo4j

```bash
# Seed the base graph
python scripts/init_neo4j.py

# Create production constraints & indexes
python scripts/init_dynamic_schema.py
```

This seeds **57+ companies** across **12 industries** with **80+ SUPPLIES_TO relationships**, plus creates uniqueness constraints for Company/Event/Region/Product nodes and performance indexes.

### 4. Train Models

#### RL Agent (auto-trains on first startup)
```bash
python -m backend.agents.rl.train --episodes 5000 --seed 42
```

#### GraphSAGE (required for hybrid risk scoring)
```bash
python scripts/train_graphsage.py
```

Trains on 6 disruption scenarios (200 epochs, ~30 seconds). Saves to `models/graphsage_risk.pt`.

#### SPERT NER+RE (optional — rule-based fallback works out of the box)
```bash
# Test extraction with rule-based fallback (no training needed)
python scripts/train_spert.py predict

# Fine-tune on supply-chain data (requires annotated examples)
cd spert
python spert.py train --config ../data/spert/supply_chain_train.conf
```

Fine-tunes from `bert-base-cased` on 12 annotated supply-chain examples. Saves to `models/spert_supply_chain/`.

### 5. Run the Application

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Open the UI

Navigate to **http://localhost:8000** in your browser.

---

## 🔌 API Endpoints

### `POST /analyze`

Standard 5-agent pipeline. Analyze a news article for supply chain disruptions.

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"news_text": "A massive earthquake struck Taiwan, causing TSMC to halt production."}'
```

### `POST /dynamic-analyze`

Dynamic KG pipeline. Auto-builds Neo4j graph neighborhood, runs GraphSAGE hybrid risk.

```bash
curl -X POST http://localhost:8000/dynamic-analyze \
  -H "Content-Type: application/json" \
  -d '{"news": "A massive earthquake struck Taiwan, causing TSMC to halt production."}'
```

**Response includes:**
- `extraction` — structured entity extraction
- `enrichment` — company ecosystem data
- `spert_extraction` — 🆕 SPERT NER+RE entities and relations
- `generated_graph_summary` — dynamic graph creation stats (includes SPERT nodes/edges)
- `rule_based_risk` — pure rule-based scores
- `graphsage_risk` — pure GraphSAGE scores + 64-dim embeddings
- `risk_scores` — hybrid blended scores (70% GraphSAGE + 30% rules)
- `decisions` — RL DQN mitigation actions
- `alternative_suppliers` — replacement supplier suggestions

### `GET /news/search?q=...`

Search for live news articles via NewsAPI.org.

### `GET /news/headlines`

Fetch curated supply chain disruption headlines.

### `GET /health`

Returns system status, LLM provider, and Neo4j connectivity.

### `GET /graph`

Returns the full supply chain graph data (nodes + edges) for visualization.

---

## 🤖 Agent Details

### Agent 1 — Disruption Detection (BERT + Groq Hybrid)

| Feature | Description |
|---------|-------------|
| **BERT Mode** | Fine-tuned multi-task BERT predicts `disruption_type` (7 classes) + `severity` (4 classes) |
| **Groq Mode** | LLaMA-3.3 70B via Groq cloud (FREE) — full structured JSON extraction |
| **Hybrid Logic** | BERT predicts first → if confidence ≥ threshold, use BERT type/severity + Groq for entities → if low confidence, full Groq fallback |
| **Output** | `disruption_type`, `severity`, `confidence`, `source`, `affected_companies[]`, `affected_countries[]`, `summary` |

**Pipeline:** BERT → confidence check → Groq enrichment/fallback → safe defaults

**Supported Disruption Types:** `natural_disaster`, `geopolitical`, `labor`, `pandemic`, `operational`, `financial`, `logistics`, `supply`, `cyber_attack`

### Agent 2 — Knowledge Graph Query

| Feature | Description |
|---------|-------------|
| Input | Affected company names from Agent 1 |
| Engine | Neo4j with parameterized Cypher queries |
| Traversal | `SUPPLIES_TO` BFS up to 4 levels deep |
| Output | Downstream supply chain paths |

### Agent 3 — Risk Assessment (Enhanced + GraphSAGE Hybrid)

#### Rule-Based Formula
```
composite = (base × 0.20 + depth_w × 0.20 + severity_w × 0.20
           + disruption_type × 0.20 + industry_crit × 0.20)
           × confidence_scale × concentration_boost

Sub-scores: supply_risk, financial_risk, operational_risk
```

#### GraphSAGE Hybrid (Dynamic Pipeline)
```
final_risk = 0.70 × graphsage_score + 0.30 × rule_based_score
```

**GraphSAGE Architecture:**
```
Input (12 features) → SAGEConv(12→64) → ReLU → Dropout
                     → SAGEConv(64→64) → ReLU → Dropout
                     → Linear(64→1) → Sigmoid
```

**Node Features (12 per node):**
| # | Feature | Description |
|---|---------|-------------|
| 0 | `country_risk` | Geopolitical risk encoding [0–1] (32 countries) |
| 1 | `industry_crit` | Industry criticality [0–1] (22 industries) |
| 2 | `in_degree` | Supplier count (normalized) |
| 3 | `out_degree` | Customer count (normalized) |
| 4 | `severity_score` | Disruption severity (0 if unaffected) |
| 5 | `depth_normalized` | Distance from epicenter (normalized) |
| 6 | `is_disrupted` | Binary flag (1 = epicenter) |
| 7 | `betweenness_centrality` | 🆕 Bridge importance (in×out / total) |
| 8 | `supplier_importance` | 🆕 Downstream dependents ratio |
| 9 | `industry_embedding` | 🆕 Hashed industry category [0–1] |
| 10 | `region_cluster` | 🆕 Geographic region encoding [0–1] |
| 11 | `edge_density` | 🆕 Local connectivity ratio [0–1] |

> **Backward Compatible:** Existing models trained on 7 features work unchanged — the inference service auto-detects and truncates inputs.

| Score | Level | Meaning |
|-------|-------|---------|
| ≥ 0.65 | 🔴 HIGH | Critical — immediate action required |
| 0.35 – 0.65 | 🟡 MEDIUM | Elevated — increased monitoring |
| < 0.35 | 🟢 LOW | Minimal — standard monitoring |

### Agent 4 — Decision Engine (Hybrid)

Supports **two decision modes**, toggled via `USE_RL_DECISION`:

#### Rule-Based Mode

| Risk Level | Action | Recommendation |
|------------|--------|----------------|
| HIGH | `REPLACE_SUPPLIER` | Immediately onboard alternatives, activate reserves |
| MEDIUM | `INCREASE_MONITORING` | Daily monitoring, evaluate backup options |
| LOW | `NO_ACTION` | Continue standard monitoring |

#### RL Mode (DQN)

**8-Dimensional State Vector:**

| Dimension | Description |
|-----------|-------------|
| `severity` | Disruption severity (0.0 / 0.5 / 1.0) |
| `num_affected_norm` | Affected companies normalized (÷50) |
| `avg_risk_score` | Mean risk score across companies |
| `max_risk_score` | Maximum individual risk score |
| `avg_depth_norm` | Average disruption depth (÷4) |
| `high_risk_ratio` | Fraction of companies at HIGH risk |
| `criticality` | Supply chain criticality factor |
| `persistence` | Disruption persistence factor |

**DQN Architecture:** `Input (8) → Dense(64, ReLU) → Dense(32, ReLU) → Output (4 actions)`

**4 Actions:** `NO_ACTION`, `INCREASE_MONITORING`, `DIVERSIFY_SUPPLY`, `REPLACE_SUPPLIER`

### Agent 5 — Alternative Supplier Finder

For HIGH and MEDIUM risk companies, queries the graph for same-industry suppliers **not** in the disrupted path. Returns up to 5 alternatives per company.

### SPERT Agent — Joint NER + Relation Extraction Middleware

Inserted between disruption classification and graph building, SpERT performs span-based joint entity and relation extraction with **entity canonicalization**.

| Feature | Description |
|---------|-------------|
| **Model** | SpERT (Span-based Entity and Relation Transformer) |
| **Fallback** | Rule-based NER+RE with 100+ company, event, country, port, logistics, industry patterns |
| **Entity Types (11)** | `COMPANY`, `COUNTRY`, `EVENT`, `SUPPLIER`, `CUSTOMER`, `PORT`, `REGION`, `PRODUCT`, `INDUSTRY`, `RAW_MATERIAL`, `LOGISTICS_PROVIDER` |
| **Relation Types (11)** | `AFFECTS`, `OCCURS_IN`, `DEPENDS_ON`, `SUPPLIES_TO`, `DELAYS`, `IMPACTS`, `PRODUCES`, `SHIPS_TO`, `MANUFACTURES`, `LOCATED_IN`, `ALTERNATIVE_TO` |
| **Validation** | 14-step pipeline: type checking, blocklist, confidence filter, canonicalization, cross-dedup, self-loop removal, semantic constraints |
| **Canonicalization** | 125+ company aliases, 45 country aliases, fuzzy matching (0.85 threshold) |
| **Output** | Standardized `{"entities": [...], "relations": [...]}` dict |

**Example:**
```
Input:  "U.S. embargo on the Strait of Hormuz impacts Saudi Aramco oil shipments via Maersk"
Output:
  Entities: [EVENT] Embargo, [PORT] Strait of Hormuz, [COMPANY] Saudi Aramco,
            [PRODUCT] Crude Oil, [LOGISTICS_PROVIDER] Maersk
  Relations: Embargo -[AFFECTS]-> Saudi Aramco, Embargo -[OCCURS_IN]-> Strait of Hormuz,
             Maersk -[SHIPS_TO]-> Strait of Hormuz
```

**Entity → Neo4j Node Mapping:**
| SPERT Type | Neo4j Label |
|------------|-------------|
| COMPANY / SUPPLIER / CUSTOMER | `:Company` |
| COUNTRY / REGION / PORT | `:Region` |
| EVENT | `:Event` |
| PRODUCT / RAW_MATERIAL | `:Product` |
| LOGISTICS_PROVIDER | `:Company` |
| INDUSTRY | Property on `:Company` |

### 🆕 Entity Canonicalization Service

Centralized data integrity layer applied **before** all Neo4j writes.

| Feature | Details |
|---------|--------|
| **Company Aliases** | 125+ mappings ("taiwan semiconductor" → TSMC, "aramco" → Saudi Aramco) |
| **Country Aliases** | 45 mappings ("us" → United States, "uk" → United Kingdom) |
| **Product Aliases** | 43 mappings ("wti" → Crude Oil, "chip" → Semiconductors) |
| **Blocklist** | Rejects news artifacts (Reuters, AP, Bloomberg) and placeholders (Unknown, None, N/A) |
| **Fuzzy Matching** | `difflib.SequenceMatcher` at 0.85 threshold for typo detection |

### 🆕 Graph Quality Engine

Automated graph maintenance runs after every dynamic graph build.

| Operation | Description |
|-----------|-------------|
| Orphan cleanup | Removes Company nodes with zero edges |
| Edge deduplication | Merges duplicate SUPPLIES_TO edges (same source→target) |
| Self-loop removal | Removes Company→Company edges where source = target |
| Low-confidence pruning | Removes edges with confidence < 0.3 |
| Health reporting | Returns orphans removed, duplicates merged, self-loops fixed |

---

## 🗃️ Neo4j Knowledge Graph

**57+ companies** across **12 industries** and **12+ countries**:

| Industry | Count | Examples |
|----------|-------|---------|
| Semiconductor | 11 | TSMC, Intel, NVIDIA, Samsung, AMD |
| Semiconductor Equipment | 2 | ASML, Tokyo Electron |
| Electronics | 9 | Apple, Microsoft, Sony, Huawei, HP |
| Contract Manufacturing | 4 | Foxconn, Pegatron, Flex, Jabil |
| Automotive | 8 | Toyota, BMW, Tesla, Hyundai |
| Auto Parts | 5 | Bosch, Denso, Continental |
| Chemicals | 3 | BASF, Dow Chemical, LG Chem |
| Pharma | 6 | Pfizer, Novartis, Roche, AstraZeneca |
| Mining | 4 | BHP, Rio Tinto, Vale, Glencore |
| Industrial | 4 | Siemens, Honeywell, 3M, Caterpillar |
| Shipping | 3 | Maersk, COSCO, Hapag-Lloyd |
| Steel / IT / Conglomerate | 3 | Tata Steel, Infosys, Reliance |

**80+ `SUPPLIES_TO` relationships** model realistic multi-tier supply chains.

### Schema Constraints & Indexes

| Type | Target | Purpose |
|------|--------|---------|
| Unique Constraint | `Company.name` | Prevent duplicate company nodes |
| Unique Constraint | `Event.event_id` | Prevent duplicate events |
| Unique Constraint | `Region.name` | Prevent duplicate regions |
| Unique Constraint | `Product.name` | 🆕 Prevent duplicate products |
| Index | `Company.industry` | 🆕 Fast industry-based lookups |
| Index | `Company.country` | 🆕 Fast country-based lookups |
| Index | `Product.name` | 🆕 Fast product search |

### Intelligence Queries

| Query | Description |
|-------|-------------|
| `find_critical_suppliers()` | 🆕 Top companies by downstream dependents (in-degree) |
| `get_upstream_chain()` | 🆕 Recursive supplier tracing (1–4 hops) |
| `get_disruption_propagation()` | 🆕 Cascading impact with depth-decayed severity |
| `get_region_risk_summary()` | 🆕 Company counts aggregated by country |
| `find_bottleneck_nodes()` | 🆕 Companies with high in×out degree (bridge nodes) |

---

## 🎨 Frontend Features

- **Dark Mode Glassmorphic UI** with animated radial gradient background mesh
- **Interactive D3.js Force-Directed Graph** with production-grade features:
  - 🆕 **Risk-based node coloring** — red (high), amber (medium), green (low), indigo (unaffected)
  - 🆕 **Degree-proportional node sizing** — bigger nodes = more supply chain connections
  - 🆕 **Focus mode** — hover a node to dim all unconnected nodes and edges
  - 🆕 **Edge labels on hover** — "supplies to" appears when hovering edges
  - 🆕 **Disrupted path highlighting** — red arrows and glowing edges between affected nodes
  - 🆕 **"Affected only" filter toggle** — hide unaffected companies
  - 🆕 **SVG glow filter** — affected nodes have Gaussian blur glow effect
  - 🆕 **Enhanced tooltips** — company name, industry, country, connections, risk %
  - 🆕 **5-item legend** — High Risk / Medium Risk / Low Risk / Unaffected / Disrupted Path
- **Risk Badges** with color-coded severity indicators (red / amber / green)
- **Pipeline Progress Bar** showing agent execution stages
- **Stats Dashboard** with affected count, risk distribution, processing time
- **Live News Feed** — search NewsAPI or browse curated supply chain headlines
- **Toast Notifications** for success and error feedback
- **Responsive Design** — works on desktop and tablet
- **Keyboard Shortcut** — Ctrl+Enter to analyze

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Groq API key (free at console.groq.com) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `NEWS_API_KEY` | *(required)* | NewsAPI.org key (free at newsapi.org) |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `USE_RL_DECISION` | `false` | Use DQN agent for decisions |
| `ENRICHMENT_MODE` | `mock` | Company enrichment: `mock` or `llm` |
| `USE_GRAPHSAGE` | `true` | Enable GraphSAGE hybrid risk scoring |
| `GRAPHSAGE_MODEL_PATH` | `models/graphsage_risk.pt` | Path to trained GraphSAGE weights |
| `GRAPHSAGE_HIDDEN_DIM` | `64` | GraphSAGE hidden dimension |
| `GRAPHSAGE_BLEND_WEIGHT` | `0.70` | GraphSAGE weight in hybrid (0.70 = 70% GS + 30% rules) |
| `BERT_MODEL_PATH` | `models/bert_supply_chain_final.pt` | Path to fine-tuned BERT weights |
| `BERT_TOKENIZER_PATH` | `models/bert_supply_chain_tokenizer` | Path to BERT tokenizer dir |
| `BERT_CONFIDENCE_THRESHOLD` | `0.8` | Min BERT confidence to use prediction |
| `USE_SPERT` | `true` | Enable SPERT NER+RE extraction |
| `SPERT_MODEL_PATH` | `models/spert_supply_chain` | Path to fine-tuned SPERT model |
| `SPERT_TYPES_PATH` | `data/spert/supply_chain_types.json` | SPERT entity/relation type definitions |
| `SPERT_CONFIDENCE_THRESHOLD` | `0.4` | Min confidence for SPERT predictions |
| `SPERT_MAX_SPAN_SIZE` | `10` | Max entity span size |
| `SPERT_REL_FILTER_THRESHOLD` | `0.4` | Relation filter threshold |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## 🧪 Testing

### Quick Test

```bash
# Start the server
uvicorn backend.main:app --reload

# Test standard pipeline
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"news_text": "A massive earthquake struck Taiwan. TSMC has halted production. Apple and NVIDIA face chip shortages."}'

# Test dynamic pipeline (with GraphSAGE)
curl -X POST http://localhost:8000/dynamic-analyze \
  -H "Content-Type: application/json" \
  -d '{"news": "A massive earthquake struck Taiwan. TSMC has halted production."}'
```

### Train & Test RL Agent

```bash
python -m backend.agents.rl.train --episodes 5000
# Enable: USE_RL_DECISION=true in .env
```

### Train & Test GraphSAGE

```bash
python scripts/train_graphsage.py
# Enable: USE_GRAPHSAGE=true in .env
```

### Train & Test SPERT

```bash
# Run extraction demo (rule-based fallback, no training needed)
python scripts/train_spert.py predict

# Fine-tune SPERT on supply-chain data
cd spert && python spert.py train --config ../data/spert/supply_chain_train.conf
# Copy final model: cp models/spert_supply_chain/.../final_model/* models/spert_supply_chain/
```

---

## 📝 License

This project is for educational and demonstration purposes.
updated
