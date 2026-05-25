"""
Supply Chain Disruption Monitoring System — FastAPI Application

Orchestrates 5 modular agents in a pipeline to analyze news articles
for supply chain disruptions and generate risk assessments.

News articles are fetched from NewsAPI.org and analyzed using
Groq (LLaMA-3) + Neo4j knowledge graph.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from loguru import logger

from backend.config import settings
from backend.agents import (
    DisruptionAgent,
    EnhancedDisruptionAgent,
    KGQueryAgent,
    RiskAssessmentAgent,
    GraphSAGERiskAgent,
    DecisionAgent,
    AlternativeSupplierAgent,
    SpertAgent,
)
from backend.services.neo4j_service import Neo4jService
from backend.services.news_service import NewsService
from backend.services.company_enrichment_service import CompanyEnrichmentService
from backend.services.dynamic_graph_builder import DynamicGraphBuilder
from backend.services.spert_validation import SpertValidationService


# ── Configure logging ────────────────────────────────────────────────────────
logger.add(
    "logs/supply_chain_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level=settings.log_level,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
)

# ── Initialize agents & services ─────────────────────────────────────────────
disruption_agent =DisruptionAgent()
enhanced_disruption_agent = EnhancedDisruptionAgent()
kg_query_agent = KGQueryAgent()
risk_agent = RiskAssessmentAgent()
decision_agent = DecisionAgent()
alt_supplier_agent = AlternativeSupplierAgent()
neo4j_service = Neo4jService()
news_service = NewsService()
enrichment_service = CompanyEnrichmentService()
graph_builder = DynamicGraphBuilder()
graphsage_risk_agent = GraphSAGERiskAgent()
spert_agent = SpertAgent()
spert_validation = SpertValidationService(confidence_threshold=settings.spert_confidence_threshold)


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Supply Chain Disruption Monitor starting up")
    logger.info("   LLM: Groq ({}) | DB: Neo4j ({})", settings.groq_model, settings.neo4j_uri)
    yield
    logger.info("🛑 Shutting down")
    from backend.db.connection import Neo4jConnection
    Neo4jConnection.close()


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Supply Chain Disruption Monitor",
    description="Multi-agent system for detecting and assessing supply chain disruptions",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    news_text: str = Field(..., min_length=10, description="News article text to analyze")


class AnalyzeResponse(BaseModel):
    status: str
    processing_time_ms: float
    disruption: dict
    supply_chain: dict
    risk_assessments: list
    decisions: list
    alternative_suppliers: list
    graph_data: dict


class DynamicAnalyzeRequest(BaseModel):
    news: str = Field(..., min_length=5, description="Disruption news headline or article")


class DynamicAnalyzeResponse(BaseModel):
    status: str
    processing_time_ms: float
    disrupted_company: str
    extraction: dict
    enrichment: dict
    spert_extraction: dict
    generated_graph_summary: dict
    affected_downstream_companies: list[str]
    supply_chain: dict
    rule_based_risk: list
    graphsage_risk: list
    risk_scores: list
    decisions: list
    alternative_suppliers: list
    graph_data: dict


# ── News Endpoints ───────────────────────────────────────────────────────────

@app.get("/news/search")
async def search_news(
    q: str = Query(..., min_length=2, description="Search query for news articles"),
    page_size: int = Query(default=10, ge=1, le=50, description="Number of articles to return"),
):
    """Search for news articles via NewsAPI.org."""
    try:
        return await news_service.search_news(query=q, page_size=page_size)
    except Exception as exc:
        logger.error("News search failed: {}", exc)
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/news/headlines")
async def get_headlines(
    page_size: int = Query(default=10, ge=1, le=50, description="Number of articles to return"),
):
    """Fetch curated supply chain disruption news headlines."""
    try:
        return await news_service.get_supply_chain_headlines(page_size=page_size)
    except Exception as exc:
        logger.error("Headlines fetch failed: {}", exc)
        raise HTTPException(status_code=502, detail=str(exc))


# ── Analysis Pipeline ────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_news(request: AnalyzeRequest):
    """
    Main pipeline endpoint.

    Orchestrates all 5 agents:
    1. Disruption detection (Groq LLM)
    2. Knowledge graph traversal (Neo4j)
    3. Risk assessment
    4. Decision generation
    5. Alternative supplier suggestion
    """
    start_time = time.time()
    logger.info("━" * 60)
    logger.info("📨 New analysis request ({} chars)", len(request.news_text))

    try:
        # ── Agent 1: Disruption Detection ────────────────────────────────
        disruption = await disruption_agent.run(request.news_text)

        # # ── SPERT: NER + RE Extraction ───────────────────────────────────
        # spert_raw = spert_agent.extract(request.news_text)
        # spert_data = spert_validation.validate(spert_raw)
        # logger.info(
        #     "SPERT ✅ Extracted {} entities, {} relations",
        #     len(spert_data["entities"]), len(spert_data["relations"]),
        # )

        # # ── Build SPERT-derived graph nodes/edges ────────────────────────
        # try:
        #     graph_builder.build_from_spert(spert_data, request.news_text)
        # except Exception as graph_exc:
        #     logger.warning("SPERT graph build failed (non-fatal): {}", graph_exc)

        # ── Agent 2: Knowledge Graph Query ───────────────────────────────
        supply_chain = kg_query_agent.run(disruption)

        # ── Agent 3: Risk Assessment ─────────────────────────────────────
        risk_assessments = risk_agent.run(disruption, supply_chain)

        # ── Agent 4: Decision Generation (Rule-based or RL) ────────────
        decisions = decision_agent.run(risk_assessments, disruption_data=disruption)

        # ── Agent 5: Alternative Suppliers ───────────────────────────────
        all_affected = supply_chain.get("all_affected_companies", [])
        alternatives = alt_supplier_agent.run(decisions, all_affected)

        # ── Get graph data for visualization ─────────────────────────────
        graph_data = neo4j_service.get_all_graph_data()

        # Mark affected nodes in graph data
        affected_set = set(all_affected)
        for node in graph_data.get("nodes", []):
            node["affected"] = node["id"] in affected_set

        processing_time = round((time.time() - start_time) * 1000, 2)

        logger.info("✅ Analysis complete in {}ms", processing_time)
        logger.info("━" * 60)

        return AnalyzeResponse(
            status="success",
            processing_time_ms=processing_time,
            disruption=disruption,
            supply_chain=supply_chain,
            risk_assessments=risk_assessments,
            decisions=decisions,
            alternative_suppliers=alternatives,
            graph_data=graph_data,
        )

    except Exception as exc:
        logger.error("Analysis failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from backend.db.connection import Neo4jConnection
    return {
        "status": "healthy",
        "llm": f"Groq ({settings.groq_model})",
        "neo4j_connected": Neo4jConnection.health_check(),
        "news_api": "NewsAPI.org",
    }


@app.get("/graph")
async def get_graph():
    """Return the full supply chain graph for visualization."""
    return neo4j_service.get_all_graph_data()


# ── Dynamic Analysis Pipeline ────────────────────────────────────────────────

@app.post("/dynamic-analyze", response_model=DynamicAnalyzeResponse)
async def dynamic_analyze(request: DynamicAnalyzeRequest):
    """
    Dynamic Knowledge Graph pipeline.

    Automatically builds a Neo4j graph neighborhood from live news,
    then runs the full 5-agent ET pipeline on it:

    1. Enhanced Disruption Extraction (company, event, location, severity)
    2. Company Ecosystem Enrichment (suppliers, customers, industry)
    3. Dynamic Neo4j Graph Construction (MERGE-based, idempotent)
    4. KG Traversal (Agent 2 — downstream impact)
    5. Risk Assessment (Agent 3)
    6. Decision Generation (Agent 4)
    7. Alternative Suppliers (Agent 5)
    """
    start_time = time.time()
    logger.info("━" * 60)
    logger.info("🔄 Dynamic analysis request: '{}'", request.news[:100])

    try:
        # ── Step 1: Enhanced Disruption Extraction ────────────────────
        extraction = await enhanced_disruption_agent.run(request.news)
        logger.info(
            "Step 1 ✅ Extracted: company={}, event={}, location={}",
            extraction.company, extraction.event_type, extraction.location,
        )

        # ── Step 1.5: SPERT NER+RE Extraction ────────────────────────
        spert_raw = spert_agent.extract(request.news)
        spert_data = spert_validation.validate(spert_raw)
        logger.info(
            "Step 1.5 ✅ SPERT: {} entities, {} relations",
            len(spert_data["entities"]), len(spert_data["relations"]),
        )

        # Merge SPERT-extracted companies into extraction's affected list
        for ent in spert_data.get("entities", []):
            if ent["type"] in ("COMPANY", "SUPPLIER"):
                if ent["text"] not in extraction.affected_companies:
                    extraction.affected_companies.append(ent["text"])
            if ent["type"] == "COUNTRY":
                if ent["text"] not in extraction.affected_countries:
                    extraction.affected_countries.append(ent["text"])

        # ── Step 2: Company Ecosystem Enrichment ─────────────────────
        ecosystem = await enrichment_service.enrich(extraction.company)
        logger.info(
            "Step 2 ✅ Enriched: {} suppliers, {} customers",
            len(ecosystem.suppliers), len(ecosystem.customers),
        )


        # ── Step 3: Dynamic Graph Construction ───────────────────────
        graph_summary = graph_builder.build_dynamic_graph(extraction, ecosystem)
        logger.info(
            "Step 3 ✅ Graph built: {} nodes, {} relationships",
            graph_summary.total_nodes, graph_summary.total_relationships,
        )

        # ── Step 3.5: SPERT → Neo4j Graph Nodes/Edges ────────────────
        try:
            spert_graph_summary = graph_builder.build_from_spert(
                spert_data, request.news,
            )
            # Merge summary counts
            graph_summary.companies_created += spert_graph_summary.companies_created
            graph_summary.regions_created += spert_graph_summary.regions_created
            graph_summary.events_created += spert_graph_summary.events_created
            graph_summary.supplies_to_created += spert_graph_summary.supplies_to_created
            graph_summary.impacts_created += spert_graph_summary.impacts_created
            graph_summary.located_in_created += spert_graph_summary.located_in_created
            graph_summary.total_nodes += spert_graph_summary.total_nodes
            graph_summary.total_relationships += spert_graph_summary.total_relationships
            logger.info(
                "Step 3.5 ✅ SPERT graph: +{} nodes, +{} rels",
                spert_graph_summary.total_nodes,
                spert_graph_summary.total_relationships,
            )
        except Exception as spert_graph_exc:
            logger.warning("SPERT graph build failed (non-fatal): {}", spert_graph_exc)

        # ── Step 4: KG Query (Agent 2) on dynamic graph ──────────────
        disruption_for_kg = {
            "disruption_type": extraction.event_type,
            "severity": extraction.severity,
            "affected_companies": [extraction.company] + extraction.affected_companies,
            "affected_countries": extraction.affected_countries,
            "summary": extraction.summary,
            "confidence": 1.0,
            "source": "enhanced_extraction+spert",
        }
        # Deduplicate company list
        seen = set()
        unique_companies = []
        for c in disruption_for_kg["affected_companies"]:
            if c not in seen:
                seen.add(c)
                unique_companies.append(c)
        disruption_for_kg["affected_companies"] = unique_companies

        supply_chain = kg_query_agent.run(disruption_for_kg)
        logger.info(
            "Step 4 ✅ KG traversal: {} paths, {} affected",
            len(supply_chain.get("supply_chain_paths", [])),
            supply_chain.get("total_downstream_affected", 0),
        )

        # ── Step 5: Hybrid Risk Assessment (GraphSAGE + Rules) ────────
        hybrid_risks, rb_risks, gs_risks = graphsage_risk_agent.run(
            disruption_for_kg, supply_chain,
        )
        risk_assessments = hybrid_risks  # backward-compatible for Agent 4/5
        logger.info(
            "Step 5 ✅ Risk assessed: {} companies (mode={})",
            len(risk_assessments),
            risk_assessments[0].get("risk_mode", "unknown") if risk_assessments else "n/a",
        )

        # ── Step 6: Decision Generation (Agent 4) ────────────────────
        decisions = decision_agent.run(risk_assessments, disruption_data=disruption_for_kg)
        logger.info("Step 6 ✅ Decisions generated: {}", len(decisions))

        # ── Step 7: Alternative Suppliers (Agent 5) ──────────────────
        all_affected = supply_chain.get("all_affected_companies", [])
        alternatives = alt_supplier_agent.run(decisions, all_affected)
        logger.info("Step 7 ✅ Alternatives found: {}", len(alternatives))

        # ── Get dynamic graph data for visualization ─────────────────
        graph_data = neo4j_service.get_dynamic_graph_data(extraction.company)

        processing_time = round((time.time() - start_time) * 1000, 2)
        logger.info("🔄 Dynamic analysis complete in {}ms", processing_time)
        logger.info("━" * 60)

        return DynamicAnalyzeResponse(
            status="success",
            processing_time_ms=processing_time,
            disrupted_company=extraction.company,
            extraction=extraction.model_dump(),
            enrichment=ecosystem.model_dump(),
            spert_extraction=spert_data,
            generated_graph_summary=graph_summary.model_dump(),
            affected_downstream_companies=all_affected,
            supply_chain=supply_chain,
            rule_based_risk=rb_risks,
            graphsage_risk=gs_risks,
            risk_scores=risk_assessments,
            decisions=decisions,
            alternative_suppliers=alternatives,
            graph_data=graph_data,
        )

    except Exception as exc:
        logger.error("Dynamic analysis failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Serve frontend ───────────────────────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))