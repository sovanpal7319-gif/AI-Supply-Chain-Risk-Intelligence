/* ════════════════════════════════════════════════════════════════════════════
   Supply Chain Disruption Monitor — React Application (v3 — Dynamic Pipeline)
   ════════════════════════════════════════════════════════════════════════════ */

const { useState, useEffect, useRef, useCallback } = React;

const PIPELINE_STEPS = [
    { icon: "🔴", label: "Extraction", key: "extraction" },
    { icon: "🏭", label: "Enrichment", key: "enrichment" },
    { icon: "🔧", label: "Graph Build", key: "graph_build" },
    { icon: "🔵", label: "KG Traversal", key: "supply_chain" },
    { icon: "🧠", label: "GraphSAGE Risk", key: "risk" },
    { icon: "🟢", label: "Decisions", key: "decisions" },
    { icon: "🟣", label: "Alternatives", key: "alternatives" },
];

// ─── Toast Component ──────────────────────────────────────────────────────────
function ToastContainer({ toasts, onDismiss }) {
    return (
        <div className="toast-container">
            {toasts.map((t) => (
                <div key={t.id} className={`toast ${t.type}`} onClick={() => onDismiss(t.id)}>
                    <span>{t.type === "error" ? "❌" : "✅"}</span>
                    <span>{t.message}</span>
                </div>
            ))}
        </div>
    );
}

// ─── Pipeline Bar ─────────────────────────────────────────────────────────────
function PipelineBar({ activeStep, completed }) {
    return (
        <div className="pipeline-bar fade-in">
            {PIPELINE_STEPS.map((step, i) => (
                <React.Fragment key={step.key}>
                    {i > 0 && <span className="pipeline-arrow">→</span>}
                    <div className={`pipeline-step ${completed ? "done" : activeStep === i ? "active" : ""}`}>
                        <span>{step.icon}</span>
                        <span>{step.label}</span>
                        {completed && <span>✓</span>}
                    </div>
                </React.Fragment>
            ))}
        </div>
    );
}

// ─── News Search Section ──────────────────────────────────────────────────────
function NewsSearch({ onArticleSelect, addToast }) {
    const [query, setQuery] = useState("");
    const [articles, setArticles] = useState([]);
    const [searching, setSearching] = useState(false);
    const [headlinesLoaded, setHeadlinesLoaded] = useState(false);

    const handleSearch = async () => {
        if (!query.trim() || query.trim().length < 2) {
            addToast("Enter at least 2 characters to search.", "error");
            return;
        }
        setSearching(true);
        try {
            const res = await fetch(`/news/search?q=${encodeURIComponent(query.trim())}&page_size=12`);
            if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Search failed"); }
            const data = await res.json();
            setArticles(data.articles || []);
            setHeadlinesLoaded(false);
            if ((data.articles || []).length === 0) addToast("No articles found. Try different keywords.", "error");
            else addToast(`Found ${data.articles.length} articles`, "success");
        } catch (err) { addToast(err.message, "error"); }
        finally { setSearching(false); }
    };

    const handleHeadlines = async () => {
        setSearching(true);
        try {
            const res = await fetch("/news/headlines?page_size=12");
            if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Failed to fetch headlines"); }
            const data = await res.json();
            setArticles(data.articles || []);
            setHeadlinesLoaded(true);
            setQuery("");
            if ((data.articles || []).length === 0) addToast("No headlines found.", "error");
            else addToast(`Loaded ${data.articles.length} supply chain headlines`, "success");
        } catch (err) { addToast(err.message, "error"); }
        finally { setSearching(false); }
    };

    const handleKeyDown = (e) => { if (e.key === "Enter") handleSearch(); };

    return (
        <div className="input-section">
            <div className="card input-card">
                <div className="card-header">
                    <div className="card-icon">📰</div>
                    <span className="card-title">Live News Feed</span>
                    <span className="card-count">NewsAPI.org</span>
                </div>

                <div className="news-search-bar">
                    <input
                        id="news-search-input"
                        type="text"
                        className="search-input"
                        placeholder="Search supply chain news… (e.g. semiconductor shortage, earthquake Taiwan)"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={handleKeyDown}
                        disabled={searching}
                    />
                    <button className="btn-search" onClick={handleSearch} disabled={searching || !query.trim()} id="search-btn">
                        {searching ? <><div className="loading-spinner" style={{width:16,height:16,borderWidth:2}}></div></> : <>🔍 Search</>}
                    </button>
                    <button className="btn-headlines" onClick={handleHeadlines} disabled={searching} id="headlines-btn">
                        {searching ? "…" : "⚡ Supply Chain Headlines"}
                    </button>
                </div>

                {headlinesLoaded && articles.length > 0 && (
                    <div className="headlines-badge">Showing curated supply chain disruption headlines</div>
                )}
            </div>

            {articles.length > 0 && (
                <div className="news-grid fade-in">
                    {articles.map((article, i) => (
                        <NewsCard key={i} article={article} onSelect={onArticleSelect} />
                    ))}
                </div>
            )}
        </div>
    );
}

// ─── News Article Card ────────────────────────────────────────────────────────
function NewsCard({ article, onSelect }) {
    const timeAgo = (dateStr) => {
        if (!dateStr) return "";
        const diff = Date.now() - new Date(dateStr).getTime();
        const hrs = Math.floor(diff / 3600000);
        if (hrs < 1) return `${Math.floor(diff / 60000)}m ago`;
        if (hrs < 24) return `${hrs}h ago`;
        return `${Math.floor(hrs / 24)}d ago`;
    };

    return (
        <div className="news-card">
            {article.image_url && (
                <div className="news-card-image">
                    <img src={article.image_url} alt="" onError={(e) => e.target.style.display='none'} />
                </div>
            )}
            <div className="news-card-body">
                <div className="news-card-source">
                    <span className="source-name">{article.source}</span>
                    <span className="source-time">{timeAgo(article.published_at)}</span>
                </div>
                <div className="news-card-title">{article.title}</div>
                {article.description && (
                    <div className="news-card-desc">{article.description.substring(0, 150)}{article.description.length > 150 ? "…" : ""}</div>
                )}
                <div className="news-card-actions">
                    <button className="btn-analyze-article" onClick={() => onSelect(article)} id={`analyze-article-${article.title?.substring(0,10)?.replace(/\s/g,'')}`}>
                        🚀 Analyze This
                    </button>
                    {article.url && (
                        <a href={article.url} target="_blank" rel="noopener noreferrer" className="btn-read-more">
                            ↗ Read Full
                        </a>
                    )}
                </div>
            </div>
        </div>
    );
}

// ─── Stats Bar ────────────────────────────────────────────────────────────────
function StatsBar({ data }) {
    const riskMode = data.risk_scores?.[0]?.risk_mode || "rule_based";
    const stats = [
        { value: data.affected_downstream_companies?.length || 0, label: "Affected Companies" },
        { value: data.risk_scores?.filter(r => r.risk_level === "HIGH").length || 0, label: "High Risk" },
        { value: data.risk_scores?.filter(r => r.risk_level === "MEDIUM").length || 0, label: "Medium Risk" },
        { value: riskMode === "hybrid" ? "🧠 Hybrid" : "📏 Rules", label: "Risk Mode" },
        { value: `${data.processing_time_ms || 0}ms`, label: "Processing Time" },
    ];
    return (
        <div className="stats-bar fade-in">
            {stats.map((s, i) => (
                <div key={i} className="stat-card">
                    <div className="stat-value">{s.value}</div>
                    <div className="stat-label">{s.label}</div>
                </div>
            ))}
        </div>
    );
}

// ─── Disruption Summary Card ──────────────────────────────────────────────────
function DisruptionCard({ extraction }) {
    if (!extraction) return null;
    const typeLabels = {
        natural_disaster: "🌋 Natural Disaster", geopolitical: "🏛️ Geopolitical",
        earthquake: "🌋 Earthquake", fire: "🔥 Fire", flood: "🌊 Flood",
        labor: "👷 Labor", strike: "👷 Strike", pandemic: "🦠 Pandemic",
        operational: "⚙️ Operational", financial: "💰 Financial",
        logistics: "🚢 Logistics", supply: "📦 Supply", sanctions: "🏛️ Sanctions",
        cyber_attack: "💻 Cyber Attack", unknown: "❓ Unknown",
    };
    return (
        <div className="card fade-in">
            <div className="card-header">
                <div className="card-icon">📰</div>
                <span className="card-title">Disruption Extraction</span>
                <span className="card-count">Step 1</span>
            </div>
            <div className="disruption-meta">
                <div className="meta-item"><div className="meta-label">Company</div><div className="meta-value">{extraction.company || "N/A"}</div></div>
                <div className="meta-item"><div className="meta-label">Event</div><div className="meta-value">{typeLabels[extraction.event_type] || extraction.event_type}</div></div>
                <div className="meta-item"><div className="meta-label">Severity</div><div className="meta-value"><span className={`risk-badge ${extraction.severity}`}>{extraction.severity?.toUpperCase()}</span></div></div>
                <div className="meta-item"><div className="meta-label">Location</div><div className="meta-value">{extraction.location || "N/A"}</div></div>
            </div>
            {extraction.summary && <div className="disruption-summary-text">{extraction.summary}</div>}
        </div>
    );
}

// ─── Dynamic Graph Stats Card ─────────────────────────────────────────────────
function GraphBuildCard({ summary, enrichment }) {
    if (!summary) return null;
    return (
        <div className="card fade-in">
            <div className="card-header">
                <div className="card-icon">🔧</div>
                <span className="card-title">Dynamic Graph Construction</span>
                <span className="card-count">Step 2-3</span>
            </div>
            <div className="disruption-meta">
                <div className="meta-item"><div className="meta-label">Companies</div><div className="meta-value">{summary.companies_created}</div></div>
                <div className="meta-item"><div className="meta-label">Relationships</div><div className="meta-value">{summary.supplies_to_created}</div></div>
                <div className="meta-item"><div className="meta-label">Events</div><div className="meta-value">{summary.events_created}</div></div>
                <div className="meta-item"><div className="meta-label">Total Nodes</div><div className="meta-value">{summary.total_nodes}</div></div>
            </div>
            {enrichment && (
                <div style={{ marginTop: "0.75rem" }}>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.5rem" }}>Enriched Ecosystem ({enrichment.source})</div>
                    <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                        <div style={{ flex: 1, minWidth: "140px" }}>
                            <div style={{ fontSize: "0.75rem", color: "var(--text-accent)", marginBottom: "0.3rem" }}>↑ Suppliers</div>
                            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.3rem" }}>
                                {(enrichment.suppliers || []).map((s, i) => <span key={i} className="alt-chip" style={{ fontSize: "0.72rem", padding: "0.2rem 0.5rem" }}>{s}</span>)}
                            </div>
                        </div>
                        <div style={{ flex: 1, minWidth: "140px" }}>
                            <div style={{ fontSize: "0.75rem", color: "var(--accent-cyan)", marginBottom: "0.3rem" }}>↓ Customers</div>
                            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.3rem" }}>
                                {(enrichment.customers || []).map((c, i) => <span key={i} className="alt-chip" style={{ fontSize: "0.72rem", padding: "0.2rem 0.5rem" }}>{c}</span>)}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

// ─── Risk Assessment Table ────────────────────────────────────────────────────
function RiskTable({ assessments }) {
    if (!assessments || assessments.length === 0) return null;
    const hasGraphSAGE = assessments.some(r => r.risk_mode === "hybrid");
    return (
        <div className="card fade-in">
            <div className="card-header">
                <div className="card-icon">⚠️</div>
                <span className="card-title">Hybrid Risk Assessment</span>
                <span className="card-count">{hasGraphSAGE ? "🧠 GraphSAGE 70% + Rules 30%" : "Rule-based"}</span>
            </div>
            <div className="risk-table-wrapper">
                <table className="risk-table" id="risk-assessment-table">
                    <thead><tr><th>Company</th><th>Country</th><th>Industry</th><th>Hybrid Score</th>{hasGraphSAGE && <th>GS Score</th>}<th>Level</th><th>Depth</th><th>Path</th></tr></thead>
                    <tbody>
                        {assessments.map((r, i) => (
                            <tr key={i}>
                                <td style={{ fontWeight: 600, color: "var(--text-primary)" }}>{r.company}</td>
                                <td>{r.country}</td>
                                <td>{r.industry}</td>
                                <td>
                                    {r.risk_score.toFixed(3)}
                                    <div className="risk-score-bar"><div className={`risk-score-fill ${r.risk_level.toLowerCase()}`} style={{ width: `${r.risk_score * 100}%` }} /></div>
                                </td>
                                {hasGraphSAGE && <td style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "0.8rem", color: "var(--text-accent)" }}>{r.graphsage_risk != null ? r.graphsage_risk.toFixed(4) : "—"}</td>}
                                <td><span className={`risk-badge ${r.risk_level.toLowerCase()}`}>{r.risk_level}</span></td>
                                <td style={{ textAlign: "center" }}>{r.depth}</td>
                                <td><div className="path-chain">{r.path?.map((node, j) => (<React.Fragment key={j}>{j > 0 && <span className="path-arrow">→</span>}<span className={`path-node ${j === 0 ? "source" : ""}`}>{node}</span></React.Fragment>))}</div></td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

// ─── Recommendations Card ─────────────────────────────────────────────────────
function RecommendationsCard({ decisions }) {
    if (!decisions || decisions.length === 0) return null;
    return (
        <div className="card fade-in">
            <div className="card-header">
                <div className="card-icon">💡</div>
                <span className="card-title">Recommendations</span>
                <span className="card-count">{decisions.length} actions</span>
            </div>
            <div className="recommendation-list">
                {decisions.map((d, i) => (
                    <div key={i} className={`recommendation-item ${d.risk_level.toLowerCase()}`}>
                        <div className="rec-icon">{d.icon}</div>
                        <div className="rec-content">
                            <div className="rec-company">{d.company}<span style={{ marginLeft: "0.5rem", fontSize: "0.75rem", color: "var(--text-muted)" }}>{d.country}</span></div>
                            <div className="rec-action">{d.action}</div>
                            <div className="rec-text">{d.recommendation}</div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

// ─── Alternative Suppliers Card ───────────────────────────────────────────────
function AlternativesCard({ alternatives }) {
    if (!alternatives || alternatives.length === 0) return null;
    return (
        <div className="card fade-in">
            <div className="card-header">
                <div className="card-icon">🔄</div>
                <span className="card-title">Alternative Suppliers</span>
                <span className="card-count">{alternatives.length} groups</span>
            </div>
            {alternatives.map((group, i) => (
                <div key={i} className="alt-supplier-group">
                    <div className="alt-supplier-header">
                        <span className={`risk-badge ${group.risk_level.toLowerCase()}`}>{group.risk_level}</span>
                        <span>Alternatives for <strong>{group.company}</strong>:</span>
                    </div>
                    <div className="alt-supplier-chips">
                        {group.alternatives.map((alt, j) => (
                            <div key={j} className="alt-chip">{alt.name}<span className="alt-chip-country">({alt.country})</span></div>
                        ))}
                    </div>
                </div>
            ))}
        </div>
    );
}

// ─── D3 Supply Chain Graph (Production-Grade) ────────────────────────────────
function SupplyChainGraph({ graphData, affectedCompanies, riskScores }) {
    const svgRef = useRef(null);
    const tooltipRef = useRef(null);
    const containerRef = useRef(null);
    const [filterAffectedOnly, setFilterAffectedOnly] = useState(false);

    useEffect(() => {
        if (!graphData || !graphData.nodes || graphData.nodes.length === 0) return;
        const container = containerRef.current;
        const width = container.clientWidth;
        const height = container.clientHeight;
        const affectedSet = new Set(affectedCompanies || []);
        const riskMap = {};
        (riskScores || []).forEach(r => { riskMap[r.company] = r.risk_score || 0; });

        d3.select(svgRef.current).selectAll("*").remove();
        const svg = d3.select(svgRef.current).attr("viewBox", [0, 0, width, height]);
        const g = svg.append("g");
        svg.call(d3.zoom().scaleExtent([0.3, 5]).on("zoom", (event) => { g.attr("transform", event.transform); }));

        const companyNodes = graphData.nodes.filter(n => {
            if (!n.id || n.id === "Unknown") return false;
            if (n.label && n.label !== "Company") return false;
            if (n.event_type || n.severity) return false;
            if (/^[a-z_]+_[A-Za-z]+_\d{14}_/.test(n.id)) return false;
            return true;
        });

        let filteredNodes = filterAffectedOnly
            ? companyNodes.filter(n => affectedSet.has(n.id)) : companyNodes;
        if (filteredNodes.length === 0) filteredNodes = companyNodes;

        const nodes = filteredNodes.map(n => ({
            ...n, affected: affectedSet.has(n.id), riskScore: riskMap[n.id] || 0,
        }));
        const nodeIds = new Set(nodes.map(n => n.id));
        const degreeMap = {};
        graphData.edges.forEach(e => {
            if (nodeIds.has(e.source) && nodeIds.has(e.target)) {
                degreeMap[e.source] = (degreeMap[e.source] || 0) + 1;
                degreeMap[e.target] = (degreeMap[e.target] || 0) + 1;
            }
        });
        const links = graphData.edges
            .filter(e => (!e.type || e.type === "SUPPLIES_TO") && nodeIds.has(e.source) && nodeIds.has(e.target))
            .map(e => ({ ...e }));

        const riskColor = (d) => {
            if (d.affected) {
                const score = d.riskScore || 0.7;
                if (score >= 0.65) return "#ef4444";
                if (score >= 0.35) return "#f59e0b";
                return "#22c55e";
            }
            return "#6366f1";
        };
        const nodeRadius = (d) => Math.min((d.affected ? 9 : 6) + (degreeMap[d.id] || 0) * 1.2, 18);

        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-280))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("x", d3.forceX(width / 2).strength(0.04))
            .force("y", d3.forceY(height / 2).strength(0.04))
            .force("collision", d3.forceCollide().radius(d => nodeRadius(d) + 8));

        const defs = svg.append("defs");
        defs.append("marker").attr("id", "arrowhead").attr("viewBox", "0 -5 10 10")
            .attr("refX", 22).attr("refY", 0).attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
            .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "rgba(255,255,255,0.2)");
        defs.append("marker").attr("id", "arrowhead-red").attr("viewBox", "0 -5 10 10")
            .attr("refX", 22).attr("refY", 0).attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
            .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "rgba(239,68,68,0.6)");
        const filter = defs.append("filter").attr("id", "glow");
        filter.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "coloredBlur");
        const fm = filter.append("feMerge"); fm.append("feMergeNode").attr("in", "coloredBlur"); fm.append("feMergeNode").attr("in", "SourceGraphic");

        const isDisrupted = (d) => {
            const s = typeof d.source === 'object' ? d.source.id : d.source;
            const t = typeof d.target === 'object' ? d.target.id : d.target;
            return affectedSet.has(s) && affectedSet.has(t);
        };

        const link = g.append("g").selectAll("line").data(links).join("line")
            .attr("stroke", d => isDisrupted(d) ? "rgba(239,68,68,0.5)" : "rgba(148,163,184,0.12)")
            .attr("stroke-width", d => isDisrupted(d) ? 2 : 0.8)
            .attr("marker-end", d => isDisrupted(d) ? "url(#arrowhead-red)" : "url(#arrowhead)");

        const edgeLabel = g.append("g").selectAll("text").data(links).join("text")
            .text("supplies to").attr("fill", "rgba(148,163,184,0)").attr("font-size", "7px")
            .attr("font-family", "Inter, sans-serif").attr("text-anchor", "middle").attr("dy", -4);

        link.on("mouseenter", function(event, d) {
            d3.select(edgeLabel.nodes()[links.indexOf(d)]).attr("fill", "rgba(148,163,184,0.7)");
            d3.select(this).attr("stroke-width", 3);
        }).on("mouseleave", function(event, d) {
            d3.select(edgeLabel.nodes()[links.indexOf(d)]).attr("fill", "rgba(148,163,184,0)");
            d3.select(this).attr("stroke-width", isDisrupted(d) ? 2 : 0.8);
        });

        const node = g.append("g").selectAll("g").data(nodes).join("g")
            .call(d3.drag()
                .on("start", (ev, d) => { if (!ev.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
                .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
                .on("end", (ev, d) => { if (!ev.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
            );

        node.append("circle").attr("r", d => nodeRadius(d))
            .attr("fill", d => riskColor(d))
            .attr("stroke", d => d.affected ? "rgba(239,68,68,0.4)" : "rgba(99,102,241,0.25)")
            .attr("stroke-width", d => d.affected ? 3 : 1.5)
            .style("filter", d => d.affected ? "url(#glow)" : "none").style("cursor", "pointer");

        node.append("text").text(d => d.id.length > 14 ? d.id.substring(0, 14) + "…" : d.id)
            .attr("x", d => nodeRadius(d) + 4).attr("y", 4)
            .attr("fill", d => d.affected ? "#fca5a5" : "#94a3b8")
            .attr("font-size", d => d.affected ? "10px" : "9px")
            .attr("font-family", "Inter, sans-serif").attr("font-weight", d => d.affected ? "600" : "400");

        const tooltip = d3.select(tooltipRef.current);
        node.on("mouseenter", (event, d) => {
            const deg = degreeMap[d.id] || 0;
            const risk = d.riskScore ? (d.riskScore * 100).toFixed(0) + "%" : "N/A";
            tooltip.style("display", "block").style("left", (event.offsetX + 15) + "px").style("top", (event.offsetY - 10) + "px")
                .html(`<div class="tooltip-name">${d.id}</div>
                       <div class="tooltip-detail">${d.industry || "—"} · ${d.country || "—"}</div>
                       <div class="tooltip-detail">Connections: ${deg} · Risk: ${risk}</div>
                       ${d.affected ? '<div class="tooltip-detail" style="color:#ef4444;">⚠ Disruption impact</div>' : ''}`);
        }).on("mouseleave", () => { tooltip.style("display", "none"); });

        // Focus mode: hover a node dims everything else
        node.on("mouseover", function(event, d) {
            const connected = new Set([d.id]);
            links.forEach(l => {
                const s = typeof l.source === 'object' ? l.source.id : l.source;
                const t = typeof l.target === 'object' ? l.target.id : l.target;
                if (s === d.id) connected.add(t); if (t === d.id) connected.add(s);
            });
            node.select("circle").style("opacity", n => connected.has(n.id) ? 1 : 0.15);
            node.select("text").style("opacity", n => connected.has(n.id) ? 1 : 0.1);
            link.style("opacity", l => {
                const s = typeof l.source === 'object' ? l.source.id : l.source;
                const t = typeof l.target === 'object' ? l.target.id : l.target;
                return (s === d.id || t === d.id) ? 1 : 0.05;
            });
        }).on("mouseout", () => {
            node.select("circle").style("opacity", 1); node.select("text").style("opacity", 1); link.style("opacity", 1);
        });

        simulation.on("tick", () => {
            link.attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y);
            edgeLabel.attr("x", d => (d.source.x + d.target.x) / 2).attr("y", d => (d.source.y + d.target.y) / 2);
            node.attr("transform", d => `translate(${d.x},${d.y})`);
        });
        return () => simulation.stop();
    }, [graphData, affectedCompanies, riskScores, filterAffectedOnly]);

    const nc = graphData?.nodes?.filter(n => n.id && n.id !== "Unknown" && !n.event_type).length || 0;
    const ec = graphData?.edges?.filter(e => !e.type || e.type === "SUPPLIES_TO").length || 0;

    return (
        <div className="card full-width fade-in">
            <div className="card-header">
                <div className="card-icon">🕸️</div>
                <span className="card-title">Supply Chain Network Graph</span>
                <span className="card-count">{nc} companies · {ec} edges</span>
            </div>
            <div className="graph-toolbar" id="graph-toolbar">
                <label className="graph-filter-toggle">
                    <input type="checkbox" checked={filterAffectedOnly} onChange={e => setFilterAffectedOnly(e.target.checked)} />
                    <span>Affected only</span>
                </label>
            </div>
            <div className="graph-container" ref={containerRef}>
                <svg ref={svgRef}></svg>
                <div className="graph-tooltip" ref={tooltipRef} style={{ display: "none" }}></div>
            </div>
            <div className="graph-legend">
                <div className="legend-item"><div className="legend-dot" style={{background:"#ef4444"}}></div> High Risk</div>
                <div className="legend-item"><div className="legend-dot" style={{background:"#f59e0b"}}></div> Medium Risk</div>
                <div className="legend-item"><div className="legend-dot" style={{background:"#22c55e"}}></div> Low Risk</div>
                <div className="legend-item"><div className="legend-dot" style={{background:"#6366f1"}}></div> Unaffected</div>
                <div className="legend-item"><div className="legend-dot" style={{background:"rgba(239,68,68,0.5)",width:"16px",borderRadius:"2px"}}></div> Disrupted Path</div>
            </div>
        </div>
    );
}

// ─── Loading / Empty States ───────────────────────────────────────────────────
function LoadingState() {
    return (
        <div className="card full-width">
            <div className="loading-overlay">
                <div className="loading-spinner"></div>
                <div className="loading-text">Agents are analyzing the disruption…</div>
                <div className="loading-agents">
                    <div className="agent-dot" title="Disruption Detection"></div>
                    <div className="agent-dot" title="Graph Traversal"></div>
                    <div className="agent-dot" title="Risk Assessment"></div>
                    <div className="agent-dot" title="Decision Engine"></div>
                    <div className="agent-dot" title="Alternative Suppliers"></div>
                </div>
            </div>
        </div>
    );
}

function EmptyState() {
    return (
        <div className="card full-width">
            <div className="empty-state">
                <div className="empty-icon">📰</div>
                <div className="empty-title">Search for Live News</div>
                <div className="empty-subtitle">
                    Search for supply chain news above, or click "⚡ Supply Chain Headlines" to load curated disruption news. Then click "Analyze This" on any article to run the full dynamic pipeline with GraphSAGE risk scoring.
                </div>
            </div>
        </div>
    );
}

// ─── Selected Article Banner ──────────────────────────────────────────────────
function SelectedArticleBanner({ article }) {
    if (!article) return null;
    return (
        <div className="selected-article-banner fade-in">
            <div className="selected-article-label">📋 Analyzing Article</div>
            <div className="selected-article-title">{article.title}</div>
            <div className="selected-article-source">{article.source} · {article.published_at ? new Date(article.published_at).toLocaleDateString() : ""}</div>
        </div>
    );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
function ManualInput({ onAnalyze, loading }) {
    const [text, setText] = useState("");
    return (
        <div className="card fade-in" style={{ marginTop: "1rem" }}>
            <div className="card-header">
                <div className="card-icon">✏️</div>
                <span className="card-title">Manual Analysis</span>
                <span className="card-count">Dynamic Pipeline</span>
            </div>
            <textarea
                id="manual-input"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste or type disruption news here… (e.g. A massive earthquake struck Taiwan, forcing TSMC to halt all semiconductor production.)"
                disabled={loading}
                style={{
                    width: "100%", minHeight: "80px", background: "var(--bg-input)",
                    border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-md)",
                    padding: "0.75rem 1rem", color: "var(--text-primary)",
                    fontFamily: "'Inter', sans-serif", fontSize: "0.88rem",
                    resize: "vertical", lineHeight: 1.6,
                }}
            />
            <button
                id="manual-analyze-btn"
                onClick={() => onAnalyze(text)}
                disabled={loading || text.trim().length < 10}
                className="btn-search"
                style={{ marginTop: "0.75rem", width: "100%", justifyContent: "center" }}
            >
                {loading ? <><div className="loading-spinner" style={{width:16,height:16,borderWidth:2}}></div> Analyzing…</> : <>🚀 Run Dynamic Pipeline (BERT → GraphSAGE → DQN)</>}
            </button>
        </div>
    );
}

function App() {
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [toasts, setToasts] = useState([]);
    const [selectedArticle, setSelectedArticle] = useState(null);

    const addToast = useCallback((message, type = "success") => {
        const id = Date.now();
        setToasts(prev => [...prev, { id, message, type }]);
        setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4000);
    }, []);

    const dismissToast = useCallback((id) => {
        setToasts(prev => prev.filter(t => t.id !== id));
    }, []);

    const runDynamicPipeline = async (newsText) => {
        if (!newsText || newsText.trim().length < 10) {
            addToast("Text too short for analysis.", "error");
            return;
        }
        setLoading(true);
        setResult(null);

        try {
            const response = await fetch("/dynamic-analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ news: newsText.trim() }),
            });
            if (!response.ok) { const err = await response.json(); throw new Error(err.detail || "Analysis failed"); }
            const data = await response.json();
            setResult(data);
            const mode = data.risk_scores?.[0]?.risk_mode || "rule_based";
            addToast(`Dynamic analysis complete in ${data.processing_time_ms}ms (${mode})`, "success");
        } catch (err) { addToast(err.message, "error"); }
        finally { setLoading(false); }
    };

    const handleArticleSelect = async (article) => {
        if (!article.full_text || article.full_text.length < 10) {
            addToast("Article text too short for analysis.", "error");
            return;
        }
        setSelectedArticle(article);
        await runDynamicPipeline(article.full_text);
    };

    return (
        <div>
            <header className="app-header">
                <div className="header-brand">
                    <div className="header-logo">🔗</div>
                    <div>
                        <div className="header-title">Supply Chain Disruption Monitor</div>
                        <div className="header-subtitle">BERT + GraphSAGE + DQN · Dynamic KG Pipeline</div>
                    </div>
                </div>
                <div className="header-status">
                    <div className="status-dot"></div>
                    <span>System Online</span>
                    {result && <span className="processing-time">⚡ {result.processing_time_ms}ms</span>}
                </div>
            </header>

            <div className="app-container">
                <NewsSearch onArticleSelect={handleArticleSelect} addToast={addToast} />
                <ManualInput onAnalyze={runDynamicPipeline} loading={loading} />

                {(loading || result) && <PipelineBar activeStep={loading ? 2 : -1} completed={!!result} />}
                {loading && <SelectedArticleBanner article={selectedArticle} />}
                {loading && <LoadingState />}

                {result && !loading && (
                    <>
                        {selectedArticle && <SelectedArticleBanner article={selectedArticle} />}
                        <StatsBar data={result} />
                        <div className="results-grid">
                            <DisruptionCard extraction={result.extraction} />
                            <GraphBuildCard summary={result.generated_graph_summary} enrichment={result.enrichment} />
                        </div>
                        <div className="results-grid"><RiskTable assessments={result.risk_scores} /></div>
                        <div className="results-grid">
                            <RecommendationsCard decisions={result.decisions} />
                            <AlternativesCard alternatives={result.alternative_suppliers} />
                        </div>
                        <div className="results-grid">
                            <SupplyChainGraph graphData={result.graph_data} affectedCompanies={result.affected_downstream_companies || []} riskScores={result.risk_scores || []} />
                        </div>
                    </>
                )}
                {!result && !loading && <EmptyState />}
            </div>
            <ToastContainer toasts={toasts} onDismiss={dismissToast} />
        </div>
    );
}

// ─── Mount ────────────────────────────────────────────────────────────────────
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
