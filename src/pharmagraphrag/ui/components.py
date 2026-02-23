"""Reusable Streamlit UI components.

Provides graph visualization, sources panel, and drug explorer widgets.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


# ---------------------------------------------------------------------------
# Graph visualization (streamlit-agraph)
# ---------------------------------------------------------------------------


def render_graph(graph_raw: dict[str, Any]) -> None:
    """Render an interactive knowledge graph from raw graph context.

    Uses streamlit-agraph to display Drug, AdverseEvent, Outcome, and
    DrugCategory nodes with their relationships.

    Args:
        graph_raw: Per-drug raw graph data from RetrievedContext.graph_raw.
    """
    if not graph_raw:
        st.info("No hay datos de grafo para visualizar.")
        return

    try:
        from streamlit_agraph import Config, Edge, Node, agraph
    except ImportError:
        st.warning("streamlit-agraph no está instalado.")
        return

    nodes_map: dict[str, Node] = {}
    edges: list[Edge] = []

    for drug_name, ctx in graph_raw.items():
        # Drug node (central)
        drug_id = f"drug_{drug_name}"
        if drug_id not in nodes_map:
            nodes_map[drug_id] = Node(
                id=drug_id,
                label=drug_name,
                size=30,
                color="#4CAF50",
                font={"color": "#ffffff", "size": 14},
                shape="dot",
            )

        # Adverse events
        for ae in (ctx.get("adverse_events") or [])[:10]:
            ae_name = ae.get("adverse_event", "")
            ae_id = f"ae_{ae_name}"
            count = ae.get("report_count", 0)
            if ae_id not in nodes_map:
                nodes_map[ae_id] = Node(
                    id=ae_id,
                    label=ae_name,
                    size=15,
                    color="#F44336",
                    font={"color": "#ffffff", "size": 11},
                    shape="dot",
                )
            edges.append(Edge(
                source=drug_id,
                target=ae_id,
                label=f"{count}",
                color="#F44336",
                width=max(1, min(5, count / 500)),
            ))

        # Interactions
        for inter in ctx.get("interactions") or []:
            other = inter.get("interacting_drug", "")
            other_id = f"drug_{other}"
            if other_id not in nodes_map:
                nodes_map[other_id] = Node(
                    id=other_id,
                    label=other,
                    size=25,
                    color="#2196F3",
                    font={"color": "#ffffff", "size": 13},
                    shape="dot",
                )
            edges.append(Edge(
                source=drug_id,
                target=other_id,
                label="INTERACTS",
                color="#FF9800",
                width=3,
            ))

        # Outcomes
        for out in ctx.get("outcomes") or []:
            out_desc = out.get("outcome_description", out.get("outcome_code", ""))
            out_id = f"out_{out_desc}"
            count = out.get("report_count", 0)
            if out_id not in nodes_map:
                nodes_map[out_id] = Node(
                    id=out_id,
                    label=out_desc,
                    size=18,
                    color="#9C27B0",
                    font={"color": "#ffffff", "size": 11},
                    shape="diamond",
                )
            edges.append(Edge(
                source=drug_id,
                target=out_id,
                label=f"{count}",
                color="#9C27B0",
                width=2,
            ))

        # Categories
        for cat in ctx.get("categories") or []:
            cat_id = f"cat_{cat}"
            if cat_id not in nodes_map:
                nodes_map[cat_id] = Node(
                    id=cat_id,
                    label=cat,
                    size=20,
                    color="#FF9800",
                    font={"color": "#ffffff", "size": 12},
                    shape="triangle",
                )
            edges.append(Edge(
                source=drug_id,
                target=cat_id,
                label="BELONGS_TO",
                color="#FF9800",
                width=2,
            ))

    if not nodes_map:
        st.info("No hay nodos para visualizar.")
        return

    config = Config(
        width=700,
        height=500,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
    )

    agraph(
        nodes=list(nodes_map.values()),
        edges=edges,
        config=config,
    )

    # Legend
    st.markdown(
        """
        <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:0.85em;margin-top:8px;">
            <span>🟢 Fármaco</span>
            <span>🔴 Evento Adverso</span>
            <span>🔵 Fármaco (interacción)</span>
            <span>🟣 Outcome</span>
            <span>🟠 Categoría</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sources / evidence panel
# ---------------------------------------------------------------------------


def render_sources(
    graph_raw: dict[str, Any],
    vector_raw: list[dict[str, Any]],
) -> None:
    """Render the sources/evidence panel.

    Shows which graph data and vector chunks contributed to the answer.

    Args:
        graph_raw: Per-drug raw graph data.
        vector_raw: Raw vector search results.
    """
    if not graph_raw and not vector_raw:
        st.info("No hay fuentes disponibles para esta consulta.")
        return

    # Graph sources
    if graph_raw:
        with st.expander("📊 Fuentes del Knowledge Graph", expanded=False):
            for drug_name, ctx in graph_raw.items():
                st.markdown(f"**{drug_name}**")
                drug_info = ctx.get("drug_info") or {}
                if drug_info:
                    brands = drug_info.get("brand_names") or []
                    route = drug_info.get("route", "")
                    if brands:
                        st.caption(f"Marcas: {', '.join(brands)}")
                    if route:
                        st.caption(f"Vía: {route}")

                ae_count = len(ctx.get("adverse_events") or [])
                inter_count = len(ctx.get("interactions") or [])
                out_count = len(ctx.get("outcomes") or [])
                cat_count = len(ctx.get("categories") or [])

                cols = st.columns(4)
                cols[0].metric("Eventos", ae_count)
                cols[1].metric("Interacc.", inter_count)
                cols[2].metric("Outcomes", out_count)
                cols[3].metric("Categ.", cat_count)

                st.divider()

    # Vector sources
    if vector_raw:
        with st.expander("📄 Fuentes de Etiquetas (Vector Search)", expanded=False):
            for i, vr in enumerate(vector_raw):
                meta = vr.get("metadata", {})
                drug = meta.get("drug_name", "Desconocido")
                section = meta.get("section", "").replace("_", " ").title()
                distance = vr.get("distance", None)
                text = vr.get("text", "")

                # Header with relevance score
                score_text = ""
                if distance is not None:
                    relevance = max(0, 1 - distance)
                    score_text = f" — Relevancia: {relevance:.0%}"

                st.markdown(f"**{i + 1}. {drug}** — _{section}_{score_text}")
                st.text(text[:300] + ("..." if len(text) > 300 else ""))
                st.divider()


# ---------------------------------------------------------------------------
# Drug explorer sidebar
# ---------------------------------------------------------------------------


def render_drug_explorer() -> str | None:
    """Render a drug search widget in the sidebar.

    Returns:
        Selected drug name, or None.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Explorador de Fármacos")

    drug_query = st.sidebar.text_input(
        "Buscar fármaco:",
        placeholder="Ej: ibuprofen, metformin...",
        key="drug_explorer_input",
    )

    if not drug_query or len(drug_query) < 2:
        return None

    try:
        from pharmagraphrag.graph.queries import search_drugs

        matches = search_drugs(drug_query, limit=10)
        if not matches:
            st.sidebar.warning("No se encontraron fármacos.")
            return None

        selected = st.sidebar.selectbox(
            "Resultados:",
            matches,
            key="drug_explorer_select",
        )
        return selected  # type: ignore[return-value]

    except Exception as exc:
        st.sidebar.error(f"Error buscando: {exc}")
        return None


def render_drug_detail(drug_name: str) -> None:
    """Render detailed drug information in the sidebar.

    Args:
        drug_name: Drug name to look up.
    """
    try:
        from pharmagraphrag.graph.queries import get_drug_full_context

        ctx = get_drug_full_context(drug_name)
    except Exception as exc:
        st.sidebar.error(f"Error: {exc}")
        return

    drug_info = ctx.get("drug_info") or {}
    if not drug_info:
        st.sidebar.warning(f"'{drug_name}' no encontrado en el grafo.")
        return

    st.sidebar.markdown(f"### {drug_info.get('name', drug_name)}")

    brands = drug_info.get("brand_names") or []
    if brands:
        st.sidebar.caption(f"**Marcas:** {', '.join(brands)}")

    route = drug_info.get("route", "")
    if route:
        st.sidebar.caption(f"**Vía:** {route}")

    categories = ctx.get("categories") or []
    if categories:
        st.sidebar.caption(f"**Clase:** {', '.join(categories)}")

    # Top adverse events
    events = ctx.get("adverse_events") or []
    if events:
        st.sidebar.markdown("**Top Eventos Adversos:**")
        for e in events[:5]:
            st.sidebar.text(f"  • {e['adverse_event']} ({e['report_count']})")

    # Interactions
    interactions = ctx.get("interactions") or []
    if interactions:
        st.sidebar.markdown("**Interacciones:**")
        for i in interactions[:5]:
            st.sidebar.text(f"  • {i['interacting_drug']}")

    # Outcomes
    outcomes = ctx.get("outcomes") or []
    if outcomes:
        st.sidebar.markdown("**Outcomes:**")
        for o in outcomes[:5]:
            desc = o.get("outcome_description", o.get("outcome_code", ""))
            st.sidebar.text(f"  • {desc} ({o['report_count']})")
