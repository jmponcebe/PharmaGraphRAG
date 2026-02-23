"""PharmaGraphRAG — Streamlit UI.

Chat-based interface for querying drug interactions and adverse events
powered by a GraphRAG pipeline (Neo4j + ChromaDB + LLM).

Usage:
    streamlit run src/pharmagraphrag/ui/app.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import streamlit as st
from loguru import logger

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PharmaGraphRAG",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Dataclass to hold a conversation turn
# ---------------------------------------------------------------------------


@dataclass
class ChatMessage:
    """A single message in the conversation."""

    role: str  # "user" | "assistant"
    content: str
    sources_graph: dict[str, Any] = field(default_factory=dict)
    sources_vector: list[dict[str, Any]] = field(default_factory=list)
    drugs_extracted: list[str] = field(default_factory=list)
    drugs_found: list[str] = field(default_factory=list)
    llm_provider: str = ""
    llm_model: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------


def _init_session() -> None:
    """Initialise Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "use_graph": True,
            "use_vector": True,
            "use_llm": True,
            "n_results": 5,
            "llm_provider": "gemini",
        }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    """Render the sidebar with settings and drug explorer."""
    st.sidebar.image(
        "https://img.icons8.com/color/96/pill.png",
        width=64,
    )
    st.sidebar.title("PharmaGraphRAG")
    st.sidebar.caption("GraphRAG para interacciones farmacológicas")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuración")

    s = st.session_state.settings

    s["use_graph"] = st.sidebar.checkbox(
        "Usar Knowledge Graph (Neo4j)",
        value=s["use_graph"],
        key="cb_graph",
    )
    s["use_vector"] = st.sidebar.checkbox(
        "Usar Vector Search (ChromaDB)",
        value=s["use_vector"],
        key="cb_vector",
    )
    s["use_llm"] = st.sidebar.checkbox(
        "Generar respuesta con LLM",
        value=s["use_llm"],
        key="cb_llm",
    )
    s["n_results"] = st.sidebar.slider(
        "Resultados vector search",
        min_value=1,
        max_value=20,
        value=s["n_results"],
        key="sl_nresults",
    )
    s["llm_provider"] = st.sidebar.selectbox(
        "Proveedor LLM",
        ["gemini", "ollama"],
        index=0 if s["llm_provider"] == "gemini" else 1,
        key="sb_provider",
    )

    # Drug explorer
    from pharmagraphrag.ui.components import (
        render_drug_detail,
        render_drug_explorer,
    )

    selected_drug = render_drug_explorer()
    if selected_drug:
        render_drug_detail(selected_drug)


# ---------------------------------------------------------------------------
# Query processing
# ---------------------------------------------------------------------------


def _process_question(question: str) -> ChatMessage:
    """Run the GraphRAG pipeline and return a ChatMessage.

    Args:
        question: User question string.

    Returns:
        ChatMessage with the assistant's response.
    """
    s = st.session_state.settings

    try:
        from pharmagraphrag.engine.query_engine import process_query
        from pharmagraphrag.llm.client import generate_answer

        # 1. Entity extraction + retrieval
        result = process_query(
            question,
            use_graph=s["use_graph"],
            use_vector=s["use_vector"],
            n_vector_results=s["n_results"],
        )

        # 2. LLM generation (optional)
        answer = ""
        llm_provider = ""
        llm_model = ""
        error = None

        if s["use_llm"]:
            llm_resp = generate_answer(
                system_prompt=result.system_prompt,
                user_prompt=result.user_prompt,
                provider=s["llm_provider"],
            )
            answer = llm_resp.text
            llm_provider = llm_resp.provider
            llm_model = llm_resp.model
            if not llm_resp.ok:
                error = llm_resp.error
        else:
            # Retrieval-only mode: show the prompt context
            answer = (
                "**Modo solo recuperación** (LLM desactivado).\n\n"
                "Contexto recuperado:\n\n"
                f"{result.user_prompt}"
            )

        return ChatMessage(
            role="assistant",
            content=answer,
            sources_graph=result.context.graph_raw,
            sources_vector=result.context.vector_raw,
            drugs_extracted=result.entities.drugs,
            drugs_found=result.context.drugs_found,
            llm_provider=llm_provider,
            llm_model=llm_model,
            error=error,
        )

    except Exception as exc:
        logger.error("Error processing question: {}", exc)
        return ChatMessage(
            role="assistant",
            content=f"❌ Error al procesar la consulta: {exc}",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Chat display
# ---------------------------------------------------------------------------


def _display_message(msg: ChatMessage) -> None:
    """Render a single chat message with optional extras."""
    with st.chat_message(msg.role):
        st.markdown(msg.content)

        if msg.role == "assistant" and msg.content:
            # Metadata badges
            badges: list[str] = []
            if msg.drugs_extracted:
                badges.append(f"🏷️ Fármacos: {', '.join(msg.drugs_extracted)}")
            if msg.drugs_found:
                badges.append(f"📊 En grafo: {', '.join(msg.drugs_found)}")
            if msg.llm_provider:
                badges.append(f"🤖 {msg.llm_provider}/{msg.llm_model}")
            if msg.error:
                badges.append(f"⚠️ {msg.error}")

            if badges:
                st.caption(" · ".join(badges))

            # Sources & graph visualisation (in tabs)
            has_sources = msg.sources_graph or msg.sources_vector
            if has_sources:
                from pharmagraphrag.ui.components import (
                    render_graph,
                    render_sources,
                )

                tab_src, tab_graph = st.tabs(["📄 Fuentes", "🕸️ Grafo"])

                with tab_src:
                    render_sources(msg.sources_graph, msg.sources_vector)

                with tab_graph:
                    render_graph(msg.sources_graph)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Streamlit app."""
    _init_session()
    _render_sidebar()

    # Header
    st.title("💊 PharmaGraphRAG")
    st.caption(
        "Consulta interacciones farmacológicas y eventos adversos "
        "con Knowledge Graph + Vector Search + LLM."
    )

    # Display chat history
    for msg in st.session_state.messages:
        _display_message(msg)

    # Chat input
    if prompt := st.chat_input("Haz una pregunta sobre fármacos…"):
        # Add user message
        user_msg = ChatMessage(role="user", content=prompt)
        st.session_state.messages.append(user_msg)

        with st.chat_message("user"):
            st.markdown(prompt)

        # Process and display assistant response
        with st.spinner("Analizando consulta…"):
            assistant_msg = _process_question(prompt)

        st.session_state.messages.append(assistant_msg)
        _display_message(assistant_msg)

    # Empty state
    if not st.session_state.messages:
        st.markdown("---")
        st.markdown("### 💡 Ejemplos de preguntas")

        examples = [
            "What are the side effects of ibuprofen?",
            "Does metformin interact with other drugs?",
            "What adverse events are associated with warfarin?",
            "Compare the safety profile of aspirin and clopidogrel",
            "What drugs cause liver damage?",
        ]

        cols = st.columns(2)
        for i, ex in enumerate(examples):
            col = cols[i % 2]
            col.markdown(f"- _{ex}_")


if __name__ == "__main__":
    main()
