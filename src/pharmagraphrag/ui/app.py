"""PharmaGraphRAG — Streamlit UI.

Chat-based interface for querying drug interactions and adverse events
powered by a GraphRAG pipeline (Neo4j + ChromaDB + LLM).

Supports two modes:
- **Local mode** (default): imports engine modules directly.
- **API mode**: calls a remote FastAPI endpoint via HTTP.
  Set the ``API_URL`` environment variable to enable (e.g. ``https://my-api.run.app``).

Usage:
    streamlit run src/pharmagraphrag/ui/app.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st
from loguru import logger

# Ensure package is importable when running as a script (e.g. Streamlit Cloud)
_src_dir = str(Path(__file__).resolve().parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


# Detect API mode: if API_URL is set, use HTTP; otherwise import engine locally.
# Supports both env vars and Streamlit Cloud secrets.
def _get_api_url() -> str | None:
    url = os.environ.get("API_URL")
    if not url:
        import contextlib

        with contextlib.suppress(Exception):
            url = st.secrets.get("API_URL")  # type: ignore[union-attr]
    return url or None


API_URL: str | None = _get_api_url()

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PharmaGraphRAG",
    page_icon="https://img.icons8.com/color/96/pill.png",
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
    agent_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    agent_tool_results: list[dict[str, Any]] = field(default_factory=list)


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
            "agent_mode": False,
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
    st.sidebar.caption("GraphRAG for drug interactions & adverse events")

    if API_URL:
        st.sidebar.success(f"🌐 API mode: {API_URL}", icon="🔗")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Settings")

    s = st.session_state.settings

    s["agent_mode"] = st.sidebar.toggle(
        "🤖 Agent Mode (ReAct)",
        value=s["agent_mode"],
        key="tg_agent",
        help="Use a LangGraph ReAct agent that autonomously decides which tools to call.",
    )

    if s["agent_mode"]:
        st.sidebar.info(
            "Agent mode: the LLM decides which tools to use "
            "(graph queries, vector search, etc.) autonomously.",
            icon="🤖",
        )
    else:
        s["use_graph"] = st.sidebar.checkbox(
            "Use Knowledge Graph (Neo4j)",
            value=s["use_graph"],
            key="cb_graph",
        )
        s["use_vector"] = st.sidebar.checkbox(
            "Use Vector Search (ChromaDB)",
            value=s["use_vector"],
            key="cb_vector",
        )
        s["use_llm"] = st.sidebar.checkbox(
            "Generate answer with LLM",
            value=s["use_llm"],
            key="cb_llm",
        )
        s["n_results"] = st.sidebar.slider(
            "Vector search results",
            min_value=1,
            max_value=20,
            value=s["n_results"],
            key="sl_nresults",
        )
        s["llm_provider"] = st.sidebar.selectbox(
            "LLM Provider",
            ["gemini", "ollama"],
            index=0 if s["llm_provider"] == "gemini" else 1,
            key="sb_provider",
            disabled=bool(API_URL),
            help="Disabled in API mode — the server chooses the provider." if API_URL else None,
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
    """Run the GraphRAG pipeline or agent depending on settings.

    Args:
        question: User question string.

    Returns:
        ChatMessage with the assistant's response.
    """
    agent_mode = st.session_state.settings.get("agent_mode", False)

    if API_URL:
        if agent_mode:
            return _process_question_agent_api(question)
        return _process_question_api(question)

    if agent_mode:
        return _process_question_agent_local(question)
    return _process_question_local(question)


# -- API mode ---------------------------------------------------------------


def _process_question_api(question: str) -> ChatMessage:
    """Call the remote FastAPI endpoint via HTTP."""
    import requests as req_lib

    s = st.session_state.settings
    base = API_URL.rstrip("/")  # type: ignore[union-attr]

    try:
        resp = req_lib.post(
            f"{base}/query",
            json={
                "question": question,
                "use_graph": s["use_graph"],
                "use_vector": s["use_vector"],
                "use_llm": s["use_llm"],
                "n_results": s["n_results"],
            },
            timeout=180,
        )

        if resp.status_code != 200:
            return ChatMessage(
                role="assistant",
                content=f"❌ API error ({resp.status_code}): {resp.text[:300]}",
                error=resp.text[:300],
            )

        data = resp.json()

        # Fetch full graph context per drug (for visualization)
        graph_raw: dict[str, Any] = {}
        for drug in data.get("drugs_found_in_graph", []):
            try:
                dr = req_lib.get(f"{base}/drug/{drug}", timeout=60)
                if dr.status_code == 200:
                    dd = dr.json()
                    graph_raw[drug] = {
                        "drug_info": {
                            "name": dd.get("name", drug),
                            "brand_names": dd.get("brand_names", []),
                            "route": dd.get("route", ""),
                        },
                        "adverse_events": dd.get("adverse_events", []),
                        "interactions": dd.get("interactions", []),
                        "outcomes": dd.get("outcomes", []),
                        "categories": dd.get("categories", []),
                    }
            except Exception:
                pass

        # Map vector sources from API response
        vector_raw: list[dict[str, Any]] = []
        for src in data.get("sources", []):
            if src.get("type") == "vector":
                vector_raw.append(
                    {
                        "text": src.get("snippet", ""),
                        "metadata": {
                            "drug_name": src.get("drug", ""),
                            "section": src.get("section", ""),
                        },
                    }
                )

        return ChatMessage(
            role="assistant",
            content=data.get("answer", ""),
            sources_graph=graph_raw,
            sources_vector=vector_raw,
            drugs_extracted=data.get("drugs_extracted", []),
            drugs_found=data.get("drugs_found_in_graph", []),
            llm_provider=data.get("llm_provider", ""),
            llm_model=data.get("llm_model", ""),
            error=data.get("error"),
        )

    except req_lib.exceptions.ConnectionError:
        return ChatMessage(
            role="assistant",
            content=f"❌ Cannot connect to API at `{base}`. Is the service running?",
            error="Connection error",
        )
    except req_lib.exceptions.ReadTimeout:
        return ChatMessage(
            role="assistant",
            content=(
                "⏳ The request timed out. The API may be experiencing a cold start "
                "(first request after inactivity can take ~50s). Please try again."
            ),
            error="Read timeout",
        )
    except Exception as exc:
        logger.error("API call failed: {}", exc)
        return ChatMessage(
            role="assistant",
            content=f"❌ Error calling API: {exc}",
            error=str(exc),
        )


# -- Agent mode (API) -------------------------------------------------------


def _process_question_agent_api(question: str) -> ChatMessage:
    """Call the agent endpoint on the remote API."""
    import requests as req_lib

    base = API_URL.rstrip("/")  # type: ignore[union-attr]

    try:
        resp = req_lib.post(
            f"{base}/agent/query",
            json={"question": question},
            timeout=180,
        )

        if resp.status_code != 200:
            return ChatMessage(
                role="assistant",
                content=f"❌ Agent API error ({resp.status_code}): {resp.text[:300]}",
                error=resp.text[:300],
            )

        data = resp.json()

        tool_calls = data.get("tool_calls", [])
        tool_results = data.get("tool_results", [])
        answer = data.get("answer", "")
        error = data.get("error")

        # Show error as content if no answer was generated
        if error and not answer:
            return ChatMessage(
                role="assistant",
                content=f"⚠️ {error}",
                error=error,
                llm_provider="agent",
                llm_model="gemini-2.5-flash-lite",
            )

        # Use structured data returned by the agent
        graph_raw = data.get("graph_data", {})
        vector_raw = data.get("vector_data", [])
        drugs = list(graph_raw.keys())

        return ChatMessage(
            role="assistant",
            content=answer,
            sources_graph=graph_raw,
            sources_vector=vector_raw,
            drugs_extracted=drugs,
            drugs_found=drugs,
            llm_provider="agent",
            llm_model="gemini-2.5-flash-lite",
            error=error,
            agent_tool_calls=tool_calls,
            agent_tool_results=tool_results,
        )

    except req_lib.exceptions.ReadTimeout:
        return ChatMessage(
            role="assistant",
            content=(
                "⏳ The agent request timed out. The API may be experiencing a cold start "
                "(~50s). Please try again."
            ),
            error="Read timeout",
        )
    except Exception as exc:
        logger.error("Agent API call failed: {}", exc)
        return ChatMessage(
            role="assistant",
            content=f"❌ Error calling agent API: {exc}",
            error=str(exc),
        )


# -- Agent mode (local) -----------------------------------------------------


def _process_question_agent_local(question: str) -> ChatMessage:
    """Run the LangGraph agent locally."""
    try:
        from pharmagraphrag.agent.graph import run_agent

        result = run_agent(question)

        # Show error as content if no answer was generated
        if result.error and not result.answer:
            return ChatMessage(
                role="assistant",
                content=f"⚠️ {result.error}",
                error=result.error,
                llm_provider="agent",
                llm_model="gemini-2.5-flash-lite",
            )

        # Format tool calls/results for structured storage
        tc_list = result.tool_calls if result.tool_calls else []
        tr_list = result.tool_results if result.tool_results else []

        # Use structured data collected by run_agent()
        drugs = list(result.graph_data.keys())

        return ChatMessage(
            role="assistant",
            content=result.answer,
            sources_graph=result.graph_data,
            sources_vector=result.vector_data,
            drugs_extracted=drugs,
            drugs_found=drugs,
            llm_provider="agent",
            llm_model="gemini-2.5-flash-lite",
            error=result.error,
            agent_tool_calls=tc_list,
            agent_tool_results=tr_list,
        )

    except Exception as exc:
        logger.error("Agent local execution failed: {}", exc)
        return ChatMessage(
            role="assistant",
            content=f"❌ Agent error: {exc}",
            error=str(exc),
        )


# -- Local mode --------------------------------------------------------------


def _process_question_local(question: str) -> ChatMessage:
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
                "**Retrieval-only mode** (LLM disabled).\n\n"
                "Retrieved context:\n\n"
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
            content=f"❌ Error processing query: {exc}",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Chat display
# ---------------------------------------------------------------------------


def _display_message(msg: ChatMessage, *, index: int = 0) -> None:
    """Render a single chat message with optional extras.

    Args:
        msg: The chat message to display.
        index: Message index used to generate unique widget keys.
    """
    with st.chat_message(msg.role):
        st.markdown(msg.content)

        if msg.role == "assistant" and msg.content:
            # Metadata badges
            badges: list[str] = []
            if msg.drugs_extracted:
                badges.append(f"🏷️ Drugs: {', '.join(msg.drugs_extracted)}")
            if msg.drugs_found:
                badges.append(f"📊 In graph: {', '.join(msg.drugs_found)}")
            if msg.llm_provider:
                badges.append(f"🤖 {msg.llm_provider}/{msg.llm_model}")
            if msg.error:
                badges.append(f"⚠️ {msg.error}")

            if badges:
                st.caption(" · ".join(badges))

            # Agent tool calls (as an expander)
            if msg.agent_tool_calls:
                with st.expander(
                    f"🔧 Agent reasoning ({len(msg.agent_tool_calls)} tool calls)",
                    expanded=False,
                ):
                    for i_tc, tc in enumerate(msg.agent_tool_calls):
                        tool_name = tc.get("tool", "unknown")
                        args = tc.get("args", {})
                        args_str = ", ".join(
                            f"{k}={v!r}" for k, v in args.items()
                        )
                        st.markdown(f"**{i_tc + 1}.** `{tool_name}({args_str})`")

                        # Show result if available
                        if i_tc < len(msg.agent_tool_results):
                            result_content = msg.agent_tool_results[i_tc].get(
                                "content", ""
                            )
                            if result_content:
                                preview = result_content[:500]
                                if len(result_content) > 500:
                                    preview += "…"
                                st.code(preview, language=None)

            # Sources & graph visualisation (in tabs)
            has_sources = msg.sources_graph or msg.sources_vector
            if has_sources:
                tab_src, tab_graph = st.tabs(["📄 Sources", "🕸️ Graph"])

                with tab_src:
                    try:
                        from pharmagraphrag.ui.components import render_sources

                        render_sources(msg.sources_graph, msg.sources_vector)
                    except Exception as e:
                        st.error(f"Error rendering sources: {e}")

                with tab_graph:
                    try:
                        from pharmagraphrag.ui.components import render_graph

                        render_graph(msg.sources_graph, key_suffix=str(index))
                    except Exception as e:
                        st.error(f"Error rendering graph: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Streamlit app."""
    _init_session()
    _render_sidebar()

    # Header
    st.markdown(
        '<h1><img src="https://img.icons8.com/color/96/pill.png" width="40" '
        'style="vertical-align: middle; margin-right: 8px;"/>PharmaGraphRAG</h1>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Query drug interactions and adverse events with Knowledge Graph + Vector Search + LLM."
    )

    # Display chat history
    for i, msg in enumerate(st.session_state.messages):
        _display_message(msg, index=i)

    # Chat input
    if prompt := st.chat_input("Ask a question about drugs…"):
        # Add user message
        user_msg = ChatMessage(role="user", content=prompt)
        st.session_state.messages.append(user_msg)

        with st.chat_message("user"):
            st.markdown(prompt)

        # Process and display assistant response
        agent_on = st.session_state.settings.get("agent_mode", False)
        if API_URL:
            import time

            mode_label = "🤖 Agent Mode" if agent_on else "Querying PharmaGraphRAG"
            with st.status(f"{mode_label}…", expanded=True) as status:
                st.write("🔌 Connecting to API…")
                st.caption(
                    "First query after inactivity may take ~50s (cold start)."
                )
                start = time.time()
                assistant_msg = _process_question(prompt)
                elapsed = time.time() - start
                status.update(
                    label=f"Done in {elapsed:.1f}s",
                    state="complete",
                    expanded=False,
                )
        else:
            with st.spinner("Analyzing query…"):
                assistant_msg = _process_question(prompt)

        st.session_state.messages.append(assistant_msg)
        _display_message(assistant_msg, index=len(st.session_state.messages) - 1)

    # Empty state
    if not st.session_state.messages:
        if API_URL:
            st.info(
                "**Powered by GraphRAG** — This system queries FDA FAERS reports "
                "(816K reports, 2024 Q3-Q4) and DailyMed drug labels (88 drugs) "
                "using a knowledge graph (Neo4j) + vector search (ChromaDB) + LLM. "
                "For educational/portfolio purposes only — not for clinical decision-making.",
                icon=":material/medication:",
            )
        else:
            st.info(
                "**Demo data notice:** This instance uses a representative subset "
                "(88 drugs, top adverse events per drug). Results may differ from "
                "the full dataset. For educational/portfolio purposes only — not "
                "for clinical decision-making.",
                icon="⚠️",
            )
        st.markdown("---")
        st.markdown("### 💡 Example Questions")

        agent_on = st.session_state.settings.get("agent_mode", False)
        if agent_on:
            examples = [
                "What drugs cause liver damage?",
                "Find interactions between warfarin and aspirin",
                "Which drugs are associated with headache?",
                "Tell me about the safety profile of metformin",
                "What adverse events does ibuprofen cause and what are its drug interactions?",
            ]
        else:
            examples = [
                "What are the side effects of ibuprofen?",
                "Does metformin interact with other drugs?",
                "What adverse events are associated with warfarin?",
                "Compare the safety profile of aspirin and clopidogrel",
            ]

        cols = st.columns(2)
        for i, ex in enumerate(examples):
            col = cols[i % 2]
            col.markdown(f"- _{ex}_")


if __name__ == "__main__":
    main()
