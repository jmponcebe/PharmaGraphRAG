"""Multi-agent system with supervisor for PharmaGraphRAG.

Implements a supervisor pattern where a coordinator agent delegates
to specialized sub-agents based on the query type:
- Drug Expert: drug profiles, interactions, categories
- Safety Analyst: adverse events, outcomes, drug comparisons
- Literature Researcher: drug label text search (vector store)

The supervisor decides which expert(s) to consult and synthesizes
their findings into a final answer.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from loguru import logger

from pharmagraphrag.agent.graph import AgentResponse, StructuredResponse
from pharmagraphrag.agent.tools import (
    compare_drugs,
    find_drugs_by_category,
    find_drugs_for_adverse_event,
    get_drug_outcomes,
    list_drug_interactions,
    search_adverse_events,
    search_drug_info,
    search_drug_labels,
    search_drugs_by_name,
)
from pharmagraphrag.config import DEFAULT_MODEL, get_settings

# ---------------------------------------------------------------------------
# Sub-agent definitions
# ---------------------------------------------------------------------------

DRUG_EXPERT_PROMPT = """\
You are a drug information expert. Your role is to provide detailed information
about specific drugs: their profiles, interactions, pharmacologic categories,
and related medications. Always use your tools to look up accurate data.
Use MedDRA terminology for adverse events and search_drugs_by_name when
unsure of exact drug names.
Never ask the user for clarification — always try multiple search strategies.
"""

DRUG_EXPERT_TOOLS = [
    search_drug_info,
    list_drug_interactions,
    find_drugs_by_category,
    search_drugs_by_name,
]

SAFETY_ANALYST_PROMPT = """\
You are a drug safety analyst specializing in adverse events and patient outcomes.
Your role is to analyze safety profiles, compare drug risks, and identify
which drugs are associated with specific side effects. Use MedDRA terminology.
When comparing drugs, use compare_drugs for structured side-by-side analysis.
Never ask the user for clarification — if an exact term is not found, try
broader searches (e.g., search the drug's adverse events and label text).
"""

SAFETY_ANALYST_TOOLS = [
    search_drug_info,
    find_drugs_for_adverse_event,
    search_adverse_events,
    get_drug_outcomes,
    compare_drugs,
]

LITERATURE_RESEARCHER_PROMPT = """\
You are a pharmaceutical literature researcher. Your role is to find relevant
information from FDA drug labels (DailyMed) using semantic search. You can
search for specific drug label content about interactions, warnings,
contraindications, dosage, and mechanism of action.
Never ask the user for clarification — try multiple search queries to find
relevant information.
"""

LITERATURE_RESEARCHER_TOOLS = [
    search_drug_labels,
    search_drugs_by_name,
]

# ---------------------------------------------------------------------------
# Build sub-agents (lazy, no checkpointer — state is in supervisor)
# ---------------------------------------------------------------------------

# Keyed by (model, agent_name) for per-model caching
_sub_agents: dict[tuple[str, str], Any] = {}

# Stores inner tool calls from the most recent sub-agent execution.
# Keyed by sub-agent name (drug_expert, safety_analyst, literature_researcher).
# Populated by _run_sub_agent, consumed by run_multi_agent.
_last_inner_tool_calls: dict[str, dict[str, Any]] = {}

# Active model for current multi-agent invocation (set by run_multi_agent)
_active_model: str = DEFAULT_MODEL


def _get_llm(model: str | None = None):
    model = model or _active_model
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
        max_output_tokens=2048,
    )


def _get_sub_agent(name: str, model: str | None = None):
    model = model or _active_model
    cache_key = (model, name)
    if cache_key not in _sub_agents:
        llm = _get_llm(model)
        configs = {
            "drug_expert": (DRUG_EXPERT_PROMPT, DRUG_EXPERT_TOOLS),
            "safety_analyst": (SAFETY_ANALYST_PROMPT, SAFETY_ANALYST_TOOLS),
            "literature_researcher": (LITERATURE_RESEARCHER_PROMPT, LITERATURE_RESEARCHER_TOOLS),
        }
        prompt, tools = configs[name]
        _sub_agents[cache_key] = create_react_agent(
            model=llm,
            tools=tools,
            prompt=prompt,
        )
    return _sub_agents[cache_key]


def _run_sub_agent(name: str, question: str) -> str:
    """Run a sub-agent and return its text answer.

    Also stores inner tool calls in _last_inner_tool_calls for the
    supervisor to include in the response.
    """
    agent = _get_sub_agent(name, _active_model)
    try:
        result = agent.invoke({"messages": [("user", question)]})
        messages = result.get("messages", [])
        if not messages:
            return "No response from sub-agent."

        # Capture inner tool calls for transparency
        from langchain_core.messages import ToolMessage as _ToolMsg

        inner_calls = []
        inner_results = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    inner_calls.append({"tool": tc.get("name", ""), "args": tc.get("args", {})})
            if isinstance(msg, _ToolMsg):
                content = msg.content
                if isinstance(content, str) and len(content) > 500:
                    content = content[:500] + "…"
                inner_results.append({"tool": msg.name or "", "content": content})

        _last_inner_tool_calls[name] = {
            "tool_calls": inner_calls,
            "tool_results": inner_results,
        }

        raw = messages[-1].content
        if isinstance(raw, list):
            return "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in raw
                if not (isinstance(b, dict) and b.get("type") == "thinking")
            ).strip()
        return str(raw)
    except Exception as exc:
        logger.error("Sub-agent {} failed: {}", name, exc)
        return f"Sub-agent error: {exc}"


# ---------------------------------------------------------------------------
# Supervisor tools (each wraps a sub-agent call)
# ---------------------------------------------------------------------------


@tool
def ask_drug_expert(question: str) -> str:
    """Delegate a question to the Drug Expert sub-agent.

    The Drug Expert specializes in drug profiles, interactions, pharmacologic
    categories, and medication search. Use this for questions like:
    - "What is metformin?"
    - "What drugs interact with warfarin?"
    - "What drugs are in the NSAID category?"
    - "Find drugs with name similar to 'aspir'"

    Args:
        question: The question to ask the Drug Expert.
    """
    return _run_sub_agent("drug_expert", question)


@tool
def ask_safety_analyst(question: str) -> str:
    """Delegate a question to the Safety Analyst sub-agent.

    The Safety Analyst specializes in adverse events, patient outcomes,
    and drug safety comparisons. Use this for questions like:
    - "What adverse events does ibuprofen cause?"
    - "Which drugs cause hepatotoxicity?"
    - "Compare aspirin vs ibuprofen safety"
    - "What are the patient outcomes for warfarin?"
    - "Search for liver-related adverse events"

    Args:
        question: The question to ask the Safety Analyst.
    """
    return _run_sub_agent("safety_analyst", question)


@tool
def ask_literature_researcher(question: str) -> str:
    """Delegate a question to the Literature Researcher sub-agent.

    The Literature Researcher searches FDA drug label text (DailyMed) for
    relevant passages. Use this for questions about:
    - Drug warnings and contraindications
    - Mechanism of action details
    - Dosage information
    - Clinical pharmacology
    - Black box warnings

    Args:
        question: The question to ask the Literature Researcher.
    """
    return _run_sub_agent("literature_researcher", question)


SUPERVISOR_TOOLS = [ask_drug_expert, ask_safety_analyst, ask_literature_researcher]

# ---------------------------------------------------------------------------
# Supervisor agent
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """\
You are a pharmaceutical knowledge supervisor coordinating a team of
specialized experts. Your job is to understand the user's question and
delegate to the right expert(s):

Your team:
- ask_drug_expert: for drug profiles, interactions, pharmacologic categories, drug name searches
- ask_safety_analyst: for adverse events of a drug, drugs causing a specific adverse event, patient outcomes, drug safety comparisons
- ask_literature_researcher: for FDA drug label text (warnings, contraindications, pharmacology, dosage)

Workflow:
1. Analyze the user's question to determine which expert(s) to consult.
2. For complex questions, consult multiple experts and combine their knowledge.
3. Synthesize their responses into a coherent, well-structured final answer.
4. Cite specific data (drug names, event names, report counts) from expert responses.
5. If one expert lacks data, try another expert before saying information is unavailable.
6. Never ask the user for clarification — use your experts to find the answer.

You have conversation memory — you can reference previous exchanges in the session.
This data is for educational purposes only, not clinical decisions.
"""

_checkpointer = MemorySaver()
_supervisors: dict[str, Any] = {}


def _get_supervisor(model: str | None = None):
    model = model or DEFAULT_MODEL
    if model not in _supervisors:
        llm = _get_llm(model)
        _supervisors[model] = create_react_agent(
            model=llm,
            tools=SUPERVISOR_TOOLS,
            prompt=SUPERVISOR_PROMPT,
            response_format=StructuredResponse,
            checkpointer=_checkpointer,
        )
    return _supervisors[model]


def run_multi_agent(
    question: str,
    thread_id: str | None = None,
    model: str | None = None,
    subagent_model: str | None = None,
) -> AgentResponse:
    """Run the multi-agent supervisor on a user question.

    Args:
        question: The user's natural language question.
        thread_id: Optional session ID for conversation memory.
        model: LLM model for the supervisor. Defaults to DEFAULT_MODEL.
        subagent_model: LLM model for sub-agents. Falls back to ``model``.

    Returns:
        AgentResponse with the synthesized answer and tool call trace.
    """
    global _active_model
    _active_model = subagent_model or model or DEFAULT_MODEL
    supervisor_model = model or DEFAULT_MODEL
    supervisor = _get_supervisor(supervisor_model)
    config = {"configurable": {"thread_id": thread_id or "default"}}

    # Add Langfuse tracing if enabled
    from pharmagraphrag.observability import build_callback_config

    config = build_callback_config(
        session_id=thread_id or "default",
        tags=["multi-agent", "supervisor"],
        existing_config=config,
    )

    try:
        from langchain_core.messages import ToolMessage

        result = supervisor.invoke(
            {"messages": [("user", question)]},
            config=config,
        )

        # Extract tool calls and results
        tool_calls = []
        tool_results = []
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(
                        {
                            "tool": tc.get("name", ""),
                            "args": tc.get("args", {}),
                        }
                    )
            if isinstance(msg, ToolMessage):
                tool_results.append(
                    {
                        "tool": msg.name or "",
                        "content": msg.content,
                    }
                )

        # Extract structured response if available
        structured = result.get("structured_response")
        drugs_mentioned: list[str] = []
        ae_mentioned: list[str] = []
        confidence = ""
        follow_ups: list[str] = []

        if structured and isinstance(structured, StructuredResponse):
            answer = structured.answer
            drugs_mentioned = structured.drugs_mentioned
            ae_mentioned = structured.adverse_events_mentioned
            confidence = structured.confidence
            follow_ups = structured.follow_up_suggestions
        else:
            # Fallback: extract from last message
            messages = result.get("messages", [])
            raw = messages[-1].content if messages else ""
            if isinstance(raw, list):
                answer = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in raw
                    if not (isinstance(b, dict) and b.get("type") == "thinking")
                ).strip()
            else:
                answer = str(raw)

        logger.info(
            "Multi-agent completed: {} tool calls, answer length={}",
            len(tool_calls),
            len(answer),
        )

        # Enrich supervisor tool calls with inner sub-agent tool calls
        # Maps supervisor tool names to sub-agent names
        _supervisor_to_sub = {
            "ask_drug_expert": "drug_expert",
            "ask_safety_analyst": "safety_analyst",
            "ask_literature_researcher": "literature_researcher",
        }
        for tc in tool_calls:
            sub_name = _supervisor_to_sub.get(tc.get("tool", ""))
            if sub_name and sub_name in _last_inner_tool_calls:
                tc["inner_tool_calls"] = _last_inner_tool_calls[sub_name].get("tool_calls", [])
                tc["inner_tool_results"] = _last_inner_tool_calls[sub_name].get("tool_results", [])

        # Clear inner tool calls for next invocation
        _last_inner_tool_calls.clear()

        # Re-collect structured data: extract drug names from sub-agent
        # questions and results for graph visualization
        graph_data, vector_data = _collect_structured_from_results(tool_calls, tool_results)

        return AgentResponse(
            answer=answer,
            tool_calls=tool_calls,
            tool_results=tool_results,
            graph_data=graph_data,
            vector_data=vector_data,
            drugs_mentioned=drugs_mentioned,
            adverse_events_mentioned=ae_mentioned,
            confidence=confidence,
            follow_up_suggestions=follow_ups,
        )

    except Exception as exc:
        error_msg = str(exc)
        if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
            logger.warning("Multi-agent rate limited: {}", error_msg[:200])
            return AgentResponse(
                error="Rate limit exceeded. Please try again later or use Classic Mode."
            )
        logger.error("Multi-agent execution failed: {}", exc)
        return AgentResponse(error=error_msg)


def _collect_structured_from_results(
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract drug names from sub-agent questions/results and fetch structured data.

    Uses the entity extractor to reliably identify drug names from:
    1. The questions delegated to sub-agents (tool call args)
    2. The text responses from sub-agents
    Then re-fetches full context from Neo4j for graph visualization,
    and runs vector search for drug label sources.
    """
    from pharmagraphrag.engine.entity_extractor import extract_entities
    from pharmagraphrag.graph import queries
    from pharmagraphrag.vectorstore import store as vs

    graph_data: dict[str, Any] = {}
    vector_data: list[dict[str, Any]] = []
    seen_drugs: set[str] = set()

    # 1. Extract drug names from the questions delegated to sub-agents
    all_questions: list[str] = []
    for tc in tool_calls:
        question = tc.get("args", {}).get("question", "")
        if question:
            all_questions.append(question)
            try:
                entities = extract_entities(question, fuzzy=True)
                for drug in entities.drugs:
                    if drug not in seen_drugs and len(seen_drugs) < 10:
                        _try_add_drug(drug, graph_data, seen_drugs, queries)
            except Exception:
                pass

    # 2. Extract drug names from sub-agent text responses
    for tr in tool_results:
        content = tr.get("content", "")
        if content:
            try:
                entities = extract_entities(content, fuzzy=False)
                for drug in entities.drugs:
                    if drug not in seen_drugs and len(seen_drugs) < 10:
                        _try_add_drug(drug, graph_data, seen_drugs, queries)
            except Exception:
                pass

    # 3. Vector search: search drug labels for context using sub-agent questions
    for question in all_questions[:3]:
        try:
            results = vs.search(question, n_results=3)
            vector_data.extend(results)
        except Exception:
            pass

    return graph_data, vector_data


def _try_add_drug(
    drug: str,
    graph_data: dict[str, Any],
    seen_drugs: set[str],
    queries: Any,
) -> None:
    try:
        ctx = queries.get_drug_full_context(drug)
        if ctx and ctx.get("drug_info"):
            graph_data[drug] = ctx
            seen_drugs.add(drug)
    except Exception:
        pass
