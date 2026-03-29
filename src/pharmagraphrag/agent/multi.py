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

from pharmagraphrag.agent.graph import AGENT_MODEL, AgentResponse
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
from pharmagraphrag.config import get_settings

# ---------------------------------------------------------------------------
# Sub-agent definitions
# ---------------------------------------------------------------------------

DRUG_EXPERT_PROMPT = """\
You are a drug information expert. Your role is to provide detailed information
about specific drugs: their profiles, interactions, pharmacologic categories,
and related medications. Always use your tools to look up accurate data.
Use MedDRA terminology for adverse events and search_drugs_by_name when
unsure of exact drug names.
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
"""

LITERATURE_RESEARCHER_TOOLS = [
    search_drug_labels,
    search_drugs_by_name,
]

# ---------------------------------------------------------------------------
# Build sub-agents (lazy, no checkpointer — state is in supervisor)
# ---------------------------------------------------------------------------

_sub_agents: dict[str, Any] = {}


def _get_llm():
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=AGENT_MODEL,
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
        max_output_tokens=2048,
    )


def _get_sub_agent(name: str):
    if name not in _sub_agents:
        llm = _get_llm()
        configs = {
            "drug_expert": (DRUG_EXPERT_PROMPT, DRUG_EXPERT_TOOLS),
            "safety_analyst": (SAFETY_ANALYST_PROMPT, SAFETY_ANALYST_TOOLS),
            "literature_researcher": (LITERATURE_RESEARCHER_PROMPT, LITERATURE_RESEARCHER_TOOLS),
        }
        prompt, tools = configs[name]
        _sub_agents[name] = create_react_agent(
            model=llm,
            tools=tools,
            prompt=prompt,
        )
    return _sub_agents[name]


def _run_sub_agent(name: str, question: str) -> str:
    """Run a sub-agent and return its text answer."""
    agent = _get_sub_agent(name)
    try:
        result = agent.invoke({"messages": [("user", question)]})
        messages = result.get("messages", [])
        if not messages:
            return "No response from sub-agent."
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
- ask_drug_expert: for drug profiles, interactions, categories, and drug name searches
- ask_safety_analyst: for adverse events, patient outcomes, drug safety comparisons
- ask_literature_researcher: for FDA drug label text (warnings, contraindications, pharmacology)

Workflow:
1. Analyze the user's question to determine which expert(s) to consult.
2. For complex questions, you may consult multiple experts.
3. Synthesize their responses into a coherent, well-structured final answer.
4. Cite specific data (drug names, event names, report counts) from expert responses.
5. If experts cannot find sufficient information, say so explicitly.

You have conversation memory — you can reference previous exchanges in the session.
This data is for educational purposes only, not clinical decisions.
"""

_checkpointer = MemorySaver()
_supervisor = None


def _get_supervisor():
    global _supervisor
    if _supervisor is None:
        llm = _get_llm()
        _supervisor = create_react_agent(
            model=llm,
            tools=SUPERVISOR_TOOLS,
            prompt=SUPERVISOR_PROMPT,
            checkpointer=_checkpointer,
        )
    return _supervisor


def run_multi_agent(question: str, thread_id: str | None = None) -> AgentResponse:
    """Run the multi-agent supervisor on a user question.

    Args:
        question: The user's natural language question.
        thread_id: Optional session ID for conversation memory.

    Returns:
        AgentResponse with the synthesized answer and tool call trace.
    """
    supervisor = _get_supervisor()
    config = {"configurable": {"thread_id": thread_id or "default"}}

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

        # Final answer
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

        # Re-collect structured data using the inner tool calls from sub-agents
        # The supervisor's tool calls are ask_drug_expert, etc. — we need inner calls
        # for graph visualization. Extract drug names mentioned in results.
        graph_data, vector_data = _collect_structured_from_results(tool_results)

        return AgentResponse(
            answer=answer,
            tool_calls=tool_calls,
            tool_results=tool_results,
            graph_data=graph_data,
            vector_data=vector_data,
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
    tool_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract drug names from sub-agent text results and fetch structured data.

    Since the supervisor delegates to sub-agents (which call the actual tools),
    we parse drug names from the text output and re-fetch from Neo4j.
    """
    from pharmagraphrag.graph import queries

    graph_data: dict[str, Any] = {}
    vector_data: list[dict[str, Any]] = []
    seen_drugs: set[str] = set()

    for tr in tool_results:
        content = tr.get("content", "")
        # Extract drug names from formatted lines like "Drug: ASPIRIN" or "- ASPIRIN: N reports"
        for line in content.split("\n"):
            line = line.strip()
            # Pattern: "Drug: NAME"
            if line.startswith("Drug: "):
                drug = line.split("Drug: ", 1)[1].strip().upper()
                if drug and drug not in seen_drugs and len(seen_drugs) < 10:
                    _try_add_drug(drug, graph_data, seen_drugs, queries)
            # Pattern: "- DRUG_NAME: N reports"
            elif line.startswith("- ") and ":" in line and "reports" in line.lower():
                drug = line.split("- ", 1)[1].split(":")[0].strip().upper()
                if drug and drug not in seen_drugs and len(seen_drugs) < 10:
                    _try_add_drug(drug, graph_data, seen_drugs, queries)
            # Pattern: "=== Comparison: DRUG1 vs DRUG2 ==="
            elif "Comparison:" in line and " vs " in line:
                parts = line.split("Comparison:", 1)[1].split("===")[0].strip()
                for d in parts.split(" vs "):
                    drug = d.strip().upper()
                    if drug and drug not in seen_drugs and len(seen_drugs) < 10:
                        _try_add_drug(drug, graph_data, seen_drugs, queries)

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
