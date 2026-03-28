"""LangGraph ReAct agent for PharmaGraphRAG.

Builds a ReAct agent that uses tools to query the knowledge graph
and vector store, then generates a final answer with an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from loguru import logger
from pydantic import BaseModel, Field

from pharmagraphrag.agent.tools import ALL_TOOLS
from pharmagraphrag.config import get_settings

SYSTEM_PROMPT = """\
You are a pharmaceutical knowledge assistant with access to tools that query
FDA FAERS adverse event reports and DailyMed drug labels.

Your tools:
- search_drug_info: get full drug profile from the knowledge graph
- find_drugs_for_adverse_event: find drugs causing a specific side effect (uses MedDRA terms)
- search_adverse_events: search for adverse event names by keyword (use when unsure of exact MedDRA term)
- search_drug_labels: semantic search over drug label text
- list_drug_interactions: get drug-drug interactions
- search_drugs_by_name: fuzzy search for drug names
- get_drug_outcomes: get patient outcomes (hospitalization, death, etc.) for a drug
- compare_drugs: compare two drugs side-by-side (adverse events, outcomes, interactions)
- find_drugs_by_category: find drugs belonging to a pharmacologic category (e.g. NSAIDs, statins)

Guidelines:
- Use one or more tools to gather relevant information before answering.
- Adverse events in the database use MedDRA medical terminology (e.g.
  "HEPATOTOXICITY" not "liver damage", "PYREXIA" not "fever"). When the user
  uses colloquial terms, use search_adverse_events first to find the correct
  MedDRA name, then use find_drugs_for_adverse_event with the exact name.
- Similarly, drug names should be searched with search_drugs_by_name when unsure
  of the exact spelling or standardized name.
- When comparing drugs, use compare_drugs for a structured comparison.
- Cite specific drugs, adverse events, and report counts from the tool results.
- If tool results are insufficient, say so explicitly.
- Be precise with medical terminology.
- Structure your answer clearly with sections if appropriate.
- You have conversation memory: you can reference previous questions and answers
  in the same session. Use this to provide contextual follow-up answers.
- This data is for educational purposes only, not clinical decisions.
"""


# ---------------------------------------------------------------------------
# Structured output model — forces the LLM to return consistent JSON
# ---------------------------------------------------------------------------


class StructuredResponse(BaseModel):
    """Structured response format for the agent's final answer."""

    answer: str = Field(description="The main answer to the user's question, in clear prose.")
    drugs_mentioned: list[str] = Field(
        default_factory=list,
        description="List of drug names mentioned in the answer (uppercase).",
    )
    adverse_events_mentioned: list[str] = Field(
        default_factory=list,
        description="List of adverse event names mentioned in the answer (uppercase, MedDRA terms).",
    )
    confidence: str = Field(
        default="medium",
        description="How confident the answer is based on available data: 'high', 'medium', or 'low'.",
    )
    follow_up_suggestions: list[str] = Field(
        default_factory=list,
        description="1-3 suggested follow-up questions the user might ask.",
    )


@dataclass
class AgentResponse:
    """Response from the agent execution."""

    answer: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    graph_data: dict[str, Any] = field(default_factory=dict)
    vector_data: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    # Structured output metadata (populated when response_format is used)
    drugs_mentioned: list[str] = field(default_factory=list)
    adverse_events_mentioned: list[str] = field(default_factory=list)
    confidence: str = ""
    follow_up_suggestions: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.answer)


# Agent uses flash-lite to save quota (separate RPD limit from classic mode's flash)
AGENT_MODEL = "gemini-2.5-flash-lite"

# Simple in-memory response cache to avoid wasting RPD on repeated questions
_response_cache: dict[str, AgentResponse] = {}
_CACHE_MAX_SIZE = 50


# In-memory checkpointer for conversation memory (multi-turn)
_checkpointer = MemorySaver()


def _build_agent():
    """Create the LangGraph ReAct agent with conversation memory."""
    settings = get_settings()

    llm = ChatGoogleGenerativeAI(
        model=AGENT_MODEL,
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
        max_output_tokens=2048,
    )

    return create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
        response_format=StructuredResponse,
        checkpointer=_checkpointer,
    )


# Lazy singleton
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def _collect_structured_data(
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Re-fetch structured data from services based on which tools were called.

    This avoids parsing formatted text: we know the tool args, so we call the
    same underlying services to get raw dicts for UI rendering (graph viz,
    sources panel).
    """
    from pharmagraphrag.graph import queries
    from pharmagraphrag.vectorstore import store as vs

    graph_data: dict[str, Any] = {}
    vector_data: list[dict[str, Any]] = []
    seen_drugs: set[str] = set()

    for tc in tool_calls:
        name = tc.get("tool", "")
        args = tc.get("args", {})

        if name in ("search_drug_info", "list_drug_interactions", "get_drug_outcomes"):
            drug = args.get("drug_name", "").upper()
            if drug and drug not in seen_drugs:
                try:
                    ctx = queries.get_drug_full_context(drug)
                    if ctx and ctx.get("drug_info"):
                        graph_data[drug] = ctx
                        seen_drugs.add(drug)
                except Exception:
                    pass

        elif name == "compare_drugs":
            for key in ("drug_name_1", "drug_name_2"):
                drug = args.get(key, "").upper()
                if drug and drug not in seen_drugs:
                    try:
                        ctx = queries.get_drug_full_context(drug)
                        if ctx and ctx.get("drug_info"):
                            graph_data[drug] = ctx
                            seen_drugs.add(drug)
                    except Exception:
                        pass

        elif name == "search_drug_labels":
            query = args.get("query", "")
            drug = args.get("drug_name", "")
            n = args.get("n_results", 5)
            if query:
                try:
                    if drug:
                        results = vs.search_by_drug(query, drug.upper(), n_results=n)
                    else:
                        results = vs.search(query, n_results=n)
                    vector_data.extend(results)
                except Exception:
                    pass

    # Also extract drug names from find_drugs_for_adverse_event results
    # by matching tool_results to tool_calls
    for i, tc in enumerate(tool_calls):
        if tc.get("tool") == "find_drugs_for_adverse_event" and i < len(tool_results):
            content = tool_results[i].get("content", "")
            # Extract drug names from "  - DRUG_NAME: NNN reports" lines
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("- ") and ": " in line and "reports" in line:
                    drug = line.split("- ", 1)[1].split(":")[0].strip().upper()
                    if drug and drug not in seen_drugs:
                        try:
                            ctx = queries.get_drug_full_context(drug)
                            if ctx and ctx.get("drug_info"):
                                graph_data[drug] = ctx
                                seen_drugs.add(drug)
                        except Exception:
                            pass
                        if len(seen_drugs) >= 5:
                            break

    return graph_data, vector_data


def run_agent(question: str, thread_id: str | None = None) -> AgentResponse:
    """Run the ReAct agent on a user question.

    Args:
        question: The user's natural language question.
        thread_id: Optional session ID for conversation memory. When provided,
            the agent remembers previous messages in the same thread.

    Returns:
        AgentResponse with the answer and tool call history.
    """
    # Check cache to save RPD quota (only for stateless queries)
    cache_key = question.strip().lower()
    if not thread_id and cache_key in _response_cache:
        logger.info("Agent cache hit for: '{}'", question[:60])
        return _response_cache[cache_key]

    agent = _get_agent()

    # Config with thread_id for checkpointer (conversation memory)
    config = {"configurable": {"thread_id": thread_id or "default"}}

    try:
        result = agent.invoke(
            {"messages": [("user", question)]},
            config=config,
        )

        # Extract tool calls and tool results from message history
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

        # Extract structured response if available (from response_format)
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
            raw_content = messages[-1].content if messages else ""

            if isinstance(raw_content, list):
                answer = "\n".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in raw_content
                    if not (isinstance(block, dict) and block.get("type") == "thinking")
                ).strip()
            else:
                answer = str(raw_content)

        logger.info(
            "Agent completed: {} tool calls, answer length={}",
            len(tool_calls),
            len(answer),
        )

        # Collect structured data for UI visualization
        graph_data, vector_data = _collect_structured_data(tool_calls, tool_results)

        response = AgentResponse(
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

        # Cache successful stateless responses only
        if response.ok and not thread_id:
            if len(_response_cache) >= _CACHE_MAX_SIZE:
                # Evict oldest entry
                _response_cache.pop(next(iter(_response_cache)))
            _response_cache[cache_key] = response

        return response

    except Exception as exc:
        error_msg = str(exc)
        if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
            logger.warning("Agent rate limited: {}", error_msg[:200])
            return AgentResponse(
                error="Rate limit exceeded. The free tier allows ~20 requests/day per model. Please try again later or use Classic Mode."
            )
        logger.error("Agent execution failed: {}", exc)
        return AgentResponse(error=error_msg)
