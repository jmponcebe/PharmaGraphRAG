"""LangGraph ReAct agent for PharmaGraphRAG.

Builds a ReAct agent that uses tools to query the knowledge graph
and vector store, then generates a final answer with an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from loguru import logger

from pharmagraphrag.agent.tools import ALL_TOOLS
from pharmagraphrag.config import get_settings

SYSTEM_PROMPT = """\
You are a pharmaceutical knowledge assistant with access to tools that query
FDA FAERS adverse event reports and DailyMed drug labels.

Your tools:
- search_drug_info: get full drug profile from the knowledge graph
- find_drugs_for_adverse_event: find drugs causing a specific side effect
- search_drug_labels: semantic search over drug label text
- list_drug_interactions: get drug-drug interactions
- search_drugs_by_name: fuzzy search for drug names

Guidelines:
- Use one or more tools to gather relevant information before answering.
- Cite specific drugs, adverse events, and report counts from the tool results.
- If tool results are insufficient, say so explicitly.
- Be precise with medical terminology.
- Structure your answer clearly with sections if appropriate.
- This data is for educational purposes only, not clinical decisions.
"""


@dataclass
class AgentResponse:
    """Response from the agent execution."""

    answer: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.answer)


def _build_agent():
    """Create the LangGraph ReAct agent."""
    settings = get_settings()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
        max_output_tokens=2048,
    )

    return create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
    )


# Lazy singleton
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def run_agent(question: str) -> AgentResponse:
    """Run the ReAct agent on a user question.

    Args:
        question: The user's natural language question.

    Returns:
        AgentResponse with the answer and tool call history.
    """
    agent = _get_agent()

    try:
        result = agent.invoke({"messages": [("user", question)]})

        # Extract tool calls from message history
        tool_calls = []
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(
                        {
                            "tool": tc.get("name", ""),
                            "args": tc.get("args", {}),
                        }
                    )

        # Last message is the final answer
        messages = result.get("messages", [])
        raw_content = messages[-1].content if messages else ""

        # Gemini 2.5 Flash may return content as a list of blocks
        # (e.g. [{"type": "text", "text": "..."}, ...]) instead of a string
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

        return AgentResponse(answer=answer, tool_calls=tool_calls)

    except Exception as exc:
        logger.error("Agent execution failed: {}", exc)
        return AgentResponse(error=str(exc))
