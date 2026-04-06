"""Signal agent — extracts a structured AlphaSignal from research context."""

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from config.settings import Settings
from graph.state import AgentState
from schemas.signal import AlphaSignal

log = logging.getLogger(__name__)

_SIGNAL_PROMPT = """\
You are a quantitative analyst extracting a structured investment signal.

Ticker: {ticker}

Research context (synthesised from SEC filings):
{research_context}

Based solely on the evidence in the research context above, extract a single
directional alpha signal. You MUST:

1. Set direction to "long", "short", or "neutral"
2. Set confidence between 0.0 and 1.0 (be conservative — only exceed 0.7 if
   the evidence is strongly one-directional with specific quantitative support)
3. Write a rationale that cites at least one specific filing section, e.g.
   "Item 7 states revenue grew 122% YoY..." or "Item 1A identifies..."
4. Include the 2-4 most relevant verbatim excerpts in supporting_chunks
5. Set filing_period to the period covered, e.g. "10-K FY2024"

If the evidence is mixed or insufficient, set direction to "neutral" and
confidence below 0.6 — do not force a directional signal.
"""


async def signal_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: extract a structured AlphaSignal from research context.

    Uses .with_structured_output(AlphaSignal) so the LLM returns a validated
    Pydantic object directly. Appends the new signal to state['signals'].

    Updates state with:
        - signals: list with the new AlphaSignal appended
        - messages: appends the structured signal as a serialised message

    Args:
        state: Current AgentState; expects research_context to be populated.

    Returns:
        Partial state dict with updated signals and messages.
    """
    settings = Settings()
    ticker = state["ticker"]
    research_context = state.get("research_context", "")

    log.info("[%s] signal_node — extracting structured signal via LLM", ticker)
    llm = ChatAnthropic(
        model=settings.anthropic_model,
        temperature=0.0,
        api_key=settings.anthropic_api_key,
    ).with_structured_output(AlphaSignal)

    prompt = _SIGNAL_PROMPT.format(
        ticker=ticker,
        research_context=research_context,
    )

    signal: AlphaSignal = await llm.ainvoke(  # type: ignore[assignment]
        [HumanMessage(content=prompt)]
    )
    log.info(
        "[%s] signal_node complete — %s confidence=%.2f",
        ticker,
        signal.direction.upper(),
        signal.confidence,
    )

    existing_signals: list[AlphaSignal] = list(state.get("signals", []))
    existing_signals.append(signal)

    summary = (
        f"Signal: {signal.direction.upper()} {ticker} "
        f"(confidence={signal.confidence:.2f}, period={signal.filing_period})"
    )

    return {
        "signals": existing_signals,
        "messages": [HumanMessage(content=summary)],
    }
