"""AgentState TypedDict — shared state passed between all nodes in the LangGraph."""

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from schemas.backtest_result import BacktestResult
from schemas.signal import AlphaSignal


class AgentState(TypedDict):
    """Shared mutable state for the AlphaSignal research graph.

    Passed through every node in the LangGraph StateGraph. Fields are
    progressively populated as the pipeline runs:
      1. research_agent populates research_context
      2. signal_agent populates signals
      3. backtest_agent populates backtest_result

    iteration_count tracks research→signal loops to prevent infinite retries.
    """

    ticker: str
    """Target stock ticker, e.g. 'NVDA'. Set by the caller before graph invocation."""

    messages: Annotated[list[Any], add_messages]
    """LangGraph message list — automatically merged by add_messages reducer."""

    research_context: str
    """Concatenated RAG chunks from the research agent, passed to signal agent."""

    signals: list[AlphaSignal]
    """Accumulated AlphaSignal objects extracted by the signal agent."""

    backtest_result: BacktestResult | None
    """BacktestResult populated by the backtest agent, or None if not yet run."""

    iteration_count: int
    """Number of times the research→signal loop has run. Caps at 3 to prevent loops."""

    error: str | None
    """Optional error message set by any node on failure; surfaced to the caller."""
