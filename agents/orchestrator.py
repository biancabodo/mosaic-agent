"""Orchestrator — public entry point for the AlphaSignal research pipeline."""

from langsmith import traceable

from graph.builder import build_graph
from graph.state import AgentState
from schemas.backtest_result import BacktestResult
from schemas.signal import AlphaSignal


@traceable(name="alphasignal-run")
async def run_pipeline(ticker: str) -> AgentState:
    """Run the full AlphaSignal research pipeline for a given ticker.

    Initialises a fresh AgentState, compiles the LangGraph, and invokes it
    asynchronously. All LLM calls and tool use within the graph are
    automatically traced in LangSmith when LANGCHAIN_TRACING_V2=true.

    Args:
        ticker: Stock ticker symbol, e.g. 'NVDA'. Case-insensitive.

    Returns:
        Final AgentState after the graph terminates. Key fields:
            - signals: list of AlphaSignal objects (latest is the final signal)
            - backtest_result: BacktestResult if confidence >= threshold, else None
            - error: non-None string if the pipeline failed at any node
    """
    graph = build_graph()

    initial_state: AgentState = {
        "ticker": ticker.upper(),
        "messages": [],
        "research_context": "",
        "signals": [],
        "backtest_result": None,
        "iteration_count": 0,
        "error": None,
    }

    final_state: AgentState = await graph.ainvoke(initial_state)  # type: ignore[assignment]
    return final_state


def extract_result(
    state: AgentState,
) -> tuple[AlphaSignal | None, BacktestResult | None]:
    """Extract the final signal and backtest result from a completed pipeline run.

    Args:
        state: Final AgentState returned by run_pipeline.

    Returns:
        Tuple of (latest AlphaSignal or None, BacktestResult or None).
    """
    signals = state.get("signals", [])
    signal = signals[-1] if signals else None
    backtest = state.get("backtest_result")
    return signal, backtest
