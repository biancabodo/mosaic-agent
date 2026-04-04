"""Tests for LangGraph conditional routing logic in graph/edges.py."""

from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from graph.edges import route_after_research, route_after_signal
from graph.state import AgentState
from schemas.signal import AlphaSignal


@pytest.fixture
def make_signal() -> Callable[..., AlphaSignal]:
    """Factory fixture for AlphaSignal with configurable confidence and direction."""

    def _make(confidence: float, direction: str = "long") -> AlphaSignal:
        return AlphaSignal(
            ticker="NVDA",
            direction=direction,
            confidence=confidence,
            rationale="Item 7 states revenue grew 122% YoY driven by data center.",
            supporting_chunks=["Item 7: Revenue grew 122% YoY."],
            filing_period="10-K FY2024",
            generated_at=datetime.now(UTC),
        )

    return _make


@pytest.fixture
def base_state() -> AgentState:
    """Minimal valid AgentState for routing tests."""
    return {
        "ticker": "NVDA",
        "messages": [],
        "research_context": "Some research context here.",
        "signals": [],
        "backtest_result": None,
        "iteration_count": 1,
        "error": None,
    }


# ---------------------------------------------------------------------------
# route_after_signal
# ---------------------------------------------------------------------------


def test_high_confidence_routes_to_backtest(
    base_state: AgentState,
    make_signal: Callable[..., AlphaSignal],
) -> None:
    """Confidence above threshold should route to backtest."""
    state = {**base_state, "signals": [make_signal(0.8)], "iteration_count": 1}
    assert route_after_signal(state) == "backtest_agent"  # type: ignore[arg-type]


def test_exactly_at_threshold_routes_to_backtest(
    base_state: AgentState,
    make_signal: Callable[..., AlphaSignal],
) -> None:
    """Confidence exactly at the threshold (0.6) should pass to backtest."""
    state = {**base_state, "signals": [make_signal(0.6)], "iteration_count": 1}
    assert route_after_signal(state) == "backtest_agent"  # type: ignore[arg-type]


def test_low_confidence_iter_1_routes_to_research(
    base_state: AgentState,
    make_signal: Callable[..., AlphaSignal],
) -> None:
    """Low confidence with iterations remaining should retry research."""
    state = {**base_state, "signals": [make_signal(0.4)], "iteration_count": 1}
    assert route_after_signal(state) == "research_agent"  # type: ignore[arg-type]


def test_low_confidence_iter_2_routes_to_research(
    base_state: AgentState,
    make_signal: Callable[..., AlphaSignal],
) -> None:
    """Low confidence with one iteration remaining should still retry."""
    state = {**base_state, "signals": [make_signal(0.4)], "iteration_count": 2}
    assert route_after_signal(state) == "research_agent"  # type: ignore[arg-type]


def test_low_confidence_iter_3_routes_to_end(
    base_state: AgentState,
    make_signal: Callable[..., AlphaSignal],
) -> None:
    """Low confidence at max iterations should terminate the graph."""
    state = {**base_state, "signals": [make_signal(0.4)], "iteration_count": 3}
    assert route_after_signal(state) == "__end__"  # type: ignore[arg-type]


def test_low_confidence_iter_above_max_routes_to_end(
    base_state: AgentState,
    make_signal: Callable[..., AlphaSignal],
) -> None:
    """Iteration count above the cap should also terminate."""
    state = {**base_state, "signals": [make_signal(0.3)], "iteration_count": 5}
    assert route_after_signal(state) == "__end__"  # type: ignore[arg-type]


def test_no_signal_iter_1_routes_to_research(base_state: AgentState) -> None:
    """Signal agent producing no output should trigger a research retry."""
    state = {**base_state, "signals": [], "iteration_count": 1}
    assert route_after_signal(state) == "research_agent"  # type: ignore[arg-type]


def test_no_signal_iter_3_routes_to_end(base_state: AgentState) -> None:
    """No signal at max iterations should terminate."""
    state = {**base_state, "signals": [], "iteration_count": 3}
    assert route_after_signal(state) == "__end__"  # type: ignore[arg-type]


def test_short_direction_high_confidence_routes_to_backtest(
    base_state: AgentState,
    make_signal: Callable[..., AlphaSignal],
) -> None:
    """Direction should not affect routing — only confidence matters."""
    signals = [make_signal(0.75, "short")]
    state = {**base_state, "signals": signals, "iteration_count": 1}
    assert route_after_signal(state) == "backtest_agent"  # type: ignore[arg-type]


def test_neutral_direction_high_confidence_routes_to_backtest(
    base_state: AgentState,
    make_signal: Callable[..., AlphaSignal],
) -> None:
    """Even neutral signals above threshold proceed to backtest."""
    signals = [make_signal(0.65, "neutral")]
    state = {**base_state, "signals": signals, "iteration_count": 1}
    assert route_after_signal(state) == "backtest_agent"  # type: ignore[arg-type]


def test_routing_uses_latest_signal(
    base_state: AgentState,
    make_signal: Callable[..., AlphaSignal],
) -> None:
    """When multiple signals exist, routing should use the most recent one."""
    old_low = make_signal(0.3)
    new_high = make_signal(0.9)
    state = {**base_state, "signals": [old_low, new_high], "iteration_count": 1}
    assert route_after_signal(state) == "backtest_agent"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# route_after_research
# ---------------------------------------------------------------------------


def test_research_no_error_routes_to_signal(base_state: AgentState) -> None:
    """Successful research should always proceed to signal extraction."""
    assert route_after_research(base_state) == "signal_agent"


def test_research_with_error_routes_to_end(base_state: AgentState) -> None:
    """Research error should terminate the graph."""
    state = {**base_state, "error": "EDGAR API timeout"}
    assert route_after_research(state) == "__end__"  # type: ignore[arg-type]


def test_research_empty_error_string_routes_to_signal(base_state: AgentState) -> None:
    """Empty string error should not trigger early termination."""
    state = {**base_state, "error": ""}
    # Empty string is falsy — treated as no error
    assert route_after_research(state) == "signal_agent"  # type: ignore[arg-type]
