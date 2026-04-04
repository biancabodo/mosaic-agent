"""Conditional routing logic for the AlphaSignal LangGraph StateGraph."""

from graph.state import AgentState


def route_after_signal(state: AgentState) -> str:
    """Determine the next node after the signal agent runs.

    Routing rules:
    - → 'backtest_agent'  if the latest signal has confidence >= threshold
    - → 'research_agent'  if confidence < threshold AND iteration_count < 3
    - → END               if confidence < threshold AND iteration_count >= 3

    The iteration cap prevents infinite loops when the LLM repeatedly produces
    low-confidence signals (e.g. due to insufficient filing evidence).

    Args:
        state: Current AgentState, expected to have at least one signal in
            state['signals'] after the signal agent has run.

    Returns:
        String key matching a node name or '__end__' for LangGraph termination.
    """
    from config.settings import Settings

    settings = Settings()

    signals = state.get("signals", [])
    iteration_count = state.get("iteration_count", 0)

    if not signals:
        # No signal produced — treat as low confidence, retry or end
        if iteration_count < settings.max_research_iterations:
            return "research_agent"
        return "__end__"

    latest_signal = signals[-1]

    if latest_signal.confidence >= settings.confidence_threshold:
        return "backtest_agent"

    if iteration_count < settings.max_research_iterations:
        return "research_agent"

    return "__end__"


def route_after_research(state: AgentState) -> str:
    """Determine the next node after the research agent runs.

    The research agent always routes to the signal agent. This function
    exists to make the routing symmetric and to allow future branching
    (e.g. routing to END on a research error).

    Args:
        state: Current AgentState after research_agent has populated
            research_context.

    Returns:
        'signal_agent' under normal operation, '__end__' on unrecoverable error.
    """
    if state.get("error"):
        return "__end__"
    return "signal_agent"
