"""Compile the full AlphaSignal LangGraph StateGraph."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.edges import route_after_research, route_after_signal
from graph.state import AgentState


def build_graph() -> CompiledStateGraph:  # type: ignore[type-arg]
    """Assemble and compile the multi-agent AlphaSignal research graph.

    Node wiring:
        START → research_agent
        research_agent → (signal_agent | END)  via route_after_research
        signal_agent   → (backtest_agent | research_agent | END)  via route_after_signal
        backtest_agent → END

    Agents are imported inside this function to avoid circular imports at
    module load time (agents import graph.state, graph imports agents).

    Returns:
        Compiled LangGraph StateGraph ready to invoke with an initial AgentState.
    """
    # Deferred imports to break circular dependency chain
    from agents.backtest_agent import backtest_node
    from agents.research_agent import research_node
    from agents.signal_agent import signal_node

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("research_agent", research_node)
    graph.add_node("signal_agent", signal_node)
    graph.add_node("backtest_agent", backtest_node)

    # Entry point
    graph.add_edge(START, "research_agent")

    # Research → signal or END (on error)
    graph.add_conditional_edges(
        "research_agent",
        route_after_research,
        {
            "signal_agent": "signal_agent",
            "__end__": END,
        },
    )

    # Signal → backtest, research (retry), or END (iteration cap / no confidence)
    graph.add_conditional_edges(
        "signal_agent",
        route_after_signal,
        {
            "backtest_agent": "backtest_agent",
            "research_agent": "research_agent",
            "__end__": END,
        },
    )

    # Backtest always terminates
    graph.add_edge("backtest_agent", END)

    return graph.compile()
