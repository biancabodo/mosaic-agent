"""Research agent — RAG-powered SEC filing researcher node for the LangGraph."""

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from config.settings import Settings
from graph.state import AgentState
from rag.ingest import ingest_ticker
from rag.retriever import build_store, load_store, retrieve

_RESEARCH_PROMPT = """\
You are a senior equity research analyst specialising in SEC filings.

Your task is to retrieve and synthesise evidence from {ticker} SEC filings
that is relevant to the following research question:

    {query}

You have been given the following filing excerpts retrieved via semantic search:

{chunks}

Produce a structured research summary that:
1. Identifies the most important quantitative facts (revenue, margins, guidance)
2. Notes any risks or red flags mentioned in risk factors or MD&A
3. Highlights any forward-looking statements or management commentary
4. Cites the specific Item section for each key point (e.g. "Item 7 states...")

Be precise and factual. Do not speculate beyond what the filings state.
"""

_DEFAULT_QUERY = (
    "What are the key financial trends, risks, and growth drivers "
    "that would inform a directional investment signal?"
)


async def research_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: fetch SEC filings and populate research_context.

    Downloads 10-K and 10-Q filings for the target ticker, builds or loads
    a FAISS index, runs MMR retrieval, and asks the LLM to synthesise the
    retrieved chunks into a structured research summary.

    Updates state with:
        - research_context: LLM-synthesised summary of retrieved evidence
        - messages: appends the research summary as an assistant message
        - iteration_count: incremented by 1
        - error: set to an error string on failure (routes to END)

    Args:
        state: Current AgentState passed by LangGraph.

    Returns:
        Partial state dict with updated fields.
    """
    settings = Settings()
    ticker = state["ticker"]
    iteration = state.get("iteration_count", 0)

    if iteration == 0:
        query = _DEFAULT_QUERY
    else:
        query = (
            f"Provide additional evidence on {ticker}'s financial performance, "
            f"segment revenue breakdown, and any quantitative guidance. "
            f"Focus on sections not yet covered in prior retrieval."
        )

    try:
        try:
            store = load_store(ticker)
        except FileNotFoundError:
            chunks = await ingest_ticker(
                ticker, form_types=["10-K", "10-Q"], max_filings=2
            )
            store = build_store(chunks, ticker)
        docs = retrieve(query, store)
    except (ValueError, OSError) as exc:
        return {
            "error": f"Research failed for '{ticker}': {exc}",
            "iteration_count": iteration + 1,
        }

    if not docs:
        return {
            "error": f"No filing chunks retrieved for '{ticker}'.",
            "iteration_count": iteration + 1,
        }

    chunks_text = "\n\n---\n\n".join(
        f"[{i + 1}] {doc.metadata.get('form_type', 'FILING')} "
        f"({doc.metadata.get('filing_date', '?')})\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    llm = ChatAnthropic(
        model=settings.anthropic_model,
        temperature=0.1,
        api_key=settings.anthropic_api_key,
    )

    prompt = _RESEARCH_PROMPT.format(ticker=ticker, query=query, chunks=chunks_text)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    research_context = str(response.content)

    return {
        "research_context": research_context,
        "messages": [response],
        "iteration_count": iteration + 1,
        "error": None,
    }
