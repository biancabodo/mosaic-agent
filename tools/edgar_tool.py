"""LangChain tool wrapping the EDGAR ingest pipeline for use by agent nodes."""

from langchain_core.tools import tool

from rag.ingest import ingest_ticker
from rag.retriever import build_store, retrieve


@tool
async def edgar_search(ticker: str, query: str) -> str:
    """Download SEC filings for a ticker and retrieve relevant chunks via RAG.

    Fetches the most recent 10-K and 10-Q filings from EDGAR, embeds them
    into a FAISS store, then runs MMR retrieval for the given query. Returns
    the top chunks as a single formatted string for the research agent context.

    Args:
        ticker: Stock ticker symbol, e.g. 'NVDA'.
        query: Natural language research question to retrieve evidence for.

    Returns:
        Formatted string of retrieved filing chunks with source metadata,
        or an error message if ingestion or retrieval fails.
    """
    try:
        chunks = await ingest_ticker(ticker, form_types=["10-K", "10-Q"], max_filings=2)
        store = build_store(chunks, ticker)
        docs = retrieve(query, store)
    except ValueError as exc:
        return f"EDGAR retrieval failed: {exc}"

    if not docs:
        return f"No relevant chunks found in SEC filings for '{ticker}'."

    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = (
            f"[{i}] {meta.get('form_type', 'FILING')} "
            f"({meta.get('filing_date', 'unknown date')}) "
            f"— chunk {meta.get('chunk_index', '?')}"
        )
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)
