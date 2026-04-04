"""FAISS vector store — build from ingested chunks and expose MMR retrieval."""

import os
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import Settings
from rag.embeddings import get_embedder


def _index_path(ticker: str, settings: Settings) -> str:
    """Return the filesystem path for a ticker's persisted FAISS index.

    Args:
        ticker: Stock ticker symbol, e.g. 'NVDA'.
        settings: Application settings (provides faiss_index_dir).

    Returns:
        Absolute directory path where the index is saved/loaded.
    """
    return os.path.join(settings.faiss_index_dir, ticker.upper())


def build_store(chunks: list[dict[str, Any]], ticker: str) -> FAISS:
    """Embed chunks and build a FAISS vector store, persisting it to disk.

    If an index already exists for the ticker it is overwritten. This is
    intentional — call load_store if you want to reuse an existing index.

    Args:
        chunks: List of chunk dicts from rag.ingest.chunk_filing, each with
            'content' (str) and 'metadata' (dict) keys.
        ticker: Used to namespace the persisted index directory.

    Returns:
        In-memory FAISS store ready for retrieval.

    Raises:
        ValueError: If chunks is empty.
    """
    if not chunks:
        raise ValueError(f"Cannot build FAISS store: no chunks provided for '{ticker}'")

    settings = Settings()
    embedder = get_embedder()

    documents = [
        Document(page_content=chunk["content"], metadata=chunk["metadata"])
        for chunk in chunks
    ]

    store = FAISS.from_documents(documents, embedder)

    index_dir = _index_path(ticker, settings)
    os.makedirs(index_dir, exist_ok=True)
    store.save_local(index_dir)

    return store


def load_store(ticker: str) -> FAISS:
    """Load a previously persisted FAISS index from disk.

    Args:
        ticker: Stock ticker whose index should be loaded.

    Returns:
        Loaded FAISS store.

    Raises:
        FileNotFoundError: If no persisted index exists for this ticker.
    """
    settings = Settings()
    index_dir = _index_path(ticker, settings)

    if not os.path.exists(index_dir):
        raise FileNotFoundError(
            f"No FAISS index found for '{ticker}' at '{index_dir}'. "
            "Run ingest_ticker() first."
        )

    embedder = get_embedder()
    return FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)


def get_or_build_store(chunks: list[dict[str, Any]], ticker: str) -> FAISS:
    """Return an existing FAISS store for the ticker, or build one from chunks.

    Prefers the persisted index to avoid re-embedding. Only builds a new
    store if no index file exists.

    Args:
        chunks: Chunks to use if building a new store. Ignored if loading.
        ticker: Target ticker symbol.

    Returns:
        FAISS store (loaded or freshly built).
    """
    settings = Settings()
    index_dir = _index_path(ticker, settings)

    if os.path.exists(index_dir):
        return load_store(ticker)

    return build_store(chunks, ticker)


def retrieve(
    query: str,
    store: FAISS,
    *,
    k: int | None = None,
    fetch_k: int | None = None,
) -> list[Document]:
    """Run MMR retrieval against a FAISS store.

    Maximal Marginal Relevance balances relevance to the query against
    diversity of results — important for SEC filings where many chunks
    may be near-duplicates.

    Args:
        query: Natural language query string.
        store: Populated FAISS store to search.
        k: Number of documents to return. Defaults to settings.retrieval_k (6).
        fetch_k: Candidate pool size before MMR re-ranking.
            Defaults to settings.retrieval_fetch_k (20).

    Returns:
        List of LangChain Document objects, each with page_content and metadata.
    """
    settings = Settings()
    return store.max_marginal_relevance_search(
        query,
        k=k if k is not None else settings.retrieval_k,
        fetch_k=fetch_k if fetch_k is not None else settings.retrieval_fetch_k,
    )
