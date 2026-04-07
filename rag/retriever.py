"""FAISS vector store — build from ingested chunks and expose MMR/hybrid retrieval."""

import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config.settings import Settings
from rag.embeddings import get_embedder

log = logging.getLogger(__name__)

_DOCS_FILE = "docs.json"
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker  # noqa: PLW0603
    if _reranker is None:
        settings = Settings()
        log.info("loading cross-encoder re-ranker: %s", settings.reranker_model)
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def _rerank(query: str, docs: list[Document], k: int) -> list[Document]:
    """Score (query, doc) pairs with a cross-encoder and return top-k."""
    if not docs:
        return docs
    reranker = _get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores: list[float] = reranker.predict(pairs).tolist()
    ranked = sorted(zip(scores, docs, strict=True), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:k]]


def _index_path(ticker: str, settings: Settings) -> str:
    return os.path.join(settings.faiss_index_dir, ticker.upper())


def _docs_path(index_dir: str) -> Path:
    return Path(index_dir) / _DOCS_FILE


def _save_documents(documents: list[Document], index_dir: str) -> None:
    data = [{"page_content": d.page_content, "metadata": d.metadata} for d in documents]
    _docs_path(index_dir).write_text(json.dumps(data))


def _load_documents(index_dir: str) -> list[Document] | None:
    path = _docs_path(index_dir)
    if not path.exists():
        return None
    raw: list[dict[str, Any]] = json.loads(path.read_text())
    return [
        Document(page_content=r["page_content"], metadata=r["metadata"]) for r in raw
    ]


def build_store(chunks: list[dict[str, Any]], ticker: str) -> FAISS:
    """Embed chunks, build a FAISS store, and persist both index and documents to disk.

    Args:
        chunks: Chunk dicts from rag.ingest.chunk_filing with 'content' and 'metadata'.
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

    log.info("[%s] embedding %d chunks into FAISS store", ticker, len(chunks))
    documents = [
        Document(page_content=chunk["content"], metadata=chunk["metadata"])
        for chunk in chunks
    ]

    store = FAISS.from_documents(documents, embedder)

    index_dir = _index_path(ticker, settings)
    os.makedirs(index_dir, exist_ok=True)
    store.save_local(index_dir)
    _save_documents(documents, index_dir)
    log.info("[%s] FAISS index saved to %s", ticker, index_dir)

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

    log.info("[%s] loading existing FAISS index from %s", ticker, index_dir)
    embedder = get_embedder()
    return FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)


def retrieve(
    query: str,
    store: FAISS,
    *,
    k: int | None = None,
    fetch_k: int | None = None,
) -> list[Document]:
    """Run MMR retrieval against a FAISS store (semantic only).

    Args:
        query: Natural language query string.
        store: Populated FAISS store to search.
        k: Number of documents to return. Defaults to settings.retrieval_k.
        fetch_k: Candidate pool size before MMR re-ranking.

    Returns:
        List of LangChain Document objects.
    """
    settings = Settings()
    return store.max_marginal_relevance_search(
        query,
        k=k if k is not None else settings.retrieval_k,
        fetch_k=fetch_k if fetch_k is not None else settings.retrieval_fetch_k,
    )


def retrieve_hybrid(
    query: str,
    store: FAISS,
    ticker: str,
    *,
    k: int | None = None,
    fetch_k: int | None = None,
    rerank: bool = True,
) -> list[Document]:
    """Hybrid BM25 + MMR retrieval with optional cross-encoder re-ranking.

    Pipeline:
      1. BM25 (sparse) — captures exact financial figures and Item citations.
      2. MMR (dense) — adds topical diversity via semantic search.
      3. RRF merge — consensus ranking across both lists.
      4. Cross-encoder re-rank (optional) — precise (query, doc) relevance scoring.

    Falls back to MMR-only if the document cache (docs.json) is absent — this
    happens when loading a FAISS index built before this feature was added.

    Args:
        query: Natural language research question.
        store: Populated FAISS store.
        ticker: Ticker symbol used to locate the document cache.
        k: Documents to return. Defaults to settings.retrieval_k (6).
        fetch_k: MMR candidate pool. Defaults to settings.retrieval_fetch_k (20).
        rerank: Apply cross-encoder re-ranking after RRF. Defaults to True.

    Returns:
        Deduplicated list of Documents, ranked by final score.
    """
    from langchain_community.retrievers import BM25Retriever

    settings = Settings()
    k_val = k if k is not None else settings.retrieval_k
    fetch_k_val = fetch_k if fetch_k is not None else settings.retrieval_fetch_k

    index_dir = _index_path(ticker, settings)
    documents = _load_documents(index_dir)

    if documents is None:
        log.info("[%s] docs cache missing — falling back to MMR only", ticker)
        candidates = store.max_marginal_relevance_search(
            query, k=k_val, fetch_k=fetch_k_val
        )
        return _rerank(query, candidates, k_val) if rerank else candidates

    log.info("[%s] hybrid retrieval (BM25 + MMR) over %d docs", ticker, len(documents))
    bm25_results = BM25Retriever.from_documents(documents, k=k_val).invoke(query)
    mmr_results = store.max_marginal_relevance_search(
        query, k=k_val, fetch_k=fetch_k_val
    )
    candidates = _rrf_merge([bm25_results, mmr_results], k_val)

    if rerank:
        log.info("[%s] cross-encoder re-ranking %d candidates", ticker, len(candidates))
        return _rerank(query, candidates, k_val)
    return candidates


_RRF_K = 60  # standard RRF constant — dampens the impact of rank position


def _rrf_merge(ranked_lists: list[list[Document]], k_final: int) -> list[Document]:
    """Reciprocal Rank Fusion over multiple ranked document lists.

    Each document's score is the sum of 1/(RRF_K + rank) across all lists.
    Documents appearing in multiple lists get a score boost; order reflects
    overall consensus ranking.
    """
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.page_content[:120]  # stable identity within a run
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            docs_by_key[key] = doc

    return [
        docs_by_key[key]
        for key in sorted(scores, key=lambda x: scores[x], reverse=True)
    ][:k_final]
