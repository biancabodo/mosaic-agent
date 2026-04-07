"""Retrieval quality eval — compare hybrid retrieval with and without re-ranking.

Loads an existing FAISS index (built by a prior pipeline run) and measures
precision@k on a set of golden queries: what fraction of the top-k returned
documents contain the expected keywords?

Usage:
    uv run python -m evals.retrieval_eval              # default: NVDA
    uv run python -m evals.retrieval_eval --ticker AAPL
"""

import argparse
import sys
from typing import NamedTuple

from langchain_core.documents import Document

from rag.retriever import load_store, retrieve_hybrid


class QueryCase(NamedTuple):
    query: str
    must_contain: list[str]  # at least one of these must appear in each hit


_QUERY_CASES: list[QueryCase] = [
    QueryCase(
        "quarterly revenue growth and gross margin percentage",
        ["revenue", "gross margin", "margin"],
    ),
    QueryCase(
        "data center segment growth and hyperscaler demand",
        ["data center", "hyperscal", "cloud"],
    ),
    QueryCase(
        "risk factors supply chain and competitive threats",
        ["risk", "supply", "competi"],
    ),
    QueryCase(
        "earnings per share net income and operating income",
        ["earnings", "net income", "operating income", "eps"],
    ),
    QueryCase(
        "forward guidance outlook and management commentary",
        ["guidance", "outlook", "expect", "forecast"],
    ),
    QueryCase(
        "research and development spending and headcount",
        ["research", "development", "r&d", "headcount", "employee"],
    ),
]


def _precision_at_k(docs: list[Document], must_contain: list[str]) -> float:
    """Fraction of docs where page_content contains at least one expected keyword."""
    if not docs:
        return 0.0
    hits = sum(
        1
        for doc in docs
        if any(kw.lower() in doc.page_content.lower() for kw in must_contain)
    )
    return hits / len(docs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precision@k comparison: hybrid retrieval vs hybrid + re-ranking"
    )
    parser.add_argument(
        "--ticker",
        default="NVDA",
        help="Ticker with a pre-built FAISS index (run main.py first)",
    )
    args = parser.parse_args()
    ticker = args.ticker.upper()

    print(f"Loading FAISS index for {ticker}...")
    try:
        store = load_store(ticker)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    col_w = 48
    print(f"\n{'Query':<{col_w}} {'BM25+MMR':>10} {'+ rerank':>10} {'Δ':>8}")
    print("─" * (col_w + 30))

    total_base = 0.0
    total_rerank = 0.0

    for case in _QUERY_CASES:
        base_docs = retrieve_hybrid(case.query, store, ticker, rerank=False)
        rerank_docs = retrieve_hybrid(case.query, store, ticker, rerank=True)

        p_base = _precision_at_k(base_docs, case.must_contain)
        p_rerank = _precision_at_k(rerank_docs, case.must_contain)
        delta = p_rerank - p_base
        total_base += p_base
        total_rerank += p_rerank

        label = (
            case.query if len(case.query) <= col_w else case.query[: col_w - 2] + ".."
        )
        sign = "+" if delta >= 0 else ""
        print(f"{label:<{col_w}} {p_base:>10.1%} {p_rerank:>10.1%} {sign}{delta:>7.1%}")

    n = len(_QUERY_CASES)
    avg_base = total_base / n
    avg_rerank = total_rerank / n
    avg_delta = avg_rerank - avg_base
    sign = "+" if avg_delta >= 0 else ""

    print("─" * (col_w + 30))
    print(
        f"{'Average precision@k':<{col_w}} "
        f"{avg_base:>10.1%} {avg_rerank:>10.1%} {sign}{avg_delta:>7.1%}"
    )
    print(
        "\nMetric: precision@k — fraction of top-k docs matching any expected keyword."
        "\nA positive Δ means re-ranking surfaced more relevant chunks."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
