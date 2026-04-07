# AlphaSignal Research Agent

A multi-agent pipeline that reads SEC EDGAR filings and generates structured investment signals with vectorbt backtesting. Built with LangGraph, FAISS RAG, and Anthropic Claude.

## Architecture

```text
START
  └─► research_agent   — downloads 10-K / 10-Q / 8-K via EDGAR API, builds FAISS
        │                 index, runs hybrid BM25+MMR retrieval with cross-encoder
        │                 re-ranking, synthesises research summary via LLM
        │
        ├─► (error) ──► END
        │
        ▼
      signal_agent     — extracts structured AlphaSignal (direction, confidence,
        │                 rationale with Item citations) via structured output
        │
        ├─► confidence ≥ 0.6 ──► backtest_agent ──► END
        │
        └─► confidence < 0.6, iterations < 3 ──► research_agent (retry)
              └─► iterations ≥ 3 ──► END
```

### Agents

- `research_agent` — RAG over SEC filings (FAISS + hybrid retrieval, `all-MiniLM-L6-v2` embeddings). Calls Claude to synthesise retrieved chunks into a structured research summary. Streams output to the terminal when running a single ticker.
- `signal_agent` — Claude structured output (`AlphaSignal` Pydantic schema). Extracts direction (`long`/`short`/`neutral`), confidence score, and Item-cited rationale.
- `backtest_agent` — vectorbt buy-and-hold backtest over a 2-year trailing window. Computes Sharpe, max drawdown, CAGR, and benchmark comparison. Short signals use a synthetic mirror portfolio.

### RAG pipeline

1. **Ingestion** — filings fetched from `data.sec.gov/submissions/` and `www.sec.gov/Archives/` in parallel. HTML is stripped; tables are converted to pipe-delimited plaintext to preserve financial data. Text is chunked with SEC-aware separators (`\n\nItem` first, then paragraphs).
2. **Sparse retrieval (BM25)** — keyword search captures exact figures, Item numbers, and ticker mentions.
3. **Dense retrieval (MMR)** — semantic search with diversity re-ranking using FAISS.
4. **RRF merge** — Reciprocal Rank Fusion combines both ranked lists into a single consensus ranking.
5. **Cross-encoder re-rank** — `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (query, chunk) pair directly and reorders the final shortlist.

The FAISS index is persisted to `.faiss_cache/<TICKER>/` — subsequent runs skip re-ingestion.

## Setup

```bash
# 1. Install dependencies
make build

# 2. Configure environment
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### .env

```bash
ANTHROPIC_API_KEY=sk-ant-...

# Optional — enable LangSmith tracing
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=mosaic-agent
```

All settings have sensible defaults; only `ANTHROPIC_API_KEY` is required.

## Usage

### CLI

```bash
# Single ticker — research synthesis streams to terminal
uv run python main.py NVDA

# Multiple tickers — runs in parallel, output buffered
uv run python main.py NVDA AAPL MSFT
```

Example output:

```text
[NVDA] Signal: LONG (confidence=0.72, period=10-K FY2024)
[NVDA] Rationale: Item 7 states revenue grew 122% YoY to $60.9B driven by data center...
[NVDA] Backtest (2023-04-03 → 2025-04-03): Sharpe=1.17, MaxDD=-36.9%, CAGR=43.2%, Return=104.0%
```

### Web UI

```bash
make ui
# or: uv run python app.py
```

Opens a Gradio interface at `http://localhost:7860`. Enter any US ticker and click **Run pipeline** — results include the signal, rationale, backtest metrics vs SPY, and the full research synthesis.

## Development

```bash
make preflight      # format + lint + typecheck + test (run before committing)
make test           # pytest only
make typecheck      # mypy strict
make evals          # signal quality + backtest sanity evaluators
make retrieval-eval # precision@k comparison: BM25+MMR vs BM25+MMR+rerank
make clean          # remove .venv and all caches
```

## Evaluation

### Signal quality (`make evals`)

LangSmith evaluators in `evals/` run against a golden dataset of 25 labelled examples:

| Evaluator | What it checks |
| --- | --- |
| `evaluate_citation_presence` | Rationale cites a specific `Item N` section |
| `evaluate_signal_structure` | AlphaSignal validates against Pydantic schema |
| `evaluate_confidence_calibration` | Neutral signals < 0.65, no signal > 0.95 |
| `evaluate_supporting_evidence` | Supporting chunks are substantive (> 50 chars) |
| `evaluate_rag_faithfulness` | LLM-as-judge: claims in rationale grounded in chunks |
| `evaluate_no_lookahead` | Backtest window doesn't reach into the future |
| `evaluate_metric_ranges` | Sharpe, CAGR, drawdown within plausible bounds |
| `evaluate_benchmark_comparison` | Strategy beats buy-and-hold on at least one metric |

### Retrieval quality (`make retrieval-eval`)

Measures precision@k on 6 golden queries against a pre-built index, comparing hybrid retrieval (BM25+MMR) with and without the cross-encoder re-ranker:

```text
Query                                            BM25+MMR   + rerank        Δ
────────────────────────────────────────────────────────────────────────────
quarterly revenue growth and gross margin..      83.3%      100.0%     +16.7%
data center segment growth and hyperscal..      100.0%      100.0%      +0.0%
...
Average precision@k                              88.9%       95.2%      +6.3%
```

## Configuration

All settings are in `config/settings.py` and loaded from `.env`:

| Variable | Default | Description |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | — | Required |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Claude model for all agents |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder re-ranker |
| `SEC_USER_AGENT` | `mosaic-agent research@example.com` | Required by SEC EDGAR |
| `EDGAR_FORM_TYPES` | `["10-K","10-Q","8-K"]` | Filing types to ingest |
| `EDGAR_MAX_FILINGS` | `3` | Max filings per run |
| `CHUNK_SIZE` | `1000` | RAG chunk size (chars) |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_K` | `6` | Documents returned per query |
| `RETRIEVAL_FETCH_K` | `20` | MMR candidate pool size |
| `FAISS_INDEX_DIR` | `.faiss_cache` | Persisted index location |
| `CONFIDENCE_THRESHOLD` | `0.6` | Minimum confidence to proceed to backtest |
| `MAX_RESEARCH_ITERATIONS` | `3` | Loop cap before giving up |

## Stack

- **LangGraph** — stateful multi-agent graph with conditional routing
- **LangChain / langchain-anthropic** — LLM calls and structured output
- **FAISS + sentence-transformers** — local vector store and cross-encoder re-ranker, no embedding API needed
- **rank-bm25** — sparse keyword retrieval
- **vectorbt** — backtesting engine
- **yfinance** — price data
- **Gradio** — web UI
- **pydantic-settings** — typed config from `.env`
- **LangSmith** — optional tracing and evaluation
