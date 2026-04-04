# AlphaSignal Research Agent

A multi-agent pipeline that reads SEC EDGAR filings and generates structured investment signals with vectorbt backtesting. Built with LangGraph, FAISS RAG, and Anthropic Claude.

## Architecture

```text
START
  └─► research_agent   — downloads 10-K/10-Q via EDGAR API, builds FAISS index,
        │                 runs MMR retrieval, synthesises research summary via LLM
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

- `research_agent` — RAG over SEC filings (FAISS + MMR retrieval, `all-MiniLM-L6-v2` embeddings). Calls Claude to synthesise retrieved chunks into a structured research summary.
- `signal_agent` — Claude structured output (`AlphaSignal` Pydantic schema). Extracts direction (`long`/`short`/`neutral`), confidence score, and Item-cited rationale.
- `backtest_agent` — vectorbt buy-and-hold backtest over a 2-year trailing window. Computes Sharpe, max drawdown, CAGR, and benchmark comparison. Short signals use a synthetic mirror portfolio.

### RAG pipeline

- Filings are fetched from `data.sec.gov/submissions/` and `www.sec.gov/Archives/`
- HTML is stripped; tables are converted to pipe-delimited plaintext to preserve financial data
- Text is chunked with SEC-aware separators (`\n\nItem` first, then paragraphs)
- FAISS index is persisted to `.faiss_cache/<TICKER>/` — subsequent runs skip re-ingestion

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

```bash
# Run the pipeline for a ticker
uv run python main.py NVDA

# Output example:
# Signal: LONG NVDA (confidence=0.78, period=10-K FY2024)
# Rationale: Item 7 states revenue grew 122% YoY to $60.9B driven by data center...
#
# Backtest (2023-04-03 → 2025-04-03): Sharpe=2.14, MaxDD=-32.1%, CAGR=61.3%, Return=149.2%
```

## Development

```bash
make preflight   # format + lint + typecheck + test (run before committing)
make test        # pytest only
make typecheck   # mypy strict
make clean       # remove .venv and all caches
```

## Evaluation

LangSmith evaluators in `evals/`:

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

A golden dataset of 25 labelled examples (`evals/signal_quality.py`) covers valid long/short/neutral signals, missing citations, overconfident signals, and miscalibrated neutral signals.

## Configuration

All settings are in `config/settings.py` and loaded from `.env`:

| Variable | Default | Description |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | — | Required |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Claude model for all agents |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `SEC_USER_AGENT` | `mosaic-agent research@example.com` | Required by SEC EDGAR |
| `CHUNK_SIZE` | `1000` | RAG chunk size (chars) |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_K` | `6` | MMR results returned |
| `RETRIEVAL_FETCH_K` | `20` | MMR candidate pool |
| `FAISS_INDEX_DIR` | `.faiss_cache` | Persisted index location |
| `CONFIDENCE_THRESHOLD` | `0.6` | Minimum confidence to proceed to backtest |
| `MAX_RESEARCH_ITERATIONS` | `3` | Loop cap before giving up |

## Stack

- **LangGraph** — stateful multi-agent graph with conditional routing
- **LangChain / langchain-anthropic** — LLM calls and structured output
- **FAISS + sentence-transformers** — local vector store, no embedding API needed
- **vectorbt** — backtesting engine
- **yfinance** — price data
- **pydantic-settings** — typed config from `.env`
- **LangSmith** — optional tracing and evaluation
