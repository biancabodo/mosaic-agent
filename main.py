"""CLI entry point — run the AlphaSignal research pipeline for one or more tickers."""

import asyncio
import logging
import sys

from agents.orchestrator import extract_result, run_pipeline
from storage import signals as signal_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
# Silence noisy third-party loggers
for _noisy in (  # noqa: E501
    "httpx",
    "httpcore",
    "huggingface_hub",
    "sentence_transformers",
    "faiss",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


async def run_one(ticker: str) -> None:
    """Run the full pipeline for a single ticker and print results."""
    state = await run_pipeline(ticker)

    if state.get("error"):
        print(f"[{ticker.upper()}] error: {state['error']}", file=sys.stderr)
        return

    signal, backtest = extract_result(state)

    if signal:
        print(
            f"\n[{signal.ticker}] Signal: {signal.direction.upper()} "
            f"(confidence={signal.confidence:.2f}, period={signal.filing_period})"
        )
        print(f"[{signal.ticker}] Rationale: {signal.rationale}")
        signal_store.save(signal, backtest)

    if signal and backtest:
        print(
            f"[{signal.ticker}] Backtest ({backtest.start_date} → {backtest.end_date}): "  # noqa: E501
            f"Sharpe={backtest.sharpe_ratio:.2f}, "
            f"MaxDD={backtest.max_drawdown:.1%}, "
            f"CAGR={backtest.cagr:.1%}, "
            f"Return={backtest.total_return:.1%}"
        )


async def main(tickers: list[str]) -> None:
    print(f"Running AlphaSignal pipeline for: {', '.join(t.upper() for t in tickers)}")
    await asyncio.gather(*[run_one(t) for t in tickers])


if __name__ == "__main__":
    if len(sys.argv) < 2:  # noqa: PLR2004
        print("Usage: python main.py <TICKER> [TICKER ...]", file=sys.stderr)
        sys.exit(1)
    asyncio.run(main(sys.argv[1:]))
