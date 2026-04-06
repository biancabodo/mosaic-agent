"""CLI entry point — run the AlphaSignal research pipeline for a given ticker."""

import asyncio
import logging
import sys

from agents.orchestrator import extract_result, run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
# Silence noisy third-party loggers
for _noisy in (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "sentence_transformers",
    "faiss",
):  # noqa: E501
    logging.getLogger(_noisy).setLevel(logging.WARNING)


async def main(ticker: str) -> None:
    print(f"Running AlphaSignal pipeline for {ticker.upper()}...")
    state = await run_pipeline(ticker)

    if state.get("error"):
        print(f"Pipeline error: {state['error']}", file=sys.stderr)
        sys.exit(1)

    signal, backtest = extract_result(state)

    if signal:
        print(
            f"\nSignal: {signal.direction.upper()} {signal.ticker} "
            f"(confidence={signal.confidence:.2f}, period={signal.filing_period})"
        )
        print(f"Rationale: {signal.rationale}")

    if backtest:
        print(
            f"\nBacktest ({backtest.start_date} → {backtest.end_date}): "
            f"Sharpe={backtest.sharpe_ratio:.2f}, "
            f"MaxDD={backtest.max_drawdown:.1%}, "
            f"CAGR={backtest.cagr:.1%}, "
            f"Return={backtest.total_return:.1%}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:  # noqa: PLR2004
        print("Usage: python main.py <TICKER>", file=sys.stderr)
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
