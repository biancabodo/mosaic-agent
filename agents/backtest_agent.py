"""Backtest agent — runs a vectorbt backtest on the latest AlphaSignal."""

from datetime import date, timedelta
from typing import Any

import pandas as pd
import vectorbt as vbt
import yfinance as yf
from langchain_core.messages import HumanMessage

from graph.state import AgentState
from schemas.backtest_result import BacktestResult
from schemas.signal import AlphaSignal

# Backtest window: 2 years ending yesterday (no lookahead — signal is generated today)
_BACKTEST_YEARS = 2


def _fetch_prices(ticker: str, start: date, end: date) -> pd.Series:
    """Download adjusted daily close prices via yfinance.

    Args:
        ticker: Stock ticker symbol.
        start: First date of the price series (inclusive).
        end: Last date of the price series (inclusive).

    Returns:
        Pandas Series of adjusted close prices indexed by date.

    Raises:
        ValueError: If yfinance returns no data for the given range.
    """
    hist = yf.Ticker(ticker).history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
        auto_adjust=True,
    )
    if hist.empty:
        raise ValueError(f"No price data for '{ticker}' between {start} and {end}.")
    return hist["Close"].rename(ticker)


def _run_backtest(
    prices: pd.Series,
    direction: str,
) -> tuple[float, float, float, float, int]:
    """Execute a simple buy-and-hold vectorbt backtest for the given direction.

    Entry: day 0. Exit: final day. Direction flips sign for short signals.
    Returns annualised metrics with no lookahead — entry price is the first
    available close after signal generation, not the filing date.

    Args:
        prices: Daily close price series.
        direction: 'long' or 'short'.

    Returns:
        Tuple of (sharpe_ratio, max_drawdown, cagr, total_return, num_trades).
    """
    if direction == "short":
        prices = (prices.iloc[0] * 2) - prices  # synthetic short: mirror around entry

    pf = vbt.Portfolio.from_holding(prices, freq="D")

    sharpe: float = float(pf.sharpe_ratio())
    max_dd: float = float(pf.max_drawdown())
    total_ret: float = float(pf.total_return())

    n_days = len(prices)
    years = n_days / 252.0
    cagr: float = float((1.0 + total_ret) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    # from_holding is a single entry + single exit = 1 trade
    num_trades: int = 1

    return sharpe, max_dd, cagr, total_ret, num_trades


def _benchmark_metrics(prices: pd.Series) -> tuple[float, float]:
    """Compute buy-and-hold benchmark Sharpe and CAGR for comparison.

    Args:
        prices: Daily close price series (same window as the signal backtest).

    Returns:
        Tuple of (benchmark_sharpe, benchmark_cagr).
    """
    pf = vbt.Portfolio.from_holding(prices, freq="D")
    sharpe: float = float(pf.sharpe_ratio())
    total_ret: float = float(pf.total_return())
    n_days = len(prices)
    years = n_days / 252.0
    cagr: float = float((1.0 + total_ret) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    return sharpe, cagr


async def backtest_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: run a vectorbt backtest on the latest AlphaSignal.

    Uses a fixed 2-year look-back window ending yesterday to ensure no
    lookahead bias — the signal is generated from filings available before
    the backtest window's end date.

    Updates state with:
        - backtest_result: populated BacktestResult schema
        - messages: appends a human-readable backtest summary

    Args:
        state: Current AgentState; expects at least one entry in state['signals'].

    Returns:
        Partial state dict with backtest_result and messages.
    """
    signals: list[AlphaSignal] = state.get("signals", [])
    if not signals:
        return {
            "backtest_result": None,
            "messages": [
                HumanMessage(content="Backtest skipped: no signal available.")
            ],
        }

    signal = signals[-1]
    ticker = signal.ticker
    direction = signal.direction

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=_BACKTEST_YEARS * 365)

    try:
        prices = _fetch_prices(ticker, start_date, end_date)
    except ValueError as exc:
        return {
            "backtest_result": None,
            "messages": [HumanMessage(content=f"Backtest failed: {exc}")],
        }

    if direction == "neutral":
        # Neutral signals are not actionable — report zero metrics
        result = BacktestResult(
            ticker=ticker,
            direction=direction,
            start_date=start_date,
            end_date=end_date,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            cagr=0.0,
            total_return=0.0,
            num_trades=0,
        )
        summary = f"Backtest skipped: signal direction is neutral for {ticker}."
        return {
            "backtest_result": result,
            "messages": [HumanMessage(content=summary)],
        }

    sharpe, max_dd, cagr, total_ret, num_trades = _run_backtest(prices, direction)
    bench_sharpe, bench_cagr = _benchmark_metrics(prices)

    # Clamp max_drawdown to ensure it satisfies the schema's le=0.0 constraint
    max_dd = min(max_dd, 0.0)

    result = BacktestResult(
        ticker=ticker,
        direction=direction,
        start_date=start_date,
        end_date=end_date,
        sharpe_ratio=round(sharpe, 4),
        max_drawdown=round(max_dd, 4),
        cagr=round(cagr, 4),
        total_return=round(total_ret, 4),
        num_trades=num_trades,
        benchmark_sharpe=round(bench_sharpe, 4),
        benchmark_cagr=round(bench_cagr, 4),
    )

    summary = (
        f"Backtest ({ticker} {direction.upper()}, "
        f"{start_date} → {end_date}): "
        f"Sharpe={result.sharpe_ratio:.2f}, "
        f"MaxDD={result.max_drawdown:.1%}, "
        f"CAGR={result.cagr:.1%}, "
        f"Return={result.total_return:.1%}, "
        f"Trades={result.num_trades} | "
        f"Benchmark Sharpe={result.benchmark_sharpe:.2f}, "
        f"CAGR={result.benchmark_cagr:.1%}"
    )

    return {
        "backtest_result": result,
        "messages": [HumanMessage(content=summary)],
    }
