"""BacktestResult Pydantic schema — structured output from the backtest agent."""

from datetime import date

from pydantic import BaseModel, Field


class BacktestResult(BaseModel):
    """Performance metrics from a vectorbt backtest of an AlphaSignal.

    Covers a single ticker/direction/period combination. All metrics are
    computed on out-of-sample price data only — no lookahead bias.
    """

    ticker: str = Field(description="Stock ticker that was backtested")
    direction: str = Field(description="Signal direction that was tested: long/short")
    start_date: date = Field(description="First date of the backtest window")
    end_date: date = Field(description="Last date of the backtest window")

    sharpe_ratio: float = Field(
        description="Annualised Sharpe ratio (risk-free rate = 0)"
    )
    max_drawdown: float = Field(
        le=0.0,
        description="Max peak-to-trough drawdown as a negative fraction, e.g. -0.23",
    )
    cagr: float = Field(
        description="Compound annual growth rate as a fraction, e.g. 0.18 = 18%"
    )
    total_return: float = Field(
        description="Total return over the backtest window as a fraction"
    )
    num_trades: int = Field(
        ge=0,
        description="Number of completed round-trip trades",
    )

    benchmark_sharpe: float | None = Field(
        default=None,
        description="Buy-and-hold Sharpe ratio for the same period (for comparison)",
    )
    benchmark_cagr: float | None = Field(
        default=None,
        description="Buy-and-hold CAGR for the same period",
    )
