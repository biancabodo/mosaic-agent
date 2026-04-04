"""Rule-based sanity checks for BacktestResult — no lookahead bias, valid ranges."""

from datetime import date
from typing import Any

from langsmith import traceable
from langsmith.evaluation import EvaluationResult

from schemas.backtest_result import BacktestResult

# Metric guard rails — values outside these ranges indicate a bug or data issue
_SHARPE_MIN = -5.0
_SHARPE_MAX = 10.0
_CAGR_MIN = -1.0  # -100% is the floor (total loss)
_CAGR_MAX = 10.0  # 1000% CAGR is implausible for a 2yr backtest
_MAX_DD_MIN = -1.0  # -100% is the floor


@traceable(name="eval-no-lookahead")
def evaluate_no_lookahead(
    run: Any,
    example: Any,
) -> EvaluationResult:
    """Verify the backtest window contains no lookahead bias.

    Three checks:
    1. start_date < end_date (window is valid)
    2. end_date <= today (window doesn't reach into the future)
    3. end_date <= signal.generated_at.date() (backtest doesn't use data
       generated after the signal was produced)

    Args:
        run: LangSmith Run with outputs containing 'backtest_result' dict
            and optionally 'signal' dict with 'generated_at'.
        example: LangSmith Example (unused).

    Returns:
        EvaluationResult with score 1.0 (no lookahead) or 0.0 (lookahead detected).
    """
    outputs = run.outputs or {}
    backtest_data = outputs.get("backtest_result")

    if backtest_data is None:
        return EvaluationResult(
            key="no_lookahead",
            score=None,
            comment="No backtest_result in run outputs",
        )

    try:
        result = BacktestResult.model_validate(backtest_data)
    except Exception as exc:  # noqa: BLE001
        return EvaluationResult(
            key="no_lookahead",
            score=0.0,
            comment=f"BacktestResult parse failed: {exc}",
        )

    today = date.today()
    issues: list[str] = []

    if result.start_date >= result.end_date:
        issues.append(
            f"start_date ({result.start_date}) >= end_date ({result.end_date})"
        )

    if result.end_date > today:
        issues.append(f"end_date ({result.end_date}) is in the future")

    # Check against signal generation time if available
    signal_data = outputs.get("signal", {})
    if isinstance(signal_data, dict) and "generated_at" in signal_data:
        from datetime import datetime

        generated_at_str = signal_data["generated_at"]
        try:
            generated_at = datetime.fromisoformat(str(generated_at_str)).date()
            if result.end_date > generated_at:
                issues.append(
                    f"end_date ({result.end_date}) is after signal "
                    f"generated_at ({generated_at})"
                )
        except ValueError:
            pass  # Can't parse timestamp — skip this check

    if issues:
        return EvaluationResult(
            key="no_lookahead",
            score=0.0,
            comment="; ".join(issues),
        )

    return EvaluationResult(key="no_lookahead", score=1.0)


@traceable(name="eval-metric-ranges")
def evaluate_metric_ranges(
    run: Any,
    example: Any,
) -> EvaluationResult:
    """Check that backtest metrics fall within plausible ranges.

    Catches implementation bugs (e.g. wrong sign on drawdown, annualisation
    errors producing implausible CAGR values, or NaN/Inf from edge cases).

    Guard rails:
        - Sharpe ratio: [-5.0, 10.0]
        - CAGR: [-1.0, 10.0]
        - max_drawdown: [-1.0, 0.0]
        - num_trades: >= 0

    Args:
        run: LangSmith Run with outputs containing 'backtest_result' dict.
        example: LangSmith Example (unused).

    Returns:
        EvaluationResult with score 1.0 (all in range) or 0.0 (out of range).
    """
    import math

    outputs = run.outputs or {}
    backtest_data = outputs.get("backtest_result")

    if backtest_data is None:
        return EvaluationResult(
            key="metric_ranges_valid",
            score=None,
            comment="No backtest_result in run outputs",
        )

    try:
        result = BacktestResult.model_validate(backtest_data)
    except Exception as exc:  # noqa: BLE001
        return EvaluationResult(
            key="metric_ranges_valid",
            score=0.0,
            comment=f"Parse failed: {exc}",
        )

    issues: list[str] = []

    metrics: list[tuple[str, float, float, float]] = [
        ("sharpe_ratio", result.sharpe_ratio, _SHARPE_MIN, _SHARPE_MAX),
        ("cagr", result.cagr, _CAGR_MIN, _CAGR_MAX),
        ("max_drawdown", result.max_drawdown, _MAX_DD_MIN, 0.0),
    ]

    for name, value, lo, hi in metrics:
        if math.isnan(value) or math.isinf(value):
            issues.append(f"{name} is {value}")
        elif not (lo <= value <= hi):
            issues.append(f"{name}={value:.4f} outside [{lo}, {hi}]")

    if result.num_trades < 0:
        issues.append(f"num_trades={result.num_trades} is negative")

    if issues:
        return EvaluationResult(
            key="metric_ranges_valid",
            score=0.0,
            comment="; ".join(issues),
        )

    return EvaluationResult(key="metric_ranges_valid", score=1.0)


@traceable(name="eval-benchmark-comparison")
def evaluate_benchmark_comparison(
    run: Any,
    example: Any,
) -> EvaluationResult:
    """Flag signals where the strategy underperforms buy-and-hold on both metrics.

    A signal-driven strategy that has both lower Sharpe and lower CAGR than
    a passive benchmark adds no value. This doesn't make the signal invalid,
    but it's a useful quality gate.

    Args:
        run: LangSmith Run with outputs containing 'backtest_result' dict.
        example: LangSmith Example (unused).

    Returns:
        EvaluationResult with score 1.0 (beats benchmark on at least one metric)
        or 0.0 (underperforms on both). Score is None if benchmark data is absent.
    """
    outputs = run.outputs or {}
    backtest_data = outputs.get("backtest_result")

    if backtest_data is None:
        return EvaluationResult(key="beats_benchmark", score=None)

    try:
        result = BacktestResult.model_validate(backtest_data)
    except Exception as exc:  # noqa: BLE001
        return EvaluationResult(
            key="beats_benchmark",
            score=0.0,
            comment=f"Parse failed: {exc}",
        )

    if result.benchmark_sharpe is None or result.benchmark_cagr is None:
        return EvaluationResult(
            key="beats_benchmark",
            score=None,
            comment="No benchmark metrics available for comparison",
        )

    beats_sharpe = result.sharpe_ratio >= result.benchmark_sharpe
    beats_cagr = result.cagr >= result.benchmark_cagr

    if not beats_sharpe and not beats_cagr:
        return EvaluationResult(
            key="beats_benchmark",
            score=0.0,
            comment=(
                f"Strategy (Sharpe={result.sharpe_ratio:.2f}, CAGR={result.cagr:.1%}) "
                f"underperforms benchmark "
                f"(Sharpe={result.benchmark_sharpe:.2f}, "
                f"CAGR={result.benchmark_cagr:.1%})"
            ),
        )

    return EvaluationResult(key="beats_benchmark", score=1.0)
