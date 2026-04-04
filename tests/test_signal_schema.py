"""Tests for AlphaSignal and BacktestResult Pydantic schema validation."""

from datetime import UTC, date, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from schemas.backtest_result import BacktestResult
from schemas.signal import AlphaSignal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_RATIONALE = (
    "Item 7 states revenue grew 122% YoY to $60.9B driven by data center demand."
)


def valid_signal_data(**overrides: object) -> dict[str, Any]:
    """Return a minimal valid AlphaSignal dict, with optional field overrides."""
    base: dict[str, Any] = {
        "ticker": "NVDA",
        "direction": "long",
        "confidence": 0.8,
        "rationale": _VALID_RATIONALE,
        "supporting_chunks": ["Item 7: Revenue grew 122% YoY."],
        "filing_period": "10-K FY2024",
        "generated_at": datetime.now(UTC),
    }
    return {**base, **overrides}


def valid_backtest_data(**overrides: object) -> dict[str, Any]:
    """Return a minimal valid BacktestResult dict, with optional field overrides."""
    base: dict[str, Any] = {
        "ticker": "NVDA",
        "direction": "long",
        "start_date": date(2022, 1, 1),
        "end_date": date(2024, 1, 1),
        "sharpe_ratio": 1.5,
        "max_drawdown": -0.23,
        "cagr": 0.18,
        "total_return": 0.38,
        "num_trades": 1,
    }
    return {**base, **overrides}


# ---------------------------------------------------------------------------
# AlphaSignal — valid cases
# ---------------------------------------------------------------------------


def test_valid_signal_parses() -> None:
    """A fully valid signal dict should parse without errors."""
    signal = AlphaSignal(**valid_signal_data())
    assert signal.ticker == "NVDA"
    assert signal.direction == "long"
    assert signal.confidence == 0.8


def test_all_directions_accepted() -> None:
    """All three valid directions must be accepted by the schema."""
    for direction in ("long", "short", "neutral"):
        signal = AlphaSignal(**valid_signal_data(direction=direction))
        assert signal.direction == direction


def test_confidence_boundary_at_zero() -> None:
    """Confidence of exactly 0.0 should be valid."""
    AlphaSignal(**valid_signal_data(confidence=0.0))


def test_confidence_boundary_at_one() -> None:
    """Confidence of exactly 1.0 should be valid."""
    AlphaSignal(**valid_signal_data(confidence=1.0))


def test_generated_at_defaults_to_utc_now() -> None:
    """generated_at should be populated automatically if not provided."""
    data = valid_signal_data()
    del data["generated_at"]
    signal = AlphaSignal(**data)
    assert signal.generated_at is not None
    assert signal.generated_at.tzinfo is not None


def test_multiple_supporting_chunks() -> None:
    """Multiple supporting chunks should be accepted."""
    signal = AlphaSignal(
        **valid_signal_data(supporting_chunks=["chunk A", "chunk B", "chunk C"])
    )
    assert len(signal.supporting_chunks) == 3


# ---------------------------------------------------------------------------
# AlphaSignal — invalid cases
# ---------------------------------------------------------------------------


def test_invalid_direction_rejected() -> None:
    """Non-Literal direction values must be rejected."""
    with pytest.raises(ValidationError):
        AlphaSignal(**valid_signal_data(direction="buy"))


def test_direction_case_sensitive() -> None:
    """Direction must be lowercase — 'Long' is not a valid Literal."""
    with pytest.raises(ValidationError):
        AlphaSignal(**valid_signal_data(direction="Long"))


def test_confidence_above_one_rejected() -> None:
    """Confidence > 1.0 must be rejected."""
    with pytest.raises(ValidationError):
        AlphaSignal(**valid_signal_data(confidence=1.01))


def test_confidence_below_zero_rejected() -> None:
    """Negative confidence must be rejected."""
    with pytest.raises(ValidationError):
        AlphaSignal(**valid_signal_data(confidence=-0.01))


def test_short_rationale_rejected() -> None:
    """Rationale shorter than 50 characters must be rejected."""
    with pytest.raises(ValidationError):
        AlphaSignal(**valid_signal_data(rationale="Too short."))


def test_empty_supporting_chunks_rejected() -> None:
    """An empty supporting_chunks list must be rejected."""
    with pytest.raises(ValidationError):
        AlphaSignal(**valid_signal_data(supporting_chunks=[]))


def test_empty_ticker_rejected() -> None:
    """Empty ticker string must be rejected (min_length=1)."""
    with pytest.raises(ValidationError):
        AlphaSignal(**valid_signal_data(ticker=""))


def test_ticker_too_long_rejected() -> None:
    """Ticker longer than 10 characters must be rejected."""
    with pytest.raises(ValidationError):
        AlphaSignal(**valid_signal_data(ticker="TOOLONGNAME"))


# ---------------------------------------------------------------------------
# BacktestResult — valid cases
# ---------------------------------------------------------------------------


def test_valid_backtest_parses() -> None:
    """A fully valid backtest dict should parse without errors."""
    result = BacktestResult(**valid_backtest_data())
    assert result.sharpe_ratio == 1.5
    assert result.max_drawdown == -0.23


def test_zero_max_drawdown_accepted() -> None:
    """Zero drawdown (flat equity curve) is a valid edge case."""
    BacktestResult(**valid_backtest_data(max_drawdown=0.0))


def test_benchmark_fields_optional() -> None:
    """benchmark_sharpe and benchmark_cagr should default to None."""
    result = BacktestResult(**valid_backtest_data())
    assert result.benchmark_sharpe is None
    assert result.benchmark_cagr is None


def test_benchmark_fields_set_when_provided() -> None:
    """Benchmark fields should be stored when provided."""
    result = BacktestResult(
        **valid_backtest_data(benchmark_sharpe=1.2, benchmark_cagr=0.15)
    )
    assert result.benchmark_sharpe == 1.2
    assert result.benchmark_cagr == 0.15


def test_zero_trades_accepted() -> None:
    """Zero trades (no signal fired) is a valid edge case for neutral signals."""
    BacktestResult(**valid_backtest_data(num_trades=0))


# ---------------------------------------------------------------------------
# BacktestResult — invalid cases
# ---------------------------------------------------------------------------


def test_positive_max_drawdown_rejected() -> None:
    """Positive max_drawdown violates the le=0.0 constraint."""
    with pytest.raises(ValidationError):
        BacktestResult(**valid_backtest_data(max_drawdown=0.01))


def test_negative_num_trades_rejected() -> None:
    """Negative num_trades must be rejected (ge=0 constraint)."""
    with pytest.raises(ValidationError):
        BacktestResult(**valid_backtest_data(num_trades=-1))
