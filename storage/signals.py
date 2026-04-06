"""SQLite persistence for AlphaSignal pipeline results."""

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from schemas.backtest_result import BacktestResult
from schemas.signal import AlphaSignal

log = logging.getLogger(__name__)

_DEFAULT_DB = Path(".mosaic.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    rationale       TEXT    NOT NULL,
    filing_period   TEXT    NOT NULL,
    generated_at    TEXT    NOT NULL,
    saved_at        TEXT    NOT NULL,
    sharpe_ratio    REAL,
    max_drawdown    REAL,
    cagr            REAL,
    total_return    REAL,
    benchmark_sharpe REAL,
    benchmark_cagr  REAL,
    backtest_start  TEXT,
    backtest_end    TEXT
)
"""


def _connect(db_path: Path = _DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def save(
    signal: AlphaSignal,
    backtest: BacktestResult | None = None,
    db_path: Path = _DEFAULT_DB,
) -> int:
    """Persist a signal (and optional backtest) to the local SQLite store.

    Returns the row id of the inserted record.
    """
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO signals (
                ticker, direction, confidence, rationale, filing_period,
                generated_at, saved_at,
                sharpe_ratio, max_drawdown, cagr, total_return,
                benchmark_sharpe, benchmark_cagr, backtest_start, backtest_end
            ) VALUES (
                :ticker, :direction, :confidence, :rationale, :filing_period,
                :generated_at, :saved_at,
                :sharpe_ratio, :max_drawdown, :cagr, :total_return,
                :benchmark_sharpe, :benchmark_cagr, :backtest_start, :backtest_end
            )
            """,
            {
                "ticker": signal.ticker,
                "direction": signal.direction,
                "confidence": signal.confidence,
                "rationale": signal.rationale,
                "filing_period": signal.filing_period,
                "generated_at": signal.generated_at.isoformat(),
                "saved_at": datetime.now(UTC).isoformat(),
                "sharpe_ratio": backtest.sharpe_ratio if backtest else None,
                "max_drawdown": backtest.max_drawdown if backtest else None,
                "cagr": backtest.cagr if backtest else None,
                "total_return": backtest.total_return if backtest else None,
                "benchmark_sharpe": backtest.benchmark_sharpe if backtest else None,
                "benchmark_cagr": backtest.benchmark_cagr if backtest else None,
                "backtest_start": backtest.start_date.isoformat() if backtest else None,
                "backtest_end": backtest.end_date.isoformat() if backtest else None,
            },
        )
        row_id: int = cursor.lastrowid or 0
        log.info("[%s] signal saved to %s (id=%d)", signal.ticker, db_path, row_id)
        return row_id


def history(
    ticker: str | None = None,
    limit: int = 50,
    db_path: Path = _DEFAULT_DB,
) -> list[sqlite3.Row]:
    """Return recent signals, optionally filtered by ticker, newest first."""
    with _connect(db_path) as conn:
        if ticker:
            return conn.execute(
                "SELECT * FROM signals WHERE ticker = ? ORDER BY saved_at DESC LIMIT ?",
                (ticker.upper(), limit),
            ).fetchall()
        return conn.execute(
            "SELECT * FROM signals ORDER BY saved_at DESC LIMIT ?", (limit,)
        ).fetchall()
