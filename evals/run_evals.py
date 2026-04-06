"""Offline evaluation harness — runs all evaluators against the golden dataset.

Usage:
    uv run python evals/run_evals.py              # all evaluators
    uv run python evals/run_evals.py --no-llm     # skip LLM-as-judge (rag_faithfulness)

Each evaluator is run against every example in GOLDEN_DATASET and results are
printed as a scorecard. Pass/fail is determined per-example against the
'expected' dict in each golden example (where provided).
"""

import argparse
import sys
from typing import Any

from evals.backtest_sanity import (
    evaluate_benchmark_comparison,
    evaluate_metric_ranges,
    evaluate_no_lookahead,
)
from evals.signal_quality import (
    GOLDEN_DATASET,
    evaluate_citation_presence,
    evaluate_confidence_calibration,
    evaluate_signal_structure,
    evaluate_supporting_evidence,
)


class _Run:
    """Minimal stand-in for a LangSmith Run object."""

    def __init__(self, outputs: dict[str, Any]) -> None:
        self.outputs = outputs


_SIGNAL_EVALUATORS: list[tuple[str, Any]] = [
    ("citation_presence", evaluate_citation_presence),
    ("signal_structure", evaluate_signal_structure),
    ("confidence_calibration", evaluate_confidence_calibration),
    ("supporting_evidence", evaluate_supporting_evidence),
]

_BACKTEST_EVALUATORS: list[tuple[str, Any]] = [
    ("no_lookahead", evaluate_no_lookahead),
    ("metric_ranges", evaluate_metric_ranges),
    ("benchmark_comparison", evaluate_benchmark_comparison),
]


def _run_signal_evals(include_llm: bool) -> None:
    print("\n── Signal quality evaluators ─────────────────────────────────────")
    print(f"{'Evaluator':<35} {'Pass':>6} {'Fail':>6} {'N/A':>6} {'Avg score':>10}")
    print("─" * 65)

    evaluators = list(_SIGNAL_EVALUATORS)
    if include_llm:
        from evals.rag_faithfulness import evaluate_rag_faithfulness

        evaluators.append(("rag_faithfulness", evaluate_rag_faithfulness))

    for name, evaluator in evaluators:
        passes = fails = skipped = 0
        scores: list[float] = []

        for example in GOLDEN_DATASET:
            run = _Run(outputs=example["outputs"])
            result = evaluator(run, example)

            if result.score is None:
                skipped += 1
                continue

            scores.append(result.score)
            expected = example.get("expected", {}).get(result.key)
            if expected is not None:
                if result.score == expected:
                    passes += 1
                else:
                    fails += 1
            else:
                passes += 1

        avg = sum(scores) / len(scores) if scores else float("nan")
        print(f"{name:<35} {passes:>6} {fails:>6} {skipped:>6} {avg:>10.3f}")


def _run_backtest_evals() -> None:
    print("\n── Backtest sanity evaluators ────────────────────────────────────")
    print("  (requires backtest_result in outputs — skipped for signal-only dataset)")

    for name, evaluator in _BACKTEST_EVALUATORS:
        n_applicable = 0
        for example in GOLDEN_DATASET:
            run = _Run(outputs=example["outputs"])
            result = evaluator(run, example)
            if result.score is not None:
                n_applicable += 1

        print(f"  {name}: {n_applicable} applicable examples in golden dataset")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AlphaSignal offline evaluations")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM-as-judge evaluators (rag_faithfulness) — no API key needed",
    )
    args = parser.parse_args()

    print("AlphaSignal evaluation harness")
    print(f"Golden dataset: {len(GOLDEN_DATASET)} examples")

    _run_signal_evals(include_llm=not args.no_llm)
    _run_backtest_evals()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
