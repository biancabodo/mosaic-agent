"""LangSmith evaluators for AlphaSignal structure, citation, and calibration."""

import re
from typing import Any

from langsmith import traceable
from langsmith.evaluation import EvaluationResult

from schemas.signal import AlphaSignal

# Regex that matches SEC Item citations: "Item 7", "Item 1A", "Item 1a", etc.
_ITEM_CITATION_RE = re.compile(r"\bItem\s+\d+[A-Za-z]?\b")

# Signals with these directions should never have confidence above this threshold
_NEUTRAL_CONFIDENCE_CAP = 0.65


# ---------------------------------------------------------------------------
# Core evaluator functions
# ---------------------------------------------------------------------------


@traceable(name="eval-citation-presence")
def evaluate_citation_presence(
    run: Any,  # langsmith Run object
    example: Any,  # langsmith Example object
) -> EvaluationResult:
    """Check whether the signal rationale cites a specific SEC filing section.

    A valid citation matches the pattern 'Item N' or 'Item NA' (e.g. 'Item 7',
    'Item 1A'). Signals without citations are not grounded in filing evidence
    and should not be trusted for trading decisions.

    Args:
        run: LangSmith Run with outputs containing 'signal' (AlphaSignal dict).
        example: LangSmith Example (unused here, pure structural check).

    Returns:
        EvaluationResult with score 1.0 (citation found) or 0.0 (missing).
    """
    outputs = run.outputs or {}
    signal_data = outputs.get("signal", {})
    rationale = (
        signal_data.get("rationale", "") if isinstance(signal_data, dict) else ""
    )

    has_citation = bool(_ITEM_CITATION_RE.search(rationale))
    return EvaluationResult(
        key="citation_present",
        score=1.0 if has_citation else 0.0,
        comment=None if has_citation else "Rationale missing Item N citation",
    )


@traceable(name="eval-signal-structure")
def evaluate_signal_structure(
    run: Any,
    example: Any,
) -> EvaluationResult:
    """Validate that the signal conforms to the AlphaSignal schema.

    Attempts to parse the output as an AlphaSignal Pydantic model. Catches
    validation errors that indicate the LLM produced a malformed response
    (e.g. confidence out of range, missing required fields).

    Args:
        run: LangSmith Run with outputs containing 'signal' (AlphaSignal dict).
        example: LangSmith Example (unused).

    Returns:
        EvaluationResult with score 1.0 (valid) or 0.0 (invalid), with
        a comment describing the first validation error.
    """
    from pydantic import ValidationError

    outputs = run.outputs or {}
    signal_data = outputs.get("signal", {})

    try:
        AlphaSignal.model_validate(signal_data)
        return EvaluationResult(key="structure_valid", score=1.0)
    except ValidationError as exc:
        first_error = exc.errors()[0]
        return EvaluationResult(
            key="structure_valid",
            score=0.0,
            comment=f"{first_error['loc']}: {first_error['msg']}",
        )


@traceable(name="eval-confidence-calibration")
def evaluate_confidence_calibration(
    run: Any,
    example: Any,
) -> EvaluationResult:
    """Flag signals where confidence appears miscalibrated.

    Two calibration rules:
    1. Neutral signals with confidence > 0.65 are contradictory — if the
       evidence is genuinely neutral, confidence should be low.
    2. Any signal with confidence > 0.95 is suspicious — that level of
       certainty is rarely warranted from SEC filings alone.

    Args:
        run: LangSmith Run with outputs containing 'signal' (AlphaSignal dict).
        example: LangSmith Example (unused).

    Returns:
        EvaluationResult with score 1.0 (calibrated) or 0.0 (flagged).
    """
    outputs = run.outputs or {}
    signal_data = outputs.get("signal", {})
    if not isinstance(signal_data, dict):
        return EvaluationResult(key="confidence_calibrated", score=0.0)

    direction = signal_data.get("direction", "")
    confidence = float(signal_data.get("confidence", 0.0))

    if direction == "neutral" and confidence > _NEUTRAL_CONFIDENCE_CAP:
        return EvaluationResult(
            key="confidence_calibrated",
            score=0.0,
            comment=(
                f"Neutral signal has confidence {confidence:.2f} > "
                f"{_NEUTRAL_CONFIDENCE_CAP} — contradictory"
            ),
        )

    if confidence > 0.95:
        return EvaluationResult(
            key="confidence_calibrated",
            score=0.0,
            comment=f"Confidence {confidence:.2f} > 0.95 — suspiciously high",
        )

    return EvaluationResult(key="confidence_calibrated", score=1.0)


@traceable(name="eval-supporting-evidence")
def evaluate_supporting_evidence(
    run: Any,
    example: Any,
) -> EvaluationResult:
    """Check that supporting_chunks are non-empty and contain substantive text.

    Args:
        run: LangSmith Run with outputs containing 'signal' (AlphaSignal dict).
        example: LangSmith Example (unused).

    Returns:
        EvaluationResult scoring the proportion of non-trivial chunks (>50 chars).
    """
    outputs = run.outputs or {}
    signal_data = outputs.get("signal", {})
    chunks = (
        signal_data.get("supporting_chunks", [])
        if isinstance(signal_data, dict)
        else []
    )

    if not chunks:
        return EvaluationResult(
            key="supporting_evidence",
            score=0.0,
            comment="No supporting chunks provided",
        )

    substantive = sum(1 for c in chunks if len(str(c).strip()) > 50)
    score = substantive / len(chunks)
    return EvaluationResult(key="supporting_evidence", score=round(score, 2))


# ---------------------------------------------------------------------------
# Golden dataset — 25 labelled examples for offline evaluation
# ---------------------------------------------------------------------------

GOLDEN_DATASET: list[dict[str, Any]] = [
    # ── Strong valid long signals ─────────────────────────────────────────
    {
        "inputs": {"ticker": "NVDA", "direction": "long"},
        "outputs": {
            "signal": {
                "ticker": "NVDA",
                "direction": "long",
                "confidence": 0.82,
                "rationale": "Item 7 states total revenue grew 122% YoY to $60.9B, driven by data center segment which grew 217%.",
                "supporting_chunks": [
                    "Item 7: Data center revenue was $47.5B, up 217% from prior year."
                ],
                "filing_period": "10-K FY2024",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    {
        "inputs": {"ticker": "NVDA", "direction": "long"},
        "outputs": {
            "signal": {
                "ticker": "NVDA",
                "direction": "long",
                "confidence": 0.75,
                "rationale": "Item 1 describes the company as the dominant provider of AI training GPUs. Item 7 shows gross margins expanding to 73%.",
                "supporting_chunks": [
                    "Item 1: NVIDIA designs GPUs used in over 90% of AI model training workloads."
                ],
                "filing_period": "10-K FY2024",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    {
        "inputs": {"ticker": "MSFT", "direction": "long"},
        "outputs": {
            "signal": {
                "ticker": "MSFT",
                "direction": "long",
                "confidence": 0.78,
                "rationale": "Item 7 reports Intelligent Cloud revenue grew 21% to $87.9B, with Azure growing 28% YoY.",
                "supporting_chunks": [
                    "Item 7 MD&A: Azure and other cloud services revenue grew 28%."
                ],
                "filing_period": "10-K FY2024",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    {
        "inputs": {"ticker": "META", "direction": "long"},
        "outputs": {
            "signal": {
                "ticker": "META",
                "direction": "long",
                "confidence": 0.69,
                "rationale": "Item 7 shows advertising revenue rebounded to $131.9B (+16% YoY) after the 2022 decline. Item 1 describes the Year of Efficiency restructuring.",
                "supporting_chunks": [
                    "Item 7: Total revenue was $134.9B, with advertising revenue of $131.9B."
                ],
                "filing_period": "10-K FY2023",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    {
        "inputs": {"ticker": "AMZN", "direction": "long"},
        "outputs": {
            "signal": {
                "ticker": "AMZN",
                "direction": "long",
                "confidence": 0.72,
                "rationale": "Item 7 reports AWS operating income of $24.6B (29.6% margin), growing 61% YoY. Item 1 identifies AWS as the primary profit driver.",
                "supporting_chunks": [
                    "Item 7: AWS segment operating income was $24.6B for fiscal 2023."
                ],
                "filing_period": "10-K FY2023",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    # ── Valid short signals ───────────────────────────────────────────────
    {
        "inputs": {"ticker": "TSLA", "direction": "short"},
        "outputs": {
            "signal": {
                "ticker": "TSLA",
                "direction": "short",
                "confidence": 0.71,
                "rationale": "Item 1A identifies intensifying competition from BYD and legacy OEMs as a key risk. Item 7 shows automotive gross margin declining to 17.6% from 25.6%.",
                "supporting_chunks": [
                    "Item 1A: Competition from BYD, GM, Ford and other EV manufacturers is intensifying."
                ],
                "filing_period": "10-K FY2023",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    {
        "inputs": {"ticker": "INTC", "direction": "short"},
        "outputs": {
            "signal": {
                "ticker": "INTC",
                "direction": "short",
                "confidence": 0.68,
                "rationale": "Item 1A states Intel faces severe competition from AMD and TSMC-manufactured chips. Item 7 reports client computing revenue declined 8% YoY.",
                "supporting_chunks": [
                    "Item 7: Client Computing Group revenue was $25.5B, down 8% year-over-year."
                ],
                "filing_period": "10-K FY2023",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    {
        "inputs": {"ticker": "AMD", "direction": "long"},
        "outputs": {
            "signal": {
                "ticker": "AMD",
                "direction": "long",
                "confidence": 0.65,
                "rationale": "Item 7 shows Data Center segment revenue grew 107% YoY driven by MI300X GPU adoption. Item 1 describes expanding hyperscaler customer base.",
                "supporting_chunks": [
                    "Item 7: Data Center segment revenue of $6.5B grew 107% year-over-year."
                ],
                "filing_period": "10-K FY2023",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    # ── Valid neutral signals ─────────────────────────────────────────────
    {
        "inputs": {"ticker": "AAPL", "direction": "neutral"},
        "outputs": {
            "signal": {
                "ticker": "AAPL",
                "direction": "neutral",
                "confidence": 0.45,
                "rationale": "Item 7 shows iPhone revenue declined 2.8% YoY but Services grew 16%. Item 1A notes ongoing China market headwinds. Mixed evidence prevents a directional view.",
                "supporting_chunks": [
                    "Item 7: iPhone net sales were $200.6B, down 2.8% versus prior year."
                ],
                "filing_period": "10-K FY2023",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    {
        "inputs": {"ticker": "GOOGL", "direction": "neutral"},
        "outputs": {
            "signal": {
                "ticker": "GOOGL",
                "direction": "neutral",
                "confidence": 0.38,
                "rationale": "Item 1A identifies AI competition from Microsoft/OpenAI as a significant risk to search revenue. Item 7 shows Google Cloud growing 28% but advertising growth slowing to 6%.",
                "supporting_chunks": [
                    "Item 1A: Our business is subject to competition from AI-powered search alternatives."
                ],
                "filing_period": "10-K FY2023",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    # ── Signals missing citations (should fail citation check) ────────────
    {
        "inputs": {"ticker": "NVDA", "direction": "long"},
        "outputs": {
            "signal": {
                "ticker": "NVDA",
                "direction": "long",
                "confidence": 0.80,
                "rationale": "Revenue grew significantly and the company is well-positioned for AI demand. Strong execution by management team expected to continue.",
                "supporting_chunks": [
                    "Revenue grew significantly year over year driven by AI demand."
                ],
                "filing_period": "10-K FY2024",
            }
        },
        "expected": {
            "citation_present": 0.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    {
        "inputs": {"ticker": "TSLA", "direction": "short"},
        "outputs": {
            "signal": {
                "ticker": "TSLA",
                "direction": "short",
                "confidence": 0.72,
                "rationale": "Competition is increasing and margins are under pressure. The business faces multiple headwinds in the current environment.",
                "supporting_chunks": [
                    "Automotive margins declined significantly due to price cuts."
                ],
                "filing_period": "10-K FY2023",
            }
        },
        "expected": {
            "citation_present": 0.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    {
        "inputs": {"ticker": "META", "direction": "long"},
        "outputs": {
            "signal": {
                "ticker": "META",
                "direction": "long",
                "confidence": 0.75,
                "rationale": "Advertising revenue is recovering strongly and the efficiency program has improved margins substantially versus the prior year period.",
                "supporting_chunks": [
                    "Advertising revenue grew 16% year over year as the ad market recovered."
                ],
                "filing_period": "10-K FY2023",
            }
        },
        "expected": {
            "citation_present": 0.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    # ── Overconfident signals ─────────────────────────────────────────────
    {
        "inputs": {"ticker": "NVDA", "direction": "long"},
        "outputs": {
            "signal": {
                "ticker": "NVDA",
                "direction": "long",
                "confidence": 0.98,
                "rationale": "Item 7 shows extraordinary revenue growth of 122%. This is a near-certain long signal.",
                "supporting_chunks": ["Item 7: Revenue grew 122% year over year."],
                "filing_period": "10-K FY2024",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 0.0,
        },
    },
    {
        "inputs": {"ticker": "AAPL", "direction": "long"},
        "outputs": {
            "signal": {
                "ticker": "AAPL",
                "direction": "long",
                "confidence": 0.97,
                "rationale": "Item 7 reports consistent Services growth. Apple's ecosystem is unassailable.",
                "supporting_chunks": ["Item 7: Services revenue grew 16% to $85.2B."],
                "filing_period": "10-K FY2023",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 0.0,
        },
    },
    # ── Neutral + high confidence (miscalibrated) ─────────────────────────
    {
        "inputs": {"ticker": "NVDA", "direction": "neutral"},
        "outputs": {
            "signal": {
                "ticker": "NVDA",
                "direction": "neutral",
                "confidence": 0.90,
                "rationale": "Item 7 shows mixed signals. Revenue is up but risks are also rising. Item 1A notes geopolitical exposure.",
                "supporting_chunks": [
                    "Item 7: Revenue grew but competition is intensifying."
                ],
                "filing_period": "10-K FY2024",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 0.0,
        },
    },
    {
        "inputs": {"ticker": "MSFT", "direction": "neutral"},
        "outputs": {
            "signal": {
                "ticker": "MSFT",
                "direction": "neutral",
                "confidence": 0.85,
                "rationale": "Item 7 shows both positive cloud growth and elevated capex. Item 1A identifies AI regulation risk.",
                "supporting_chunks": [
                    "Item 7: Capital expenditure increased 79% to $55.7B."
                ],
                "filing_period": "10-K FY2024",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 0.0,
        },
    },
    # ── Additional valid signals ──────────────────────────────────────────
    {
        "inputs": {"ticker": "JPM", "direction": "long"},
        "outputs": {
            "signal": {
                "ticker": "JPM",
                "direction": "long",
                "confidence": 0.71,
                "rationale": "Item 7 shows net interest income grew 34% YoY to $89.3B benefiting from higher rates. Item 1 describes diversified revenue across investment banking and retail.",
                "supporting_chunks": [
                    "Item 7: Net interest income was $89.3B, up 34% due to higher interest rates."
                ],
                "filing_period": "10-K FY2023",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    {
        "inputs": {"ticker": "GS", "direction": "short"},
        "outputs": {
            "signal": {
                "ticker": "GS",
                "direction": "short",
                "confidence": 0.66,
                "rationale": "Item 1A identifies M&A advisory revenue risk from declining deal activity. Item 7 shows investment banking revenue declined 20% YoY to $6.1B.",
                "supporting_chunks": [
                    "Item 7: Investment Banking net revenues decreased 20% to $6.1B."
                ],
                "filing_period": "10-K FY2023",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    {
        "inputs": {"ticker": "NVDA", "direction": "long"},
        "outputs": {
            "signal": {
                "ticker": "NVDA",
                "direction": "long",
                "confidence": 0.60,
                "rationale": "Item 7 shows data center revenue surpassing gaming for the first time. Item 1 positions NVIDIA in the AI infrastructure stack.",
                "supporting_chunks": [
                    "Item 7: Data Center revenue of $47.5B exceeded Gaming revenue of $10.4B for the first time."
                ],
                "filing_period": "10-K FY2024",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    {
        "inputs": {"ticker": "AMD", "direction": "short"},
        "outputs": {
            "signal": {
                "ticker": "AMD",
                "direction": "short",
                "confidence": 0.59,
                "rationale": "Item 1A notes AMD faces intense competition from NVIDIA in the AI accelerator market. Item 7 shows gaming segment revenue declined 9% YoY.",
                "supporting_chunks": [
                    "Item 1A: We face substantial competition from NVIDIA in the data center GPU market."
                ],
                "filing_period": "10-K FY2023",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    {
        "inputs": {"ticker": "MSFT", "direction": "long"},
        "outputs": {
            "signal": {
                "ticker": "MSFT",
                "direction": "long",
                "confidence": 0.80,
                "rationale": "Item 7 reports Microsoft Cloud revenue exceeded $135B, growing 23% YoY. Item 1 describes Copilot as an AI monetisation layer across all product lines.",
                "supporting_chunks": [
                    "Item 7: Microsoft Cloud revenue was $135.0B and grew 23%."
                ],
                "filing_period": "10-K FY2024",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    {
        "inputs": {"ticker": "AMZN", "direction": "long"},
        "outputs": {
            "signal": {
                "ticker": "AMZN",
                "direction": "long",
                "confidence": 0.73,
                "rationale": "Item 7 shows AWS revenue of $90.8B growing 17% YoY with improving margins. Item 1 describes the strategic focus on AI infrastructure and Bedrock.",
                "supporting_chunks": [
                    "Item 7: AWS segment net sales were $90.8B, up 17% year-over-year."
                ],
                "filing_period": "10-K FY2023",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    {
        "inputs": {"ticker": "TSLA", "direction": "neutral"},
        "outputs": {
            "signal": {
                "ticker": "TSLA",
                "direction": "neutral",
                "confidence": 0.42,
                "rationale": "Item 7 shows revenue growth of 19% but automotive gross margin compressed to 17.6% from 25.6%. Item 1A notes significant regulatory and competition risks.",
                "supporting_chunks": [
                    "Item 7: Automotive gross margin was 17.6%, down from 25.6% in the prior year."
                ],
                "filing_period": "10-K FY2023",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
    {
        "inputs": {"ticker": "NVDA", "direction": "long"},
        "outputs": {
            "signal": {
                "ticker": "NVDA",
                "direction": "long",
                "confidence": 0.77,
                "rationale": "Item 7 and Item 1 together paint a picture of dominant market position: 80%+ GPU market share in AI training per Item 1, and 122% revenue growth per Item 7.",
                "supporting_chunks": [
                    "Item 1: NVIDIA's H100 GPU is used in the majority of AI training workloads.",
                    "Item 7: Total revenue was $60.9B for fiscal 2024, up 122% year-over-year.",
                ],
                "filing_period": "10-K FY2024",
            }
        },
        "expected": {
            "citation_present": 1.0,
            "structure_valid": 1.0,
            "confidence_calibrated": 1.0,
        },
    },
]


# ---------------------------------------------------------------------------
# LangSmith dataset creation utility
# ---------------------------------------------------------------------------


def create_langsmith_dataset(dataset_name: str = "alphasignal-golden-set") -> str:
    """Create or update the golden evaluation dataset in LangSmith.

    Uploads all GOLDEN_DATASET examples to LangSmith. If a dataset with the
    same name already exists, it is reused (examples are appended).

    Requires LANGCHAIN_API_KEY to be set in the environment.

    Args:
        dataset_name: Name for the LangSmith dataset.

    Returns:
        The dataset ID string.
    """
    from langsmith import Client

    client = Client()

    existing = {d.name: d for d in client.list_datasets()}
    if dataset_name in existing:
        dataset = existing[dataset_name]
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Golden evaluation set for AlphaSignal quality checks (25 examples)",
        )

    client.create_examples(
        inputs=[ex["inputs"] for ex in GOLDEN_DATASET],
        outputs=[ex["outputs"] for ex in GOLDEN_DATASET],
        dataset_id=dataset.id,
    )

    return str(dataset.id)
