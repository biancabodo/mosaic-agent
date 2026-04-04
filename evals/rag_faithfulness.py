"""RAGAS-style faithfulness evaluator — LLM-as-judge for RAG grounding."""

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langsmith import traceable
from langsmith.evaluation import EvaluationResult
from pydantic import BaseModel, Field

from config.settings import Settings

_FAITHFULNESS_PROMPT = """\
You are evaluating whether a financial analyst's rationale is faithfully
grounded in the provided source chunks from SEC filings.

RATIONALE:
{rationale}

SOURCE CHUNKS:
{chunks}

For each factual claim in the rationale, determine whether it is directly
supported by at least one of the source chunks.

Return:
- faithful_claims: number of claims directly supported by the source chunks
- total_claims: total number of factual claims in the rationale
- unsupported_claims: list of claims not found in the source chunks (max 3)
"""


class _FaithfulnessResponse(BaseModel):
    """Structured output for the faithfulness LLM judge."""

    faithful_claims: int = Field(ge=0, description="Claims supported by source chunks")
    total_claims: int = Field(ge=1, description="Total factual claims in rationale")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims not grounded in source chunks",
    )


@traceable(name="eval-rag-faithfulness")
def evaluate_rag_faithfulness(
    run: Any,
    example: Any,
) -> EvaluationResult:
    """Score whether the signal rationale is grounded in retrieved chunks.

    Uses an LLM judge (same model as the pipeline) to decompose the rationale
    into individual factual claims and check each against the supporting_chunks.
    The faithfulness score is faithful_claims / total_claims.

    This is the RAGAS faithfulness metric adapted for SEC filing signals:
    a score of 1.0 means every claim in the rationale appears in the retrieved
    chunks; 0.0 means none do.

    Args:
        run: LangSmith Run with outputs containing 'signal' (AlphaSignal dict).
        example: LangSmith Example (unused — pure output quality check).

    Returns:
        EvaluationResult with key 'rag_faithfulness' and score in [0.0, 1.0].
        Score is None if the LLM judge call fails (to avoid masking real errors).
    """
    import asyncio

    outputs = run.outputs or {}
    signal_data = outputs.get("signal", {})

    if not isinstance(signal_data, dict):
        return EvaluationResult(
            key="rag_faithfulness",
            score=None,
            comment="No signal data in run outputs",
        )

    rationale = signal_data.get("rationale", "")
    chunks = signal_data.get("supporting_chunks", [])

    if not rationale or not chunks:
        return EvaluationResult(
            key="rag_faithfulness",
            score=0.0,
            comment="Missing rationale or supporting_chunks",
        )

    chunks_text = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))

    settings = Settings()
    llm = ChatAnthropic(
        model=settings.anthropic_model,
        temperature=0.0,
        api_key=settings.anthropic_api_key,
    ).with_structured_output(_FaithfulnessResponse)

    prompt = _FAITHFULNESS_PROMPT.format(
        rationale=rationale,
        chunks=chunks_text,
    )

    try:
        response: _FaithfulnessResponse = asyncio.get_event_loop().run_until_complete(  # type: ignore[assignment]
            llm.ainvoke([HumanMessage(content=prompt)])
        )
    except Exception as exc:  # noqa: BLE001
        return EvaluationResult(
            key="rag_faithfulness",
            score=None,
            comment=f"LLM judge call failed: {exc}",
        )

    score = response.faithful_claims / response.total_claims
    comment = None
    if response.unsupported_claims:
        listed = "; ".join(response.unsupported_claims[:3])
        comment = f"Unsupported: {listed}"

    return EvaluationResult(
        key="rag_faithfulness",
        score=round(score, 3),
        comment=comment,
    )


@traceable(name="eval-chunk-relevance")
def evaluate_chunk_relevance(
    run: Any,
    example: Any,
) -> EvaluationResult:
    """Score average length and content quality of supporting chunks.

    A simple heuristic: chunks shorter than 100 characters are unlikely to
    contain substantive evidence. This catches cases where the retriever
    returns boilerplate headers or empty sections.

    Args:
        run: LangSmith Run with outputs containing 'signal' (AlphaSignal dict).
        example: LangSmith Example (unused).

    Returns:
        EvaluationResult with key 'chunk_relevance' and score in [0.0, 1.0].
    """
    outputs = run.outputs or {}
    signal_data = outputs.get("signal", {})
    is_dict = isinstance(signal_data, dict)
    chunks = signal_data.get("supporting_chunks", []) if is_dict else []

    if not chunks:
        return EvaluationResult(key="chunk_relevance", score=0.0)

    min_length = 100
    substantive = sum(1 for c in chunks if len(str(c).strip()) >= min_length)
    score = substantive / len(chunks)

    return EvaluationResult(
        key="chunk_relevance",
        score=round(score, 2),
        comment=f"{substantive}/{len(chunks)} chunks exceed {min_length} chars",
    )
