"""AlphaSignal Pydantic schema — structured output for the signal extraction agent."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class AlphaSignal(BaseModel):
    """A structured alpha signal extracted from SEC filing analysis.

    All signals must include cited evidence from a specific filing section
    (e.g. 'Item 7 states...') in the rationale field.
    """

    ticker: str = Field(
        description="Stock ticker symbol, e.g. 'NVDA'",
        min_length=1,
        max_length=10,
    )
    direction: Literal["long", "short", "neutral"] = Field(
        description="Directional bias for the signal"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0",
    )
    rationale: str = Field(
        description=(
            "Explanation of the signal. Must cite a specific filing section, "
            "e.g. 'Item 7 states that revenue grew 122% YoY driven by...'"
        ),
        min_length=50,
    )
    supporting_chunks: list[str] = Field(
        description="Raw RAG chunks retrieved from the FAISS store used as evidence",
        min_length=1,
    )
    filing_period: str = Field(
        description="Filing type and period, e.g. '10-K FY2023' or '10-Q Q2 2024'"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the signal was generated",
    )
