"""Tests for RAG ingestion, chunking, and HTML extraction logic."""

import pytest
from bs4 import BeautifulSoup

from rag.ingest import FilingMetadata, _extract_text, _table_to_text, chunk_filing


@pytest.fixture
def sample_metadata() -> FilingMetadata:
    """A realistic FilingMetadata instance for NVDA 10-K."""
    return FilingMetadata(
        accession_number="0001045810-24-000008",
        form_type="10-K",
        filing_date="2024-01-26",
        cik="0001045810",
        primary_document="nvda-20240128.htm",
    )


# ---------------------------------------------------------------------------
# chunk_filing
# ---------------------------------------------------------------------------


def test_chunk_filing_produces_chunks(sample_metadata: FilingMetadata) -> None:
    """Non-empty text should produce at least one chunk."""
    text = "\n\nItem 1\nBusiness section.\n\nItem 7\nMD&A with revenue details."
    chunks = chunk_filing(text, "NVDA", sample_metadata)
    assert len(chunks) >= 1


def test_chunk_filing_structure(sample_metadata: FilingMetadata) -> None:
    """Each chunk must have 'content' and 'metadata' keys."""
    chunks = chunk_filing("Item 1\nText.", "NVDA", sample_metadata)
    for chunk in chunks:
        assert "content" in chunk
        assert "metadata" in chunk
        assert isinstance(chunk["content"], str)
        assert isinstance(chunk["metadata"], dict)


def test_chunk_metadata_fields(sample_metadata: FilingMetadata) -> None:
    """Chunk metadata must include ticker, form_type, filing_date, and chunk_index."""
    chunks = chunk_filing("Item 1\nText.", "NVDA", sample_metadata)
    meta = chunks[0]["metadata"]
    assert meta["ticker"] == "NVDA"
    assert meta["form_type"] == "10-K"
    assert meta["filing_date"] == "2024-01-26"
    assert meta["accession_number"] == "0001045810-24-000008"
    assert meta["chunk_index"] == 0


def test_chunk_indices_are_sequential(sample_metadata: FilingMetadata) -> None:
    """Chunk indices should start at 0 and increment by 1."""
    long_text = "\n\nItem 1\n" + ("A" * 500) + "\n\nItem 7\n" + ("B" * 500)
    chunks = chunk_filing(long_text, "NVDA", sample_metadata)
    if len(chunks) > 1:
        indices = [c["metadata"]["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))


def test_chunk_filing_uses_ticker(sample_metadata: FilingMetadata) -> None:
    """Ticker passed to chunk_filing should appear in each chunk's metadata."""
    chunks = chunk_filing("Item 1\nContent.", "TSLA", sample_metadata)
    for chunk in chunks:
        assert chunk["metadata"]["ticker"] == "TSLA"


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


def test_extract_text_returns_plaintext() -> None:
    """Output should contain no HTML tags."""
    html = "<html><body><p>Revenue grew <b>122%</b> year over year.</p></body></html>"
    text = _extract_text(html)
    assert "<b>" not in text
    assert "<p>" not in text
    assert "Revenue grew" in text


def test_extract_text_removes_script_tags() -> None:
    """Script tag content must be stripped entirely."""
    html = "<html><body><script>alert('xss')</script><p>Filing text.</p></body></html>"
    text = _extract_text(html)
    assert "alert" not in text
    assert "Filing text" in text


def test_extract_text_removes_style_tags() -> None:
    """Style tag content must be stripped."""
    html = (
        "<html><head><style>body{color:red}</style></head>"
        "<body><p>Text.</p></body></html>"
    )
    text = _extract_text(html)
    assert "color" not in text
    assert "Text" in text


def test_extract_text_collapses_whitespace() -> None:
    """Excessive blank lines should be collapsed to at most 3 newlines."""
    html = "<html><body><p>A</p>\n\n\n\n\n\n<p>B</p></body></html>"
    text = _extract_text(html)
    assert "\n\n\n\n" not in text


def test_extract_text_converts_table_to_pipe_delimited() -> None:
    """Tables should be converted to pipe-delimited rows, not dropped."""
    html = """
    <html><body>
    <table>
        <tr><th>Metric</th><th>FY2024</th></tr>
        <tr><td>Revenue</td><td>$60.9B</td></tr>
    </table>
    </body></html>
    """
    text = _extract_text(html)
    assert "$60.9B" in text
    assert "|" in text


# ---------------------------------------------------------------------------
# _table_to_text
# ---------------------------------------------------------------------------


def test_table_to_text_preserves_numbers() -> None:
    """Revenue figures in table cells must appear in the output."""
    html = """
    <table>
        <tr><th>Metric</th><th>FY2024</th><th>FY2023</th></tr>
        <tr><td>Revenue</td><td>$60.9B</td><td>$26.9B</td></tr>
        <tr><td>Net Income</td><td>$29.8B</td><td>$4.4B</td></tr>
    </table>
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    text = _table_to_text(table)  # type: ignore[arg-type]
    assert "$60.9B" in text
    assert "$29.8B" in text
    assert "Revenue" in text


def test_table_to_text_uses_pipe_delimiter() -> None:
    """Columns must be separated by ' | '."""
    html = "<table><tr><td>A</td><td>B</td><td>C</td></tr></table>"
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    text = _table_to_text(table)  # type: ignore[arg-type]
    assert "|" in text


def test_table_to_text_handles_empty_cells() -> None:
    """Empty cells should be replaced with '-' not cause errors."""
    html = "<table><tr><td>Revenue</td><td></td><td>$60.9B</td></tr></table>"
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    text = _table_to_text(table)  # type: ignore[arg-type]
    assert "-" in text
    assert "$60.9B" in text


def test_table_to_text_returns_empty_for_empty_table() -> None:
    """A table with no rows should return an empty string."""
    soup = BeautifulSoup("<table></table>", "html.parser")
    table = soup.find("table")
    text = _table_to_text(table)  # type: ignore[arg-type]
    assert text == ""


def test_table_to_text_drops_all_dash_rows() -> None:
    """Rows where every cell is empty (all dashes) should be excluded."""
    html = (
        "<table>"
        "<tr><td></td><td></td></tr>"
        "<tr><td>Revenue</td><td>$1B</td></tr>"
        "</table>"
    )
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    text = _table_to_text(table)  # type: ignore[arg-type]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # Only the non-empty row should appear
    assert all("Revenue" in ln or "$1B" in ln for ln in lines)


# ---------------------------------------------------------------------------
# FilingMetadata
# ---------------------------------------------------------------------------


def test_filing_metadata_is_immutable() -> None:
    """FilingMetadata is frozen=True — mutation should raise AttributeError."""
    meta = FilingMetadata(
        accession_number="0001045810-24-000008",
        form_type="10-K",
        filing_date="2024-01-26",
        cik="0001045810",
        primary_document="nvda-20240128.htm",
    )
    with pytest.raises((AttributeError, TypeError)):
        meta.form_type = "10-Q"  # type: ignore[misc]
