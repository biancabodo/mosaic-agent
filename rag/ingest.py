"""SEC EDGAR filing ingestion — async download, HTML stripping, SEC-aware chunking."""

import re
from dataclasses import dataclass
from typing import Any

import httpx
from bs4 import BeautifulSoup, Tag
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import Settings

EDGAR_BASE_URL = "https://data.sec.gov"
EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar"
TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"

# Separators tuned for SEC 10-K/10-Q structure: split on Item breaks first
SEC_SEPARATORS: list[str] = ["\n\nItem ", "\n\n", "\n"]


@dataclass(frozen=True)
class FilingMetadata:
    """Lightweight descriptor for a single SEC filing."""

    accession_number: str
    form_type: str
    filing_date: str
    cik: str
    primary_document: str  # filename of the primary document, e.g. 'nvda-20240128.htm'


async def fetch_cik(ticker: str, client: httpx.AsyncClient) -> str:
    """Resolve a stock ticker to its zero-padded 10-digit SEC CIK number.

    Args:
        ticker: Stock ticker symbol, e.g. 'NVDA'.
        client: Shared httpx async client (caller manages lifecycle).

    Returns:
        Zero-padded 10-digit CIK string, e.g. '0001045810'.

    Raises:
        ValueError: If the ticker is not found in SEC's company_tickers.json.
        httpx.HTTPStatusError: On non-2xx response from SEC.
    """
    response = await client.get(TICKER_CIK_URL)
    response.raise_for_status()
    data: dict[str, Any] = response.json()

    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"].upper() == ticker_upper:
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR company_tickers.json")


async def fetch_filing_metadata(
    cik: str,
    form_types: list[str],
    client: httpx.AsyncClient,
    max_filings: int = 5,
) -> list[FilingMetadata]:
    """Fetch recent filing metadata from the EDGAR submissions API.

    Args:
        cik: Zero-padded 10-digit CIK string.
        form_types: Filing types to include, e.g. ['10-K', '10-Q'].
        client: Shared httpx async client.
        max_filings: Maximum number of filings to return (most recent first).

    Returns:
        List of FilingMetadata ordered newest-first.

    Raises:
        httpx.HTTPStatusError: On non-2xx response from SEC.
    """
    url = f"{EDGAR_BASE_URL}/submissions/CIK{cik}.json"
    response = await client.get(url)
    response.raise_for_status()
    data: dict[str, Any] = response.json()

    recent = data.get("filings", {}).get("recent", {})
    accession_numbers: list[str] = recent.get("accessionNumber", [])
    forms: list[str] = recent.get("form", [])
    dates: list[str] = recent.get("filingDate", [])
    primary_docs: list[str] = recent.get("primaryDocument", [])

    results: list[FilingMetadata] = []
    for accession, form, date, primary_doc in zip(
        accession_numbers, forms, dates, primary_docs, strict=False
    ):
        if form in form_types:
            results.append(
                FilingMetadata(
                    accession_number=accession,
                    form_type=form,
                    filing_date=date,
                    cik=cik,
                    primary_document=primary_doc,
                )
            )
        if len(results) >= max_filings:
            break

    return results


async def fetch_filing_text(
    metadata: FilingMetadata,
    client: httpx.AsyncClient,
) -> str:
    """Download and parse the full text of an SEC filing.

    Uses the primary_document filename from the submissions API directly —
    no index fetch required. Downloads the HTML, strips markup, converts
    tables to pipe-delimited text, and returns clean plaintext.

    Args:
        metadata: Filing descriptor from fetch_filing_metadata, including
            the primary_document filename sourced from the submissions API.
        client: Shared httpx async client.

    Returns:
        Cleaned plaintext of the primary filing document.

    Raises:
        httpx.HTTPStatusError: On non-2xx response from SEC Archives.
    """
    accession_clean = metadata.accession_number.replace("-", "")
    cik_int = int(metadata.cik)

    doc_url = (
        f"{EDGAR_ARCHIVES_URL}/data/{cik_int}/"
        f"{accession_clean}/{metadata.primary_document}"
    )
    doc_response = await client.get(doc_url)
    doc_response.raise_for_status()

    return _extract_text(doc_response.text)


def _table_to_text(table_tag: Tag) -> str:
    """Convert an HTML table to pipe-delimited plaintext rows.

    Preserves the quantitative data (revenue, margins, EPS, debt levels)
    that financial tables contain — critical for the signal agent to produce
    evidence-backed, high-confidence directional calls.

    Empty cells are replaced with '-'. Rows with no meaningful content
    (all dashes) are dropped to reduce noise.

    Args:
        table_tag: A BeautifulSoup <table> element.

    Returns:
        Multi-line string of pipe-delimited rows, or empty string if the
        table contains no usable data.
    """
    rows: list[str] = []
    for tr in table_tag.find_all("tr"):
        cells = [
            re.sub(r"\s+", " ", td.get_text(separator=" ")).strip() or "-"
            for td in tr.find_all(["td", "th"])
        ]
        if any(c != "-" for c in cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _extract_text(html_content: str) -> str:
    """Strip HTML markup from SEC filing and normalise whitespace.

    Tables are converted to pipe-delimited plaintext rather than dropped —
    financial tables contain the quantitative evidence (revenue, EPS, margins)
    the signal agent needs. Script and style tags are removed entirely.

    Args:
        html_content: Raw HTML string of the filing document.

    Returns:
        Cleaned plaintext string with tables preserved as readable rows.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    for table in soup.find_all("table"):
        table_text = _table_to_text(table)
        table.replace_with(f"\n{table_text}\n" if table_text else "")

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r" {3,}", " ", text)
    return text.strip()


def chunk_filing(
    text: str, ticker: str, metadata: FilingMetadata
) -> list[dict[str, Any]]:
    """Split filing text into overlapping chunks with source metadata attached.

    Uses SEC-aware separators that split on Item boundaries first, then
    paragraph breaks, to keep semantically related content together.

    Args:
        text: Cleaned plaintext of a single filing.
        ticker: Stock ticker, attached to each chunk's metadata.
        metadata: Filing descriptor, attached to each chunk's metadata.

    Returns:
        List of dicts with keys 'content' (str) and 'metadata' (dict).
    """
    settings = Settings()
    splitter = RecursiveCharacterTextSplitter(
        separators=SEC_SEPARATORS,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    raw_chunks = splitter.split_text(text)
    return [
        {
            "content": chunk,
            "metadata": {
                "ticker": ticker,
                "form_type": metadata.form_type,
                "filing_date": metadata.filing_date,
                "accession_number": metadata.accession_number,
                "chunk_index": i,
            },
        }
        for i, chunk in enumerate(raw_chunks)
    ]


async def ingest_ticker(
    ticker: str,
    form_types: list[str] | None = None,
    max_filings: int = 3,
) -> list[dict[str, Any]]:
    """Download, parse, and chunk SEC filings for a ticker end-to-end.

    This is the main entry point for the ingest pipeline. Fetches the most
    recent filings matching form_types, parses each one, and returns all
    chunks merged into a single list ready for embedding and FAISS insertion.

    Args:
        ticker: Stock ticker symbol, e.g. 'NVDA'.
        form_types: Filing types to ingest. Defaults to ['10-K', '10-Q'].
        max_filings: Maximum number of filings to process per form type.

    Returns:
        Flat list of chunk dicts with 'content' and 'metadata' keys.

    Raises:
        ValueError: If the ticker is not found or no matching filings exist.
        httpx.HTTPStatusError: On SEC API failures.
    """
    if form_types is None:
        form_types = ["10-K", "10-Q"]

    settings = Settings()
    headers = {"User-Agent": settings.sec_user_agent}

    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        cik = await fetch_cik(ticker, client)
        filings = await fetch_filing_metadata(cik, form_types, client, max_filings)

        if not filings:
            raise ValueError(
                f"No {form_types} filings found for ticker '{ticker}' (CIK: {cik})"
            )

        all_chunks: list[dict[str, Any]] = []
        for filing_meta in filings:
            text = await fetch_filing_text(filing_meta, client)
            chunks = chunk_filing(text, ticker, filing_meta)
            all_chunks.extend(chunks)

        return all_chunks
