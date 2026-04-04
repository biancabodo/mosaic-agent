"""LangChain tool wrapping yfinance for historical price data retrieval."""

import yfinance as yf
from langchain_core.tools import tool


@tool
def get_price_history(ticker: str, period: str = "2y") -> str:
    """Fetch historical OHLCV price data for a ticker via yfinance.

    Returns a compact CSV-formatted string of daily close prices and volume,
    usable by the backtest agent or as supplementary context for the signal agent.

    Args:
        ticker: Stock ticker symbol, e.g. 'NVDA'.
        period: yfinance period string. Valid values: '1mo', '3mo', '6mo',
            '1y', '2y', '5y', '10y', 'ytd', 'max'. Defaults to '2y'.

    Returns:
        CSV string with columns: Date, Close, Volume.
        Returns an error message string if the download fails or returns no data.
    """
    try:
        hist = yf.Ticker(ticker).history(period=period)
    except Exception as exc:
        return f"Price data fetch failed for '{ticker}': {exc}"

    if hist.empty:
        return f"No price data returned for '{ticker}' over period '{period}'."

    hist = hist[["Close", "Volume"]].round(4)
    return hist.to_csv(date_format="%Y-%m-%d") or ""


@tool
def get_ticker_info(ticker: str) -> str:
    """Fetch key fundamental metadata for a ticker via yfinance.

    Returns a newline-separated list of key metrics (market cap, sector,
    trailing PE, forward PE, revenue, profit margins). Useful as supplementary
    context for the signal agent when SEC filing data is ambiguous.

    Args:
        ticker: Stock ticker symbol, e.g. 'NVDA'.

    Returns:
        Formatted string of key metrics, or an error message on failure.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as exc:
        return f"Ticker info fetch failed for '{ticker}': {exc}"

    fields = [
        ("Sector", "sector"),
        ("Industry", "industry"),
        ("Market Cap", "marketCap"),
        ("Trailing PE", "trailingPE"),
        ("Forward PE", "forwardPE"),
        ("Revenue (TTM)", "totalRevenue"),
        ("Profit Margin", "profitMargins"),
        ("52w High", "fiftyTwoWeekHigh"),
        ("52w Low", "fiftyTwoWeekLow"),
    ]

    lines: list[str] = [f"=== {ticker.upper()} ==="]
    for label, key in fields:
        value = info.get(key, "N/A")
        lines.append(f"{label}: {value}")

    return "\n".join(lines)
