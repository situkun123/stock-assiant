import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from backend.stock_fetcher import CompanyData

# ============================================================================
# CACHE
# ============================================================================
_company_cache = {}

def get_company_client(ticker: str) -> CompanyData:
    """Get or create a cached CompanyData instance."""
    ticker = ticker.upper()
    if ticker not in _company_cache:
        _company_cache[ticker] = CompanyData(ticker)
    return _company_cache[ticker]

def get_cached_companies():
    """Return list of currently cached company tickers."""
    return list(_company_cache.keys())

# ============================================================================
# TOOLS
# ============================================================================
@tool
def validate_stock_symbol(symbol: str) -> str:
    """
    Validate if a stock symbol is correct for Yahoo Finance and suggest corrections if needed use LLM.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""Validate the stock ticker symbol '{symbol}' for Yahoo Finance.

        Instructions:
        1. If '{symbol}' is a valid Yahoo Finance ticker, respond with: "VALID: {symbol}"
        2. If '{symbol}' is incorrect but you know the correct ticker, respond with: "CORRECTED: [correct_symbol] (was: {symbol})"
        3. If '{symbol}' looks like a company name instead of a ticker, respond with: "SYMBOL: [ticker] (for company: {symbol})"
        4. If you're not sure or it's not publicly traded, respond with: "UNKNOWN: {symbol}"

        Examples:
        - Input: "AAPL" → "VALID: AAPL"
        - Input: "APPL" → "CORRECTED: AAPL (was: APPL)"
        - Input: "Apple" → "SYMBOL: AAPL (for company: Apple)"
        - Input: "GOOG" → "CORRECTED: GOOGL (was: GOOG)" [Note: GOOGL is the primary ticker]
        - Input: "XYZ123" → "UNKNOWN: XYZ123"

        Respond with ONLY one of the formats above, nothing else."""

    response = llm.invoke(prompt)
    return response.content.strip()

@tool
def search_stock_symbol(company_name: str) -> str:
    """Search for a stock ticker symbol for a given company name."""
    try: 
        output = CompanyData.search_stock_symbol(company_name)
        return output['found']['symbol'][:3]  # Return the best match symbol, truncated to 3 for cost control
    except Exception:
        return "UNKNOWN"
@tool
def get_company_info(ticker: str):
    """Fetch key company metrics like P/E ratio, Market Cap, and business summary."""
    client = get_company_client(ticker)
    return client.get_info().to_string()

@tool
def get_stock_history(ticker: str, period: str = "1mo", interval: str = "1d"):
    """Fetch daily price history (OHLCV) for a given ticker and period.

    IMPORTANT: The period must be one of the following valid values:
    1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

    If the user requests a period that is NOT in the list above (e.g. '1w', '2w', '3m', 'week',
    'month', 'year'), you MUST call correct_period_parameter first to get the valid equivalent,
    then pass the corrected value here.
    """
    client = get_company_client(ticker)
    return client.get_ticker_data(period=period, interval=interval).tail(10).to_string()

@tool
def get_financial_statements(ticker: str):
    """Fetch the annual income statement and financial metrics."""
    client = get_company_client(ticker)
    return client.get_financials().to_string()

@tool
def correct_period_parameter(invalid_period: str) -> str:
    """
    Convert a user-supplied period string (e.g. '1w', '2w', '3m', 'week', 'month', 'year')
    into the nearest valid yfinance period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max).
    Call this BEFORE get_stock_history whenever the requested period is not already a valid value.
    Returns the corrected valid period string.
    """
    # Direct mapping for common cases
    period_mapping = {
        '1w': '5d',
        '2w': '1mo',
        '3w': '1mo',
        '4w': '1mo',
        'week': '5d',
        'month': '1mo',
        'year': '1y',
        '3m': '3mo',
        '6m': '6mo',
        '2y': '2y',
        '5y': '5y',
        '10y': '10y',
    }

    # check for direct mapping first
    period_lower = invalid_period.lower().strip()
    if period_lower in period_mapping:
        return period_mapping[period_lower]

    # LLM fallback for edge cases
    correction_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

    correction_prompt = f"""Given the invalid period '{invalid_period}'.
        Map it to the nearest valid option, ALWAYS choosing one larger than the invalid value to ensure we get enough data. Valid periods are: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        Always choose a period that is larger than the invalid input to ensure sufficient data is returned.
        Return ONLY the corrected period value, nothing else."""

    correction_response = correction_llm.invoke([
        SystemMessage(content="Map invalid period to nearest valid option."),
        HumanMessage(content=correction_prompt)
    ])

    corrected = correction_response.content.strip()

    # Validate the response is actually valid
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    if corrected in valid_periods:
        return corrected

    # Default fallback
    return '1mo'
