from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from backend.stock_fetcher import CompanyData
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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
def get_stock_symbol(company_name: str) -> str:
    """Convert a company name to its stock ticker symbol using LLM knowledge. The stock ticker symbol (e.g., 'AAPL', 'MSFT')"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""What is the stock ticker symbol for {company_name}? 
            Return ONLY the ticker symbol, nothing else. 
            If you're not sure or the company is not publicly traded, return 'UNKNOWN'."""
    
    response = llm.invoke(prompt)
    return response.content.strip()

@tool
def get_company_info(ticker: str):
    """Fetch key company metrics like P/E ratio, Market Cap, and business summary."""
    client = get_company_client(ticker)
    return client.get_info().to_string()

@tool
def get_stock_history(ticker: str, period: str = "1mo", interval: str = "1d"):
    """Fetch daily price history (OHLCV) for a given ticker and period."""
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
    Correct invalid yfinance period parameters to the nearest valid option.
    The corrected valid period parameter
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
    
    correction_prompt = f"""Given the invalid period '{invalid_period}', map it to the nearest valid option.
        Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
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