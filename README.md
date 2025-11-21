# Simple-stock-analyzer
Let users analyze a stock by ticker and date range.


 from google.colab import drive
drive.mount('/content/drive')

pip install pandas numpy yfinance

# --- Core Libraries for Stock Analyzer Backend ---

import yfinance as yf      # Fetch historical stock data and fundamentals from Yahoo Finance
import pandas as pd
import numpy as np
import requests            # For making API requests (used to look up tickers by company name)
from datetime import datetime, timedelta   # Work with dates and time ranges (e.g., last 1 year, custom ranges)

# ---------- Indicator Calculations ----------

def calculate_rsi(data, period: int = 14):
    """
    Calculate the Relative Strength Index (RSI).

    Parameters:
        data (pd.DataFrame): Stock data containing at least a 'Close' column.
        period (int): Lookback period for RSI calculation (default = 14).

    Returns:
        pd.Series: RSI values for the given period.
    """
    # 1. Calculate day-to-day price changes (positive = gain, negative = loss)
    delta = data['Close'].diff()

    # 2. Separate positive and negative changes
    gain = delta.clip(lower=0)            # keep only gains, set losses to 0
    loss = -delta.clip(upper=0)           # keep only losses (as positive values), set gains to 0

    # 3. Calculate average gains and losses over the lookback period
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # 4. Compute Relative Strength (RS)
    rs = avg_gain / avg_loss

    # 5. Convert RS to RSI using the standard formula
    rsi = 100 - (100 / (1 + rs))

    return rsi

    def calculate_macd(data):
    """
    Calculate the Moving Average Convergence Divergence (MACD) indicator.

    Parameters:
        data (pd.DataFrame): Stock data containing at least a 'Close' column.

    Returns:
        tuple: (macd, signal, hist)
            - macd: Difference between 12-day EMA and 26-day EMA
            - signal: 9-day EMA of the MACD line (signal line)
            - hist: Difference between MACD and Signal (histogram)
    """

    # 1. Calculate the 12-day Exponential Moving Average (short-term trend)
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()

    # 2. Calculate the 26-day Exponential Moving Average (long-term trend)
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()

    # 3. MACD Line = 12-day EMA - 26-day EMA
    macd = exp1 - exp2

    # 4. Signal Line = 9-day EMA of the MACD line
    signal = macd.ewm(span=9, adjust=False).mean()

    # 5. Histogram = MACD - Signal (shows momentum strength)
    hist = macd - signal

    return macd, signal, hist

def calculate_bollinger_bands(data, period: int = 20):
    """
    Calculate Bollinger Bands for a stock.

    Parameters:
        data (pd.DataFrame): Stock data containing at least a 'Close' column.
        period (int): Lookback period for the moving average (default = 20).

    Returns:
        tuple: (upper, lower)
            - upper: Upper Bollinger Band (SMA + 2*STD)
            - lower: Lower Bollinger Band (SMA - 2*STD)
    """

    # 1. Calculate the Simple Moving Average (SMA) of closing prices
    sma = data['Close'].rolling(window=period).mean()

    # 2. Calculate the standard deviation of closing prices over the same period
    std = data['Close'].rolling(window=period).std()

    # 3. Upper Band = SMA + (2 × standard deviation)
    upper = sma + (2 * std)

    # 4. Lower Band = SMA - (2 × standard deviation)
    lower = sma - (2 * std)

    return upper, lower

# ---------- Format Helpers ----------

def format_number(value):
    """
    Format a numeric value as a dollar currency string.

    Parameters:
        value (int | float | any): The number to format.

    Returns:
        str: Formatted string like "$1,234" or "N/A" if not numeric.
    """
    if isinstance(value, (int, float)):
        # Format with commas as thousands separators, no decimal places
        return f"${value:,.0f}"
    return "N/A"   # Return fallback if value is not a number


def format_percent(value):
    """
    Format a numeric value as a percentage string.

    Parameters:
        value (int | float | any): The number to format (e.g., 0.1234 for 12.34%).

    Returns:
        str: Formatted string like "12.34%" or "N/A" if not numeric.
    """
    if isinstance(value, (int, float)):
        # Multiply by 100 to convert ratio -> percent, round to 2 decimals
        return f"{value * 100:.2f}%"
    return "N/A"   # Return fallback if value is not a number

    # ---------- Ticker Lookup ----------

def get_ticker_from_company(company_name):
    """
    Try to find a stock ticker symbol from a company name using Yahoo Finance's search API.

    Parameters:
        company_name (str): The company name entered by the user (e.g., "Microsoft").

    Returns:
        str | None: The first matching ticker symbol (e.g., "MSFT"),
                    or None if no match is found.
    """

    # Try multiple variants of the company name to improve search accuracy
    for name_variant in [company_name, company_name + " Inc.", company_name + " Corp."]:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        user_agent = "Mozilla/5.0"  # Fake user-agent to avoid request blocking
        params = {
            "q": name_variant,     # Search query (company name variant)
            "quotes_count": 1,     # Return only the top result
            "country": "US"        # Restrict search to US companies
        }
        try:
            # Send GET request to Yahoo Finance's search API
            response = requests.get(url, params=params, headers={'User-Agent': user_agent})
            data = response.json()

            # If a valid result is returned, extract the ticker symbol
            if 'quotes' in data and len(data['quotes']) > 0:
                return data['quotes'][0]['symbol']
        except Exception:
            # Fail silently and try the next variant if request fails
            pass

    # If no ticker is found after trying all variants
    return None

    # ---------- Core Stock Analyzer Backend ----------

class StockAnalyzer:
    """
    A backend class to fetch stock data, calculate technical indicators,
    and retrieve key company fundamentals using Yahoo Finance.
    """

    def __init__(self, ticker: str, start=None, end=None):
        """
        Initialize StockAnalyzer with ticker and date range.

        Parameters:
            ticker (str): Stock ticker symbol (e.g., "MSFT").
            start (datetime): Start date for fetching data (default = 1 year ago).
            end (datetime): End date for fetching data (default = today).
        """
        self.ticker = ticker.upper()
        self.start = start or (datetime.now() - timedelta(days=365))  # Default: last 1 year
        self.end = end or datetime.now()
        self.data = None   # Will hold historical stock price data
        self.info = {}     # Will hold company fundamentals

    def fetch_data(self):
        """
        Fetch historical price data and fundamentals for the stock.

        Returns:
            pd.DataFrame: Historical OHLCV stock data.
        """
        stock = yf.Ticker(self.ticker)

        # Make end date inclusive by adding +1 day
        end_inclusive = self.end + timedelta(days=1)
        self.data = stock.history(start=self.start, end=end_inclusive)

        # Try fetching company info (can sometimes fail due to API limits)
        try:
            self.info = stock.info
        except Exception:
            self.info = {}

        return self.data

    def add_indicators(self):
        """
        Add technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands) to stock data.

        Returns:
            pd.DataFrame: DataFrame with added technical indicator columns.
        """
        if self.data is None or self.data.empty:
            return None

        df = self.data.copy()

        # Moving Averages
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

        # RSI and its 9-period signal line
        df['RSI'] = calculate_rsi(df)
        df['RSI_Signal'] = df['RSI'].ewm(span=9, adjust=False).mean()

        # MACD (line, signal, histogram)
        df['MACD'], df['Signal'], df['Hist'] = calculate_macd(df)

        # Bollinger Bands (upper & lower)
        df['UpperBand'], df['LowerBand'] = calculate_bollinger_bands(df)

        self.data = df
        return self.data

    def get_summary(self):
        """
        Generate a summary of stock performance (last close, returns, etc.).

        Returns:
            dict: Summary stats including last price, return %, and period counts.
        """
        if self.data is None or self.data.empty:
            return {}

        close = self.data["Close"]
        summary = {}

        # Basic price performance
        summary["last_close"] = float(close.iloc[-1])
        summary["first_close"] = float(close.iloc[0])
        summary["periods"] = len(close)
        summary["return_pct"] = ((close.iloc[-1] / close.iloc[0]) - 1) * 100 if close.iloc[0] != 0 else np.nan

        # Quick performance snapshots
        summary["1M"] = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else np.nan
        summary["3M"] = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) > 63 else np.nan
        summary["1Y"] = (close.iloc[-1] / close.iloc[-252] - 1) * 100 if len(close) > 252 else np.nan

        return summary

    def fundamentals(self):
        """
        Get structured company fundamentals and key ratios.

        Returns:
            dict: Dictionary of company financial and valuation metrics.
        """
        return {
            "name": self.info.get("longName"),
            "industry": self.info.get("industry"),
            "sector": self.info.get("sector"),
            "country": self.info.get("country"),

            # Market and valuation metrics
            "market_cap": format_number(self.info.get("marketCap")),
            "eps": self.info.get("trailingEps"),
            "dividend_yield": format_percent(self.info.get("dividendYield")),
            "beta": self.info.get("beta"),
            "52w_range": (
                self.info.get("fiftyTwoWeekLow"),
                self.info.get("fiftyTwoWeekHigh"),
            ),
            "avg_volume": format_number(self.info.get("averageVolume")),

            # Earnings and profitability
            "revenue": format_number(self.info.get("totalRevenue")),
            "net_income": format_number(self.info.get("netIncomeToCommon")),
            "profit_margin": format_percent(self.info.get("profitMargins")),
            "free_cashflow": format_number(self.info.get("freeCashflow")),

            # Valuation ratios
            "trailing_pe": self.info.get("trailingPE"),
            "forward_pe": self.info.get("forwardPE"),
            "peg_ratio": self.info.get("pegRatio"),
            "price_to_sales": self.info.get("priceToSalesTrailing12Months"),

            # Balance sheet
            "assets": format_number(self.info.get("totalAssets")),
            "liabilities": format_number(self.info.get("totalLiab")),
            "de_ratio": self.info.get("debtToEquity"),
            "roe": format_percent(self.info.get("returnOnEquity")),

            # Analyst insights
            "recommendation": self.info.get("recommendationKey"),
            "target_price": format_number(self.info.get("targetMeanPrice")),
            "analyst_count": self.info.get("numberOfAnalystOpinions"),
            "insider_own": format_percent(self.info.get("heldPercentInsiders")),
            "institutional_own": format_percent(self.info.get("heldPercentInstitutions")),
        }

Visualizations

# Import the yfinance library to fetch financial data such as stock prices
import yfinance as yf

# Import plotly's graph_objects module for creating interactive visualizations
import plotly.graph_objects as go

# Define the stock ticker symbol you want to fetch data for (in this case, Apple Inc.)
ticker = "AAPL"

# Create a Ticker object using yfinance for the given symbol
stock = yf.Ticker(ticker)

# Fetch the stock's historical price data for the last 6 months
data = stock.history(period="6mo")

# Calculate the 20-day Simple Moving Average (SMA) of the closing price
# This smooths out short-term price fluctuations and shows short-term trends
data['SMA20'] = data['Close'].rolling(window=20).mean()

# Calculate the 50-day Simple Moving Average (SMA) of the closing price
# This provides a longer-term trend indicator
data['SMA50'] = data['Close'].rolling(window=50).mean()

# --- Price Chart ---

# Create an instance of StockAnalyzer and fetch data
analyzer = StockAnalyzer(ticker="AAPL", start=datetime(2023, 1, 1), end=datetime.now())
data = analyzer.fetch_data()

if data is not None and not data.empty:
    # Add technical indicators to the data
    data = analyzer.add_indicators()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
        name='Candlesticks', increasing_line_color='green', decreasing_line_color='red'))

    # Add indicators based on toggles (assuming these are defined elsewhere or will be added)
    # For now, we'll add them directly for demonstration after adding indicators
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name="SMA20", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name="SMA50", line=dict(color="orange")))
    # Corrected to use data['EMA20'] after add_indicators() is called
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA20'], name="EMA20", line=dict(color="cyan", dash="dash")))
    fig.add_trace(go.Scatter(x=data.index, y=data['UpperBand'], name="Upper Band", line=dict(color="gray", dash="dot")))
    fig.add_trace(go.Scatter(x=data.index, y=data['LowerBand'], name="Lower Band", line=dict(color="gray", dash="dot")))


    fig.update_layout(title='AAPL Price Chart with Indicators', xaxis_rangeslider_visible=False, height=600, legend_title_text='Indicators')
    # st.plotly_chart(fig, use_container_width=True) # This line is for Streamlit and will cause an error in Colab
    fig.show() # Use fig.show() to display the plot in Colab

else:
    print("Could not fetch data for AAPL.")

    # --- RSI Chart ---

# Create an instance of StockAnalyzer and fetch data
analyzer = StockAnalyzer(ticker="AAPL", start=datetime(2023, 1, 1), end=datetime.now())
data = analyzer.fetch_data()

if data is not None and not data.empty:
    # Add technical indicators (RSI and RSI Signal included here)
    data = analyzer.add_indicators()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, y=data['RSI'],
        name="RSI", line=dict(color="red")
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data['RSI_Signal'],
        name="RSI Signal (9-EMA)", line=dict(color="yellow")
    ))

    # Overbought & Oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="gray", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="gray", annotation_text="Oversold")

    fig.update_layout(
        title="RSI (Relative Strength Index)",
        height=300,
        yaxis_title="RSI",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # st.plotly_chart(fig, use_container_width=True)  # For Streamlit
    fig.show()  # Works in Colab / Jupyter

else:
    print("Could not fetch data for AAPL.")

    # --- MACD Chart ---

# Create an instance of StockAnalyzer and fetch data
analyzer = StockAnalyzer(ticker="AAPL", start=datetime(2023, 1, 1), end=datetime.now())
data = analyzer.fetch_data()

if data is not None and not data.empty:
    # Add technical indicators (MACD, Signal, Histogram included here)
    data = analyzer.add_indicators()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, y=data['MACD'],
        name="MACD", line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Signal'],
        name="Signal", line=dict(color="orange")
    ))
    fig.add_trace(go.Bar(
        x=data.index, y=data['Hist'],
        name="Histogram", marker_color="gray"
    ))

    fig.update_layout(
        title="MACD (Trend Strength)",
        height=300,
        yaxis_title="MACD",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # st.plotly_chart(fig, use_container_width=True)  # For Streamlit
    fig.show()  # Works in Jupyter/Colab

else:
    print("Could not fetch data for AAPL.")

# --- Volume Chart ---

# Create an instance of StockAnalyzer and fetch data
analyzer = StockAnalyzer(ticker="AAPL", start=datetime(2023, 1, 1), end=datetime.now())
data = analyzer.fetch_data()

if data is not None and not data.empty:
    # Add indicators (not strictly needed for Volume, but keeps flow consistent)
    data = analyzer.add_indicators()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data.index, y=data['Volume'],
        name="Volume", marker_color="purple"
    ))

    fig.update_layout(
        title="Trading Volume",
        height=300,
        yaxis_title="Volume"
    )

    # st.plotly_chart(fig, use_container_width=True)  # For Streamlit
    fig.show()  # Works in Jupyter/Colab

else:
    print("Could not fetch data for AAPL.")


