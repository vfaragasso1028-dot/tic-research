"""
Schwab data layer for TIC Research.
Uses schwab-py to fetch real-time quotes, 2yr daily history, and fundamentals.
Falls back to yfinance if Schwab is unavailable.
"""

import os
import json
import tempfile
import pandas as pd
import numpy as np

# ── Schwab credentials ────────────────────────────────────────────────
CLIENT_ID     = os.environ.get("SCHWAB_CLIENT_ID",     "EgTGgrXUKiUUc5jnb9MKqMC5sm0VMllOasA5SDOE2uPhGSGb")
CLIENT_SECRET = os.environ.get("SCHWAB_CLIENT_SECRET", "QHpGo0MGf27WOTzYCr34JjNWxG3oTKeXAnejLr2zR4t0XngXYc1oK9FJyvhJGFpG")

# Local token file paths to try in order
LOCAL_TOKEN_PATHS = [
    r"C:\Users\vince\Desktop\orb_bot\schwab_token.json",  # freshest token
    r"C:\Users\vince\schwab_token.json",
    "/tmp/schwab_token.json",  # Render / Linux
]

_schwab_client = None  # cached


def _get_token_path():
    """Return the first existing token file, or write from env var."""
    # Check local paths first
    for p in LOCAL_TOKEN_PATHS:
        if os.path.exists(p):
            return p

    # On Render: try env var SCHWAB_TOKEN_JSON
    token_json = os.environ.get("SCHWAB_TOKEN_JSON", "")
    if token_json:
        tmp = "/tmp/schwab_token.json"
        with open(tmp, "w") as f:
            f.write(token_json)
        return tmp

    return None


def get_schwab_client():
    """Return a cached Schwab client, or None if unavailable."""
    global _schwab_client
    if _schwab_client is not None:
        return _schwab_client

    token_path = _get_token_path()
    if not token_path:
        return None

    try:
        from schwab.auth import client_from_token_file
        _schwab_client = client_from_token_file(
            token_path=token_path,
            api_key=CLIENT_ID,
            app_secret=CLIENT_SECRET,
        )
        return _schwab_client
    except Exception as e:
        print(f"[Schwab] Client init failed: {e}")
        return None


def schwab_price_history(symbol):
    """
    Fetch 2yr daily OHLCV from Schwab.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    Index: DatetimeIndex (UTC)
    """
    c = get_schwab_client()
    if c is None:
        return None

    try:
        from schwab.client import Client
        resp = c.get_price_history(
            symbol,
            period_type=Client.PriceHistory.PeriodType.YEAR,
            period=Client.PriceHistory.Period.TWO_YEARS,
            frequency_type=Client.PriceHistory.FrequencyType.DAILY,
            frequency=Client.PriceHistory.Frequency.DAILY,
            need_extended_hours_data=False,
        )
        data = resp.json()
        candles = data.get("candles", [])
        if not candles:
            return None

        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
        df = df.set_index("datetime")
        df = df.rename(columns={
            "open":   "Open",
            "high":   "High",
            "low":    "Low",
            "close":  "Close",
            "volume": "Volume",
        })
        return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    except Exception as e:
        print(f"[Schwab] price_history error for {symbol}: {e}")
        return None


def schwab_quote(symbol):
    """
    Fetch real-time quote from Schwab.
    Returns a flat dict with standardized keys matching what app.py expects from yfinance info.
    """
    c = get_schwab_client()
    if c is None:
        return None

    try:
        resp = c.get_quote(symbol)
        raw  = resp.json()

        # Response: { "TSLA": { "quote": {...}, "fundamental": {...}, ... } }
        data = raw.get(symbol, {})
        q    = data.get("quote",       {})
        f    = data.get("fundamental", {})
        ref  = data.get("reference",   {})

        price       = q.get("lastPrice") or q.get("mark") or 0
        prev_close  = q.get("closePrice") or price
        change_val  = q.get("netChange", price - prev_close)
        change_pct  = q.get("netPercentChangeInDouble", 0)
        high52      = q.get("52WeekHigh") or f.get("high52") or 0
        low52       = q.get("52WeekLow")  or f.get("low52")  or 0
        mkt_cap     = f.get("marketCap", 0) or 0
        avg_vol     = f.get("avg10DaysVolume") or q.get("totalVolume") or 0
        pe_ratio    = f.get("peRatio") or 0
        eps         = f.get("eps") or 0
        description = ref.get("description", symbol)

        # Build an info-like dict (same keys app.py reads from yfinance info)
        info = {
            "symbol":                symbol,
            "longName":              description,
            "currentPrice":          price,
            "regularMarketPrice":    price,
            "previousClose":         prev_close,
            "regularMarketChange":   change_val,
            "regularMarketChangePercent": change_pct,
            "fiftyTwoWeekHigh":      high52,
            "fiftyTwoWeekLow":       low52,
            "marketCap":             mkt_cap * 1e6 if mkt_cap < 1e9 else mkt_cap,
            "averageVolume":         avg_vol,
            "averageVolume10days":   avg_vol,
            "regularMarketVolume":   q.get("totalVolume", 0),
            "trailingPE":            pe_ratio if pe_ratio else None,
            "forwardPE":             None,   # Schwab doesn't provide fwd PE
            "beta":                  None,   # supplement from yfinance
            "targetMeanPrice":       None,   # supplement from yfinance
            "recommendationKey":     "N/A",
            "sector":                ref.get("exchange", "N/A"),
            "industry":              "N/A",
            "floatShares":           f.get("marketCapFloat", 0) or 0,
            "institutionPercentHeld": None,
            "returnOnEquity":        None,
            "profitMargins":         None,
            "recommendationMean":    None,
            "_source": "schwab",
        }
        return info

    except Exception as e:
        print(f"[Schwab] quote error for {symbol}: {e}")
        return None


def supplement_with_yfinance(symbol, schwab_info):
    """
    Fill in gaps in Schwab info (fwd PE, beta, analyst target, sector/industry)
    with a lightweight yfinance fetch. Much faster than full yfinance since
    we only need the info dict, not history.
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(symbol)
        yf_info = stock.info or {}

        # Only fill in what Schwab didn't provide
        fill_keys = [
            "forwardPE", "beta", "targetMeanPrice", "recommendationKey",
            "recommendationMean", "sector", "industry", "longName",
            "institutionPercentHeld", "returnOnEquity", "profitMargins",
        ]
        for k in fill_keys:
            if schwab_info.get(k) is None and yf_info.get(k) is not None:
                schwab_info[k] = yf_info[k]

        # Keep Schwab longName if it's meaningful, else use yfinance
        if len(schwab_info.get("longName", "")) < 3:
            schwab_info["longName"] = yf_info.get("longName", symbol)

    except Exception as e:
        print(f"[YF supplement] {e}")

    return schwab_info


def yf_fetch_robust(symbol):
    """
    Fetch history + info + financials from yfinance.
    Uses yfinance's built-in session management (curl_cffi internally).
    Includes retry logic for transient failures.
    """
    import time
    import yfinance as yf
    import pandas as pd

    for attempt in range(3):
        try:
            # Let yfinance manage its own curl_cffi session — don't override it
            stock = yf.Ticker(symbol)
            df    = stock.history(period="2y", interval="1d")

            if df is not None and not df.empty:
                try:
                    info = stock.info or {}
                except Exception:
                    info = {}
                try:    qf = stock.quarterly_income_stmt
                except: qf = getattr(stock, "quarterly_financials", pd.DataFrame())
                try:    af = stock.income_stmt
                except: af = getattr(stock, "financials", pd.DataFrame())
                if qf is None: qf = pd.DataFrame()
                if af is None: af = pd.DataFrame()
                return df, info, qf, af
            else:
                print(f"[YF] Empty data on attempt {attempt+1} for {symbol}")
        except Exception as e:
            print(f"[YF] Attempt {attempt+1} failed for {symbol}: {e}")
        if attempt < 2:
            time.sleep(2)

    return None, {}, pd.DataFrame(), pd.DataFrame()


def fetch_stock_data(symbol):
    """
    Main entry point for TIC Research data fetching.
    Returns (df, info, qf, af) — same signature as the old yfinance fetch.

    Priority:
      1. Schwab for OHLCV + real-time quote
      2. yfinance (with cloud-safe headers) for fundamentals + financials
      3. Full yfinance fallback if Schwab unavailable
    """
    import pandas as pd

    schwab_available = get_schwab_client() is not None

    if schwab_available:
        print(f"[Data] Using Schwab for {symbol}")
        df   = schwab_price_history(symbol)
        info = schwab_quote(symbol)

        if df is not None and info is not None:
            info = supplement_with_yfinance(symbol, info)

            # Financials always from yfinance (Schwab has no income statements)
            _, _, qf, af = yf_fetch_robust(symbol)
            return df, info, qf, af

    # Full yfinance fallback (cloud-safe)
    print(f"[Data] Schwab unavailable — using Yahoo Finance for {symbol}")
    try:
        df, info, qf, af = yf_fetch_robust(symbol)
        return df, info, qf, af
    except Exception as e:
        import pandas as pd
        return None, {}, pd.DataFrame(), pd.DataFrame()
