import pandas as pd
import numpy as np


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(window=period).mean()

def calc_atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_adx(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)
    plus_dm  = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    plus_dm[plus_dm < minus_dm]   = 0
    minus_dm[minus_dm < plus_dm]  = 0
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(span=period, adjust=False).mean()
    pdi   = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr14
    ndi   = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr14
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-10)
    adx   = dx.ewm(span=period, adjust=False).mean()
    return adx, pdi, ndi

def swing_highs(series, lookback=5):
    highs = []
    for i in range(lookback, len(series) - lookback):
        window = series.iloc[i - lookback:i + lookback + 1]
        if series.iloc[i] == window.max():
            highs.append((i, float(series.iloc[i])))
    return highs[-6:]

def swing_lows(series, lookback=5):
    lows = []
    for i in range(lookback, len(series) - lookback):
        window = series.iloc[i - lookback:i + lookback + 1]
        if series.iloc[i] == window.min():
            lows.append((i, float(series.iloc[i])))
    return lows[-6:]

def detect_gap_zones(df, min_gap_pct=0.005):
    gaps = []
    for i in range(1, len(df)):
        prev_high = df["High"].iloc[i - 1]
        curr_low  = df["Low"].iloc[i]
        if curr_low > prev_high * (1 + min_gap_pct):
            gaps.append({"bottom": float(prev_high), "top": float(curr_low)})
    return gaps[-3:]

def detect_pocket_pivot(df, lookback=10):
    if len(df) < lookback + 1:
        return False
    today = df.iloc[-1]
    if today["Close"] <= today["Open"]:
        return False
    down_vols = [df.iloc[-i]["Volume"] for i in range(2, lookback + 2) if df.iloc[-i]["Close"] < df.iloc[-i]["Open"]]
    return bool(down_vols) and today["Volume"] > max(down_vols)

def detect_lower_highs(highs):
    if len(highs) < 3:
        return False, []
    vals = [h[1] for h in highs[-3:]]
    return (vals[0] > vals[1] > vals[2]), vals

def rs_rating(price, week52_high, week52_low):
    if week52_high == week52_low:
        return 50
    pct = (price - week52_low) / (week52_high - week52_low) * 100
    return max(0, min(99, int(pct * 0.9 + 5)))


def _cluster_prices(prices, tolerance=0.015):
    """Group nearby prices within tolerance % into clusters."""
    prices = sorted(prices)
    clusters = []
    current = [prices[0]]
    for p in prices[1:]:
        if (p - current[0]) / current[0] <= tolerance:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
    clusters.append(current)
    return clusters


def compute_sr_zones(close, atr_val, lookback=5, tolerance=0.015):
    """
    Cluster swing highs/lows into S/R zones with a price band and strength score.
    Returns list of zone dicts sorted by price descending.
    """
    sh = swing_highs(close, lookback)
    sl = swing_lows(close, lookback)
    zones = []

    res_prices = [v for _, v in sh]
    if res_prices:
        for cluster in _cluster_prices(res_prices, tolerance):
            mid = sum(cluster) / len(cluster)
            pad = atr_val * 0.15
            strength = min(5, len(cluster) + 1)
            label = {5: "Very Strong", 4: "Strong", 3: "Moderate", 2: "Weak"}.get(strength, "Weak")
            zones.append({
                "type": "resistance",
                "low":  round(min(cluster) - pad, 2),
                "high": round(max(cluster) + pad, 2),
                "mid":  round(mid, 2),
                "strength": strength,
                "strength_label": label,
                "touches": len(cluster),
            })

    sup_prices = [v for _, v in sl]
    if sup_prices:
        for cluster in _cluster_prices(sup_prices, tolerance):
            mid = sum(cluster) / len(cluster)
            pad = atr_val * 0.15
            strength = min(5, len(cluster) + 1)
            label = {5: "Very Strong", 4: "Strong", 3: "Moderate", 2: "Weak"}.get(strength, "Weak")
            zones.append({
                "type": "support",
                "low":  round(min(cluster) - pad, 2),
                "high": round(max(cluster) + pad, 2),
                "mid":  round(mid, 2),
                "strength": strength,
                "strength_label": label,
                "touches": len(cluster),
            })

    zones.sort(key=lambda z: z["mid"], reverse=True)
    return zones[:8]
