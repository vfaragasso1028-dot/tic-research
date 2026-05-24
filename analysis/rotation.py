"""
Sector Rotation module for TIC Research.

For the analyzed ticker's sector, fetches a peer basket, scores each peer
on RS Rating + cross type + stage, then returns ranked Leaders / Laggers
to show where capital is rotating within the sector.
"""

import pandas as pd
import numpy as np

# ── Sector peer baskets ────────────────────────────────────────────────────
# 10–12 liquid large-cap names per sector. Balanced across sub-industries.
SECTOR_PEERS = {
    "Technology": [
        "NVDA", "MSFT", "AAPL", "META", "GOOGL",
        "AVGO", "AMD", "ORCL", "CRM", "ADBE", "QCOM", "AMAT",
    ],
    "Consumer Cyclical": [
        "AMZN", "TSLA", "HD", "MCD", "NKE",
        "SBUX", "TJX", "LOW", "BKNG", "CMG", "GM", "F",
    ],
    "Healthcare": [
        "LLY", "UNH", "JNJ", "ABBV", "MRK",
        "TMO", "ABT", "DHR", "AMGN", "BMY", "CVS", "ISRG",
    ],
    "Financial Services": [
        "BRK-B", "JPM", "V", "MA", "BAC",
        "WFC", "GS", "MS", "BLK", "AXP", "C", "SCHW",
    ],
    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS", "T",
        "VZ", "CMCSA", "SNAP", "SPOT", "PINS", "ROKU", "TTWO",
    ],
    "Industrials": [
        "CAT", "DE", "HON", "UPS", "RTX",
        "LMT", "GE", "MMM", "BA", "ETN", "EMR", "PH",
    ],
    "Consumer Defensive": [
        "WMT", "PG", "KO", "PEP", "COST",
        "MDLZ", "CL", "GIS", "KHC", "SYY", "TSN", "MKC",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "MPC",
        "PSX", "VLO", "OXY", "DVN", "HAL", "BKR", "FANG",
    ],
    "Basic Materials": [
        "LIN", "APD", "ECL", "NEM", "FCX",
        "NUE", "CF", "MOS", "ALB", "DD", "PPG", "IFF",
    ],
    "Real Estate": [
        "PLD", "AMT", "EQIX", "CCI", "SPG",
        "O", "VICI", "PSA", "EXR", "WELL", "AVB", "EQR",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "AEP", "SRE",
        "EXC", "XEL", "D", "PCG", "ED", "AEE", "WEC",
    ],
}

# Fallback basket for unknown sectors
BROAD_MARKET = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLU"]


def _rs(price, w52h, w52l):
    if w52h == w52l or w52h == 0:
        return 50
    pct = (price - w52l) / (w52h - w52l) * 100
    return max(0, min(99, int(pct * 0.9 + 5)))


def _sma(series, n):
    return series.rolling(n, min_periods=max(1, n // 2)).mean()


def _momentum_score(prices):
    """
    Composite momentum: weighted blend of 1M, 3M, 6M returns.
    Returns float — higher is stronger momentum.
    """
    if len(prices) < 20:
        return 0.0
    try:
        p_now = float(prices.iloc[-1])
        p_1m  = float(prices.iloc[-21])  if len(prices) >= 21  else p_now
        p_3m  = float(prices.iloc[-63])  if len(prices) >= 63  else p_now
        p_6m  = float(prices.iloc[-126]) if len(prices) >= 126 else p_now
        ret_1m = (p_now / p_1m - 1) * 100
        ret_3m = (p_now / p_3m - 1) * 100
        ret_6m = (p_now / p_6m - 1) * 100
        return round(ret_1m * 0.5 + ret_3m * 0.3 + ret_6m * 0.2, 1)
    except Exception:
        return 0.0


def get_rotation_data(ticker: str, sector: str, max_peers: int = 10) -> dict:
    """
    Fetch peer basket for the sector, score each name, return leaders/laggers.

    Returns:
    {
      "sector": str,
      "sector_signal": "Strong" | "Mixed" | "Weak",
      "sector_signal_color": "green" | "yellow" | "red",
      "leaders": [ {ticker, rs, cross, stage_num, momentum, price, pct_from_52h, change_pct} ],
      "laggers":  [ ... ],
      "all_peers": [ ... ],
      "ticker_rank": int,       # 1-based rank among all peers (lower = better)
      "total_peers": int,
      "golden_count": int,
      "death_count": int,
    }
    """
    peers = SECTOR_PEERS.get(sector, BROAD_MARKET)
    # Put the analyzed ticker first so it's always included
    all_tickers = [ticker] + [p for p in peers if p.upper() != ticker.upper()]
    all_tickers = all_tickers[:max_peers + 1]

    scored = []
    try:
        import yfinance as yf
        # Single batch download — much faster than per-ticker calls
        raw = yf.download(
            " ".join(all_tickers),
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        # Normalize to dict of { ticker: close_series }
        close_dict = {}
        if isinstance(raw.columns, pd.MultiIndex):
            for t in all_tickers:
                try:
                    s = raw["Close"][t].dropna()
                    if len(s) > 20:
                        close_dict[t] = s
                except Exception:
                    pass
        else:
            # Single ticker returned as flat DataFrame
            if "Close" in raw.columns and len(raw) > 20:
                close_dict[all_tickers[0]] = raw["Close"].dropna()

        for t, close in close_dict.items():
            try:
                price  = float(close.iloc[-1])
                w52h   = float(close.rolling(252, min_periods=100).max().iloc[-1])
                w52l   = float(close.rolling(252, min_periods=100).min().iloc[-1])
                s50    = float(_sma(close, 50).iloc[-1])
                s150   = float(_sma(close, 150).iloc[-1])
                s200   = float(_sma(close, 200).iloc[-1])

                rs     = _rs(price, w52h, w52l)
                cross  = "golden" if s50 > s200 else "death"
                mom    = _momentum_score(close)

                # Simple stage number
                above_150 = price > s150
                s150_trend = float(_sma(close, 150).iloc[-1]) > float(_sma(close, 150).iloc[-6]) if len(close) >= 6 else True
                if above_150 and s150_trend:
                    stage_num = 2
                elif above_150 and not s150_trend:
                    stage_num = 3
                elif not above_150 and not s150_trend:
                    stage_num = 4
                else:
                    stage_num = 1

                # 1-day change pct (approximate)
                prev = float(close.iloc[-2]) if len(close) > 1 else price
                chg_pct = round((price / prev - 1) * 100, 2)
                pct_from_52h = round((price / w52h - 1) * 100, 1) if w52h else 0

                scored.append({
                    "ticker":        t.upper(),
                    "rs":            rs,
                    "cross":         cross,
                    "stage_num":     stage_num,
                    "momentum":      mom,
                    "price":         round(price, 2),
                    "pct_from_52h":  pct_from_52h,
                    "change_pct":    chg_pct,
                    "is_subject":    t.upper() == ticker.upper(),
                })
            except Exception as e:
                print(f"[Rotation] {t}: {e}")
                continue

    except Exception as e:
        print(f"[Rotation] Batch download failed: {e}")
        return _empty_result(ticker, sector)

    if not scored:
        return _empty_result(ticker, sector)

    # Sort by composite score: RS (60%) + momentum direction (40%)
    scored.sort(key=lambda x: x["rs"] * 0.6 + (1 if x["momentum"] > 0 else -1) * 20 + x["momentum"] * 0.4, reverse=True)

    # Rank & sector signal
    for i, p in enumerate(scored):
        p["rank"] = i + 1

    golden_count = sum(1 for p in scored if p["cross"] == "golden")
    death_count  = len(scored) - golden_count
    stage2_count = sum(1 for p in scored if p["stage_num"] == 2)

    total = len(scored)
    if golden_count / total >= 0.60 or stage2_count / total >= 0.55:
        sector_signal, sector_color = "Strong", "green"
    elif death_count / total >= 0.60:
        sector_signal, sector_color = "Weak", "red"
    else:
        sector_signal, sector_color = "Mixed", "yellow"

    # Subject ticker rank
    subject = next((p for p in scored if p["is_subject"]), None)
    ticker_rank = subject["rank"] if subject else None

    leaders = [p for p in scored if not p["is_subject"]][:3]
    laggers = list(reversed([p for p in scored if not p["is_subject"]]))[:3]

    return {
        "sector":              sector,
        "sector_signal":       sector_signal,
        "sector_signal_color": sector_color,
        "leaders":             leaders,
        "laggers":             laggers,
        "all_peers":           scored,
        "ticker_rank":         ticker_rank,
        "total_peers":         total,
        "golden_count":        golden_count,
        "death_count":         death_count,
        "stage2_count":        stage2_count,
    }


def _empty_result(ticker, sector):
    return {
        "sector": sector, "sector_signal": "No Data", "sector_signal_color": "gray",
        "leaders": [], "laggers": [], "all_peers": [],
        "ticker_rank": None, "total_peers": 0,
        "golden_count": 0, "death_count": 0, "stage2_count": 0,
    }
