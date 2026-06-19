"""
News confirmation layer for TIC Research.
Fetches recent headlines from yfinance + Finviz.
Returns structured sentiment signal to confirm / contradict thesis.
"""

import re

# ── Sentiment keyword lists ────────────────────────────────────────────────
BULL_WORDS = {
    "beat", "beats", "record", "surge", "surges", "rally", "rallies",
    "upgrade", "upgraded", "outperform", "buy", "strong", "growth",
    "profit", "profits", "gain", "gains", "bullish", "breakout",
    "exceeds", "raise", "raised", "positive", "upside", "expansion",
    "jumps", "jump", "soars", "soar", "high", "revenue growth",
    "earnings beat", "tops", "top", "launches", "wins", "won",
    "partnership", "deal", "contract", "approval", "approved",
    "accelerate", "accelerating", "momentum", "demand",
}

BEAR_WORDS = {
    "miss", "misses", "missed", "loss", "losses", "decline", "declines",
    "fall", "falls", "plunge", "plunges", "downgrade", "downgraded",
    "sell", "weak", "weakness", "concern", "concerns", "disappoints",
    "disappointing", "cuts", "cut", "warns", "warning", "below",
    "risk", "risks", "layoffs", "lawsuit", "investigation", "probe",
    "slows", "slow", "slowdown", "debt", "recall", "halted",
    "bearish", "lower", "guidance cut", "shortfall", "pressure",
    "decline", "shrinks", "shrink", "negative", "miss expectations",
}


def _score_headline(text: str) -> str:
    """Return 'bull', 'bear', or 'neutral' based on keyword scan."""
    t = text.lower()
    words = set(re.findall(r'\b\w+\b', t))
    bull_hits = len(words & BULL_WORDS)
    bear_hits = len(words & BEAR_WORDS)
    if bull_hits > bear_hits:
        return "bull"
    elif bear_hits > bull_hits:
        return "bear"
    return "neutral"


def _score_headline_numeric(text: str) -> float:
    """Return a numeric sentiment score from -1.0 (very bearish) to +1.0 (very bullish)."""
    t = text.lower()
    words = set(re.findall(r'\b\w+\b', t))
    bull_hits = len(words & BULL_WORDS)
    bear_hits = len(words & BEAR_WORDS)
    total = bull_hits + bear_hits
    if total == 0:
        return 0.0
    return round((bull_hits - bear_hits) / total, 2)


def _fetch_yfinance_news(symbol: str, max_items: int = 6) -> list:
    """Pull recent news from yfinance."""
    headlines = []
    try:
        import yfinance as yf
        stock = yf.Ticker(symbol)
        news  = stock.news or []
        for item in news[:max_items]:
            # yfinance returns dicts; handle both old + new schema
            content = item.get("content") or item
            title   = (
                content.get("title") or
                item.get("title") or
                ""
            ).strip()
            url = (
                content.get("canonicalUrl", {}).get("url") or
                content.get("url") or
                item.get("link") or
                "#"
            )
            if not title:
                continue
            headlines.append({
                "source":    "Yahoo Finance",
                "title":     title,
                "url":       url,
                "sentiment": _score_headline(title),
                "score":     _score_headline_numeric(title),
            })
    except Exception as e:
        print(f"[News/YF] {e}")
    return headlines


def _fetch_finviz_news(symbol: str, max_items: int = 6) -> list:
    """Scrape recent headlines from Finviz news table."""
    headlines = []
    try:
        import urllib.request
        url = f"https://finviz.com/quote.ashx?t={symbol.upper()}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                                   "Chrome/124.0.0.0 Safari/537.36"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Extract news table rows: <tr class="nn-row ... "> ... <a class="tab-link-news" href="...">TITLE</a>
        # Finviz news rows contain links with class "tab-link-news"
        pattern = re.compile(
            r'<a[^>]+class="tab-link-news"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL
        )
        matches = pattern.findall(html)

        seen = set()
        for href, raw_title in matches[:max_items]:
            title = re.sub(r'<[^>]+>', '', raw_title).strip()
            if not title or title in seen:
                continue
            seen.add(title)
            # Finviz hrefs are external — keep as-is
            link = href if href.startswith("http") else f"https://finviz.com/{href}"
            headlines.append({
                "source":    "Finviz",
                "title":     title,
                "url":       link,
                "sentiment": _score_headline(title),
                "score":     _score_headline_numeric(title),
            })
    except Exception as e:
        print(f"[News/Finviz] {e}")
    return headlines


def _overall_signal(headlines: list) -> dict:
    """
    Aggregate sentiment across all headlines.
    Returns signal dict with badge, color, and counts.
    """
    bull  = sum(1 for h in headlines if h["sentiment"] == "bull")
    bear  = sum(1 for h in headlines if h["sentiment"] == "bear")
    total = len(headlines)
    neutral = total - bull - bear

    if total == 0:
        return {"label": "No Data", "color": "gray", "bull": 0, "bear": 0, "neutral": 0, "total": 0}

    bull_pct = bull / total
    bear_pct = bear / total

    if bull_pct >= 0.60:
        label, color = "✅ Confirming", "green"
    elif bear_pct >= 0.60:
        label, color = "❌ Contradicting", "red"
    else:
        label, color = "⚠️ Mixed", "yellow"

    return {
        "label":   label,
        "color":   color,
        "bull":    bull,
        "bear":    bear,
        "neutral": neutral,
        "total":   total,
    }


def fetch_news(symbol: str, max_per_source: int = 5) -> dict:
    """
    Main entry point. Returns:
    {
      "signal":    {"label", "color", "bull", "bear", "neutral", "total"},
      "headlines": [ {"source", "title", "url", "sentiment"}, ... ],
      "sources":   ["Yahoo Finance", "Finviz"],
    }
    """
    yf_news  = _fetch_yfinance_news(symbol, max_items=max_per_source)
    fvz_news = _fetch_finviz_news(symbol, max_items=max_per_source)

    all_headlines = yf_news + fvz_news
    signal = _overall_signal(all_headlines)

    return {
        "signal":    signal,
        "headlines": all_headlines,
        "sources":   [s for s in ["Yahoo Finance", "Finviz"] if any(h["source"] == s for h in all_headlines)],
    }
