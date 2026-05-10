import anthropic
import json


def build_context(ticker, info, zen_total, zen_bd, stage, stage_sub,
                  atr, adx, di_plus, di_minus, rsi, macd, sig, hist,
                  e9, e21, s50, s150, s200, support, resistance,
                  pocket_pivot, lower_highs, lh_vals, gap_zones, quarterly_rows):
    price       = info.get("currentPrice") or info.get("regularMarketPrice", 0) or 0
    w52h        = info.get("fiftyTwoWeekHigh", 0)
    w52l        = info.get("fiftyTwoWeekLow", 0)
    mkt_cap     = info.get("marketCap", 0)
    trail_pe    = info.get("trailingPE", "N/A")
    fwd_pe      = info.get("forwardPE", "N/A")
    beta        = info.get("beta", "N/A")
    target      = info.get("targetMeanPrice", 0)
    rec         = info.get("recommendationKey", "N/A")
    avg_vol     = info.get("averageVolume", 0)
    company     = info.get("longName", ticker)
    sector      = info.get("sector", "N/A")
    industry    = info.get("industry", "N/A")

    tgt_delta = f"{((target/price-1)*100):.1f}%" if price and target else "N/A"
    fin_lines = ""
    for r in quarterly_rows[:4]:
        fin_lines += f"  {r['quarter']}: Revenue={r['revenue']}, Net Income={r['net_income']}\n"

    ctx = f"""
TICKER: {ticker} — {company}
SECTOR: {sector} | INDUSTRY: {industry}
PRICE: ${price:.2f} | 52W HIGH: ${w52h:.2f} | 52W LOW: ${w52l:.2f}
MKT CAP: ${mkt_cap/1e9:.2f}B | TRAIL PE: {trail_pe} | FWD PE: {fwd_pe} | BETA: {beta}
ANALYST TARGET: ${target:.2f} ({tgt_delta} from price) | CONSENSUS: {rec}
AVG VOLUME: {avg_vol:,}

TICSCORE: {zen_total}/100 (G:{zen_bd['G']} V:{zen_bd['V']} Q:{zen_bd['Q']} S:{zen_bd['S']})
TICSTAGE: {stage} — {stage_sub}

TECHNICALS:
  9EMA={e9:.2f} ({((price/e9-1)*100):+.1f}%)  21EMA={e21:.2f} ({((price/e21-1)*100):+.1f}%)
  50SMA={s50:.2f} ({((price/s50-1)*100):+.1f}%)  150SMA={s150:.2f}  200SMA={s200:.2f} ({((price/s200-1)*100):+.1f}%)
  Death Cross: {s50 < s200} | Golden Cross: {s50 > s200}
  RSI(14): {rsi:.1f} | MACD: {macd:.3f} | Signal: {sig:.3f} | Hist: {hist:.3f}
  ATR(14): ${atr:.2f} ({(atr/price*100):.1f}% of price)
  ADX(14): {adx:.1f} | DI+: {di_plus:.1f} | DI-: {di_minus:.1f}

SUPPORT: ${support:.2f} | RESISTANCE: ${resistance:.2f}
POCKET PIVOT: {pocket_pivot}
LOWER HIGH PATTERN: {lower_highs} ({[f'${v:.2f}' for v in lh_vals]})
GAP ZONES: {[f'${g["bottom"]:.2f}-${g["top"]:.2f}' for g in gap_zones]}

QUARTERLY FINANCIALS:
{fin_lines}
""".strip()
    return ctx


def generate_thesis(api_key, context_block, ticker, company):
    client = anthropic.Anthropic(api_key=api_key)

    system = """You are TIC Research — a precision equity research engine combining CAN SLIM fundamentals,
Stan Weinstein stage analysis, and institutional-grade technical analysis.

Output: direct, dense, professional. No filler. Every sentence adds signal.
Use exact dollar figures, percentages, and price levels from the data provided.
Write for sophisticated swing traders — maximize signal-to-noise ratio.

Return ONLY a valid JSON object with these exact keys:
{
  "tic_signal": "2-4 sentence technical signal. Cover active patterns, exact price levels, directional bias for aggressive vs conservative traders.",
  "tic_stock": "3-5 sentence holistic narrative. Valuation, revenue trend, net income trajectory, analyst positioning, beta implications. Balanced — name risks AND opportunity.",
  "thesis": ["point 1", "point 2", "point 3"],
  "bull_cases": ["bull 1", "bull 2", "bull 3"],
  "bear_cases": ["bear 1", "bear 2", "bear 3"],
  "catalysts": [
    {"date": "Month YYYY", "text": "catalyst description"},
    {"date": "Month YYYY", "text": "catalyst description"},
    {"date": "Month YYYY", "text": "catalyst description"}
  ]
}

Rules: Each point 2-3 sentences. Reference specific levels/figures. No fluff."""

    prompt = f"Analyze {ticker} ({company}) and produce full TIC Research output.\n\nDATA:\n{context_block}\n\nReturn ONLY the JSON object."

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip().rstrip("```").strip()
    return json.loads(raw)
