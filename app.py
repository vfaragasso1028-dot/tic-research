import os
import warnings
import numpy as np
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from analysis.schwab_data import fetch_stock_data, get_schwab_client
from analysis.technical import (
    ema, sma, calc_atr, calc_rsi, calc_macd, calc_adx,
    swing_highs, swing_lows, detect_gap_zones,
    detect_pocket_pivot, detect_lower_highs, rs_rating
)
from analysis.canslim import ticscore, is_exempt
from analysis.stage import ticstage
from analysis.ai_thesis import build_context, generate_thesis
from analysis.news import fetch_news
from analysis.rotation import get_rotation_data

warnings.filterwarnings("ignore")

app = Flask(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def fmt_currency(val):
    try:
        val = float(val)
        if abs(val) >= 1e12: return f"${val/1e12:.2f}T"
        if abs(val) >= 1e9:  return f"${val/1e9:.2f}B"
        if abs(val) >= 1e6:  return f"${val/1e6:.2f}M"
        return f"${val:,.0f}"
    except Exception:
        return "N/A"

def fmt_large(val):
    try:
        val = float(val)
        if abs(val) >= 1e12: return f"${val/1e12:.2f}T"
        if abs(val) >= 1e9:  return f"${val/1e9:.2f}B"
        if abs(val) >= 1e6:  return f"${val/1e6:.2f}M"
        if abs(val) >= 1e3:  return f"${val/1e3:.1f}K"
        return f"${val:,.0f}"
    except Exception:
        return "N/A"

def safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except Exception:
        return default

def parse_quarterly(qf):
    rows = []
    try:
        if qf is None or qf.empty:
            return rows
        idx = [str(i) for i in qf.index]
        rev_key = next((k for k in qf.index if "Revenue" in str(k) or "revenue" in str(k)), None)
        ni_key  = next((k for k in qf.index if "Net Income" in str(k) or "net income" in str(k).lower()), None)
        for i, col in enumerate(qf.columns[:4]):
            try:
                rev = float(qf.loc[rev_key, col]) if rev_key is not None else None
            except Exception:
                rev = None
            try:
                ni = float(qf.loc[ni_key, col]) if ni_key is not None else None
            except Exception:
                ni = None
            label = col.strftime("%b '%y") if hasattr(col, "strftime") else str(col)[:7]
            rows.append({
                "quarter":    label,
                "revenue":    fmt_currency(rev) if rev is not None else "N/A",
                "net_income": fmt_currency(ni)  if ni  is not None else "N/A",
            })
    except Exception:
        pass
    return rows


def get_financials(stock):
    """Fetch quarterly + annual financials, handling yfinance API differences."""
    import pandas as pd
    qf = af = pd.DataFrame()
    # quarterly — try multiple attribute names across yfinance versions
    for attr in ("quarterly_income_stmt", "quarterly_financials"):
        try:
            data = getattr(stock, attr, None)
            if data is not None and not data.empty:
                qf = data
                break
        except Exception:
            pass
    # annual
    for attr in ("income_stmt", "financials"):
        try:
            data = getattr(stock, attr, None)
            if data is not None and not data.empty:
                af = data
                break
        except Exception:
            pass
    return qf, af

def safe_df(df):
    """Flatten MultiIndex columns from yfinance if present."""
    import pandas as pd
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    return df

def analyze_ticker(ticker, api_key=""):
    import pandas as pd

    # ── Fetch data (Schwab → yfinance fallback) ───────────────────────
    try:
        df, info, qf, af = fetch_stock_data(ticker)
    except Exception as e:
        return None, f"Data fetch error: {e}"

    if df is None or (hasattr(df, 'empty') and df.empty):
        return None, f"No price data found for {ticker}. Check the symbol."

    if qf is None:
        import pandas as pd
        qf = pd.DataFrame()
    if af is None:
        import pandas as pd
        af = pd.DataFrame()

    df = safe_df(df)

    # Flatten Close if it's a DataFrame (MultiIndex yfinance response)
    if hasattr(df["Close"], "squeeze"):
        close = df["Close"].squeeze()
    else:
        close = df["Close"]
    # Ensure it's a 1D Series
    import pandas as pd
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    price      = safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or close.iloc[-1])
    prev_close = safe_float(info.get("previousClose") or (close.iloc[-2] if len(close) > 1 else close.iloc[-1]))
    change_val = price - prev_close
    change_pct = (change_val / prev_close * 100) if prev_close else 0

    # Moving averages
    e9   = float(ema(close, 9).iloc[-1])
    e21  = float(ema(close, 21).iloc[-1])
    s50  = float(sma(close, 50).iloc[-1])
    s150 = float(sma(close, 150).iloc[-1])
    s200 = float(sma(close, 200).iloc[-1])

    # Indicators
    atr_val = float(calc_atr(df).iloc[-1])
    rsi_val = float(calc_rsi(close).iloc[-1])
    macd_l, sig_l, hist_l = calc_macd(close)
    macd_val = float(macd_l.iloc[-1])
    sig_val  = float(sig_l.iloc[-1])
    hist_val = float(hist_l.iloc[-1])
    adx_s, pdi_s, ndi_s = calc_adx(df)
    adx_val = float(adx_s.iloc[-1])
    di_plus = float(pdi_s.iloc[-1])
    di_minus = float(ndi_s.iloc[-1])

    # Support / Resistance
    sh = swing_highs(close)
    sl = swing_lows(close)
    resistance = sh[-1][1] if sh else float(close.rolling(20).max().iloc[-1])
    support    = sl[-1][1] if sl else float(close.rolling(20).min().iloc[-1])

    # Patterns
    gap_zones    = detect_gap_zones(df)
    pocket_pivot = detect_pocket_pivot(df)
    lh_det, lh_vals = detect_lower_highs(sh)

    # Active gap near support
    active_gap = next((g for g in reversed(gap_zones) if support < g["bottom"] < price), None)

    # Scores
    exempt, exempt_reason = is_exempt(info)
    if not exempt:
        tic_total, tic_bd, tic_pass, tic_detail = ticscore(info, qf if qf is not None else __import__('pandas').DataFrame(), af if af is not None else __import__('pandas').DataFrame())
    else:
        tic_total, tic_bd, tic_pass, tic_detail = 0, {"G":0,"V":0,"Q":0,"S":0}, False, {}

    stage, stage_sub, stage_desc, stage_color = ticstage(df)

    w52h = safe_float(info.get("fiftyTwoWeekHigh"))
    w52l = safe_float(info.get("fiftyTwoWeekLow"))
    rs   = rs_rating(price, w52h, w52l)

    cross_type   = "death" if s50 < s200 else "golden"
    full_stack   = e9 > e21 > s50
    mkt_cap      = safe_float(info.get("marketCap"))
    trail_pe     = info.get("trailingPE")
    fwd_pe       = info.get("forwardPE")
    beta         = info.get("beta")
    target       = safe_float(info.get("targetMeanPrice"))
    rec_key      = (info.get("recommendationKey") or "N/A").replace("_", " ").title()
    avg_vol      = safe_float(info.get("averageVolume"))
    vol_today    = safe_float(info.get("regularMarketVolume") or df["Volume"].iloc[-1])
    company      = info.get("longName", ticker)
    sector       = info.get("sector", "N/A")
    industry     = info.get("industry", "N/A")

    quarterly_rows = parse_quarterly(qf)

    # AI thesis
    thesis_data = None
    ai_error    = None
    effective_key = api_key or ANTHROPIC_API_KEY
    if effective_key:
        try:
            ctx = build_context(
                ticker, info, tic_total, tic_bd, stage, stage_sub,
                atr_val, adx_val, di_plus, di_minus, rsi_val,
                macd_val, sig_val, hist_val, e9, e21, s50, s150, s200,
                support, resistance, pocket_pivot, lh_det, lh_vals,
                gap_zones, quarterly_rows
            )
            thesis_data = generate_thesis(effective_key, ctx, ticker, company)
        except Exception as e:
            ai_error = str(e)

    # Auto-generated signal if no AI
    if not thesis_data:
        pp_txt  = "Pocket Pivot active — volume confirms institutional accumulation. " if pocket_pivot else ""
        gap_txt = f"Gap Support zone ${active_gap['bottom']:.2f}–${active_gap['top']:.2f} holding as structural floor. " if active_gap else ""
        dc_txt  = ("Death Cross (50 SMA below 200 SMA) establishes bearish macro structure. "
                   if cross_type == "death" else "Golden Cross (50 SMA above 200 SMA) supports bullish macro structure. ")
        lh_txt  = f"Lower High pattern active ({' → '.join([f'${v:.2f}' for v in lh_vals[-3:]])}) — distribution pressure present. " if lh_det else ""
        bias    = (f"Stage 2 structure supports long exposure on pullback to ${support:.2f}."
                   if "2" in stage else f"Stage {stage} — caution warranted. Require confirmed breakout above ${resistance:.2f}.")
        tic_signal = pp_txt + gap_txt + dc_txt + lh_txt + bias

        tgt_delta = f"{((target/price-1)*100):.1f}%" if price and target else "N/A"
        tic_stock = (
            f"{company} trades at ${price:.2f}, between its 52-week low of ${w52l:.2f} and high of ${w52h:.2f}. "
            + (f"Trailing PE of {trail_pe:.1f}x and forward PE of {fwd_pe:.1f}x embed significant growth expectations. " if trail_pe and fwd_pe else "")
            + (f"Analyst consensus target of ${target:.2f} sits {tgt_delta} from current price. " if target else "")
            + (f"Beta of {beta:.2f} — {'high-volatility vehicle, position sizing is the trade' if beta and beta > 1.5 else 'moderate-beta name'}." if beta else "")
        )

        # Auto company bio — only show if we have real content
        raw_bio = info.get("longBusinessSummary", "") or ""
        has_sector = sector and sector not in ("N/A", "", None)
        if raw_bio:
            sentences = [s.strip() for s in raw_bio.replace("  ", " ").split(". ") if s.strip()]
            bio_text = ". ".join(sentences[:2]) + "."
            sector_label = sector if has_sector else "Equity"
            company_bio = f"[{sector_label}] {bio_text}"
        elif has_sector:
            company_bio = f"[{sector}] {company} operates in the {industry} industry." if industry and industry != "N/A" else f"[{sector}] {company}."
        else:
            company_bio = ""  # nothing useful to show — hide the block

        thesis_data = {
            "company_bio": company_bio,
            "tic_signal": tic_signal,
            "tic_stock":  tic_stock,
            "thesis": [
                f"Price at ${price:.2f} is {((price/s50-1)*100):+.1f}% vs 50 SMA (${s50:.2f}) — key trend gauge for continuation or reversal thesis.",
                f"Key resistance at ${resistance:.2f} — a sustained close above this level with volume would confirm trend and attract momentum buyers.",
                f"Support at ${support:.2f} is the critical risk level — a break below invalidates the current setup and calls for capital preservation."
            ],
            "bull_cases": [
                f"A confirmed close above ${resistance:.2f} activates breakout momentum with {((w52h/price-1)*100):.1f}% upside to the 52-week high of ${w52h:.2f}.",
                f"Beta of {beta:.2f} provides amplified upside in risk-on market rotations — any broad rally produces outsized torque." if beta else f"Stage {stage} structure supports continuation if market conditions improve.",
                f"Analyst consensus target of ${target:.2f} provides a fundamental anchor — institutional re-rating above target drives further upside." if target else f"Technical strength in the PowerStack ({('aligned' if full_stack else 'unaligned')}) supports momentum continuation."
            ],
            "bear_cases": [
                f"Death Cross active (50 SMA ${s50:.2f} below 200 SMA ${s200:.2f}) — bearish macro structure remains headwind until a confirmed Golden Cross." if cross_type == "death" else f"ADX at {adx_val:.1f} signals {'weak' if adx_val < 20 else 'moderate'} trend conviction — susceptible to reversal.",
                f"Lower High pattern ({' → '.join([f'${v:.2f}' for v in lh_vals[-3:]])}) indicates distribution — sellers defending swing highs." if lh_det else f"RSI at {rsi_val:.1f} {'approaching overbought — mean reversion risk increases' if rsi_val >= 65 else 'in neutral zone — limited directional catalyst'}.",
                f"Break below support at ${support:.2f} exposes the ${w52l:.2f} 52-week low — a {((w52l/price-1)*100):.1f}% drawdown scenario."
            ],
            "catalysts": []
        }

    # ── Sector rotation ───────────────────────────────────────────────────
    try:
        rotation = get_rotation_data(ticker, sector)
    except Exception:
        rotation = {"sector": sector, "sector_signal": "No Data", "sector_signal_color": "gray",
                    "leaders": [], "laggers": [], "all_peers": [],
                    "ticker_rank": None, "total_peers": 0,
                    "golden_count": 0, "death_count": 0, "stage2_count": 0}

    # ── News confirmation ──────────────────────────────────────────────────
    try:
        news_data = fetch_news(ticker)
    except Exception:
        news_data = {"signal": {"label": "No Data", "color": "gray", "bull": 0, "bear": 0, "neutral": 0, "total": 0}, "headlines": [], "sources": []}

    def pct_vs(p, m): return f"{((p/m-1)*100):+.1f}%" if m else "N/A"

    result = {
        # Header
        "ticker":    ticker,
        "data_source": "schwab" if info.get("_source") == "schwab" else "yahoo",
        "company":   company,
        "sector":    sector,
        "industry":  industry,
        "price":     f"{price:,.2f}",
        "change_val": f"{abs(change_val):.2f}",
        "change_pct": f"{abs(change_pct):.2f}",
        "change_dir": "up" if change_val >= 0 else "down",
        "change_arrow": "▲" if change_val >= 0 else "▼",

        # TicScore
        "exempt":       exempt,
        "exempt_reason": exempt_reason or "",
        "tic_total":    tic_total,
        "tic_g":        tic_bd["G"],
        "tic_v":        tic_bd["V"],
        "tic_q":        tic_bd["Q"],
        "tic_s":        tic_bd["S"],
        "tic_pass":     tic_pass,
        "tic_color":    "green" if tic_total >= 70 else ("yellow" if tic_total >= 45 else "red"),

        # TicStage
        "stage":        stage,
        "stage_sub":    stage_sub,
        "stage_desc":   stage_desc,
        "stage_color":  stage_color,

        # ATR / ADX / RS
        "atr":      f"{atr_val:.2f}",
        "atr_pct":  f"{(atr_val/price*100):.1f}",
        "adx":      f"{adx_val:.1f}",
        "adx_str":  ("Very Strong" if adx_val >= 40 else ("Strong" if adx_val >= 25 else ("Developing" if adx_val >= 20 else "Weak"))),
        "di_plus":  f"{di_plus:.1f}",
        "di_minus": f"{di_minus:.1f}",
        "rs":       rs,
        "rs_label": ("Leader" if rs >= 80 else ("Laggard" if rs < 60 else "Neutral")),
        "rs_color": ("green" if rs >= 80 else ("red" if rs < 60 else "yellow")),
        "full_stack": full_stack,
        "cross_type": cross_type,

        # Key levels
        "support":    f"{support:,.2f}",
        "resistance": f"{resistance:,.2f}",
        "w52h":       f"{w52h:,.2f}",
        "w52l":       f"{w52l:,.2f}",
        "w52h_pct":   pct_vs(price, w52h),
        "w52l_pct":   pct_vs(price, w52l),

        # Market data
        "mkt_cap":   fmt_large(mkt_cap),
        "trail_pe":  f"{trail_pe:.1f}x" if trail_pe else "N/A",
        "fwd_pe":    f"{fwd_pe:.1f}x"   if fwd_pe  else "N/A",
        "beta":      f"{beta:.2f}" if beta else "N/A",
        "target":    f"{target:,.2f}" if target else "N/A",
        "target_pct": pct_vs(price, target) if target else "N/A",
        "rec_key":   rec_key,
        "avg_vol":   f"{avg_vol/1e6:.1f}M" if avg_vol else "N/A",
        "vol_today": f"{vol_today/1e6:.1f}M",
        "vol_ratio": f"{(vol_today/avg_vol):.1f}x" if avg_vol else "N/A",

        # MAs
        "e9":     f"{e9:,.2f}",   "e9_pct":  pct_vs(price, e9),
        "e21":    f"{e21:,.2f}",  "e21_pct": pct_vs(price, e21),
        "s50":    f"{s50:,.2f}",  "s50_pct": pct_vs(price, s50),
        "s150":   f"{s150:,.2f}", "s150_pct": pct_vs(price, s150),
        "s200":   f"{s200:,.2f}", "s200_pct": pct_vs(price, s200),

        # Indicators
        "rsi":      f"{rsi_val:.1f}",
        "rsi_label": ("Overbought" if rsi_val >= 70 else ("Oversold" if rsi_val <= 30 else ("Elevated" if rsi_val >= 60 else ("Depressed" if rsi_val <= 40 else "Neutral")))),
        "rsi_color": ("red" if rsi_val >= 70 else ("green" if rsi_val <= 30 else "yellow")),
        "macd":     f"{macd_val:.3f}",
        "sig":      f"{sig_val:.3f}",
        "hist":     f"{hist_val:+.3f}",
        "macd_dir": "Bullish" if macd_val > sig_val else "Bearish",
        "macd_color": "green" if macd_val > sig_val else "red",

        # Patterns
        "pocket_pivot": pocket_pivot,
        "lower_highs":  lh_det,
        "lh_vals":      [f"${v:.2f}" for v in lh_vals[-3:]],
        "active_gap":   active_gap,

        # Financials
        "quarterly": quarterly_rows,

        # AI thesis
        "company_bio": thesis_data.get("company_bio", ""),
        "tic_signal": thesis_data.get("tic_signal", ""),
        "tic_stock":  thesis_data.get("tic_stock", ""),
        "thesis":     thesis_data.get("thesis", []),
        "bull_cases": thesis_data.get("bull_cases", []),
        "bear_cases": thesis_data.get("bear_cases", []),
        "catalysts":  thesis_data.get("catalysts", []),
        # Rotation
        "rotation":       rotation,

        # News
        "news_signal":    news_data["signal"],
        "news_headlines": news_data["headlines"],
        "news_sources":   news_data["sources"],

        "ai_error":   ai_error,
        "has_ai":     bool(effective_key and not ai_error),
    }
    return result, None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    ticker = (request.args.get("ticker") or request.form.get("ticker") or "").strip().upper()
    api_key = (request.args.get("api_key") or request.form.get("api_key") or "").strip()
    if not ticker:
        return render_template("index.html", error="Please enter a ticker symbol.")
    result, err = analyze_ticker(ticker, api_key)
    if err:
        return render_template("index.html", error=err)
    return render_template("report.html", d=result)


@app.route("/healthcheck")
def healthcheck():
    """Quick diagnostic — shows env, package versions, and a live data test."""
    import yfinance as yf
    import pandas as pd
    import sys

    diag = {
        "python": sys.version,
        "yfinance": yf.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "schwab_token_found": any(
            __import__("os").path.exists(p) for p in [
                r"C:\Users\vince\Desktop\orb_bot\schwab_token.json",
                "/tmp/schwab_token.json"
            ]
        ),
        "schwab_env_var": bool(__import__("os").environ.get("SCHWAB_TOKEN_JSON")),
    }

    # Quick yfinance data test
    try:
        test = yf.Ticker("SPY").history(period="5d", interval="1d")
        diag["yfinance_test"] = "OK" if not test.empty else "EMPTY"
        diag["yfinance_rows"] = len(test)
    except Exception as e:
        diag["yfinance_test"] = f"FAIL: {e}"

    return jsonify(diag)


@app.route("/api/analyze")
def api_analyze():
    ticker  = request.args.get("ticker", "").strip().upper()
    api_key = request.args.get("api_key", "").strip()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    result, err = analyze_ticker(ticker, api_key)
    if err:
        return jsonify({"error": err}), 404
    return jsonify(result)


@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    tb = traceback.format_exc()
    # Show detailed error in response so we can debug
    return f"<pre style='background:#111;color:#ef4444;padding:20px;font-size:13px;'><b>Error:</b> {e}\n\n{tb}</pre>", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
