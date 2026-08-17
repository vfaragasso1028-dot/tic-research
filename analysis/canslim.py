import numpy as np

EXEMPT_SECTORS = {"Financial Services", "Real Estate", "Energy", "Basic Materials", "Utilities"}
EXEMPT_INDUSTRIES = {
    "Banks", "REITs", "Insurance", "Asset Management", "Gold", "Silver",
    "Oil & Gas", "Mining", "Utilities Regulated", "Marine Shipping", "Biotechnology"
}

def is_exempt(info):
    sector   = info.get("sector", "")
    industry = info.get("industry", "")
    for ex in EXEMPT_SECTORS:
        if ex.lower() in sector.lower():
            return True, sector
    for ex in EXEMPT_INDUSTRIES:
        if ex.lower() in industry.lower():
            return True, industry
    return False, None

def safe_pct(a, b):
    try:
        if b and b != 0:
            return (float(a) - float(b)) / abs(float(b)) * 100
    except Exception:
        pass
    return None

def ticscore(info, quarterly_financials, annual_financials):
    """
    100-pt CAN SLIM score.
    G=Growth(40) V=Supply&Demand(20) Q=Quality(20) S=Sponsorship(20)
    """
    G = V = Q = S = 0
    detail = {}

    try:
        # ── G: GROWTH (40) ──────────────────────────────
        eps_q = None
        try:
            if "Basic EPS" in quarterly_financials.index:
                eps_q = quarterly_financials.loc["Basic EPS"]
        except Exception:
            pass

        # G1: Current qtr EPS YoY ≥25% (+10)
        try:
            if eps_q is not None and len(eps_q) >= 5:
                pct = safe_pct(eps_q.iloc[0], eps_q.iloc[4])
                detail["eps_q_yoy"] = pct
                if pct is not None:
                    G += 10 if pct >= 25 else (5 if pct >= 10 else 0)
        except Exception:
            pass

        # G2: Current qtr Revenue YoY ≥25% (+10)
        try:
            rev_key = next((k for k in quarterly_financials.index if "Revenue" in k), None)
            if rev_key:
                rq = quarterly_financials.loc[rev_key]
                if len(rq) >= 5:
                    pct = safe_pct(rq.iloc[0], rq.iloc[4])
                    detail["rev_q_yoy"] = pct
                    if pct is not None:
                        G += 10 if pct >= 25 else (5 if pct >= 10 else 0)
        except Exception:
            pass

        # G3: Annual EPS avg growth ≥25% (+10)
        try:
            if "Basic EPS" in annual_financials.index:
                eps_a = annual_financials.loc["Basic EPS"]
                if len(eps_a) >= 3:
                    growths = [safe_pct(eps_a.iloc[i], eps_a.iloc[i+1]) for i in range(min(3, len(eps_a)-1))]
                    growths = [g for g in growths if g is not None]
                    avg = np.mean(growths) if growths else None
                    detail["eps_annual_avg"] = avg
                    if avg is not None:
                        G += 10 if avg >= 25 else (5 if avg >= 15 else 0)
        except Exception:
            pass

        # G4: EPS acceleration (+10)
        try:
            if eps_q is not None and len(eps_q) >= 8:
                yoy = [safe_pct(eps_q.iloc[i], eps_q.iloc[i+4]) for i in range(3)]
                yoy = [g for g in yoy if g is not None]
                detail["eps_accelerating"] = len(yoy) >= 2 and yoy[0] > yoy[1]
                if len(yoy) >= 3 and yoy[0] > yoy[1] > yoy[2]:
                    G += 10
                elif len(yoy) >= 2 and yoy[0] > yoy[1]:
                    G += 5
        except Exception:
            pass

        # ── V: SUPPLY & DEMAND (20) ──────────────────────
        # V1: Float size (+10)
        try:
            fs = info.get("floatShares")
            detail["float_shares"] = fs
            if fs:
                V += 10 if fs < 50e6 else (6 if fs < 150e6 else (3 if fs < 500e6 else 0))
        except Exception:
            pass

        # V2: Relative volume (+10)
        try:
            av  = info.get("averageVolume")
            av10 = info.get("averageVolume10days")
            if av and av10 and av > 0:
                rv = av10 / av
                detail["rel_vol"] = rv
                V += 10 if rv >= 1.5 else (6 if rv >= 1.1 else (3 if rv >= 0.9 else 0))
        except Exception:
            pass

        # ── Q: QUALITY (20) ─────────────────────────────
        # Q1: ROE ≥17% (+10)
        try:
            roe = info.get("returnOnEquity")
            detail["roe"] = roe * 100 if roe else None
            if roe:
                Q += 10 if roe >= 0.17 else (5 if roe >= 0.10 else 0)
        except Exception:
            pass

        # Q2: Profit margin (+10)
        try:
            pm = info.get("profitMargins")
            detail["profit_margin"] = pm * 100 if pm else None
            if pm:
                Q += 10 if pm >= 0.15 else (6 if pm >= 0.05 else (3 if pm > 0 else 0))
        except Exception:
            pass

        # ── S: SPONSORSHIP (20) ──────────────────────────
        # S1: Institutional ownership (+10)
        try:
            ip = info.get("institutionPercentHeld")
            detail["inst_pct"] = ip * 100 if ip else None
            if ip:
                S += 10 if ip >= 0.60 else (6 if ip >= 0.40 else (3 if ip >= 0.20 else 0))
        except Exception:
            pass

        # S2: Analyst consensus (+10)
        try:
            rec = info.get("recommendationMean")  # 1=Strong Buy .. 5=Sell
            detail["analyst_rec"] = rec
            if rec:
                S += 10 if rec <= 1.5 else (8 if rec <= 2.0 else (5 if rec <= 2.5 else (2 if rec <= 3.0 else 0)))
        except Exception:
            pass

    except Exception as e:
        detail["error"] = str(e)

    total  = G + V + Q + S
    passed = total >= 50
    return total, {"G": G, "V": V, "Q": Q, "S": S}, passed, detail
