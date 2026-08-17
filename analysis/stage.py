from analysis.technical import ema, sma


def ticstage(df):
    """
    Stan Weinstein Stage Analysis using 150-day SMA.
    Returns: (stage_label, sub_label, description, color_class)
    """
    if len(df) < 200:
        return "Unknown", "Insufficient Data", "Need 200+ days of history", "gray"

    close     = df["Close"]
    sma150    = sma(close, 150)
    slope_20  = float(sma150.iloc[-1]) - float(sma150.iloc[-20])

    price     = float(close.iloc[-1])
    ma150     = float(sma150.iloc[-1])
    ema9_v    = float(ema(close, 9).iloc[-1])
    ema21_v   = float(ema(close, 21).iloc[-1])
    sma50_v   = float(sma(close, 50).iloc[-1])
    sma200_v  = float(sma(close, 200).iloc[-1])

    above_150   = price > ma150
    rising_150  = slope_20 > 0
    full_stack  = ema9_v > ema21_v > sma50_v
    golden_cross = sma50_v > sma200_v
    death_cross  = sma50_v < sma200_v

    if above_150 and rising_150:
        if full_stack and golden_cross:
            if price > sma50_v * 1.05:
                return "Stage 2B", "Extended / Advancing", "Extended above 50 SMA — trend intact but stretched", "green"
            return "Stage 2A", "Early Breakout", "Full stack bullish · Golden Cross confirmed", "green"
        elif full_stack:
            return "Stage 2A", "Advancing / Unconfirmed", "Stack aligned · Death Cross still active", "yellow"
        return "Stage 2", "Advancing", "Price above rising 150 SMA", "green"

    elif not above_150 and not rising_150:
        if price < ma150 * 0.90:
            return "Stage 4B", "Declining / Extended", "Deeply below falling 150 SMA — avoid longs", "red"
        return "Stage 4A", "Early Decline", "Price below falling 150 SMA — capital preservation", "red"

    elif not above_150 and rising_150:
        return "Stage 1", "Basing", "Flat/rising 150 SMA · price below · await breakout", "blue"

    else:
        return "Stage 3", "Topping", "150 SMA flattening · reduce exposure · tighten stops", "yellow"
