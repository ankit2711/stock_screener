# =============================================================================
# SCREEN: 52-Week High Breakout
# Price at or breaking above its 52-week high — one of the strongest momentum signals
#
# *** PASTE YOUR PINE SCRIPT LOGIC HERE ***
# =============================================================================

import pandas as pd
from config import HIGH_52W


def screen_52w_high(df: pd.DataFrame) -> dict:
    try:
        return _run_52w_high(df)
    except Exception as e:
        return {"passed": False, "score": 0.0, "detail": f"Error: {e}"}


def _run_52w_high(df: pd.DataFrame) -> dict:
    cfg = HIGH_52W
    close = df["close"]
    high  = df["high"]

    lookback = min(cfg["lookback_days"], len(df) - 1)
    if lookback < 50:
        return {"passed": False, "score": 0.0, "detail": "Insufficient history"}

    yearly_high  = high.iloc[-lookback:].max()
    latest_close = close.iloc[-1]
    pct_from_high = (yearly_high - latest_close) / yearly_high * 100

    # 50-day MA check
    ma50 = close.rolling(50).mean().iloc[-1]
    pct_above_ma50 = (latest_close - ma50) / ma50 * 100 if ma50 > 0 else 0

    within_high   = pct_from_high  <= cfg["within_pct"]
    above_ma50    = pct_above_ma50 >= cfg["min_above_50dma_pct"]
    new_high      = latest_close   >= yearly_high  # actual breakout

    score = (
        (0.5 if within_high else max(0, 0.5 - pct_from_high / 20)) +
        (0.3 if above_ma50  else 0) +
        (0.2 if new_high    else 0)
    )
    passed = within_high and above_ma50

    detail = (
        f"52W high ✓ | {pct_from_high:.1f}% from high | {pct_above_ma50:.1f}% above 50MA"
        if passed
        else f"{pct_from_high:.1f}% from 52W high ({yearly_high:.2f})"
    )

    return {"passed": passed, "score": round(score, 2), "detail": detail}
