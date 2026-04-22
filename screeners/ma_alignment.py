# =============================================================================
# SCREEN: Moving Average Alignment
# Bullish stack: price > 20 MA > 50 MA > 200 MA, with 50MA sloping up
#
# *** PASTE YOUR PINE SCRIPT LOGIC HERE ***
# =============================================================================

import pandas as pd
from config import MA_ALIGNMENT


def screen_ma_alignment(df: pd.DataFrame) -> dict:
    try:
        return _run_ma_alignment(df)
    except Exception as e:
        return {"passed": False, "score": 0.0, "detail": f"Error: {e}"}


def _run_ma_alignment(df: pd.DataFrame) -> dict:
    cfg   = MA_ALIGNMENT
    close = df["close"]

    if len(df) < cfg["long_ma"] + 10:
        return {"passed": False, "score": 0.0, "detail": "Insufficient history for 200MA"}

    ma_short  = close.rolling(cfg["short_ma"]).mean()
    ma_mid    = close.rolling(cfg["mid_ma"]).mean()
    ma_long   = close.rolling(cfg["long_ma"]).mean()

    price      = close.iloc[-1]
    short_val  = ma_short.iloc[-1]
    mid_val    = ma_mid.iloc[-1]
    long_val   = ma_long.iloc[-1]

    # 50MA slope over last 10 days
    mid_10d_ago   = ma_mid.iloc[-11]
    slope_pct     = (mid_val - mid_10d_ago) / mid_10d_ago * 100 if mid_10d_ago > 0 else 0

    checks = []
    if cfg["require_price_above_short"]: checks.append(("price>20MA", price > short_val))
    if cfg["require_short_above_mid"]:   checks.append(("20MA>50MA",  short_val > mid_val))
    if cfg["require_mid_above_long"]:    checks.append(("50MA>200MA", mid_val > long_val))
    checks.append(("50MA slope", slope_pct >= cfg["min_slope_pct"]))

    n_passed = sum(v for _, v in checks)
    score    = n_passed / len(checks)
    passed   = all(v for _, v in checks)

    if passed:
        detail = f"MA stack ✓ | {price:.2f} > {short_val:.2f} > {mid_val:.2f} > {long_val:.2f} | slope +{slope_pct:.1f}%"
    else:
        failed = [label for label, v in checks if not v]
        detail = "Failed: " + ", ".join(failed)

    return {"passed": passed, "score": round(score, 2), "detail": detail}
