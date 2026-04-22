# =============================================================================
# SCREEN: Volume Breakout
# Price moves significantly on unusually high volume — institutional accumulation
#
# *** PASTE YOUR PINE SCRIPT LOGIC HERE ***
# =============================================================================

import pandas as pd
from config import VOLUME_BREAKOUT


def screen_volume_breakout(df: pd.DataFrame) -> dict:
    try:
        return _run_volume_breakout(df)
    except Exception as e:
        return {"passed": False, "score": 0.0, "detail": f"Error: {e}"}


def _run_volume_breakout(df: pd.DataFrame) -> dict:
    cfg = VOLUME_BREAKOUT
    close  = df["close"]
    high   = df["high"]
    volume = df["volume"]

    avg_vol_days = cfg["avg_volume_days"]
    if len(df) < avg_vol_days + 5:
        return {"passed": False, "score": 0.0, "detail": "Insufficient history"}

    avg_vol      = volume.rolling(avg_vol_days).mean()
    latest_vol   = volume.iloc[-1]
    latest_avg   = avg_vol.iloc[-2]          # use previous day's avg to avoid lookahead
    vol_ratio    = latest_vol / latest_avg if latest_avg > 0 else 0

    # Price change on breakout day
    prev_close   = close.iloc[-2]
    latest_close = close.iloc[-1]
    price_chg    = (latest_close - prev_close) / prev_close * 100

    # Price near N-day high
    lookback     = cfg["price_high_lookback_days"]
    n_day_high   = high.iloc[-lookback:].max()
    near_high    = latest_close >= n_day_high * (1 - cfg["price_near_high_pct"] / 100)

    checks = [
        vol_ratio    >= cfg["breakout_volume_ratio"],
        price_chg    >= cfg["min_price_change_pct"],
        near_high,
        latest_vol   >= 100_000,           # minimum absolute volume floor
    ]
    score  = sum(checks) / len(checks)
    passed = checks[0] and checks[1]       # volume + price move are mandatory

    detail = (
        f"Vol breakout ✓ | {vol_ratio:.1f}x avg | +{price_chg:.1f}%"
        if passed
        else f"vol {vol_ratio:.1f}x (need {cfg['breakout_volume_ratio']}x) | Δ{price_chg:+.1f}%"
    )

    return {"passed": passed, "score": round(score, 2), "detail": detail}
