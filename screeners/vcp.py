# =============================================================================
# SCREEN: Volatility Contraction Pattern (VCP)
# Mark Minervini's signature setup — prior uptrend, multiple contractions,
# declining volume during contractions, breakout on expanding volume.
#
# *** PASTE YOUR PINE SCRIPT LOGIC HERE ***
# Replace the body of screen_vcp() with your translated logic.
# All parameters are in config.py under VCP = {...}
# =============================================================================

import numpy as np
import pandas as pd
from config import VCP


def screen_vcp(df: pd.DataFrame) -> dict:
    """
    Evaluate whether a ticker shows a VCP setup.

    Args:
        df: DataFrame with columns [open, high, low, close, volume], DatetimeIndex,
            sorted ascending, at least 100 rows.

    Returns:
        {
            "passed": bool,
            "score":  float (0.0 - 1.0, partial credit for near-setups),
            "detail": str  (human-readable reason)
        }
    """
    try:
        result = _run_vcp(df)
    except Exception as e:
        result = {"passed": False, "score": 0.0, "detail": f"Error: {e}"}
    return result


# -----------------------------------------------------------------------------
# IMPLEMENTATION — replace this with your Pine Script translation
# -----------------------------------------------------------------------------

def _run_vcp(df: pd.DataFrame) -> dict:
    cfg = VCP
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    avg_vol = volume.rolling(50).mean()

    # --- Step 1: Prior uptrend ---
    lookback = cfg["trend_lookback_days"]
    if len(df) < lookback + 30:
        return {"passed": False, "score": 0.0, "detail": "Insufficient history"}

    trend_start_price = close.iloc[-(lookback + 30)]
    trend_peak_price  = close.iloc[-30:].max()
    trend_gain_pct    = (trend_peak_price - trend_start_price) / trend_start_price * 100

    if trend_gain_pct < cfg["min_trend_gain_pct"]:
        return {
            "passed": False,
            "score":  max(0.0, trend_gain_pct / cfg["min_trend_gain_pct"] * 0.3),
            "detail": f"Uptrend only {trend_gain_pct:.1f}% (need {cfg['min_trend_gain_pct']}%)"
        }

    # --- Step 2: Identify contractions in the last 60 days ---
    recent = df.iloc[-60:].copy()
    contractions = _find_contractions(recent, cfg["max_contraction_range_pct"])

    if len(contractions) < cfg["num_contractions"]:
        score = 0.3 + (len(contractions) / cfg["num_contractions"]) * 0.3
        return {
            "passed": False,
            "score":  round(score, 2),
            "detail": f"Only {len(contractions)}/{cfg['num_contractions']} contractions found"
        }

    # --- Step 3: Volume dry-up during contractions ---
    vol_ok = _check_volume_dryup(recent, contractions, avg_vol.iloc[-60:], cfg["volume_dry_up_ratio"])

    # --- Step 4: Latest contraction is the tightest (contractions narrowing) ---
    ranges = [c["range_pct"] for c in contractions]
    narrowing = all(ranges[i] >= ranges[i+1] for i in range(len(ranges)-1))

    # --- Step 5: Breakout check (optional — today's close near pivot high) ---
    pivot_high = high.iloc[-cfg["pivot_lookback_days"]:].max()
    latest_close = close.iloc[-1]
    near_pivot = latest_close >= pivot_high * 0.98

    breakout_vol = volume.iloc[-1] / avg_vol.iloc[-1] if avg_vol.iloc[-1] > 0 else 0
    vol_expansion = breakout_vol >= cfg["breakout_volume_ratio"]

    # --- Composite score ---
    checks = [
        trend_gain_pct >= cfg["min_trend_gain_pct"],
        len(contractions) >= cfg["num_contractions"],
        narrowing,
        vol_ok,
        near_pivot,
        vol_expansion,
    ]
    score = sum(checks) / len(checks)
    passed = score >= 0.75  # passes if 4+ of 6 checks clear

    detail_parts = []
    if not narrowing:    detail_parts.append("contractions not narrowing")
    if not vol_ok:       detail_parts.append("volume not drying up")
    if not near_pivot:   detail_parts.append(f"price {latest_close:.2f} not near pivot {pivot_high:.2f}")
    if not vol_expansion:detail_parts.append(f"breakout vol {breakout_vol:.1f}x (need {cfg['breakout_volume_ratio']}x)")

    detail = "VCP setup ✓" if passed else " | ".join(detail_parts) or "Partial setup"

    return {"passed": passed, "score": round(score, 2), "detail": detail}


def _find_contractions(df: pd.DataFrame, max_range_pct: float) -> list[dict]:
    """
    Identify consolidation periods where high-low range contracts below threshold.
    Uses a rolling 10-day window to find tight consolidation zones.
    """
    contractions = []
    window = 10

    for i in range(window, len(df) - window, window):
        segment = df.iloc[i:i+window]
        seg_high = segment["high"].max()
        seg_low  = segment["low"].min()
        range_pct = (seg_high - seg_low) / seg_low * 100

        if range_pct <= max_range_pct:
            contractions.append({
                "start_idx": i,
                "end_idx":   i + window,
                "range_pct": range_pct,
                "high":      seg_high,
                "low":       seg_low,
            })

    return contractions


def _check_volume_dryup(df: pd.DataFrame, contractions: list, avg_vol: pd.Series, ratio: float) -> bool:
    """Check if volume during contractions is below the ratio threshold."""
    if not contractions:
        return False
    for c in contractions:
        seg_vol = df["volume"].iloc[c["start_idx"]:c["end_idx"]].mean()
        avg     = avg_vol.iloc[c["start_idx"]:c["end_idx"]].mean()
        if avg > 0 and seg_vol / avg >= ratio:
            return False  # volume not dried up in this contraction
    return True
