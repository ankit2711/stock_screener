# =============================================================================
# SCREEN: Darvas Box Breakout
# Nicolas Darvas's method — price consolidates in a box, then breaks out
# on high volume.
#
# *** PASTE YOUR PINE SCRIPT LOGIC HERE ***
# Replace the body of _run_darvas() with your translated logic.
# All parameters are in config.py under DARVAS = {...}
# =============================================================================

import pandas as pd
from config import DARVAS


def screen_darvas(df: pd.DataFrame) -> dict:
    """
    Evaluate whether a ticker is breaking out of a Darvas Box.

    Args:
        df: DataFrame with columns [open, high, low, close, volume], DatetimeIndex.

    Returns:
        {"passed": bool, "score": float, "detail": str}
    """
    try:
        return _run_darvas(df)
    except Exception as e:
        return {"passed": False, "score": 0.0, "detail": f"Error: {e}"}


# -----------------------------------------------------------------------------
# IMPLEMENTATION — replace this with your Pine Script translation
# -----------------------------------------------------------------------------

def _run_darvas(df: pd.DataFrame) -> dict:
    cfg = DARVAS
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    if len(df) < cfg["box_lookback_days"] + 20:
        return {"passed": False, "score": 0.0, "detail": "Insufficient history"}

    avg_vol = volume.rolling(20).mean()

    # --- Find the most recent Darvas Box ---
    box = _find_latest_box(df, cfg)
    if box is None:
        return {"passed": False, "score": 0.1, "detail": "No valid Darvas box found"}

    box_top    = box["top"]
    box_bottom = box["bottom"]
    box_days   = box["duration_days"]

    # --- Check minimum box duration ---
    if box_days < cfg["min_box_duration_days"]:
        return {
            "passed": False,
            "score":  0.2,
            "detail": f"Box too short: {box_days}d (need {cfg['min_box_duration_days']}d)"
        }

    # --- Check for breakout: today's close above box top ---
    latest_close = close.iloc[-1]
    latest_vol   = volume.iloc[-1]
    latest_avg   = avg_vol.iloc[-1]

    above_box    = latest_close > box_top * (1 + cfg["box_top_tolerance_pct"] / 100)
    vol_ratio    = latest_vol / latest_avg if latest_avg > 0 else 0
    vol_confirm  = vol_ratio >= cfg["breakout_volume_ratio"]

    # --- Score ---
    checks = [
        box_days >= cfg["min_box_duration_days"],
        above_box,
        vol_confirm,
        latest_close > box_bottom,   # not collapsed into box
    ]
    score = sum(checks) / len(checks)
    passed = above_box and vol_confirm

    detail_parts = []
    if not above_box:   detail_parts.append(f"price {latest_close:.2f} not above box top {box_top:.2f}")
    if not vol_confirm: detail_parts.append(f"vol {vol_ratio:.1f}x avg (need {cfg['breakout_volume_ratio']}x)")

    detail = (
        f"Darvas breakout ✓ | box {box_top:.2f}-{box_bottom:.2f} | vol {vol_ratio:.1f}x"
        if passed
        else " | ".join(detail_parts) or f"Box found ({box_top:.2f}-{box_bottom:.2f}), no breakout yet"
    )

    return {"passed": passed, "score": round(score, 2), "detail": detail}


def _find_latest_box(df: pd.DataFrame, cfg: dict) -> dict | None:
    """
    Scan backwards from the most recent data to find the latest Darvas Box.
    A box is defined as a period where:
    - The high doesn't exceed the box top (within tolerance)
    - The low doesn't go below the box bottom (within tolerance)
    """
    lookback = cfg["box_lookback_days"]
    tol_top  = cfg["box_top_tolerance_pct"] / 100
    tol_bot  = cfg["box_bottom_tolerance_pct"] / 100

    # Start from recent history (exclude the last 3 days — the potential breakout)
    segment = df.iloc[-(lookback + 3):-3]
    if len(segment) < cfg["min_box_duration_days"]:
        return None

    box_top    = segment["high"].max()
    box_bottom = segment["low"].min()

    # Validate box: most days should stay within tolerance of top/bottom
    within_top    = (segment["high"] <= box_top * (1 + tol_top)).mean()
    within_bottom = (segment["low"]  >= box_bottom * (1 - tol_bot)).mean()

    if within_top < 0.85 or within_bottom < 0.85:
        return None  # too volatile — not a proper box

    # Find actual box duration (consecutive days contained in the box)
    duration = 0
    for i in range(len(segment) - 1, -1, -1):
        row = segment.iloc[i]
        if row["high"] <= box_top * (1 + tol_top) and row["low"] >= box_bottom * (1 - tol_bot):
            duration += 1
        else:
            break

    return {
        "top":           box_top,
        "bottom":        box_bottom,
        "duration_days": duration,
    }
