# =============================================================================
# SHARED UTILITY — Weinstein Weekly Stage Analysis (30-week SMA)
# =============================================================================
#
# All three rankers (stage, sepa, trade) must classify weekly stages
# identically or a stock could pass/fail different gates in different sheets.
#
# Canonical method (Weinstein 1988 + 2018–2024 seminars):
#   - Weekly OHLCV bars (not daily)
#   - 30-WEEK SMA — Weinstein's published exact specification
#   - NOT EMA40 — which moves faster and can flip during a 4-6 week VCP base,
#     producing false Stage 3 classifications while the weekly 30w SMA stays
#     solidly bullish.  EMA40 is the TradingView visual approximation;
#     30w SMA is the original Weinstein signal.
#
# Import this from any ranker:
#   from screeners.weekly_stage import to_weekly, get_weekly_stage_weinstein
#
# Return value from get_weekly_stage_weinstein:
#   (stage_int, label_str, sma30_value, vol_dry_in_base, breakout_vol_confirm)
#
#   stage_int:
#     0 = Unknown (insufficient history — < 32 weekly bars)
#     1 = W-S1 Accum  — price below 30w SMA but SMA turning up
#     2 = W-S2 ✓      — price above rising 30w SMA  (ideal — buy zone)
#     3 = W-S3 Dist   — price above flat/falling SMA (distribution — caution)
#     4 = W-S4 Decline — price below falling SMA      (avoid)
#
#   label_str: human-readable string matching stage_int values above
#   sma30_value: float — current value of the 30-week SMA
#   vol_dry_in_base: bool — recent 13-week avg volume < prior 13-week avg volume
#                   by ≥15% (institutional patience, right side of base)
#   breakout_vol_confirm: bool — latest weekly bar has volume ≥ 90% of 13-week
#                   volume high (institutional conviction behind the move)
#
# STAGE MAP (int → "Weekly Stage" column string):
#   Kept in sync with ranker_sepa._result_to_row → weekly_stage_labels dict.
#   0 → "Unknown"
#   1 → "W-S1 Accum"
#   2 → "W-S2 ✓"
#   3 → "W-S3 Dist"
#   4 → "W-S4 Decline"
# =============================================================================

import pandas as pd


# ── Public label map — used wherever an int stage needs to become a string ───
WEEKLY_STAGE_LABELS = {
    0: "Unknown",
    1: "W-S1 Accum",
    2: "W-S2 ✓",
    3: "W-S3 Dist",
    4: "W-S4 Decline",
}


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a daily OHLCV DataFrame to weekly (Friday close).

    Expects columns: open, high, low, close, volume (lower-case).
    Returns a weekly DataFrame with the same columns, or an empty DataFrame
    if resampling fails (e.g. index is not DatetimeIndex).
    """
    try:
        weekly = df.resample("W-FRI").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna(subset=["close"])
        return weekly[weekly["close"] > 0]
    except Exception:
        return pd.DataFrame()


def get_weekly_stage_weinstein(weekly_df: pd.DataFrame) -> tuple:
    """
    Classify a stock's weekly Weinstein stage using the canonical 30-week SMA.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly OHLCV with columns open/high/low/close/volume.
        Must have a DatetimeIndex.  Produced by to_weekly().

    Returns
    -------
    (stage, label, sma30_value, vol_dry, breakout_vol)

    stage : int
        0  = unknown (< 32 weekly bars)
        1  = W-S1 Accum  (below rising SMA — transitioning)
        2  = W-S2 ✓      (above rising SMA — ideal buy zone)
        3  = W-S3 Dist   (above flat/falling SMA — distribution)
        4  = W-S4 Decline (below falling SMA — avoid)

    label : str
        Human-readable stage string.  Matches WEEKLY_STAGE_LABELS.

    sma30_value : float
        Current value of the 30-week SMA (useful for chart annotation).

    vol_dry : bool
        True when right-side base volume dries up:
        avg weekly volume in the last 13 weeks < avg of the prior 13 weeks × 0.85.
        Signals sellers exhausted — stock is coiling for breakout.

    breakout_vol : bool
        True when the latest weekly bar's volume ≥ 90% of the 13-week volume max.
        Weinstein requires heavy institutional volume on the breakout week.

    Notes
    -----
    Slope is measured over 4 weekly bars (≈ 1 month).  Short-term oscillations
    (2-3 weeks) do not flip the slope reading — avoids false Stage-3 labels
    during brief consolidations inside an ongoing Stage-2 advance.

    Minimum 32 weekly bars required (30 for SMA + 2 for slope comparison).
    You need approximately 160 days of daily history to produce 32 weekly bars.
    """
    if weekly_df is None or len(weekly_df) < 32:
        return 0, "Unknown", 0.0, False, False

    close_w = weekly_df["close"]
    vol_w   = weekly_df["volume"]

    # ── 30-week SMA — Weinstein's exact specification ─────────────────────────
    sma30  = close_w.rolling(30).mean()
    price  = float(close_w.iloc[-1])
    ma_now = float(sma30.iloc[-1]) if not pd.isna(sma30.iloc[-1]) else 0.0

    if ma_now == 0.0:
        return 0, "Unknown", 0.0, False, False

    # ── Slope: 4-week comparison (≈ 1 calendar month) ────────────────────────
    ma_4w      = float(sma30.iloc[-4]) if len(sma30) >= 4 and not pd.isna(sma30.iloc[-4]) else ma_now
    ma_rising  = ma_now > ma_4w * 1.001   # must lift ≥ 0.1% to count as rising
    ma_falling = ma_now < ma_4w * 0.999

    above_ma = price > ma_now

    # ── Volume dry-up in base (right-side of base) ────────────────────────────
    # Weinstein (2018+ seminars): sellers exhausted when recent-13w avg volume
    # is ≥15% below prior-13w avg.  Confirms institutional patience / accumulation.
    if len(vol_w) >= 26:
        vol_recent = float(vol_w.iloc[-13:].mean())
        vol_prior  = float(vol_w.iloc[-26:-13].mean())
        vol_dry    = vol_recent < vol_prior * 0.85
    else:
        vol_dry = False

    # ── Breakout volume confirmation ──────────────────────────────────────────
    # Latest weekly bar should be the heaviest volume week in 13 weeks.
    # ≥90% of the 13-week max is the threshold (allows for minor shortfall).
    if len(vol_w) >= 13:
        vol_this    = float(vol_w.iloc[-1])
        vol_13w_max = float(vol_w.iloc[-13:].max())
        breakout_vol = vol_this >= vol_13w_max * 0.90
    else:
        breakout_vol = False

    # ── Stage classification ──────────────────────────────────────────────────
    if above_ma and ma_rising:
        return 2, "W-S2 ✓",     ma_now, vol_dry, breakout_vol
    if above_ma and not ma_rising:
        return 3, "W-S3 Dist",  ma_now, vol_dry, breakout_vol
    if not above_ma and ma_rising:
        return 1, "W-S1 Accum", ma_now, vol_dry, breakout_vol
    return 4, "W-S4 Decline",   ma_now, vol_dry, breakout_vol


# =============================================================================
# THEWRAP TA RULES — Weekly 10W / 20W / 40W EMA Decision Tree
# =============================================================================
#
# Two-path decision tree:
#   Path A — EMAs Converging (squeeze): EMAs narrowing toward each other,
#             producing compression that resolves in a strong directional move.
#   Path B — EMAs Trending (not converging): EMAs spread out, trend clearly
#             established; signal tracks where price sits in the EMA stack.
#
# Convergence definition (price-normalized, scale-agnostic):
#   spread_now_pct = |ema10w - ema40w| / price
#   Converging = spread > 2% of price (not sideways noise) AND
#                spread narrowed >15% vs 20 weeks ago.
#
# Signal codes (returned as first element of tuple):
#   TW_BULLISH   — price above 10W, 10W rising; bullish structure confirmed
#   TW_MAINTAIN  — full bull stack (px > 10W > 20W > 40W) AND 40W rising
#   TW_WAIT      — squeeze with no clear direction / pullback to 20W support
#   TW_FADING    — full bull stack but 40W flattening (aging/extended trend)
#   TW_CAUTIOUS  — below 10W or 20W; structure deteriorating
#   TW_EXIT_40W  — below 20W with 40W rolling over (structural warning)
#   TW_EXIT      — below all EMAs or below 40W (structural breakdown)
#   TW_NONE      — insufficient weekly data (< 42 complete weeks)
#
# Partial-week handling:
#   If the current week is incomplete (today is not Friday), the last weekly
#   bar is excluded so it doesn't distort EMA calculations.
#
# IMPORTANT: Uses EMA (exponential) not SMA, so 40W ≈ EMA200 daily.
#   This is SEPARATE from the Weinstein 30-week SMA — it uses shorter spans
#   that are more responsive to trend changes.  Both analyses are run in
#   parallel: Weinstein for structural stage, TheWrap for dynamic momentum.
# =============================================================================

# Human-readable labels for each TheWrap signal code
THEWRAP_LABELS: dict[str, str] = {
    "TW_BULLISH":  "🟢 TW: Bullish",
    "TW_MAINTAIN": "✅ TW: Maintain",
    "TW_WAIT":     "🟡 TW: Wait",
    "TW_FADING":   "🟡 TW: Fading",
    "TW_CAUTIOUS": "🟡 TW: Caution",
    "TW_EXIT_40W": "🔴 TW: 40W Break",
    "TW_EXIT":     "🔴 TW: Exit",
    "TW_NONE":     "— TW: No data",
}


def compute_thewrap_signal(weekly_df: pd.DataFrame) -> tuple:
    """
    Compute TheWrap TA Rules signal using weekly 10W / 20W / 40W EMAs.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly OHLCV produced by to_weekly().  Must have a DatetimeIndex.

    Returns
    -------
    (signal_code, signal_label, ema10w, ema20w, ema40w)

    signal_code  : str   — one of the TW_* codes above
    signal_label : str   — emoji + text label (from THEWRAP_LABELS)
    ema10w       : float — current 10-week EMA value
    ema20w       : float — current 20-week EMA value
    ema40w       : float — current 40-week EMA value
    """
    if weekly_df is None or len(weekly_df) < 45:
        return "TW_NONE", THEWRAP_LABELS["TW_NONE"], 0.0, 0.0, 0.0

    # ── Exclude incomplete current week (partial bar distorts EMAs) ───────────
    wdf = weekly_df.copy()
    if len(wdf) >= 2 and wdf.index[-1].weekday() != 4:  # 4 = Friday
        wdf = wdf.iloc[:-1]

    if len(wdf) < 42:
        return "TW_NONE", THEWRAP_LABELS["TW_NONE"], 0.0, 0.0, 0.0

    close_w = wdf["close"]
    px = float(close_w.iloc[-1])

    # ── Weekly EMAs ───────────────────────────────────────────────────────────
    ema10_s = close_w.ewm(span=10, adjust=False).mean()
    ema20_s = close_w.ewm(span=20, adjust=False).mean()
    ema40_s = close_w.ewm(span=40, adjust=False).mean()

    ema10w = float(ema10_s.iloc[-1])
    ema20w = float(ema20_s.iloc[-1])
    ema40w = float(ema40_s.iloc[-1])

    if ema10w <= 0 or ema20w <= 0 or ema40w <= 0 or px <= 0:
        return "TW_NONE", THEWRAP_LABELS["TW_NONE"], 0.0, 0.0, 0.0

    # ── EMA slopes (4-week lookback ≈ 1 calendar month) ──────────────────────
    _lag = 4
    def _at(s: pd.Series, lag: int) -> float:
        idx = -(lag + 1)
        return float(s.iloc[idx]) if len(s) >= abs(idx) else float(s.iloc[0])

    ema10w_4ago = _at(ema10_s, _lag)
    ema20w_4ago = _at(ema20_s, _lag)
    ema40w_4ago = _at(ema40_s, _lag)

    ema10w_rising = ema10w > ema10w_4ago * 1.001   # lifted ≥ 0.1% in 4 weeks
    ema20w_rising = ema20w > ema20w_4ago * 1.001
    ema40w_rising = ema40w > ema40w_4ago * 1.001

    # ── Convergence detection (price-normalised spread, 20-week lookback) ─────
    _clag = 20
    if len(ema10_s) >= _clag + 1 and len(ema40_s) >= _clag + 1:
        ema10_lag20 = _at(ema10_s, _clag)
        ema40_lag20 = _at(ema40_s, _clag)
        px_lag20    = float(close_w.iloc[-(_clag + 1)]) if len(close_w) >= _clag + 1 else px

        spread_now = abs(ema10w    - ema40w)    / px          if px > 0        else 0.0
        spread_old = abs(ema10_lag20 - ema40_lag20) / px_lag20 if px_lag20 > 0 else 0.0

        # Converging: spread meaningful (>2% of price) AND narrowed >15% in 20w
        converging = (
            spread_now > 0.02           and   # not already flat / inside noise
            spread_old > spread_now     and   # was wider 20 weeks ago
            spread_now < spread_old * 0.85    # narrowed by >15%
        )
    else:
        converging = False

    # =========================================================================
    # PATH A — EMA Squeeze (Converging)
    # =========================================================================
    if converging:
        if px > ema10w:
            if ema10w_rising:
                return (
                    "TW_BULLISH",
                    "🟢 TW: Bullish Squeeze — EMAs converging, price > 10W↑",
                    ema10w, ema20w, ema40w,
                )
            else:
                return (
                    "TW_WAIT",
                    "🟡 TW: Wait — squeeze above 10W, no direction yet",
                    ema10w, ema20w, ema40w,
                )
        elif px > ema40w:
            return (
                "TW_CAUTIOUS",
                "🟡 TW: Caution — between 10W/40W in squeeze",
                ema10w, ema20w, ema40w,
            )
        else:
            return (
                "TW_EXIT",
                "🔴 TW: Exit — below all EMAs in squeeze (distribution)",
                ema10w, ema20w, ema40w,
            )

    # =========================================================================
    # PATH B — Trending (Not Converging)
    # =========================================================================

    # ── Full bull stack: price > 10W > 20W > 40W ──────────────────────────────
    if px > ema10w and ema10w > ema20w and ema20w > ema40w:
        if ema40w_rising:
            return (
                "TW_MAINTAIN",
                "✅ TW: Maintain — full bull stack, 40W rising",
                ema10w, ema20w, ema40w,
            )
        else:
            return (
                "TW_FADING",
                "🟡 TW: Fading — bull stack intact but 40W flattening (aging trend)",
                ema10w, ema20w, ema40w,
            )

    # ── Price above 10W but not full stack ────────────────────────────────────
    if px > ema10w:
        if ema10w > ema20w and ema10w_rising:
            return (
                "TW_BULLISH",
                "🟢 TW: Bullish — price > 10W, 10W > 20W and rising",
                ema10w, ema20w, ema40w,
            )
        else:
            return (
                "TW_WAIT",
                "🟡 TW: Wait — above 10W but EMA stack unordered",
                ema10w, ema20w, ema40w,
            )

    # ── Price below 10W: between 10W and 20W ─────────────────────────────────
    if px > ema20w:
        if ema20w_rising and ema40w_rising:
            return (
                "TW_WAIT",
                "🟡 TW: Wait — testing 20W support, 40W still rising",
                ema10w, ema20w, ema40w,
            )
        else:
            return (
                "TW_CAUTIOUS",
                "🟡 TW: Caution — below 10W, 20W/40W losing momentum",
                ema10w, ema20w, ema40w,
            )

    # ── Price below 20W: between 20W and 40W ─────────────────────────────────
    if px > ema40w:
        if ema40w_rising:
            return (
                "TW_CAUTIOUS",
                "🟡 TW: Caution — below 20W, testing 40W support",
                ema10w, ema20w, ema40w,
            )
        else:
            return (
                "TW_EXIT_40W",
                "🔴 TW: 40W Break — below 20W with 40W rolling over",
                ema10w, ema20w, ema40w,
            )

    # ── Price below 40W — structural breakdown ────────────────────────────────
    return (
        "TW_EXIT",
        "🔴 TW: Exit — below all weekly EMAs (structural breakdown)",
        ema10w, ema20w, ema40w,
    )
