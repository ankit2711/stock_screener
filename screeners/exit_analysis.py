# =============================================================================
# EXIT ANALYSIS ENGINE — TheWrap TA Rules (weekly 10W / 20W / 40W EMAs)
# =============================================================================
#
# PHILOSOPHY:
#   Exit decisions are driven entirely by TheWrap two-path EMA decision tree.
#   Weekly 10W / 20W / 40W EMAs tell you exactly where the trend stands.
#   All signals are gain-aware and reference concrete EMA price levels.
#
# HOW IT WORKS:
#   Path A — EMAs Converging (squeeze): narrowing spreads signal compression
#             before a directional move — direction determines action.
#   Path B — EMAs Trending: price position within the EMA stack determines
#             hold / reduce / exit decision.
#
# SIGNAL PRIORITY (urgency 0→100):
#   TW_EXIT     (90)  — below all weekly EMAs → EXIT FULL POSITION
#   TW_EXIT_40W (75)  — below 20W + 40W turning down → REDUCE 50-75%
#   TW_CAUTIOUS (55)  — below 10W/20W EMA → TIGHTEN STOP
#   TW_FADING   (40)  — full bull stack but 40W flattening → RAISE STOP
#   TW_WAIT     (20)  — testing 20W support → WATCH
#   TW_BULLISH  ( 8)  — above 10W, rising → HOLD
#   TW_MAINTAIN ( 5)  — full bull stack + 40W rising → HOLD (ideal)
#
# Only run on held positions from Om-Holdings. Not a universe scan.
# =============================================================================

from __future__ import annotations
import logging
import pandas as pd
from dataclasses import dataclass

from screeners.weekly_stage import compute_thewrap_signal

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class ExitAnalysisResult:
    # ── Primary TheWrap signal ─────────────────────────────────────────────────
    signal_code:   str  = "TW_NONE"   # TW_MAINTAIN / TW_BULLISH / TW_WAIT /
    signal_label:  str  = "— No data" # TW_FADING / TW_CAUTIOUS / TW_EXIT_40W /
    urgency_score: int  = 0           # TW_EXIT / TW_NONE
    action:        str  = "—"         # concrete recommended action

    # ── EMA reference values (for stop-level context) ─────────────────────────
    ema10w:       float = 0.0    # current 10-week EMA
    ema20w:       float = 0.0    # current 20-week EMA
    ema40w:       float = 0.0    # current 40-week EMA

    # ── Price distance from each EMA (positive = above, negative = below) ─────
    dist_10w_pct: float = 0.0    # % from 10W EMA
    dist_20w_pct: float = 0.0    # % from 20W EMA
    dist_40w_pct: float = 0.0    # % from 40W EMA

    # ── 40W EMA velocity — key trend health indicator ─────────────────────────
    ema40w_slope: float = 0.0    # % change of 40W EMA over last 8 weeks
                                  # >+1% = accelerating up | <-1% = falling fast

    # ── Holdings context ──────────────────────────────────────────────────────
    gain_pct:     float = 0.0    # % gain vs buy price (0 if unknown)
    has_buy_price: bool = False   # True if buy price in Om-Holdings


# =============================================================================
# URGENCY BASE SCORES
# =============================================================================

_URGENCY_BASE: dict[str, int] = {
    "TW_EXIT":     90,
    "TW_EXIT_40W": 75,
    "TW_CAUTIOUS": 55,
    "TW_FADING":   40,
    "TW_WAIT":     20,
    "TW_BULLISH":   8,
    "TW_MAINTAIN":  5,
    "TW_NONE":      0,
}

# Severity order for sorting (lower index = more urgent)
SEVERITY_ORDER: list[str] = [
    "TW_EXIT", "TW_EXIT_40W", "TW_CAUTIOUS",
    "TW_FADING", "TW_WAIT", "TW_BULLISH", "TW_MAINTAIN", "TW_NONE",
]


def exit_severity(signal_code: str) -> int:
    """Return numeric severity for sorting (lower = more urgent)."""
    try:
        return SEVERITY_ORDER.index(signal_code) + 1
    except ValueError:
        return len(SEVERITY_ORDER) + 1


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def analyse_exit(
    weekly_df:     pd.DataFrame | None,
    holdings_info: dict | None = None,
) -> ExitAnalysisResult:
    """
    Compute TheWrap-based exit analysis for a single held position.

    Args:
        weekly_df:     Weekly OHLCV DataFrame from to_weekly(). Needs ≥ 45 bars.
        holdings_info: Dict from holdings_reader — contains buy_price, portfolio,
                       qty, name. Pass None if buy price is unknown.

    Returns:
        ExitAnalysisResult with signal, urgency, action, and EMA context.
    """
    er = ExitAnalysisResult()

    if weekly_df is None or len(weekly_df) < 45:
        return er

    # ── TheWrap signal (10W / 20W / 40W EMA decision tree) ───────────────────
    try:
        tw_code, tw_label, ema10w, ema20w, ema40w = compute_thewrap_signal(weekly_df)
    except Exception as e:
        logger.debug(f"TheWrap computation failed: {e}")
        return er

    er.signal_code  = tw_code
    er.signal_label = tw_label
    er.ema10w       = ema10w
    er.ema20w       = ema20w
    er.ema40w       = ema40w

    # ── Extended metrics — compute directly from weekly data ──────────────────
    try:
        wdf = weekly_df.copy()
        if len(wdf) >= 2 and wdf.index[-1].weekday() != 4:   # exclude partial week
            wdf = wdf.iloc[:-1]

        close_w = wdf["close"]
        px = float(close_w.iloc[-1])   # last complete week's close

        # EMA distances (signed %)
        if ema10w > 0:
            er.dist_10w_pct = round((px - ema10w) / ema10w * 100, 1)
        if ema20w > 0:
            er.dist_20w_pct = round((px - ema20w) / ema20w * 100, 1)
        if ema40w > 0:
            er.dist_40w_pct = round((px - ema40w) / ema40w * 100, 1)

        # 40W EMA slope: % change over last 8 weeks (velocity of the longest trend)
        ema40_s    = close_w.ewm(span=40, adjust=False).mean()
        ema40_now  = float(ema40_s.iloc[-1])
        ema40_8ago = float(ema40_s.iloc[-9]) if len(ema40_s) >= 9 else ema40_now
        if ema40_8ago > 0:
            er.ema40w_slope = round((ema40_now - ema40_8ago) / ema40_8ago * 100, 2)

    except Exception as e:
        logger.debug(f"Extended metrics failed: {e}")
        px = 0.0

    # ── Gain % from holdings info ─────────────────────────────────────────────
    # Priority 1: Change% column read directly from the Om-Holdings sheet.
    #             The sheet already tracks this accurately (handles corp actions,
    #             partial fills, FIFO cost etc.) — trust it over recomputation.
    # Priority 2: Compute from buy_price vs current price (fallback when the
    #             Change% column is absent or blank in the sheet).
    if holdings_info:
        change_pct = holdings_info.get("change_pct")
        if change_pct is not None:
            er.gain_pct      = change_pct
            er.has_buy_price = True
        elif px > 0:
            buy_price = holdings_info.get("buy_price", 0.0)
            if buy_price and buy_price > 0:
                er.gain_pct      = round((px - buy_price) / buy_price * 100, 1)
                er.has_buy_price = True

    # ── Urgency score ─────────────────────────────────────────────────────────
    er.urgency_score = _compute_urgency(er)

    # ── Concrete action recommendation ────────────────────────────────────────
    er.action = _compute_action(er)

    return er


# =============================================================================
# URGENCY SCORE (0–100)
# =============================================================================

def _compute_urgency(er: ExitAnalysisResult) -> int:
    """
    Urgency = base score for signal + modifiers for slope velocity and gain context.

    40W slope modifiers (velocity of the most important EMA):
      slope < -1.5%  → +10  (accelerating breakdown)
      slope < -0.5%  → +5   (clearly falling)
      slope <  0.0%  → +2   (starting to turn down)
      slope >  1.0%  → -4   (strong uptrend, less urgent)

    Holdings gain modifiers:
      gain < -15%    → +15  (significant loss — exit immediately)
      gain <  -5%    → +10  (in the red — thesis broken)
      gain <   0%    → +5   (slightly underwater)
      gain > 100%    → +8   (large profit to protect aggressively)
      gain >  50%    → +4   (meaningful profit)
    """
    base = _URGENCY_BASE.get(er.signal_code, 0)
    mod  = 0

    # 40W slope velocity
    if er.ema40w_slope < -1.5:
        mod += 10
    elif er.ema40w_slope < -0.5:
        mod += 5
    elif er.ema40w_slope < 0.0:
        mod += 2
    elif er.ema40w_slope > 1.0:
        mod -= 4   # rising trend reduces urgency of mild signals

    # Holdings gain modifiers
    if er.has_buy_price:
        if er.gain_pct < -15:
            mod += 15
        elif er.gain_pct < -5:
            mod += 10
        elif er.gain_pct < 0:
            mod += 5
        elif er.gain_pct > 100:
            mod += 8
        elif er.gain_pct > 50:
            mod += 4

    return min(100, max(0, base + mod))


# =============================================================================
# ACTION RECOMMENDATION (gain-aware, EMA-referenced)
# =============================================================================

def _compute_action(er: ExitAnalysisResult) -> str:
    """
    Generate a concrete, gain-aware action string with EMA distance context.

    Format: VERB — context. EMA reference (distance %).
    """
    sc   = er.signal_code
    gain = er.gain_pct if er.has_buy_price else None
    sl   = er.ema40w_slope
    d10  = er.dist_10w_pct
    d20  = er.dist_20w_pct
    d40  = er.dist_40w_pct

    slope_ctx = f"40W slope {sl:+.1f}%/8wk"

    # ── TW_MAINTAIN — full bull stack, 40W rising ─────────────────────────────
    if sc == "TW_MAINTAIN":
        if gain is not None and gain > 100:
            return (f"HOLD CORE, TRIM 25% — Protect {gain:+.0f}% gain. "
                    f"Trail stop: 10W EMA ({d10:+.1f}%)")
        if gain is not None and gain > 50:
            return (f"HOLD — Full bull stack. Trail stop to 10W EMA ({d10:+.1f}%)")
        return f"HOLD — Full bull stack, 40W rising. Trail stop: 10W EMA ({d10:+.1f}%)"

    # ── TW_BULLISH — above 10W, bullish structure ─────────────────────────────
    if sc == "TW_BULLISH":
        if gain is not None and gain < 0:
            return (f"WATCH CLOSELY — Bullish structure but {gain:+.0f}% underwater. "
                    f"Hard stop below 20W EMA ({d20:+.1f}%)")
        if gain is not None and gain > 50:
            return f"HOLD — Bullish. Trail stop to 20W EMA ({d20:+.1f}%). Protect {gain:+.0f}% gain"
        return f"HOLD — Bullish. Key support: 20W EMA ({d20:+.1f}%). Stop below 20W"

    # ── TW_WAIT — testing 20W or squeeze with no direction ───────────────────
    if sc == "TW_WAIT":
        if gain is not None and gain < 0:
            return (f"REDUCE — Testing support while {gain:+.0f}% underwater. "
                    f"Exit on any bounce. Stop: 20W EMA ({d20:+.1f}%)")
        if gain is not None and gain > 50:
            return (f"TIGHTEN STOP — At 20W EMA ({d20:+.1f}%). "
                    f"Protect {gain:+.0f}% gain. Exit if 20W breaks")
        return (f"WATCH — Testing 20W EMA ({d20:+.1f}%). "
                f"Set hard stop below 20W. Reduce if broken")

    # ── TW_FADING — bull stack intact but 40W flattening ─────────────────────
    if sc == "TW_FADING":
        if gain is not None and gain < 0:
            return f"EXIT — Fading trend while {gain:+.0f}% underwater. {slope_ctx}"
        if gain is not None and gain > 50:
            return (f"TRIM 25-33% — Protect {gain:+.0f}% gain. 40W EMA flattening. "
                    f"Raise stop to 10W EMA ({d10:+.1f}%). {slope_ctx}")
        if gain is not None and gain > 20:
            return (f"RAISE STOP to 10W EMA ({d10:+.1f}%) — "
                    f"Trend aging, protect {gain:+.0f}% gain. {slope_ctx}")
        return (f"RAISE STOP to 10W EMA ({d10:+.1f}%) — "
                f"40W EMA flattening. Reduce if 10W breaks. {slope_ctx}")

    # ── TW_CAUTIOUS — below 10W or 20W, 40W still supports ───────────────────
    if sc == "TW_CAUTIOUS":
        if gain is not None and gain < -10:
            return (f"EXIT NOW — Below 10W EMA while {gain:+.0f}% underwater. "
                    f"Stop loss. No waiting")
        if gain is not None and gain < 0:
            return (f"EXIT — Below 10W EMA ({d10:+.1f}%) while in loss. "
                    f"Exit on any bounce")
        if gain is not None and gain > 50:
            return (f"REDUCE 25-50% — Protect {gain:+.0f}% gain. "
                    f"Below 10W EMA ({d10:+.1f}%). Stop: 20W EMA ({d20:+.1f}%)")
        return (f"TIGHTEN STOP to 20W EMA ({d20:+.1f}%) — "
                f"Below 10W EMA ({d10:+.1f}%). Reduce if 20W breaks")

    # ── TW_EXIT_40W — below 20W, 40W rolling over ────────────────────────────
    if sc == "TW_EXIT_40W":
        if gain is not None and gain <= -10:
            return (f"EXIT NOW — Stop loss. {gain:+.0f}% position. "
                    f"40W EMA breaking down. {slope_ctx}")
        if gain is not None and gain < 0:
            return (f"EXIT — 40W EMA rolling over while {gain:+.0f}% underwater. "
                    f"Exit on next bounce. {slope_ctx}")
        if gain is not None and gain > 0:
            return (f"REDUCE 50-75% — Protect {gain:+.0f}% gain. "
                    f"Exit on bounces — 40W EMA turning down. {slope_ctx}")
        return (f"REDUCE 50-75% — 40W EMA rolling over. "
                f"Exit position on any bounce. {slope_ctx}")

    # ── TW_EXIT — below all weekly EMAs ──────────────────────────────────────
    if sc == "TW_EXIT":
        if gain is not None and gain < 0:
            return (f"EXIT NOW — Below all weekly EMAs. "
                    f"Stop loss at {gain:+.0f}%. No weekly support left")
        if gain is not None and gain > 0:
            return (f"EXIT FULL POSITION — Protect {gain:+.0f}% gain. "
                    f"All weekly EMAs broken. No floor")
        return "EXIT FULL POSITION — Below all weekly EMAs. Structural breakdown"

    # TW_NONE
    return "— Insufficient weekly data (need ≥ 42 complete weeks)"


# =============================================================================
# SLOPE LABEL HELPER
# =============================================================================

def slope_label(slope: float) -> str:
    """Convert 40W slope % to human-readable label."""
    if slope > 1.5:   return "Rising ↑↑"
    if slope > 0.3:   return "Rising ↑"
    if slope > -0.3:  return "Flat →"
    if slope > -1.0:  return "Falling ↓"
    return "Falling ↓↓"
