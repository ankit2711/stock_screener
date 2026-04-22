# =============================================================================
# RS LEADERS — Relative Strength Analysis Engine
# =============================================================================
#
# Purpose:
#   Identify stocks whose RS line (stock / benchmark) is at or near 52-week
#   highs even while the broader market is correcting.  These are the stocks
#   that lead the next bull market rally — Minervini calls them "RS Leaders".
#
# Philosophy (Minervini):
#   "The stocks that hold up the best during a correction are the ones that
#    go up the most when the market recovers."
#   The RS line making new highs BEFORE price makes new highs is the single
#   most powerful leading indicator of institutional accumulation.
#
# Key signals computed (all used in scoring):
#   rs_line            daily ratio: stock close / benchmark close (aligned)
#   rs_at_52w_high     rs_line within 3% of its 52-week high
#   rs_52w_high_pct    how close (0–100%) to 52-week RS high
#   resilience         benchmark % off its 52w high MINUS stock % off its 52w high
#                      positive = stock holding up better than the market
#   accum_ratio        up-day volume share in last 20 bars (>0.55 = accumulation)
#   vol_dry            avg vol 5d / avg vol 20d (<0.70 = constructive dry-up)
#   above_ema200       price > EMA200
#   above_ema50        price > EMA50
#   bullish_ema_stack  EMA10 > EMA21 > EMA50 > EMA200
#   stage              1/2/3/4 from stage_analysis (2 = best, but 1/Accum OK here)
#   base_forming       price range last 20 bars < 10% (coiling/consolidation)
#   ftd_signal         Follow-Through Day on benchmark: +1.5%+ with higher vol on Day 4-7
#                      of an attempted rally (signals potential new uptrend)
#
# Scoring (0–100):
#   RS at 52-week high              35 pts  (primary signal)
#   Relative resilience vs bench    25 pts  (holding up better during correction)
#   Volume accumulation             20 pts  (institutional buying)
#   Structural integrity            15 pts  (EMA200, EMA stack, slope)
#   Base formation quality           5 pts  (coiling / tight consolidation)
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class RSLeaderResult:
    ticker:             str = ""
    market_cap:         float = 0.0

    # --- RS Line metrics ---
    rs_line_current:    float = 0.0    # stock/bench ratio today
    rs_52w_high:        float = 0.0    # 52-week high of the RS line
    rs_52w_high_pct:    float = 0.0    # % below 52w RS high (0 = AT high, positive = below)
    rs_at_52w_high:     bool  = False  # within 3% of 52w RS high
    rs_new_high:        bool  = False  # RS line at all-time high in the dataset
    rs_leads_price:     bool  = False  # ★ RS line at new high while price is still >5% below its own
                                       #   52w high. Minervini's highest-priority pre-breakout signal:
                                       #   institutions are accumulating before the price breakout shows.
                                       #   These stocks most often lead the next bull phase.
    rs_vs_bench_1m:     float = 0.0    # 1-month RS vs benchmark (%)
    rs_vs_bench_3m:     float = 0.0    # 3-month RS vs benchmark (%)

    # --- Resilience ---
    stock_pct_off_52w:  float = 0.0    # % stock is below its own 52w high
    bench_pct_off_52w:  float = 0.0    # % benchmark is below its 52w high
    resilience:         float = 0.0    # bench_off - stock_off (positive = leading)
    resilience_label:   str   = ""     # "Strong Leader" / "Leader" / "Neutral" / "Laggard"

    # --- Volume ---
    accum_ratio:        float = 0.0    # up-day vol / total vol (20 bars)
    vol_dry:            bool  = False  # recent 5d avg vol < 70% of 20d avg
    vol_dry_ratio:      float = 0.0    # 5d vol / 20d vol

    # --- Structure ---
    above_ema200:       bool  = False
    above_ema50:        bool  = False
    bullish_ema_stack:  bool  = False
    ema200_slope:       float = 0.0    # 10-day slope of EMA200 (% change)
    price_vs_ema200:    float = 0.0    # % above/below EMA200
    stage:              int   = 0      # 1/2/3/4 from stage analysis (0 = unknown)
    stage_label:        str   = ""     # e.g. "Stage 2 ↑"

    # --- Base ---
    base_forming:       bool  = False  # price range last 20 bars < 10%
    base_depth_pct:     float = 0.0    # % range of last 20-bar price action
    consolidation_bars: int   = 0      # how many consecutive bars price has been in range

    # --- Prices ---
    price:              float = 0.0
    avg_dollar_vol:     float = 0.0    # 20-day avg dollar volume

    # --- Market context ---
    bench_pct_off_52w_label: str = ""  # "At High" / "-5%" / "-10%+" etc.
    ftd_detected:       bool  = False  # Follow-Through Day on benchmark in last 7 bars

    # --- Composite score ---
    rs_score:           float = 0.0    # 0–100 composite RS Leader score
    score_breakdown:    dict  = field(default_factory=dict)


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_rs_leaders_analysis(
    df:           pd.DataFrame,
    benchmark_df: pd.DataFrame,
    ticker:       str,
    market:       str = "india",
) -> RSLeaderResult:
    """
    Compute RS Leader metrics and composite score for a single ticker.

    Args:
        df:           OHLCV DataFrame for the stock (index=date, cols=open/high/low/close/volume)
        benchmark_df: OHLCV DataFrame for the benchmark (same structure)
        ticker:       ticker string (used in result)
        market:       "india" or "us"

    Returns:
        RSLeaderResult with all metrics filled in.
    """
    result = RSLeaderResult(ticker=ticker)

    close  = df["close"].dropna()
    volume = df["volume"].dropna()
    high   = df["high"].dropna()

    if len(close) < 60:
        return result

    # ── Align stock and benchmark on shared dates ─────────────────────────────
    bench_close = benchmark_df["close"].dropna()
    common      = close.index.intersection(bench_close.index)
    if len(common) < 60:
        return result

    s_close = close.reindex(common)
    b_close = bench_close.reindex(common)

    # ── RS Line ───────────────────────────────────────────────────────────────
    rs_line = s_close / b_close

    rs_now       = float(rs_line.iloc[-1])
    rs_52w_high  = float(rs_line.iloc[-252:].max()) if len(rs_line) >= 252 else float(rs_line.max())
    rs_pct_below = (rs_52w_high - rs_now) / rs_52w_high * 100 if rs_52w_high > 0 else 99.0

    result.rs_line_current = round(rs_now, 6)
    result.rs_52w_high     = round(rs_52w_high, 6)
    result.rs_52w_high_pct = round(rs_pct_below, 2)
    result.rs_at_52w_high  = rs_pct_below <= 3.0
    result.rs_new_high     = rs_now >= rs_line.max() * 0.995  # AT all-time RS high in dataset
    # rs_leads_price is computed after stock_off is available (see below)

    # RS vs benchmark over 1m / 3m
    def _pct(n: int) -> float:
        if len(s_close) < n + 1 or len(b_close) < n + 1:
            return 0.0
        s_ret = (s_close.iloc[-1] / s_close.iloc[-n] - 1) * 100
        b_ret = (b_close.iloc[-1] / b_close.iloc[-n] - 1) * 100
        return round(s_ret - b_ret, 2)

    result.rs_vs_bench_1m = _pct(21)
    result.rs_vs_bench_3m = _pct(63)

    # ── Resilience ────────────────────────────────────────────────────────────
    price    = float(close.iloc[-1])
    s_52w_hi = float(high.iloc[-252:].max()) if len(high) >= 252 else float(high.max())
    b_52w_hi = float(b_close.iloc[-252:].max()) if len(b_close) >= 252 else float(b_close.max())

    stock_off = (s_52w_hi - price)     / s_52w_hi * 100
    bench_off = (b_52w_hi - b_close.iloc[-1]) / b_52w_hi * 100

    resilience = bench_off - stock_off   # positive = stock holding up better

    result.price             = round(price, 2)
    result.stock_pct_off_52w = round(stock_off, 2)
    result.bench_pct_off_52w = round(bench_off, 2)
    result.resilience        = round(resilience, 2)
    result.bench_pct_off_52w_label = _bench_label(bench_off)

    # ── RS Leads Price ────────────────────────────────────────────────────────
    # True when the RS line is already at its own 52-week high while the stock
    # price is still more than 5% below its 52-week price high.
    # This divergence = institutions are accumulating quietly before the stock's
    # own price breakout occurs. It is the single most predictive pre-breakout
    # signal in Minervini's methodology.
    # 5% gap threshold: avoids false positives where RS line "leads" by a trivial
    # intraday difference — at least 5% price gap means a meaningful divergence.
    result.rs_leads_price = result.rs_at_52w_high and (result.stock_pct_off_52w > 5.0)

    if resilience >= 10:
        result.resilience_label = "Strong Leader"
    elif resilience >= 4:
        result.resilience_label = "Leader"
    elif resilience >= -2:
        result.resilience_label = "Neutral"
    else:
        result.resilience_label = "Laggard"

    # ── Volume accumulation ───────────────────────────────────────────────────
    vol = volume.reindex(common).fillna(0)
    recent_vol = vol.iloc[-20:]
    price_changes = s_close.iloc[-20:].pct_change().fillna(0)

    up_vol   = float(recent_vol[price_changes >= 0].sum())
    total_vol = float(recent_vol.sum())
    accum    = up_vol / total_vol if total_vol > 0 else 0.5

    vol_5d  = float(vol.iloc[-5:].mean())
    vol_20d = float(vol.iloc[-20:].mean())
    dry_ratio = vol_5d / vol_20d if vol_20d > 0 else 1.0

    result.accum_ratio  = round(accum, 3)
    result.vol_dry      = dry_ratio < 0.70
    result.vol_dry_ratio = round(dry_ratio, 3)
    result.avg_dollar_vol = round(float(vol.iloc[-20:].mean()) * price, 0)

    # ── Structural integrity ──────────────────────────────────────────────────
    ema10  = float(close.ewm(span=10,  adjust=False).mean().iloc[-1])
    ema21  = float(close.ewm(span=21,  adjust=False).mean().iloc[-1])
    ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
    ema200_series = close.ewm(span=200, adjust=False).mean()
    ema200 = float(ema200_series.iloc[-1])

    # EMA200 slope: 10-day % change
    if len(ema200_series) >= 11:
        ema200_10d_ago = float(ema200_series.iloc[-11])
        slope_pct = (ema200 - ema200_10d_ago) / ema200_10d_ago * 100 if ema200_10d_ago > 0 else 0.0
    else:
        slope_pct = 0.0

    result.above_ema200    = price > ema200
    result.above_ema50     = price > ema50
    result.bullish_ema_stack = (ema10 > ema21 > ema50 > ema200) and (price > ema10)
    result.ema200_slope    = round(slope_pct, 3)
    result.price_vs_ema200 = round((price - ema200) / ema200 * 100, 2)

    # Stage (simplified — reuse EMA alignment logic without full stage_analysis import)
    # Full run_stage_analysis is expensive per-ticker; use quick heuristic for RS scan.
    result.stage, result.stage_label = _quick_stage(price, ema50, ema200, slope_pct)

    # ── Base formation ────────────────────────────────────────────────────────
    recent_20 = close.iloc[-20:]
    base_range = (float(recent_20.max()) - float(recent_20.min())) / float(recent_20.max()) * 100
    result.base_forming  = base_range < 10.0
    result.base_depth_pct = round(base_range, 2)

    # How many consecutive bars price has stayed within 8% of current price
    result.consolidation_bars = _count_consolidation_bars(close, tolerance_pct=8.0)

    # ── Follow-Through Day detection on benchmark ─────────────────────────────
    result.ftd_detected = _detect_ftd(b_close, benchmark_df.get("volume", pd.Series()))

    # ── Composite score (0–100) ───────────────────────────────────────────────
    result.rs_score, result.score_breakdown = _composite_score(result)

    return result


# =============================================================================
# COMPOSITE SCORE
# =============================================================================

def _composite_score(r: RSLeaderResult) -> tuple[float, dict]:
    """
    Compute 0–100 RS Leader composite score.

    Weights:
        RS at 52-week high              35 pts
        Relative resilience vs bench    25 pts
        Volume accumulation             20 pts
        Structural integrity            15 pts
        Base formation quality           5 pts
    """
    breakdown = {}

    # ── 1. RS Line at/near 52-week high (35 pts) ─────────────────────────────
    # Score scales with closeness to 52w RS high.
    # AT high (0% below): 35 pts
    # 3% below:           28 pts (still qualifies as "at high")
    # 5% below:           20 pts
    # 10% below:          10 pts
    # 15%+ below:          0 pts
    pct_below = r.rs_52w_high_pct
    if pct_below <= 3.0:
        rs_pts = 35.0 * (1.0 - pct_below / 3.0 * 0.20)   # 35→28 across 0–3%
    elif pct_below <= 10.0:
        rs_pts = 28.0 * max(0, (10.0 - pct_below) / 7.0)  # 28→0 across 3–10%
    else:
        rs_pts = 0.0
    breakdown["rs_at_high"] = round(rs_pts, 2)

    # Bonus: RS line at new all-time high in dataset
    if r.rs_new_high:
        rs_pts = min(35.0, rs_pts + 3.0)
        breakdown["rs_new_high_bonus"] = 3.0

    # ── 2. Relative resilience vs benchmark (25 pts) ──────────────────────────
    # resilience = bench_off_52w - stock_off_52w
    # +20% better than bench → 25 pts (full)
    # +10%: ~18 pts | +4%: ~9 pts | neutral: 5 pts | laggard: 0 pts
    res = r.resilience
    if res >= 20:
        res_pts = 25.0
    elif res >= 0:
        res_pts = 5.0 + 20.0 * (res / 20.0)
    elif res >= -5:
        res_pts = max(0.0, 5.0 + res)   # 5 → 0 across 0 to -5%
    else:
        res_pts = 0.0
    breakdown["resilience"] = round(res_pts, 2)

    # ── 3. Volume accumulation (20 pts) ──────────────────────────────────────
    # accum_ratio >0.65 = strong accumulation (20 pts)
    # >0.55: moderate | 0.50: neutral | <0.45: distribution
    acc = r.accum_ratio
    if acc >= 0.65:
        vol_pts = 20.0
    elif acc >= 0.55:
        vol_pts = 10.0 + 10.0 * (acc - 0.55) / 0.10
    elif acc >= 0.50:
        vol_pts = 5.0 + 5.0 * (acc - 0.50) / 0.05
    else:
        vol_pts = 0.0

    # Bonus: volume dry-up (constructive quiet before potential breakout)
    if r.vol_dry:
        vol_pts = min(20.0, vol_pts + 2.0)
        breakdown["vol_dry_bonus"] = 2.0

    breakdown["volume"] = round(vol_pts, 2)

    # ── 4. Structural integrity (15 pts) ─────────────────────────────────────
    struct_pts = 0.0
    if r.above_ema200:    struct_pts += 5.0    # above structural MA
    if r.above_ema50:     struct_pts += 3.0    # above intermediate MA
    if r.bullish_ema_stack: struct_pts += 5.0  # full bullish EMA alignment
    if r.ema200_slope > 0:  struct_pts += 2.0  # EMA200 turning up
    struct_pts = min(15.0, struct_pts)
    breakdown["structure"] = round(struct_pts, 2)

    # ── 5. Base formation (5 pts) ─────────────────────────────────────────────
    base_pts = 0.0
    if r.base_forming:
        # Tighter base = more points (< 5% range = full 5 pts)
        tightness = max(0, (10.0 - r.base_depth_pct) / 10.0)
        base_pts  = tightness * 5.0
    if r.consolidation_bars >= 10:
        base_pts = min(5.0, base_pts + 1.0)   # bonus for extended consolidation
    breakdown["base"] = round(base_pts, 2)

    # ── 6. RS Leads Price bonus (0 or 8 pts) ─────────────────────────────────
    # RS line at new 52w high while the stock price is still >5% below its own
    # 52w high. This is the highest-conviction pre-breakout signal: the RS line
    # diverges positively, showing institutional accumulation before the price
    # breakout that will follow.
    #
    # Why 8 pts: enough to elevate a solid RS leader (score ~55) with this signal
    # above a higher-score stock (score ~60) that lacks it. The signal should
    # visibly lift rank, not just appear as a footnote.
    #
    # Critically: this bonus does NOT filter any stock out. Every stock that
    # passes the existing gates still appears — some just rank higher.
    lead_pts = 0.0
    if r.rs_leads_price:
        lead_pts = 8.0
        breakdown["rs_leads_price_bonus"] = 8.0

    # ── Total ─────────────────────────────────────────────────────────────────
    total = rs_pts + res_pts + vol_pts + struct_pts + base_pts + lead_pts
    total = round(min(max(total, 0.0), 100.0), 2)
    breakdown["total"] = total

    return total, breakdown


# =============================================================================
# HELPERS
# =============================================================================

def _quick_stage(price: float, ema50: float, ema200: float, ema200_slope: float) -> tuple[int, str]:
    """
    Fast heuristic stage classification (no full stage_analysis import).
    Used in RS Leaders scan where per-ticker speed matters.
    """
    above_200 = price > ema200
    above_50  = price > ema50
    ma_rising = ema200_slope > 0

    if above_200 and above_50 and ma_rising:
        return 2, "Stage 2 ↑"
    elif above_200 and above_50:
        return 2, "Stage 2 (flat)"
    elif above_200 and not above_50:
        return 1, "Stage 1 Accum"
    elif not above_200 and not above_50 and not ma_rising:
        return 4, "Stage 4 ↓"
    else:
        return 3, "Stage 3 Dist"


def _count_consolidation_bars(close: pd.Series, tolerance_pct: float = 8.0) -> int:
    """
    Count consecutive bars (from most recent) where price stayed within
    tolerance_pct of the most recent closing price.
    """
    current = float(close.iloc[-1])
    lo = current * (1 - tolerance_pct / 100)
    hi = current * (1 + tolerance_pct / 100)

    count = 0
    for price in reversed(close.values):
        if lo <= price <= hi:
            count += 1
        else:
            break
    return count


def _detect_ftd(bench_close: pd.Series, bench_volume: pd.Series = None) -> bool:
    """
    Detect a Follow-Through Day (FTD) on the benchmark in the last 7 bars.

    FTD criteria (Minervini / O'Neil):
      - Benchmark up ≥1.5% on Day 4-7 of an attempted rally.
      - Volume higher than the prior session.
      - An attempted rally starts from a low (price turns up after downtrend).

    Simplified version: look for any +1.5% up day in last 7 bars where
    volume was higher than the prior day (when volume data available).
    Returns True if an FTD signal is present.
    """
    if len(bench_close) < 8:
        return False

    recent = bench_close.iloc[-8:]
    pct_changes = recent.pct_change().fillna(0)

    has_volume = bench_volume is not None and len(bench_volume) >= 8
    if has_volume:
        vol_recent = bench_volume.iloc[-8:]

    for i in range(1, len(pct_changes)):
        if pct_changes.iloc[i] >= 0.015:   # +1.5%+
            if has_volume and i >= 1:
                if float(vol_recent.iloc[i]) > float(vol_recent.iloc[i - 1]):
                    return True
            else:
                return True   # no volume data — flag on price alone

    return False


def _bench_label(pct_off: float) -> str:
    if pct_off <= 2.0:   return "At High"
    if pct_off <= 5.0:   return f"-{pct_off:.0f}%"
    if pct_off <= 10.0:  return f"-{pct_off:.0f}%"
    if pct_off <= 20.0:  return f"-{pct_off:.0f}% (Correction)"
    return f"-{pct_off:.0f}% (Bear)"
