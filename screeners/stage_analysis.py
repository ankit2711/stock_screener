# =============================================================================
# Stage Analysis Screener — Python translation of Pine Script
# Translated from: "Stage Analysis Screener (TheWrap Clone)"
#
# Original Pine Script
#   Author:  Atlas (OpenClaw)
#   For:     Ankit
#   Version: 1.0
#
# Python translation by Claude (Anthropic) — faithful port of all logic
#
# Computes for each ticker:
#   1. Weinstein Stage (S1/S2/S3/S4) via scoring system
#   2. Relative Strength vs benchmark (SPX for US, CNX500 for India)
#   3. Momentum (ROC fast/slow + acceleration)
#   4. Average Dollar Volume + trend
#   5. EMA Distance (fast/medium/slow)
#   6. Cheat Entry (VCP-style pullback in S2)
#   7. Beta (correlation-based)
#   8. PEAD — Post-Earnings Announcement Drift (approximated)
#   9. Market Cap
#  10. Volume Conviction (volume ratio vs avg)
# =============================================================================

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal

# -----------------------------------------------------------------------------
# CONFIG — mirrors Pine Script inputs
# -----------------------------------------------------------------------------

@dataclass
class StageAnalysisConfig:
    # Stage settings
    sensitivity:      Literal["Aggressive", "Normal", "Conservative"] = "Aggressive"
    ma_length:        int   = 200    # Stage MA length in DAILY bars — implemented as EMA.
                                     # 200-day EMA = 40-week EMA = the universal institutional
                                     # reference line visible on every charting platform.
                                     # This is what traders mean when they say "40-week EMA"
                                     # on TradingView. Aligns code output with visual chart checks.
                                     #
                                     # Weinstein's original = 30-week SMA ≈ 150-day SMA.
                                     # Both represent the same structural trend concept;
                                     # EMA200 is preferred here because:
                                     #   1. Matches TradingView visual reference exactly
                                     #   2. EMA weights recent price more — catches turns faster
                                     #   3. Standard reference for NSE institutional activity
                                     #
                                     # DO NOT use values < 100 — a 30-day MA is only 6 weeks
                                     # and turns on any short bounce, producing false S2/S4.
    slope_lookback:   int   = 10     # Bars for slope measurement on the 200-day EMA.
                                     # 10 bars gives a 2-week slope — responsive without noise.

    # EMA settings
    ema_fast:         int   = 10
    ema_medium:       int   = 21
    ema_slow:         int   = 50

    # RS settings
    rs_ma_length:     int   = 52

    # Volume settings
    vol_avg_len:      int   = 50

    # Momentum settings
    mom_fast:         int   = 10
    mom_slow:         int   = 20

    # Beta settings
    beta_length:      int   = 52

    # PEAD settings
    pead_threshold:   float = 10.0   # % for "Strong"
    pead_window:      int   = 5      # bars after earnings

    # Cheat entry
    cheat_pullback_pct: float = 2.5  # % distance from medium EMA
    cheat_vol_ratio:    float = 0.7  # volume below X * avg
    vol_expansion_ratio:float = 1.5  # volume expansion for breakout


# -----------------------------------------------------------------------------
# RESULT DATACLASS
# -----------------------------------------------------------------------------

@dataclass
class StageAnalysisResult:
    ticker: str = ""

    # Stage
    stage:            int   = 0      # 1, 2, 3, 4
    stage_label:      str   = "S?"
    stage_full:       str   = ""
    stage_duration:   int   = 0
    stage_score:      dict  = field(default_factory=dict)
    stage_transition: bool  = False
    prev_stage:       int   = 0

    # RS
    rs_vs_benchmark:  float = 0.0
    rs_is_strong:     bool  = False
    rs_rising:        bool  = False
    rs_status:        str   = ""

    # Momentum
    roc_fast:         float = 0.0
    roc_slow:         float = 0.0
    mom_accel:        float = 0.0
    mom_label:        str   = ""

    # Dollar volume
    avg_dollar_vol:   float = 0.0
    vol_trend:        bool  = False
    vol_trend_str:    str   = ""

    # EMA distances
    ema_dist_fast:    float = 0.0
    ema_dist_medium:  float = 0.0
    ema_dist_slow:    float = 0.0

    # Cheat entry
    is_cheat_entry:   bool  = False
    cheat_detail:     str   = ""

    # Beta
    beta:             float = 1.0
    beta_label:       str   = ""

    # PEAD (approximated — no earnings calendar without premium API)
    pead_pct:         float = 0.0
    pead_label:       str   = ""

    # Volume conviction
    vol_ratio:        float = 1.0
    vol_conviction:   str   = ""

    # Exit signals — computed for every stock regardless of stage
    exit_signal:      str   = ""    # code: EXIT_NOW / EXIT_WEEKLY / WARN_STAGE3 / WARN_EMA21 / WARN_MOMENTUM / PROFIT_TAKE / CLEAN
    exit_label:       str   = ""    # human-readable: "🔴 EXIT — Stage 4" etc.

    # Market cap (passed in from metadata)
    market_cap:       float = 0.0

    # Composite screener score (0–1, for ranking)
    score:            float = 0.0

    # For sheet output
    passed:           bool  = False   # True if S2 or Cheat Entry

    # Weekly Stage — filled by ranker_stage.py after weekly analysis
    # Weinstein's primary classification uses the 30-week SMA on weekly bars.
    # The stage_analysis.py daily analysis is the SECONDARY (entry timing) view.
    weekly_stage:       int   = 0     # 0=unknown, 1=accum, 2=advancing, 3=dist, 4=decline
    weekly_stage_label: str   = ""    # "W-S2 ✓", "W-S1 Accum", "W-S3 Dist", "W-S4 Decline"
    weekly_sma30:       float = 0.0   # current 30-week SMA value (₹ price level)
    weekly_vol_dry:     bool  = False  # True if right-side base vol < left-side (accumulation pattern)
    mansfield_rs:       float = 0.0   # Mansfield RS = (RS_line / RS_MA − 1) × 100
                                      # Positive = outperforming. rs_vs_benchmark already computes
                                      # this — mansfield_rs is an alias for explicit display.


# -----------------------------------------------------------------------------
# MAIN ANALYSIS FUNCTION
# -----------------------------------------------------------------------------

def run_stage_analysis(
    df:           pd.DataFrame,
    benchmark_df: pd.DataFrame,
    ticker:       str = "",
    market:       str = "us",          # 'us' or 'india'
    cfg:          StageAnalysisConfig = None,
) -> StageAnalysisResult:
    """
    Run full stage analysis on a single ticker.

    Args:
        df:           Daily OHLCV DataFrame, columns [open,high,low,close,volume],
                      DatetimeIndex ascending. At least 100 rows recommended.
        benchmark_df: Benchmark OHLCV (SPX for US, CNX500/Nifty500 for India).
        ticker:       Ticker symbol string.
        market:       'us' or 'india' — affects benchmark and currency labels.
        cfg:          StageAnalysisConfig — uses defaults if None.

    Returns:
        StageAnalysisResult with all metrics populated.
    """
    if cfg is None:
        cfg = StageAnalysisConfig()

    result = StageAnalysisResult(ticker=ticker)

    # Minimum bars required:
    # • For EMA-based stage MA (ma_length ≥ 100): EWM initialises from the first
    #   price point and converges within ~3× its effective half-life, so it is
    #   numerically valid with far fewer bars than ma_length.  We require
    #   max(100, ma_length // 2) so recently-listed stocks (6–12 months) with a
    #   genuine rising trend are not silently discarded.
    # • For SMA-based stage MA: full ma_length + 10 bars are required.
    # Other indicators (EMA50, RS52, Beta52) need ≤ 62 bars — always satisfied
    # when the stage MA minimum is met.
    use_ema_early = cfg.ma_length >= 100
    if use_ema_early:
        min_bars = max(100, cfg.ma_length // 2)
    else:
        min_bars = cfg.ma_length + 10
    if len(df) < min_bars:
        result.stage_label = "Insufficient data"
        return result

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # -------------------------------------------------------------------------
    # 1. Stage MA + slope
    # -------------------------------------------------------------------------
    # We use an EMA rather than SMA so the line matches exactly what traders
    # see on TradingView when they add EMA200 to a daily chart (= 40-week EMA).
    # SMA150 (Weinstein's original 30-week SMA) and EMA200 (40-week EMA) both
    # represent roughly the same concept but EMA200 is the universal reference
    # used by institutions and visible on every charting platform.
    use_ema_for_stage = cfg.ma_length >= 100   # only for long-period (structural) MAs
    if use_ema_for_stage:
        stage_ma = close.ewm(span=cfg.ma_length, adjust=False).mean()
    else:
        stage_ma = close.rolling(cfg.ma_length).mean()
    ma_slope = (
        (stage_ma - stage_ma.shift(cfg.slope_lookback))
        / stage_ma.shift(cfg.slope_lookback) * 100
    )

    sens_mult = _sensitivity_mult(cfg.sensitivity)
    slope_thresh = 0.5 * sens_mult

    ma_rising  = ma_slope.iloc[-1] >  slope_thresh
    ma_falling = ma_slope.iloc[-1] < -slope_thresh
    ma_flat    = not ma_rising and not ma_falling

    # -------------------------------------------------------------------------
    # 2. EMAs
    # -------------------------------------------------------------------------
    ema_fast   = close.ewm(span=cfg.ema_fast,   adjust=False).mean()
    ema_medium = close.ewm(span=cfg.ema_medium,  adjust=False).mean()
    ema_slow   = close.ewm(span=cfg.ema_slow,    adjust=False).mean()

    ef = ema_fast.iloc[-1]
    em = ema_medium.iloc[-1]
    es = ema_slow.iloc[-1]
    px = close.iloc[-1]

    ema_dist_fast   = (px - ef) / ef * 100
    ema_dist_medium = (px - em) / em * 100
    ema_dist_slow   = (px - es) / es * 100

    result.ema_dist_fast   = round(ema_dist_fast, 2)
    result.ema_dist_medium = round(ema_dist_medium, 2)
    result.ema_dist_slow   = round(ema_dist_slow, 2)

    # -------------------------------------------------------------------------
    # 3. Relative Strength vs benchmark
    # -------------------------------------------------------------------------
    rs_raw, rs_ma, rs_vs_bm, rs_is_strong, rs_rising, rs_falling = _calc_rs(
        close, benchmark_df["close"], cfg.rs_ma_length
    )
    result.rs_vs_benchmark = round(rs_vs_bm, 2)
    result.mansfield_rs    = round(rs_vs_bm, 2)   # alias — same value, explicit name
    result.rs_is_strong    = rs_is_strong
    result.rs_rising       = rs_rising

    if rs_is_strong and rs_rising:
        result.rs_status = "Strong ↑"
    elif rs_is_strong and not rs_rising:
        result.rs_status = "Moderate ↑"
    elif not rs_is_strong and rs_falling:
        result.rs_status = "Weak ↓"
    else:
        result.rs_status = "Fading ↓"

    # -------------------------------------------------------------------------
    # 4. Momentum (ROC fast/slow + acceleration)
    # -------------------------------------------------------------------------
    roc_fast_s = close.pct_change(cfg.mom_fast) * 100
    roc_slow_s = close.pct_change(cfg.mom_slow) * 100
    mom_accel  = roc_fast_s.iloc[-1] - roc_fast_s.iloc[-4]  # accel over 3 bars

    roc_f = roc_fast_s.iloc[-1]
    roc_s = roc_slow_s.iloc[-1]

    result.roc_fast   = round(roc_f, 2)
    result.roc_slow   = round(roc_s, 2)
    result.mom_accel  = round(mom_accel, 2)

    if roc_f > 0 and mom_accel > 0:
        result.mom_label = "Strong ↑↑"
    elif roc_f > 0 and mom_accel <= 0:
        result.mom_label = "Rising ↑"
    elif roc_f < 0 and mom_accel >= 0:
        result.mom_label = "Weak ↓"
    else:
        result.mom_label = "Weak ↓↓"

    # -------------------------------------------------------------------------
    # 5. Average Dollar Volume
    # -------------------------------------------------------------------------
    dollar_vol     = close * volume
    avg_dollar_vol = dollar_vol.rolling(cfg.vol_avg_len).mean()
    prev_avg_dv    = dollar_vol.shift(10).rolling(cfg.vol_avg_len).mean()

    adv = avg_dollar_vol.iloc[-1]
    vol_trend = adv > prev_avg_dv.iloc[-1]

    result.avg_dollar_vol = adv
    result.vol_trend      = vol_trend
    result.vol_trend_str  = "Expanding ↑" if vol_trend else "Declining ↓"

    # -------------------------------------------------------------------------
    # 6. Beta (correlation-based, matching Pine Script)
    # -------------------------------------------------------------------------
    result.beta = _calc_beta(close, benchmark_df["close"], cfg.beta_length)
    b = result.beta
    if b < 0.5:   result.beta_label = "Very Low"
    elif b < 0.8: result.beta_label = "Low"
    elif b < 1.2: result.beta_label = "Normal"
    elif b < 1.5: result.beta_label = "High"
    else:         result.beta_label = "Very High"

    # -------------------------------------------------------------------------
    # 7. Volume Conviction
    # -------------------------------------------------------------------------
    avg_vol  = volume.rolling(cfg.vol_avg_len).mean()
    vol_ratio = volume.iloc[-1] / avg_vol.iloc[-1] if avg_vol.iloc[-1] > 0 else 1.0
    result.vol_ratio = round(vol_ratio, 2)

    if vol_ratio > 2.0:   result.vol_conviction = "Very High"
    elif vol_ratio > 1.3: result.vol_conviction = "High"
    elif vol_ratio > 0.7: result.vol_conviction = "Normal"
    else:                 result.vol_conviction = "Low"

    # -------------------------------------------------------------------------
    # 8. PEAD (approximated — no live earnings calendar)
    # Since we don't have earnings dates via free API, we detect large single-day
    # gaps (>5%) as proxy for earnings events, then track drift.
    # For accurate PEAD: integrate an earnings calendar API (e.g. Alpha Vantage
    # earnings endpoint, or store your own dates).
    # -------------------------------------------------------------------------
    result.pead_pct, result.pead_label = _calc_pead_approx(
        close, cfg.pead_threshold, cfg.pead_window
    )

    # -------------------------------------------------------------------------
    # 9. Stage Determination — scoring system (faithful to Pine)
    # -------------------------------------------------------------------------
    price_above_ma  = px > stage_ma.iloc[-1]
    price_below_ma  = px < stage_ma.iloc[-1]
    bullish_ema     = ef > em and em > es
    bearish_ema     = ef < em and em < es
    vol_expansion   = volume.iloc[-1] > avg_vol.iloc[-1] * cfg.vol_expansion_ratio

    # % price is above the 200-day EMA — used to graduate s2_core credit
    stage_ma_val = float(stage_ma.iloc[-1]) if not np.isnan(stage_ma.iloc[-1]) else 0.0
    stage_ma_dist_pct = (px - stage_ma_val) / stage_ma_val * 100 if stage_ma_val > 0 else 0.0

    # Higher highs / lower lows over 40-bar windows (≈ 8 weeks).
    # Weinstein's higher-high / lower-low structure is a multi-week pattern.
    # The old 10-bar (2-week) window was too short — a 2-day rally in a downtrend
    # triggered hh=True, adding noise to S2/S4 scoring.
    hh = high.rolling(40).max().iloc[-1] > high.rolling(40).max().iloc[-41]
    ll = low.rolling(40).min().iloc[-1]  < low.rolling(40).min().iloc[-41]

    stage, scores, transition, prev_stage, duration = _calc_stage(
        ma_rising, ma_falling, ma_flat,
        price_above_ma, price_below_ma,
        bullish_ema, bearish_ema,
        rs_is_strong, rs_rising, rs_falling,
        roc_f, roc_s,
        vol_expansion, hh, ll,
        ema_dist_slow,
        ma_slope,
        stage_ma_dist_pct,
        cfg,
    )

    result.stage          = stage
    result.stage_label    = f"S{stage}"
    result.stage_full     = ["", "S1 (Accumulation)", "S2 (Advancing)",
                              "S3 (Distribution)", "S4 (Declining)"][stage]
    result.stage_score    = scores
    result.stage_transition = transition
    result.prev_stage     = prev_stage
    result.stage_duration = duration

    # -------------------------------------------------------------------------
    # 10. Cheat Entry (VCP-style pullback within Stage 2)
    # -------------------------------------------------------------------------
    atr14       = _calc_atr(high, low, close, 14)
    atr_avg     = atr14.rolling(20).mean()
    atr_contracting = atr14.iloc[-1] < atr_avg.iloc[-1]
    vol_dry_up   = volume.iloc[-1] < avg_vol.iloc[-1] * cfg.cheat_vol_ratio
    cheat_pullback = abs(ema_dist_medium) < cfg.cheat_pullback_pct

    # CRITICAL FIX: cheat entry requires price above the 150-day stage MA (the structural
    # support), not just above EMA50. A stock below its 30-week equivalent MA has no
    # business being flagged as a cheat entry — it's in a downtrend, not a pullback.
    stage_ma_val = stage_ma.iloc[-1] if not np.isnan(stage_ma.iloc[-1]) else 0.0
    cheat_above_structural_ma = px > stage_ma_val and stage_ma_val > 0

    is_cheat = (stage == 2 and cheat_pullback and vol_dry_up
                and atr_contracting and cheat_above_structural_ma)

    result.is_cheat_entry = is_cheat
    result.cheat_detail   = "Pullback Zone — VCP setup" if is_cheat else "Not in Pullback Zone"

    # -------------------------------------------------------------------------
    # Composite score + passed flag
    # -------------------------------------------------------------------------
    result.score  = _composite_score(result, scores)
    result.passed = (stage == 2) or is_cheat

    return result


# -----------------------------------------------------------------------------
# HELPER: Sensitivity multiplier
# -----------------------------------------------------------------------------

def _sensitivity_mult(sensitivity: str) -> float:
    # Multiplied with 0.5 base → final slope_thresh:
    #   Aggressive:   1.0 × 0.5 = 0.5%  (EMA200 must move 0.5% over 10 bars = meaningful upward turn)
    #   Normal:       1.5 × 0.5 = 0.75% (moderate confirmation)
    #   Conservative: 2.5 × 0.5 = 1.25% (requires clear, sustained slope)
    #
    # OLD Aggressive was 0.5 × 0.5 = 0.25% — EMA200 barely ticked up by 3 rupees on a ₹1000
    # stock counted as "rising". Any stock not actively declining passed. Fixed.
    return {"Aggressive": 1.0, "Normal": 1.5, "Conservative": 2.5}.get(sensitivity, 1.0)


# -----------------------------------------------------------------------------
# HELPER: Relative Strength
# -----------------------------------------------------------------------------

def _calc_rs(
    stock_close: pd.Series,
    bench_close: pd.Series,
    ma_length: int,
) -> tuple:
    """Returns (rs_raw, rs_ma, rs_vs_bm_pct, is_strong, rising, falling)"""
    common = stock_close.index.intersection(bench_close.index)
    if len(common) < ma_length + 5:
        return pd.Series(), pd.Series(), 0.0, False, False, False

    sc = stock_close.loc[common]
    bc = bench_close.loc[common]

    rs_raw = sc / bc
    rs_ma  = rs_raw.rolling(ma_length).mean()

    rs_now = rs_raw.iloc[-1]
    rs_ma_now = rs_ma.iloc[-1]
    rs_vs_bm = (rs_now - rs_ma_now) / rs_ma_now * 100 if rs_ma_now > 0 else 0.0

    is_strong = rs_now > rs_ma_now
    rising    = (rs_raw.iloc[-1] > rs_raw.iloc[-2]) and (rs_raw.iloc[-2] > rs_raw.iloc[-3])
    falling   = (rs_raw.iloc[-1] < rs_raw.iloc[-2]) and (rs_raw.iloc[-2] < rs_raw.iloc[-3])

    return rs_raw, rs_ma, rs_vs_bm, is_strong, rising, falling


# -----------------------------------------------------------------------------
# HELPER: Beta (matching Pine Script log-return correlation method)
# -----------------------------------------------------------------------------

def _calc_beta(
    stock_close: pd.Series,
    bench_close: pd.Series,
    length: int,
) -> float:
    common = stock_close.index.intersection(bench_close.index)
    sc = stock_close.loc[common]
    bc = bench_close.loc[common]

    if len(sc) < length + 5:
        return 1.0

    stock_ret = np.log(sc / sc.shift(1)).dropna().iloc[-length:]
    bench_ret = np.log(bc / bc.shift(1)).dropna().iloc[-length:]

    # Align
    common_idx = stock_ret.index.intersection(bench_ret.index)
    sr = stock_ret.loc[common_idx]
    br = bench_ret.loc[common_idx]

    if len(sr) < 20:
        return 1.0

    stock_std = sr.std()
    bench_std = br.std()

    if bench_std == 0:
        return 1.0

    correlation = sr.corr(br)
    beta = correlation * stock_std / bench_std
    return round(float(beta) if not np.isnan(beta) else 1.0, 2)


# -----------------------------------------------------------------------------
# HELPER: ATR
# -----------------------------------------------------------------------------

def _calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()


# -----------------------------------------------------------------------------
# HELPER: PEAD approximation
# Pine Script uses request.earnings() which needs TradingView premium.
# We approximate: detect large gap days (>5%) as earnings proxies,
# then measure drift over pead_window bars from that event.
# For real PEAD: supply an earnings_date parameter from your data source.
# -----------------------------------------------------------------------------

def _calc_pead_approx(
    close: pd.Series,
    threshold: float,
    window: int,
    earnings_date: pd.Timestamp = None,
) -> tuple[float, str]:
    """
    Returns (pead_pct, pead_label).
    If earnings_date is provided (pd.Timestamp), uses that directly.
    Otherwise falls back to gap detection.
    """
    pead_pct = 0.0

    if earnings_date is not None and earnings_date in close.index:
        idx = close.index.get_loc(earnings_date)
        if idx > 0 and idx + 1 < len(close):
            pre_earnings_close = close.iloc[idx - 1]
            post_close = close.iloc[min(idx + window, len(close) - 1)]
            if pre_earnings_close > 0:
                pead_pct = (post_close - pre_earnings_close) / pre_earnings_close * 100
    else:
        # Gap detection fallback
        daily_chg = close.pct_change() * 100
        large_gaps = daily_chg[daily_chg.abs() > 5].index

        if len(large_gaps) > 0:
            last_gap_date = large_gaps[-1]
            gap_idx = close.index.get_loc(last_gap_date)

            if gap_idx > 0:
                pre_close = close.iloc[gap_idx - 1]
                post_idx  = min(gap_idx + window, len(close) - 1)
                post_close = close.iloc[post_idx]
                if pre_close > 0:
                    pead_pct = (post_close - pre_close) / pre_close * 100

    abs_pead = abs(pead_pct)
    if abs_pead > threshold:
        label = "Strong"
    elif abs_pead > threshold / 2:
        label = "Moderate"
    else:
        label = "Weak"

    return round(pead_pct, 2), label


# -----------------------------------------------------------------------------
# HELPER: Stage scoring (faithful translation of Pine scoring system)
# This is stateless — computes stage from current bar data only.
# Pine's var (persistent) variables are approximated via the score margin logic.
# -----------------------------------------------------------------------------

def _calc_stage(
    ma_rising, ma_falling, ma_flat,
    price_above_ma, price_below_ma,
    bullish_ema, bearish_ema,
    rs_is_strong, rs_rising, rs_falling,
    roc_f, roc_s,
    vol_expansion, hh, ll,
    ema_dist_slow,
    ma_slope,
    stage_ma_dist_pct: float,
    cfg: StageAnalysisConfig,
) -> tuple[int, dict, bool, int, int]:
    """
    Returns (stage, scores_dict, is_transition, prev_stage, duration).

    Note: Pine Script uses persistent `var` to track previous stage and duration
    across bars. In a batch screener we only see the final bar, so:
    - We compute stage from scores (stateless)
    - Duration is estimated from MA slope consistency
    - Transition flag is approximated from slope direction change
    """

    # --- Score each stage exactly as Pine does ---
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0

    # S1: Accumulation
    s1 += 3.0 if ma_flat else 0.0
    s1 += 2.0 if abs(ema_dist_slow) < 5 else 0.0
    s1 += 1.0 if (not hh and not ll) else 0.0
    # "coming from S4, slope improving" — approximated by slope recently turning positive
    slope_improving = ma_slope.iloc[-1] > ma_slope.iloc[-cfg.slope_lookback - 1]
    s1 += 2.0 if (ma_falling is False and slope_improving and ma_flat) else 0.0
    s1 += 1.0 if rs_rising else 0.0

    # S2: Advancing
    # Max possible = 10 pts (breakout_proxy removed — see note below)
    #
    # s2_core uses a THREE-TIER graduated condition for the structural MA:
    #
    #   Tier 1 (4 pts) — Confirmed Stage 2:
    #     price_above_ma AND ma_rising         (EMA200 actively rising, textbook S2)
    #     OR price_above_ma AND dist ≥ 15%     (strong breakout: price 15%+ above MA,
    #                                           MA will follow — Minervini fresh base breakout)
    #     OR price_above_ma AND roc_f > 25%    (explosive momentum: 25%+ in 10 days,
    #                                           supernova/surge-type Stage 2 entry)
    #
    #   Tier 2 (3 pts) — Probable Stage 2:
    #     price_above_ma AND 5% ≤ dist < 15%   (meaningfully above MA, MA not yet risen)
    #
    #   Tier 3 (2 pts) — Possible Stage 2:
    #     price_above_ma, dist < 5%             (barely crossed, MA not rising — ambiguous)
    #
    #   0 pts — price below MA (cannot be Stage 2)
    #
    # Rationale: Weinstein's primary S2 condition is price above the structural MA.
    # The MA rising is lagged CONFIRMATION. Fresh breakouts above flat/declining MAs
    # (Minervini first-base pattern) deserve full credit when price is clearly extended
    # above the MA — the MA WILL eventually follow. Stocks barely above a declining MA
    # need secondary confirmation to reach S2_MIN=5.
    #
    # Threshold ≥8.5%: A stock that far above its structural MA (even flat/declining)
    # has clearly separated — the long-term trend is net positive regardless of recent
    # slope. Combined with bullish EMA alignment + RS, this is a full Stage-2 signal.
    if price_above_ma and (ma_rising or stage_ma_dist_pct >= 8.5 or roc_f > 25):
        s2 += 4.0   # confirmed or strong breakout
    elif price_above_ma and stage_ma_dist_pct >= 5:
        s2 += 3.0   # probable: meaningfully above flat/declining MA
    elif price_above_ma:
        s2 += 2.0   # possible: barely above MA, not rising
    # else: price below MA → 0
    s2 += 2.0 if bullish_ema else 0.0                       # short/medium EMA alignment (EF>EM>ES)
    # Minervini SEPA condition #3: EMA50 > EMA200.
    # Mathematically equivalent to stage_ma_dist_pct > ema_dist_slow (both already available).
    # Gives 1 pt when EMA50 has crossed above EMA200 even if EMA21 hasn't yet crossed
    # EMA50 (which would fail the stricter bullish_ema check above). Common in early S2.
    s2 += 1.0 if stage_ma_dist_pct > ema_dist_slow else 0.0  # EMA50 above EMA200
    s2 += 2.0 if rs_is_strong else 0.0                      # outperforming benchmark
    s2 += 1.0 if hh else 0.0                                # 8-week higher high structure
    s2 += 1.0 if roc_f > 0 else 0.0                        # positive short-term momentum
    #
    # breakout_proxy REMOVED.
    # Prior code: s2 += 2.0 if (price_above_ma and vol_expansion and ma_rising).
    # This added +2 just for a single high-volume day on any stock above a rising EMA200,
    # letting vol_expansion alone push S2 from 4 → 6 = S2_MIN without RS or EMA alignment.
    # An earnings-day volume spike or index rebalancing could single-handedly classify a
    # stock as Stage 2. Removed to keep S2 classification clean and methodology-driven.

    # S3: Distribution
    s3 += 3.0 if (ma_flat and (s2 > 0 or price_above_ma)) else 0.0  # "currentStage >= 2" proxy
    s3 += 1.0 if (price_above_ma and not ma_rising) else 0.0
    s3 += 2.0 if (not rs_is_strong) else 0.0
    s3 += 1.5 if (roc_f < 0 and roc_s > 0) else 0.0   # momentum divergence
    s3 += 1.0 if (not bullish_ema and not bearish_ema) else 0.0   # EMAs tangled
    slope_deteriorating = ma_slope.iloc[-1] < ma_slope.iloc[-cfg.slope_lookback - 1]
    s3 += 2.0 if (slope_deteriorating and not ma_falling) else 0.0

    # S4: Declining
    s4 += 4.0 if (price_below_ma and ma_falling) else 0.0
    s4 += 2.0 if bearish_ema else 0.0
    s4 += 1.0 if (not rs_is_strong) else 0.0
    s4 += 1.0 if ll else 0.0
    s4 += 1.0 if roc_f < 0 else 0.0
    # Breakdown from S3
    breakdown_proxy = price_below_ma and vol_expansion and ma_falling
    s4 += 2.0 if breakdown_proxy else 0.0

    scores = {"s1": s1, "s2": s2, "s3": s3, "s4": s4}
    max_score = max(s1, s2, s3, s4)

    # Determine stage from highest score (Pine tie-break: prefer current — we use max)
    #
    # MINIMUM SCORE THRESHOLDS — prevent weak classifications.
    #
    # S2 max possible = 10 (breakout_proxy removed).
    # S2_MIN = 5 means at minimum the stock needs:
    #   Tier 1 (s2_core=4): price above rising MA or strong breakout → just 1 more pt needed
    #   Tier 2 (s2_core=3): 5%+ above flat MA → 2 more pts needed (e.g. rs_strong + roc_f)
    #   Tier 3 (s2_core=2): barely above MA → 3 more pts needed (bullish_ema + rs_strong + roc_f)
    # This means a stock barely crossing the MA needs RS outperformance + EMA alignment
    # to reach S2_MIN, preventing false positives from short-lived MA crossings.
    # Old S2_MIN was 6 with max=12 (including breakout_proxy). Same effective bar.
    #
    # S4 max possible = 11. Requiring ≥ 4 means price below falling MA at minimum.
    # S1/S3 have no minimum — they're accumulation/distribution states that can be ambiguous.
    S2_MIN = 5.0   # below this → classify as S1 or S3 depending on context, not S2
    S4_MIN = 4.0   # below this → not a confirmed downtrend

    if max_score == s2 and s2 >= S2_MIN:   stage = 2
    elif max_score == s4 and s4 >= S4_MIN: stage = 4
    elif max_score == s1:                  stage = 1
    else:                                  stage = 3

    # Transition approximation: did slope direction change recently?
    margin_required = {"Aggressive": 0.5, "Normal": 1.5, "Conservative": 3.0}[cfg.sensitivity]
    # Second-highest score
    sorted_scores = sorted([s1, s2, s3, s4], reverse=True)
    transition = (sorted_scores[0] - sorted_scores[1]) < margin_required * 2

    # Duration approximation: how many bars has MA been consistently rising/falling/flat?
    duration = _estimate_stage_duration(ma_slope, stage, cfg.slope_lookback)

    return stage, scores, transition, 0, duration


def _estimate_stage_duration(ma_slope: pd.Series, stage: int, lookback: int) -> int:
    """
    Estimate how many bars the current stage has persisted by checking
    how long MA slope has been in the same direction.
    """
    direction = "rising" if stage == 2 else "falling" if stage == 4 else "flat"
    count = 0
    for slope_val in reversed(ma_slope.dropna().values):
        if direction == "rising"  and slope_val >  0.25: count += 1
        elif direction == "falling" and slope_val < -0.25: count += 1
        elif direction == "flat" and abs(slope_val) <= 0.5: count += 1
        else: break
    return count


# -----------------------------------------------------------------------------
# HELPER: Composite score for ranking (0–1)
# S2 + cheat entry = highest weight; other metrics add signal
# -----------------------------------------------------------------------------

def _composite_score(result: StageAnalysisResult, scores: dict) -> float:
    total = 0.0

    # Stage 2 is the primary signal — normalise s2 score (max possible = 10 after
    # breakout_proxy removal; was 12 before).
    s2_norm = min(scores.get("s2", 0) / 10.0, 1.0)
    total += s2_norm * 0.35

    # RS strength
    total += 0.15 if result.rs_is_strong else 0.0
    total += 0.05 if result.rs_rising    else 0.0

    # Momentum
    if "↑↑" in result.mom_label:   total += 0.15
    elif "↑"  in result.mom_label:  total += 0.08

    # Cheat entry bonus
    total += 0.15 if result.is_cheat_entry else 0.0

    # Volume conviction
    if result.vol_conviction == "Very High": total += 0.08
    elif result.vol_conviction == "High":    total += 0.04

    # Dollar volume trend
    total += 0.05 if result.vol_trend else 0.0

    # Beta penalty: very high beta (>2) is riskier
    if result.beta > 2.0: total -= 0.05

    return round(min(max(total, 0.0), 1.0), 4)
