# =============================================================================
# SEPA — Specific Entry Point Analysis  v3
# Minervini + Weinstein + Livermore methodology
# =============================================================================
#
# ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
#
#   Stage Analysis (daily) — already run by ranker — passes only Stage 2
#       ↓
#   Market Regime multiplier — applied once per run in ranker
#       ↓
#   PATH DECISION  (based on stage_duration):
#
#   Path A  stage_duration ≤ 20 bars → "Fresh Stage 2 Breakout"
#   ────────────────────────────────────────────────────────────
#   Score the QUALITY of the initial Stage 1 → Stage 2 breakout.
#   Weekly MA gate: weekly 30-wk MA must be turning up (not falling).
#
#   Components:
#     Vol surge on breakout day   40%
#     RS line leading price       25%
#     Stage 1 base tightness      20%
#     Extension above pivot       15%
#
#   Multipliers: regime × base_count (always 1 for first base)
#
#
#   Path B  stage_duration > 20 bars → "VCP Base Scoring"
#   ────────────────────────────────────────────────────────────
#   Detect if the stock is CURRENTLY in a valid consolidation base, then
#   score the quality of that base.
#
#   Hard gates (any failure → score 0):
#     1. detect_base()          — must find valid consolidation (not a trend)
#     2. Weekly stage check     — weekly must not be Stage 3 or 4
#     3. Distribution gate      — accumulation_ratio < 0.8 AND churn ≥ 2 → fail
#
#   Components:
#     VCP contractions          25%   — number + quality of shrinking swings
#     Volume character          20%   — up-day vs down-day vol + churn detection
#     ATR contraction           15%   — ATR first half vs second half of base
#     RS line leading           15%   — RS new high before price breaks out
#     Volume dry-up             10%   — recent 5d vs pre-base 20d average
#     Current tightness         10%   — last-third CV of closes
#     Pivot proximity            5%   — distance from base high (pivot)
#
#   Multipliers: regime × base_count_mult × weekly_cap
#
#   Base count multipliers:
#     1st base  × 1.00  (highest probability — Minervini's #1 setup)
#     2nd base  × 0.85
#     3rd base  × 0.65
#     4th base  × 0.40
#     5th+      × 0.15
#
# =============================================================================

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from screeners.stage_analysis import StageAnalysisResult


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SEPAConfig:
    # Base detection
    base_min_bars:         int   = 35    # minimum consolidation length (bars) — 7 weeks minimum
    base_max_bars:         int   = 65    # maximum (~13 weeks)
    base_max_depth_pct:    float = 35.0  # (high - low) / high must be < this
    base_max_up_slope:     float = 0.20  # %/bar — steeper = trending, not basing
    base_max_down_slope:   float = 0.30  # %/bar — strong downtrend = Stage 4 risk

    # VCP contraction detection
    vcp_swing_window:      int   = 4     # bars either side for local high/low detection (4 = less noise)
    vcp_min_prominence:    float = 3.0   # swing amplitude must be ≥ this % to count
    vcp_contraction_req:   float = 0.80  # each contraction must be < this × prior

    # Volume character
    churn_vol_mult:        float = 1.5   # vol > this × base avg = high-vol bar
    churn_range_pct:       float = 0.5   # range < this % of price = narrow (churn signal)
    dist_acc_threshold:    float = 0.8   # accumulation_ratio below this = bearish
    dist_churn_threshold:  int   = 2     # churn_count ≥ this → distribution hard-fail

    # RS lookback
    rs_lookback:           int   = 252

    # Volume dry-up
    vol_recent_bars:       int   = 5     # recent vol window (bars)
    vol_pre_base_bars:     int   = 20    # pre-base vol window (bars)

    # Current tightness
    tightness_third:       float = 0.33  # use last X fraction of base for CV
    tightness_min_bars:    int   = 5

    # Path A / B threshold
    fresh_breakout_bars:   int   = 20    # stage_duration ≤ this → Path A

    # Stage duration ideal range (for base count approximation)
    stage_min_bars:        int   = 10
    stage_max_bars:        int   = 60


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BaseResult:
    valid:           bool  = False
    length_bars:     int   = 0
    base_high:       float = 0.0
    base_low:        float = 0.0
    depth_pct:       float = 0.0
    atr_first_half:  float = 0.0
    atr_second_half: float = 0.0
    atr_contraction: float = 1.0   # second_half / first_half; < 1.0 = coiling
    pre_base_vol:    float = 0.0
    in_base_vol:     float = 0.0
    reason:          str   = ""


@dataclass
class VCPResult:
    num_contractions:      int   = 0
    last_is_tightest:      bool  = False
    contraction_pcts:      list  = field(default_factory=list)
    final_contraction_pct: float = 0.0
    vcp_score:             float = 0.0


@dataclass
class SEPAResult:
    ticker: str = ""

    # ── From Stage Analysis ──────────────────────────────────────────────────
    stage:            int   = 0
    stage_full:       str   = ""
    stage_score_s2:   float = 0.0
    stage_duration:   int   = 0
    rs_vs_benchmark:  float = 0.0
    rs_status:        str   = ""
    mom_label:        str   = ""
    roc_fast:         float = 0.0
    avg_dollar_vol:   float = 0.0
    ema_dist_fast:    float = 0.0
    ema_dist_medium:  float = 0.0
    ema_dist_slow:    float = 0.0
    beta:             float = 1.0
    beta_label:       str   = ""
    vol_conviction:   str   = ""
    vol_ratio:        float = 1.0
    market_cap:       float = 0.0
    is_cheat_entry:   bool  = False

    # ── Scoring path + stage context ─────────────────────────────────────────
    scoring_path:    str = ""      # "A: Fresh Breakout" | "B: VCP Base"
    weekly_stage:    int = 0       # Weinstein 40-week EMA stage: 0=unknown 1=accum 2=advancing 3=dist 4=decline
    n_template:      int = 0       # Minervini template conditions met (0–5); 4+ = passed

    # ── Base detection (Path B) ───────────────────────────────────────────────
    base_valid:       bool  = False
    base_length_bars: int   = 0
    base_high:        float = 0.0
    base_low:         float = 0.0
    base_depth_pct:   float = 0.0
    atr_contraction:  float = 1.0
    pre_base_vol:     float = 0.0
    in_base_vol:      float = 0.0
    base_reason:      str   = ""

    # ── VCP (Path B) ─────────────────────────────────────────────────────────
    num_contractions:      int   = 0
    last_is_tightest:      bool  = False
    final_contraction_pct: float = 0.0
    vcp_score:             float = 0.0

    # ── Volume character (Path B) ─────────────────────────────────────────────
    accumulation_ratio: float = 1.0
    churn_count:        int   = 0
    vol_char_score:     float = 0.0

    # ── RS leading (both paths) ───────────────────────────────────────────────
    rs_leading:          bool  = False
    rs_new_high_in_base: bool  = False
    rs_rising_in_base:   bool  = False
    rs_leading_score:    float = 0.0

    # ── Volume dry-up (Path B) ────────────────────────────────────────────────
    vol_dry_ratio:  float = 1.0
    vol_dry_score:  float = 0.0

    # ── Current tightness (Path B — last third of base) ───────────────────────
    current_cv_pct:  float = 0.0
    tightness_score: float = 0.0

    # ── ATR contraction score (Path B) ────────────────────────────────────────
    atr_score: float = 0.0

    # ── Pivot proximity ───────────────────────────────────────────────────────
    pivot_dist_pct: float = 0.0
    pivot_score:    float = 0.0

    # ── Breakout state ────────────────────────────────────────────────────────
    breakout_state: str = ""  # AT_PIVOT | IN_BASE | BREAKOUT | WEAK_BREAKOUT | FADING | EXTENDED

    # ── Base count (Path B) ───────────────────────────────────────────────────
    base_count:      int   = 1
    base_count_mult: float = 1.0

    # ── Path A specific ───────────────────────────────────────────────────────
    vol_surge_ratio: float = 0.0
    s1_cv_pct:       float = 0.0
    extension_pct:   float = 0.0

    # ── Absolute prices (for trade execution output) ─────────────────────────
    price:         float = 0.0   # current close price
    entry_price:   float = 0.0   # computed buy-stop level (base_high + 0.5%)
    raw_sepa_score: float = 0.0  # SEPA score BEFORE regime multiplier (for trade ranking)

    # ── Stop (display only, not scored) ──────────────────────────────────────
    stop_price:    float = 0.0
    stop_dist_pct: float = 0.0

    # ── Setup classification ──────────────────────────────────────────────────
    setup_stage: str = ""

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi_14:     float = 50.0  # RSI(14) at analysis time — entry timing signal

    # ── Final scores ──────────────────────────────────────────────────────────
    sepa_score: float = 0.0
    score:      float = 0.0
    passed:     bool  = False


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_sepa_analysis(
    df:           pd.DataFrame,
    benchmark_df: pd.DataFrame,
    stage_result: StageAnalysisResult,
    ticker:       str       = "",
    market:       str       = "india",
    cfg:          SEPAConfig = None,
    weights:      dict      = None,
    regime_mult:  float     = 1.0,
    weekly_stage: int       = 0,
) -> SEPAResult:
    """
    Compute SEPA entry-quality score for a Stage-2 ticker.

    Args:
        df:           Daily OHLCV (DatetimeIndex ascending).
        benchmark_df: Benchmark OHLCV for RS line calculation.
        stage_result: Pre-computed StageAnalysisResult.
        ticker:       Symbol string.
        market:       'india' or 'us'.
        cfg:          SEPAConfig (defaults used if None).
        weights:      Path B component weights (from config.SEPA_WEIGHTS if None).
        regime_mult:  Market regime multiplier (0.0–1.0), computed once in ranker.
        weekly_stage: Stage from weekly chart (0 = unknown, 2 = confirmed, 3/4 = bad).
    """
    if cfg is None:
        cfg = SEPAConfig()
    if weights is None:
        weights = _default_weights()

    r = SEPAResult(ticker=ticker)
    _copy_stage_fields(r, stage_result)

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    price  = float(close.iloc[-1])

    if price <= 0 or len(df) < 60:
        return r

    # ── RS Line (used in both paths) ──────────────────────────────────────────
    bench_close = (
        benchmark_df["close"]
        .reindex(df.index, method="ffill")
        .replace(0, np.nan)
    )
    rs_line = (close / bench_close).dropna()

    # ── Natural stop (display only) ───────────────────────────────────────────
    slb = min(20, len(low))
    low_20d = float(low.iloc[-slb:].min())
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    r.stop_price = max(low_20d, ema50) if ema50 < price else low_20d
    if r.stop_price >= price:
        r.stop_price = low_20d
    r.stop_dist_pct = (price - r.stop_price) / price * 100 if price > 0 else 15.0

    # ── Base detection — run once, used to route to correct path ────────────────
    # A trending stock naturally fails (highest high is very recent → base too short).
    # A stock in a real consolidation passes (highest high is 20+ bars ago).
    base = detect_base(high, low, close, volume, cfg)

    # ── Path decision based on base state + price position vs pivot ─────────────
    #
    #   Path A: price has broken ABOVE a detected base (just broke out), OR the
    #           stock is truly fresh Stage 2 (≤ fresh_breakout_bars in S2).
    #
    #   Path B: price is INSIDE a valid consolidation — score VCP quality.
    #
    #   No Setup: established Stage 2 trending stock with no current base.
    #             Scores near-zero so it appears at the bottom of the list only
    #             when no better setups exist.
    #
    if base.valid and price > base.base_high:
        # Broke out above the detected consolidation pivot → entry quality score
        r.scoring_path = "A: Fresh Breakout"
        _score_path_a(r, close, high, low, volume, rs_line, price, cfg, pre_base=base, weekly_stage=weekly_stage)
    elif base.valid:
        # Still inside the consolidation → VCP / coiling quality score
        r.scoring_path = "B: VCP Base"
        _score_path_b(r, close, high, low, volume, rs_line, price, cfg, weights, weekly_stage, pre_base=base)
    elif r.stage_duration <= cfg.fresh_breakout_bars:
        # No detectable base yet — truly fresh Stage 2 (≤ 20 bars in)
        r.scoring_path = "A: Fresh Breakout"
        _score_path_a(r, close, high, low, volume, rs_line, price, cfg, pre_base=None, weekly_stage=weekly_stage)
    else:
        # Established Stage 2 trend with no consolidation base — not actionable
        r.scoring_path  = "C: Trending (No Setup)"
        r.setup_stage   = "🚫 No Base (Trending)"
        r.sepa_score    = 2.0
        r.base_count    = 1
        r.base_count_mult = 1.0

    # ── Store stage context for display ──────────────────────────────────────
    r.weekly_stage = weekly_stage

    # ── BREAKOUT stop override ────────────────────────────────────────────────
    # For confirmed BREAKOUT / WEAK_BREAKOUT: tighten stop to just below the
    # pivot (base_high × 0.99) instead of the 20d_low or EMA50 deep inside the
    # old base.  Minervini: once above the pivot the pivot BECOMES the support.
    # Without this, a stock at ₹102 over a ₹100 pivot gets a ₹85 stop (17%) and
    # fails the 9% stop gate in the trade ranker entirely.
    if r.breakout_state in ("BREAKOUT", "WEAK_BREAKOUT") and r.base_high > 0:
        pivot_stop = r.base_high * 0.99          # 1% below the pivot line
        if pivot_stop > r.stop_price:            # only tighten, never widen
            r.stop_price    = pivot_stop
            r.stop_dist_pct = (price - r.stop_price) / price * 100 if price > 0 else 15.0

    # ── RSI(14) — entry timing modifier ──────────────────────────────────────
    # Computed AFTER path scoring so the path already set r.sepa_score.
    # Applied as a multiplier: ideal zone (50-65) = slight bonus; very
    # extended (>82) = meaningful penalty; momentum absent (<40) = penalty.
    r.rsi_14 = _calc_rsi(close)
    r.sepa_score = round(r.sepa_score * _rsi_sepa_modifier(r.rsi_14, r.breakout_state), 1)

    # ── Absolute prices for trade execution ──────────────────────────────────
    r.price        = price
    # Entry = 0.5% above the pivot (base_high). For BREAKOUT state where price
    # is already above pivot, use current price as the entry reference.
    pivot = r.base_high if r.base_high > 0 else price
    if r.breakout_state in ("BREAKOUT", "WEAK_BREAKOUT"):
        r.entry_price = round(price, 2)          # already above pivot — enter at market
    else:
        r.entry_price = round(pivot * 1.005, 2)  # buy-stop 0.5% above pivot

    # ── Capture raw score BEFORE regime multiplier ────────────────────────────
    # Used by trade ranker to rank stocks on their own merits independently
    # of current market conditions (regime is shown as a separate warning).
    r.raw_sepa_score = round(r.sepa_score * r.base_count_mult, 1)

    # ── External multipliers ──────────────────────────────────────────────────
    r.sepa_score = round(r.sepa_score * r.base_count_mult * regime_mult, 1)
    r.score      = r.sepa_score

    r.passed = (r.stage >= 2) or stage_result.is_cheat_entry
    return r


# =============================================================================
# PATH A — FRESH STAGE 2 BREAKOUT  (stage_duration ≤ 20 bars)
# =============================================================================

def _score_path_a(
    r:            SEPAResult,
    close:        pd.Series,
    high:         pd.Series,
    low:          pd.Series,
    volume:       pd.Series,
    rs_line:      pd.Series,
    price:        float,
    cfg:          SEPAConfig,
    pre_base:     "BaseResult | None" = None,
    weekly_stage: int = 0,
) -> None:
    """
    Score the quality of a fresh Stage 1 → Stage 2 breakout.

    Two distinct volume regimes — the key Minervini principle:
      PRE-BREAKOUT  (AT_PIVOT / IN_BASE): volume should DRY UP — sellers exhausted,
                    stock coiling before explosive move. Low vol near pivot = bullish.
      ON BREAKOUT   (BREAKOUT state): volume must SURGE ≥ 1.4× avg — institutions
                    absorbing supply. Low vol on breakout = suspect / failed.

    Key signals:
      1. Volume character — dry-up (pre-breakout) or surge (on breakout) (40%)
      2. RS line was already at new highs before/at the breakout (25%)
      3. Tightness of the Stage 1 base the stock broke out of (20%)
      4. Extension above pivot — how much chasing is required (15%)

    pre_base: a valid BaseResult if the stock just broke out of a detected
              consolidation base. When provided, base stats come from detect_base
              output rather than the stage_duration heuristic.
    weekly_stage: Weinstein weekly stage (0=unknown, 1=transition, 2=advancing, 3/4=bad)
    """
    # ── Weekly gate (same as Path B) ─────────────────────────────────────────
    if weekly_stage in (3, 4):
        r.setup_stage = "🚫 Weekly Stage 3/4"
        r.sepa_score  = 0.0
        return
    # weekly_stage=1 (transitioning) → cap composite at 60%; fresh breakouts CAN be
    # in weekly Stage 1 transitioning to Stage 2 — this is actually ideal but warrants caution
    weekly_cap = 1.0 if weekly_stage != 1 else 0.6

    n   = len(close)
    dur = max(r.stage_duration, 1)

    vol_50d = float(volume.rolling(50, min_periods=20).mean().iloc[-1])

    # ── Step 1: Resolve pivot and base stats (FIRST — needed for state detection) ──
    # We compute base_high before the breakout state so the vol regime
    # decision below can correctly identify AT_PIVOT vs BREAKOUT.
    if pre_base is not None:
        # Detected base — use accurate stats from detect_base()
        r.base_valid       = True
        r.base_length_bars = pre_base.length_bars
        r.base_high        = pre_base.base_high
        r.base_low         = pre_base.base_low
        r.base_depth_pct   = pre_base.depth_pct
        r.atr_contraction  = pre_base.atr_contraction
        all_closes  = close.iloc[-pre_base.length_bars:]
        # Exclude post-breakout bars (above base_high): they inflate CV and
        # understate base tightness for stocks that have already broken out.
        in_base = all_closes[all_closes <= pre_base.base_high * 1.01]
        base_closes = in_base if len(in_base) >= 5 else all_closes
        mean_c      = float(base_closes.mean())
        r.s1_cv_pct = float(base_closes.std() / mean_c * 100) if mean_c > 0 else 10.0
    else:
        # No detected base — estimate Stage 1 boundary from stage_duration
        s1_end   = max(0, n - dur)
        s1_start = max(0, s1_end - 40)
        if s1_end > s1_start + 5:
            s1_closes = close.iloc[s1_start:s1_end]
            r.s1_cv_pct = (
                float(s1_closes.std() / s1_closes.mean() * 100)
                if float(s1_closes.mean()) > 0 else 10.0
            )
            r.base_high = float(high.iloc[s1_start:s1_end].max())
        else:
            r.s1_cv_pct = 5.0
            r.base_high = price * 0.90
        r.base_valid = r.base_high > 0

    s1_tightness_score = _score_vcp_cv(r.s1_cv_pct)

    # ── Step 2: Extension / pivot distance ────────────────────────────────────
    r.extension_pct  = max(0.0, (price - r.base_high) / r.base_high * 100) if r.base_high > 0 else 5.0
    r.pivot_dist_pct = (price - r.base_high) / r.base_high * 100 if r.base_high > 0 else 0.0
    extension_score  = _score_extension(r.extension_pct)

    # ── Step 3: Breakout state (now that we have pivot) ───────────────────────
    r.breakout_state = detect_breakout_state(price, r.base_high, volume, vol_50d)

    # ── Step 4: Volume — TWO REGIMES based on state ───────────────────────────
    #
    # Minervini's core principle for setup identification:
    #
    # PRE-BREAKOUT (AT_PIVOT / IN_BASE / EARLY):
    #   Volume should be QUIET — sellers have dried up, supply exhausted.
    #   A low recent-volume ratio (0.4–0.6× avg) near the pivot signals the stock
    #   is coiling like a spring before an explosive move.
    #   → Use _score_vol_dry(): lower ratio = higher score
    #   → r.vol_surge_ratio stores the dry-up ratio (< 1.0 = drying up)
    #
    # POST-BREAKOUT (BREAKOUT / WEAK_BREAKOUT / FADING / EXTENDED):
    #   Volume must SURGE on the specific bar price crosses the pivot.
    #   We scan backward to find that exact breakout bar (not max of any recent bar).
    #   → Use _score_vol_surge(): higher ratio = higher score
    #
    if r.breakout_state in ("AT_PIVOT", "IN_BASE", "EARLY"):
        # Pre-breakout: measure 5-bar avg vol vs 50d avg (smooth out single-day spikes)
        recent_5d_vol     = float(volume.iloc[-5:].mean()) if n >= 5 else float(volume.iloc[-1])
        r.vol_surge_ratio = recent_5d_vol / vol_50d if vol_50d > 0 else 1.0
        vol_surge_score   = _score_vol_dry(r.vol_surge_ratio)

    elif pre_base is not None:
        # Post-breakout with a detected base: find the EXACT bar where price
        # first crossed above the pivot. That bar's volume is the confirmation signal.
        # (Scan up to 20 bars back; if no crossing found, fall back to current bar.)
        breakout_vol = float(volume.iloc[-1])
        for j in range(n - 1, max(n - 20, -1), -1):
            if float(close.iloc[j]) <= pre_base.base_high:
                bo_bar       = min(j + 1, n - 1)
                breakout_vol = float(volume.iloc[bo_bar])
                break
        r.vol_surge_ratio = breakout_vol / vol_50d if vol_50d > 0 else 1.0
        vol_surge_score   = _score_vol_surge(r.vol_surge_ratio)

    else:
        # No detected base, post-breakout: use the peak volume in the first 5 Stage-2 bars
        # (the point where the stock broke above Stage 1 resistance)
        s2_start          = max(0, n - dur)
        s2_breakout       = volume.iloc[s2_start:min(n, s2_start + 5)]
        r.vol_surge_ratio = (
            float(s2_breakout.max()) / vol_50d
            if vol_50d > 0 and len(s2_breakout) > 0
            else 1.0
        )
        vol_surge_score = _score_vol_surge(r.vol_surge_ratio)

    # ── Step 5: RS line leading ───────────────────────────────────────────────
    if len(rs_line) >= 20:
        # When we have a detected base, the Stage-2 boundary is the start of that
        # base — more accurate than the unreliable stage_duration estimate.
        stage2_idx = -(pre_base.length_bars + 1) if pre_base is not None else -(dur + 1)
        rs_led, rs_new_high, rs_rising = _detect_rs_leading_path_a(rs_line, stage2_idx)
        r.rs_leading          = rs_led
        r.rs_new_high_in_base = rs_new_high
        r.rs_rising_in_base   = rs_rising
    rs_leading_score = (
        100.0 if r.rs_leading else
        70.0  if r.rs_new_high_in_base else
        40.0  if r.rs_rising_in_base else
        10.0
    )
    r.rs_leading_score = rs_leading_score

    # ── Step 6: Composite score ───────────────────────────────────────────────
    raw = (
        0.40 * vol_surge_score    +
        0.25 * rs_leading_score   +
        0.20 * s1_tightness_score +
        0.15 * extension_score
    )
    raw_score = round(raw * weekly_cap, 1)

    # ── Step 7: Breakout quality gate (applies AFTER raw score) ───────────────
    #
    # EXTENDED  (>10% above pivot):      TOO LATE — stop too far, risk/reward blown
    # FADING    (above pivot, <avg vol): FAILED BREAKOUT — above pivot but no buyers.
    #                                    One of the worst signals — do not trade.
    # WEAK_BREAKOUT (above pivot, ~1×):  SUSPECT — possible distribution. Reduce.
    # BREAKOUT  (above pivot, ≥1.4×):   CONFIRMED — full score
    # AT_PIVOT / IN_BASE:                Pre-breakout — full score (vol dry-up already scored)
    #
    if r.breakout_state == "EXTENDED":
        r.sepa_score = min(raw_score, 10.0)           # cap hard — too late to buy
    elif r.breakout_state == "FADING":
        r.sepa_score = round(raw_score * 0.15, 1)    # near-zero — failed breakout warning
    elif r.breakout_state == "WEAK_BREAKOUT":
        r.sepa_score = round(raw_score * 0.60, 1)    # moderate penalty — uncertain volume
    else:
        r.sepa_score = raw_score                      # BREAKOUT / AT_PIVOT / IN_BASE: full

    # Base count always 1 for first Stage 2 base
    r.base_count      = 1
    r.base_count_mult = 1.0

    r.setup_stage = _classify_path_a(
        r.vol_surge_ratio, r.extension_pct, r.rs_leading, r.breakout_state
    )


# =============================================================================
# PATH B — VCP BASE SCORING  (stage_duration > 20 bars)
# =============================================================================

def _score_path_b(
    r:            SEPAResult,
    close:        pd.Series,
    high:         pd.Series,
    low:          pd.Series,
    volume:       pd.Series,
    rs_line:      pd.Series,
    price:        float,
    cfg:          SEPAConfig,
    weights:      dict,
    weekly_stage: int,
    pre_base:     "BaseResult | None" = None,
) -> None:
    """
    Score an established Stage 2 stock that is currently inside a VCP base.

    Hard gates (any failure → score 0 and early return):
      1. detect_base()   — must find a valid consolidation (pre_base skips this)
      2. Weekly check    — weekly must not be Stage 3 or 4
      3. Distribution    — accumulation_ratio < 0.8 AND churn ≥ 2

    Component scoring within detected base boundaries.
    """
    # ── Weekly gate ────────────────────────────────────────────────────────────
    if weekly_stage in (3, 4):
        r.setup_stage = "🚫 Weekly Stage 3/4"
        r.sepa_score  = 0.0
        return
    # weekly_stage=1 (transition) → cap composite at 50; weekly_stage=0 (unknown) → no penalty
    weekly_cap = 1.0 if weekly_stage != 1 else 0.5

    # ── Base detection (use pre-detected result if available) ─────────────────
    base = pre_base if pre_base is not None else detect_base(high, low, close, volume, cfg)
    r.base_valid       = base.valid
    r.base_reason      = base.reason

    if not base.valid:
        r.setup_stage = f"🚫 No Base ({base.reason})"
        r.sepa_score  = 0.0
        return

    r.base_length_bars = base.length_bars
    r.base_high        = base.base_high
    r.base_low         = base.base_low
    r.base_depth_pct   = base.depth_pct
    r.atr_contraction  = base.atr_contraction
    r.pre_base_vol     = base.pre_base_vol
    r.in_base_vol      = base.in_base_vol

    base_idx = -base.length_bars   # negative index for slicing

    # ── Breakout state ─────────────────────────────────────────────────────────
    vol_50d = float(volume.rolling(50, min_periods=20).mean().iloc[-1])
    r.breakout_state = detect_breakout_state(price, base.base_high, volume, vol_50d)
    r.pivot_dist_pct = (price - base.base_high) / base.base_high * 100 if base.base_high > 0 else 0.0

    if r.breakout_state == "EXTENDED":
        r.setup_stage = "🔴 Extended"
        r.sepa_score  = 5.0
        r.base_count_mult = _base_count_multiplier(_count_bases_in_stage2(high, close, r.stage_duration))
        return

    # ── VCP contractions ────────────────────────────────────────────────────────
    vcp = count_vcp_contractions(high, low, base_idx, cfg)
    r.num_contractions      = vcp.num_contractions
    r.last_is_tightest      = vcp.last_is_tightest
    r.final_contraction_pct = vcp.final_contraction_pct
    r.vcp_score             = vcp.vcp_score

    # ── Volume character ───────────────────────────────────────────────────────
    acc_ratio, churn_cnt = analyze_base_volume_character(close, high, low, volume, base_idx, cfg)
    r.accumulation_ratio = acc_ratio
    r.churn_count        = churn_cnt

    # Distribution hard gate
    if acc_ratio < cfg.dist_acc_threshold and churn_cnt >= cfg.dist_churn_threshold:
        r.setup_stage = "🔴 Distribution"
        r.sepa_score  = 5.0
        r.base_count_mult = _base_count_multiplier(_count_bases_in_stage2(high, close, r.stage_duration))
        return

    r.vol_char_score = _score_vol_character(acc_ratio, churn_cnt)

    # ── RS leading ─────────────────────────────────────────────────────────────
    if len(rs_line) >= 20:
        rs_led, rs_new_high, rs_rising = _detect_rs_leading(rs_line, base_idx, base.base_high, price)
        r.rs_leading          = rs_led
        r.rs_new_high_in_base = rs_new_high
        r.rs_rising_in_base   = rs_rising
    r.rs_leading_score = (
        100.0 if r.rs_leading else
        70.0  if r.rs_new_high_in_base else
        40.0  if r.rs_rising_in_base else
        10.0
    )

    # ── Volume dry-up (recent vs pre-base) ────────────────────────────────────
    recent_vol = float(volume.iloc[-cfg.vol_recent_bars:].mean())
    pre_vol    = base.pre_base_vol if base.pre_base_vol > 0 else vol_50d
    r.vol_dry_ratio = recent_vol / pre_vol if pre_vol > 0 else 1.0
    r.vol_dry_score = _score_vol_dry(r.vol_dry_ratio)

    # ── Current tightness — COILING not flat-stock bias ──────────────────────
    # Minervini's "tightness" means the final portion of the base is COILING TIGHTER
    # than the earlier part. A perpetually flat stock (NTPC, utility) will have low
    # CV throughout — that is NOT a VCP signal. We score the coiling ratio:
    #   last-third CV / first-two-thirds CV → lower = more genuine coiling.
    third     = max(cfg.tightness_min_bars, int(base.length_bars * cfg.tightness_third))
    full_base_cls  = close.iloc[-base.length_bars:]
    early_cls      = full_base_cls.iloc[:-third] if len(full_base_cls) > third else full_base_cls
    late_cls       = close.iloc[-third:]

    mean_late  = float(late_cls.mean())
    mean_early = float(early_cls.mean())
    late_cv    = float(late_cls.std()  / mean_late  * 100) if mean_late  > 0 else 99.0
    early_cv   = float(early_cls.std() / mean_early * 100) if mean_early > 0 else 99.0

    # Coiling ratio: how much tighter is the last third vs the earlier portion?
    # < 0.5 = genuine compression; > 1.0 = expanding (anti-VCP signal)
    coiling_ratio = late_cv / early_cv if early_cv > 0 else 1.0

    r.current_cv_pct  = late_cv
    r.tightness_score = _score_tightness_coiling(late_cv, coiling_ratio)

    # ── ATR contraction score ─────────────────────────────────────────────────
    # Pass the pre-base ATR baseline so the scoring can verify that ATR was
    # actually elevated before the base (not a perpetually low-vol stock).
    # pre_base ATR proxy: look at ATR during the 20-bar pre-base period.
    pre_b_end   = len(close) - base.length_bars
    pre_b_start = max(0, pre_b_end - 20)
    if pre_b_end > pre_b_start + 5:
        pb_h = high.iloc[pre_b_start:pre_b_end]
        pb_l = low.iloc[pre_b_start:pre_b_end]
        pb_c = close.iloc[pre_b_start:pre_b_end]
        pre_atr_series = _calc_atr_series(pb_h, pb_l, pb_c)
        pre_base_atr = float(pre_atr_series.mean())
    else:
        pre_base_atr = base.atr_first_half   # fallback

    r.atr_score = _score_atr_contraction(base.atr_contraction, base.atr_second_half,
                                          base.base_high, pre_base_atr)

    # ── Pivot proximity ────────────────────────────────────────────────────────
    r.pivot_score = _score_pivot_dist(r.pivot_dist_pct)

    # ── Base count ─────────────────────────────────────────────────────────────
    r.base_count      = _count_bases_in_stage2(high, close, r.stage_duration)
    r.base_count_mult = _base_count_multiplier(r.base_count)

    # ── Composite score ────────────────────────────────────────────────────────
    raw = (
        weights.get("vcp_contractions",  0.25) * r.vcp_score       +
        weights.get("vol_character",     0.20) * r.vol_char_score   +
        weights.get("atr_contraction",   0.15) * r.atr_score        +
        weights.get("rs_leading",        0.15) * r.rs_leading_score +
        weights.get("vol_dry_up",        0.10) * r.vol_dry_score    +
        weights.get("current_tightness", 0.10) * r.tightness_score  +
        weights.get("pivot_proximity",   0.05) * r.pivot_score
    )

    r.sepa_score  = round(raw * weekly_cap, 1)
    r.setup_stage = _classify_path_b(r.pivot_dist_pct, r.vol_dry_ratio, r.atr_contraction,
                                     r.breakout_state, r.num_contractions)


# =============================================================================
# BASE DETECTION
# =============================================================================

def detect_base(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    volume: pd.Series,
    cfg:    SEPAConfig,
) -> BaseResult:
    """
    Find the most recent valid consolidation base within the last max_bars bars.

    Algorithm:
      1. Find the highest high in the search window [-(max_bars) to -3].
         Excluding the last 3 bars avoids counting a current breakout spike.
      2. Everything from that peak to today = the base window.
      3. Validate:
         a. Base length between min_bars and max_bars.
         b. Max drawdown (high - low) / high < max_depth_pct.
         c. Slope of closes within base < max_up_slope %/bar
            (steeper = still trending, not consolidating).
            Strong downslope < -max_down_slope also fails (Stage 4 within base).
         d. ATR first half vs second half (measured, not a hard gate — used in scoring).

    A trending stock naturally fails because its highest high is very recent
    (< min_bars ago), giving base_length < min_bars.
    """
    n = len(close)
    max_bars = min(cfg.base_max_bars, n - 5)

    if n < cfg.base_min_bars + 10 or max_bars < cfg.base_min_bars:
        return BaseResult(reason="Insufficient data")

    # Search window: exclude last 3 bars to avoid counting current extension
    search_highs = high.iloc[-max_bars:-3]
    if len(search_highs) == 0:
        return BaseResult(reason="Empty search window")

    reset_highs = search_highs.reset_index(drop=True)

    # ── Find the most recent significant pivot high (base ceiling) ────────────
    # Livermore / Minervini: the base forms below the MOST RECENT pivot high.
    # Using argmax() (absolute highest bar) is wrong for V-shaped recoveries:
    #   stock peaked at ₹130 → fell to ₹90 → recovered to ₹125 → consolidating
    #   argmax() picks ₹130 as base_high, spanning the entire V as the "base".
    #   The real base started at ₹125 (the most recent local high).
    #
    # Algorithm: scan RIGHT-TO-LEFT for the most recent bar that is:
    #   1. A local maximum (higher than its 5-bar neighbours)
    #   2. At least 90% of the window's absolute maximum (significant, not noise)
    # This finds the most recent shoulder / resistance level, not the all-time high.
    abs_max_val = float(reset_highs.max())
    significance_thresh = abs_max_val * 0.90
    sw = 5   # swing-high detection window

    peak_pos = int(reset_highs.argmax())   # fallback: absolute max

    n_sh = len(reset_highs)
    for i in range(n_sh - sw - 1, sw - 1, -1):   # right to left
        h = float(reset_highs.iloc[i])
        if h < significance_thresh:
            continue
        # Local max check: higher than sw neighbours on each side
        left_max  = float(reset_highs.iloc[max(0, i - sw):i].max()) if i > 0 else 0.0
        right_max = float(reset_highs.iloc[i + 1:min(n_sh, i + sw + 1)].max())
        if h >= left_max and h >= right_max:
            peak_pos = i   # most recent significant local peak
            break

    # Translate peak_pos → bars since pivot from today
    # search_highs spans -(max_bars) to -4 inclusive
    # element 0 = max_bars bars ago; element peak_pos = (max_bars - peak_pos) bars ago
    bars_since_pivot = max_bars - peak_pos
    base_length = bars_since_pivot

    if base_length < cfg.base_min_bars:
        return BaseResult(reason=f"Pivot too recent ({base_length} bars)")
    if base_length > cfg.base_max_bars:
        return BaseResult(reason=f"Pivot too old ({base_length} bars)")

    # Extract base window
    bh = high.iloc[-base_length:]
    bl = low.iloc[-base_length:]
    bc = close.iloc[-base_length:]

    base_high_val = float(bh.max())
    base_low_val  = float(bl.min())

    # Depth check
    depth_pct = (base_high_val - base_low_val) / base_high_val * 100
    if depth_pct > cfg.base_max_depth_pct:
        return BaseResult(reason=f"Too deep ({depth_pct:.1f}%)")

    # Slope check: is price moving sideways?
    x = np.arange(base_length, dtype=float)
    bc_vals = bc.values.astype(float)
    mean_close = float(bc.mean())
    if mean_close > 0:
        slope_coeff = np.polyfit(x, bc_vals, 1)[0]
        slope_pct_per_bar = slope_coeff / mean_close * 100
        if slope_pct_per_bar > cfg.base_max_up_slope:
            return BaseResult(reason=f"Still trending up ({slope_pct_per_bar:.2f}%/bar)")
        if slope_pct_per_bar < -cfg.base_max_down_slope:
            return BaseResult(reason=f"Declining in base ({slope_pct_per_bar:.2f}%/bar)")

    # ── Prior advance check ───────────────────────────────────────────────────
    # The base must follow a meaningful prior uptrend — Minervini's #1 condition.
    # A stock that's been flat for 6 months and now forms a "base" is NOT a VCP.
    # Look at up to 65 bars immediately before the base started and verify that
    # price appreciated at least 15% to reach the base pivot.
    pre_base_end = n - base_length                          # bar index of base start
    prior_lb     = min(65, pre_base_end)                    # how far back we look
    if prior_lb >= 15 and pre_base_end > 0:
        prior_ref_close = float(close.iloc[max(0, pre_base_end - prior_lb)])
        advance_pct     = (base_high_val - prior_ref_close) / prior_ref_close * 100 if prior_ref_close > 0 else 0.0
        if advance_pct < 30.0:
            return BaseResult(reason=f"No prior advance ({advance_pct:.1f}% in {prior_lb}d)")

    # ATR contraction measurement
    atr_s      = _calc_atr_series(bh, bl, bc)
    half       = max(3, base_length // 2)
    atr_first  = float(atr_s.iloc[:half].mean())
    atr_second = float(atr_s.iloc[half:].mean())
    atr_contraction = atr_second / atr_first if atr_first > 0 else 1.0

    # Pre-base volume (20 bars before base started)
    pre_end   = n - base_length
    pre_start = max(0, pre_end - cfg.vol_pre_base_bars)
    pre_base_vol = float(volume.iloc[pre_start:pre_end].mean()) if pre_end > pre_start else 0.0
    in_base_vol  = float(volume.iloc[-base_length:].mean())

    return BaseResult(
        valid           = True,
        length_bars     = base_length,
        base_high       = base_high_val,
        base_low        = base_low_val,
        depth_pct       = depth_pct,
        atr_first_half  = atr_first,
        atr_second_half = atr_second,
        atr_contraction = atr_contraction,
        pre_base_vol    = pre_base_vol,
        in_base_vol     = in_base_vol,
        reason          = "Valid",
    )


# =============================================================================
# VCP CONTRACTION COUNTER
# =============================================================================

def count_vcp_contractions(
    high:          pd.Series,
    low:           pd.Series,
    base_start_idx: int,
    cfg:           SEPAConfig,
) -> VCPResult:
    """
    Count Minervini-style VCP contractions within the base.

    A contraction is a swing-high → swing-low amplitude that is progressively
    smaller than the prior one. Each valid contraction means the stock is
    coiling tighter — the spring is being compressed.

    Detection:
      1. Find local swing highs (bar whose high > neighbors in swing_window)
      2. For each swing high, find the deepest low before the next swing high
      3. amplitude = (swing_high - swing_low) / swing_high × 100
      4. Count consecutive pairs where amplitude[i] < amplitude[i-1] × contraction_req
      5. Check if the LAST contraction is the tightest (Minervini requirement)
    """
    if base_start_idx >= 0 or abs(base_start_idx) > len(high):
        return VCPResult()

    bh = high.iloc[base_start_idx:]
    bl = low.iloc[base_start_idx:]
    n  = len(bh)

    if n < 10:
        return VCPResult()

    sw = cfg.vcp_swing_window

    # ── Find swing highs ──────────────────────────────────────────────────────
    swing_highs = []  # (position, value)
    for i in range(sw, n - sw):
        h = float(bh.iloc[i])
        if (h >= float(bh.iloc[i - sw:i].max()) and
                h >= float(bh.iloc[i + 1:i + sw + 1].max())):
            swing_highs.append((i, h))

    if len(swing_highs) < 2:
        return VCPResult(vcp_score=_score_vcp(0, False))

    # ── Build contraction amplitudes ──────────────────────────────────────────
    contractions = []   # (position, amplitude_pct)
    for idx, (sh_pos, sh_val) in enumerate(swing_highs):
        # Low range: from this swing high to the next (or end of base)
        next_sh_pos = swing_highs[idx + 1][0] if idx + 1 < len(swing_highs) else n
        low_slice   = bl.iloc[sh_pos:next_sh_pos]
        if len(low_slice) == 0:
            continue
        sl_val = float(low_slice.min())
        amp    = (sh_val - sl_val) / sh_val * 100
        if amp >= cfg.vcp_min_prominence:
            contractions.append((sh_pos, amp))

    if len(contractions) < 2:
        n_c = len(contractions)
        return VCPResult(
            num_contractions      = n_c,
            contraction_pcts      = [c[1] for c in contractions],
            final_contraction_pct = contractions[-1][1] if contractions else 0.0,
            vcp_score             = _score_vcp(n_c, False),
        )

    # ── Count valid (shrinking) contractions ──────────────────────────────────
    valid_count = 0
    for i in range(1, len(contractions)):
        if contractions[i][1] < contractions[i - 1][1] * cfg.vcp_contraction_req:
            valid_count += 1

    amps = [c[1] for c in contractions]
    last_is_tightest = amps[-1] == min(amps)

    return VCPResult(
        num_contractions      = valid_count,
        last_is_tightest      = last_is_tightest,
        contraction_pcts      = amps,
        final_contraction_pct = amps[-1],
        vcp_score             = _score_vcp(valid_count, last_is_tightest),
    )


# =============================================================================
# VOLUME CHARACTER ANALYSIS
# =============================================================================

def analyze_base_volume_character(
    close:          pd.Series,
    high:           pd.Series,
    low:            pd.Series,
    volume:         pd.Series,
    base_start_idx: int,
    cfg:            SEPAConfig,
) -> tuple:
    """
    Measure whether volume character within the base is accumulation or distribution.

    Returns:
        accumulation_ratio  float  — avg up-day vol / avg down-day vol
                                     > 1.2 = accumulation; < 0.8 = distribution
        churn_count         int    — bars with high volume but tiny price range
                                     (institutions selling into buying pressure)
    """
    if base_start_idx >= 0 or abs(base_start_idx) > len(close):
        return 1.0, 0

    bc  = close.iloc[base_start_idx:]
    bv  = volume.iloc[base_start_idx:]
    bh  = high.iloc[base_start_idx:]
    bl  = low.iloc[base_start_idx:]

    if len(bc) < 5:
        return 1.0, 0

    delta = bc.diff()
    up_mask = delta > 0
    dn_mask = delta < 0

    up_vol = float(bv[up_mask].mean()) if up_mask.sum() > 0 else 0.0
    dn_vol = float(bv[dn_mask].mean()) if dn_mask.sum() > 0 else 1.0
    acc_ratio = up_vol / dn_vol if dn_vol > 0 else 2.0

    # Churn detection: high-volume bar with very narrow price range.
    # Uses relative range (< 50% of average base range) so the threshold
    # scales with the stock's normal daily movement rather than a fixed %.
    avg_vol_base = float(bv.mean())
    if avg_vol_base > 0:
        high_vol = bv > avg_vol_base * cfg.churn_vol_mult
        avg_range_base = float((bh - bl).mean())
        narrow_range = (bh - bl) < avg_range_base * 0.50
        churn_count = int((high_vol & narrow_range).sum())
    else:
        churn_count = 0

    return acc_ratio, churn_count


# =============================================================================
# BREAKOUT STATE DETECTOR
# =============================================================================

def detect_breakout_state(
    price:       float,
    base_high:   float,
    volume:      pd.Series,
    vol_50d_avg: float,
) -> str:
    """
    Classify the stock's current position relative to its base.

    States:
      AT_PIVOT      — within 5% below base high (buy-stop zone)
      IN_BASE       — 5–15% below base high (still building)
      EARLY         — > 15% below base high (too early)
      BREAKOUT      — above base high, volume surge ≥ 1.4×50d avg
      WEAK_BREAKOUT — above base high, vol < 1.4× (suspect)
      FADING        — above base high, vol drying (failed breakout risk)
      EXTENDED      — > 10% above base high (too late)
    """
    if base_high <= 0:
        return "UNKNOWN"

    dist = (price - base_high) / base_high * 100

    if dist > 10:
        return "EXTENDED"
    if dist > 0:
        # Use 5-bar peak volume so a confirmed breakout 3 days ago doesn't
        # downgrade to FADING just because today is a quiet follow-through day.
        lookback_bars = min(5, len(volume))
        vol_recent_peak = float(volume.iloc[-lookback_bars:].max())
        vol_surge_peak = vol_recent_peak / vol_50d_avg if vol_50d_avg > 0 else 1.0
        if vol_surge_peak >= 1.4:
            return "BREAKOUT"
        elif vol_surge_peak >= 1.0:
            return "WEAK_BREAKOUT"
        else:
            return "FADING"
    if dist >= -5:
        return "AT_PIVOT"
    if dist >= -15:
        return "IN_BASE"
    return "EARLY"


# =============================================================================
# RS LINE LEADING SIGNAL
# =============================================================================

def _detect_rs_leading(
    rs_line:        pd.Series,
    base_start_idx: int,
    base_high:      float,
    price:          float,
) -> tuple:
    """
    Check if the RS line made a new high BEFORE price broke above the base.

    Minervini's most reliable leading indicator: when the RS line breaks to
    new highs while price is still inside the base, institutions are already
    accumulating. Price then follows.

    Returns: (rs_leading, rs_new_high_in_base, rs_rising_in_base)
    """
    if base_start_idx >= 0 or abs(base_start_idx) > len(rs_line):
        return False, False, False

    rs_pre   = rs_line.iloc[:base_start_idx]
    rs_in    = rs_line.iloc[base_start_idx:]

    if len(rs_pre) == 0 or len(rs_in) == 0:
        return False, False, False

    # Limit to 52-week window: all-time RS peaks 18+ months ago are irrelevant
    rs_pre_window = rs_pre.iloc[-252:] if len(rs_pre) >= 252 else rs_pre
    rs_pre_high = float(rs_pre_window.max())
    rs_current  = float(rs_line.iloc[-1])

    # RS at new all-time high while price still below pivot?
    rs_new_high  = rs_current > rs_pre_high
    price_in_base = price <= base_high * 1.05

    rs_leading       = rs_new_high and price_in_base
    rs_rising_in_base = float(rs_in.iloc[-1]) > float(rs_in.iloc[0])

    return rs_leading, rs_new_high, rs_rising_in_base


def _detect_rs_leading_path_a(rs_line: pd.Series, stage2_start_idx: int) -> tuple:
    """
    Path A version: was RS line at new highs at or before the Stage 2 breakout?
    """
    if stage2_start_idx >= 0 or abs(stage2_start_idx) >= len(rs_line):
        return False, False, False

    rs_pre    = rs_line.iloc[:stage2_start_idx]
    rs_in_s2  = rs_line.iloc[stage2_start_idx:]

    if len(rs_pre) == 0:
        return False, False, False

    # Limit to 52-week window: all-time RS peaks 18+ months ago are irrelevant
    rs_pre_window = rs_pre.iloc[-252:] if len(rs_pre) >= 252 else rs_pre
    rs_pre_high      = float(rs_pre_window.max())
    rs_at_breakout   = float(rs_line.iloc[stage2_start_idx])
    rs_current       = float(rs_line.iloc[-1])

    rs_led      = rs_at_breakout >= rs_pre_high * 0.98   # RS was at highs when stock broke out
    rs_new_high = rs_current > rs_pre_high                # RS still at highs now
    rs_rising   = len(rs_in_s2) > 1 and float(rs_in_s2.iloc[-1]) > float(rs_in_s2.iloc[0])

    return rs_led, rs_new_high, rs_rising


# =============================================================================
# BASE COUNT ESTIMATION
# =============================================================================

def _count_bases_in_stage2(high: pd.Series, close: pd.Series, stage_duration: int) -> int:
    """
    Estimate how many base-breakout cycles have occurred in Stage 2.

    Approach: within the Stage 2 window, count distinct events where price
    made a new 25-bar high (= a breakout from a base), with at least 20 bars
    between events to separate distinct cycles.
    """
    if stage_duration < 25:
        return 1

    n       = len(close)
    lookback = min(stage_duration, n - 5)

    s2_high  = high.values[-lookback:]
    s2_close = close.values[-lookback:]
    total    = len(s2_close)

    base_count    = 1
    last_breakout = None

    for i in range(25, total):
        prior_max = float(np.max(s2_high[max(0, i - 25):i]))
        # Require price to CLEARLY break above the prior 25-bar high (not just be near it).
        # 0.98 threshold was too loose — fired constantly on trending stocks, inflating base_count.
        if s2_close[i] >= prior_max * 1.005:
            if last_breakout is None:
                last_breakout = i
            elif (i - last_breakout) >= 20:
                base_count   += 1
                last_breakout = i

    return min(base_count, 5)


def _base_count_multiplier(base_count: int) -> float:
    return {1: 1.00, 2: 0.85, 3: 0.65, 4: 0.40}.get(base_count, 0.15)


# =============================================================================
# SCORING FUNCTIONS  (0–100)
# =============================================================================

def _score_vcp(num_contractions: int, last_is_tightest: bool) -> float:
    if num_contractions >= 3 and last_is_tightest: return 100.0
    if num_contractions >= 3:                      return 75.0
    if num_contractions == 2 and last_is_tightest: return 85.0
    if num_contractions == 2:                      return 55.0
    if num_contractions == 1 and last_is_tightest: return 45.0
    if num_contractions == 1:                      return 25.0
    return 0.0


def _score_atr_contraction(
    ratio:         float,
    atr_late:      float = 0.0,
    base_high:     float = 0.0,
    pre_base_atr:  float = 0.0,
) -> float:
    """
    Score ATR contraction within the base.

    ratio        — atr_second_half / atr_first_half: < 1.0 = coiling, good.
    atr_late     — absolute ATR in the second half of base (price units).
    base_high    — base pivot price (used to normalise ATR to %).
    pre_base_atr — ATR in the 20 bars BEFORE the base started.

    Two-part score:
      1. Ratio score: how much did ATR contract within the base?
      2. Baseline check: was ATR in the base LOWER than pre-base ATR?
         If a stock was always quiet (ATR first = ATR second = low), there is
         no genuine compression — this is a flat boring stock, not a VCP coil.
    """
    # Part 1: within-base contraction ratio
    if ratio < 0.40: ratio_score = 100.0
    elif ratio < 0.60: ratio_score = 85.0
    elif ratio < 0.80: ratio_score = 65.0
    elif ratio < 1.00: ratio_score = 40.0
    elif ratio < 1.20: ratio_score = 15.0
    else:              ratio_score = 0.0

    # Part 2: validate that ATR actually contracted FROM the pre-base level
    # atr_pct_late = ATR as % of price (normalises across different price levels)
    if atr_late > 0 and base_high > 0 and pre_base_atr > 0:
        atr_vs_prebase = atr_late / pre_base_atr   # < 1.0 = compressed vs before base
        if atr_vs_prebase < 0.50:
            baseline_mult = 1.15    # ATR in late base is <50% of pre-base ATR — strong coil
        elif atr_vs_prebase < 0.75:
            baseline_mult = 1.05
        elif atr_vs_prebase < 1.00:
            baseline_mult = 1.00    # ATR contracted vs pre-base — normal
        elif atr_vs_prebase < 1.30:
            baseline_mult = 0.75    # ATR similar to pre-base — weak signal
        else:
            baseline_mult = 0.50    # ATR HIGHER than pre-base — no coil at all
    else:
        baseline_mult = 1.0         # no baseline data available

    return min(100.0, ratio_score * baseline_mult)


def _score_vol_character(acc_ratio: float, churn_count: int) -> float:
    if acc_ratio >= 1.5 and churn_count == 0: return 100.0
    if acc_ratio >= 1.3 and churn_count <= 1: return 80.0
    if acc_ratio >= 1.0 and churn_count <= 2: return 60.0
    if acc_ratio >= 0.8:                      return 35.0
    return 10.0


def _score_vol_dry(ratio: float) -> float:
    """ratio = recent vol / pre-base vol. Lower = more dried up = better."""
    if ratio < 0.40: return 100.0
    if ratio < 0.55: return 85.0
    if ratio < 0.70: return 65.0
    if ratio < 0.85: return 40.0
    if ratio < 1.00: return 20.0
    return 5.0


def _score_vcp_cv(cv_pct: float) -> float:
    """CV = std / mean × 100. Lower = tighter closes = better. Used for Stage 1 base quality (Path A)."""
    if cv_pct < 1.0: return 100.0
    if cv_pct < 1.5: return 90.0
    if cv_pct < 2.0: return 75.0
    if cv_pct < 3.0: return 55.0
    if cv_pct < 4.0: return 30.0
    if cv_pct < 6.0: return 10.0
    return 0.0


def _score_tightness_coiling(late_cv: float, coiling_ratio: float) -> float:
    """
    Score the COILING quality of the last third of the base.

    Unlike a pure CV score, this penalises stocks that were ALWAYS tight
    (perpetually flat low-volatility names) while rewarding genuine compression:
    stock was volatile earlier in the base but has coiled down in the final phase.

    late_cv       — CV (%) of closes in last third of base
    coiling_ratio — late_cv / early_cv  (< 1.0 = tightening, > 1.0 = expanding)
    """
    # Base score from absolute CV of the final phase
    if late_cv < 1.0:   abs_score = 90.0
    elif late_cv < 1.5: abs_score = 75.0
    elif late_cv < 2.0: abs_score = 60.0
    elif late_cv < 3.0: abs_score = 40.0
    elif late_cv < 4.5: abs_score = 20.0
    else:               abs_score =  5.0

    # Coiling multiplier: reward genuine tightening; penalise expansion
    if coiling_ratio < 0.40:    coil_mult = 1.30    # strong compression
    elif coiling_ratio < 0.60:  coil_mult = 1.20
    elif coiling_ratio < 0.80:  coil_mult = 1.10
    elif coiling_ratio <= 1.05: coil_mult = 1.00    # flat / very slight tightening — neutral
    elif coiling_ratio < 1.30:  coil_mult = 0.70    # noticeably expanding — penalty
    else:                       coil_mult = 0.40    # strongly expanding — anti-VCP signal

    return min(100.0, abs_score * coil_mult)


def _score_pivot_dist(d: float) -> float:
    """d = (price - base_high) / base_high × 100. Negative = below pivot."""
    if d < -20:  return 0.0
    if d < -15:  return 10.0
    if d < -10:  return 25.0
    if d < -5:   return 55.0
    if d < -2:   return 80.0
    if d < 0:    return 100.0
    if d <= 2:   return 90.0
    if d <= 5:   return 70.0
    if d <= 10:  return 40.0
    if d <= 15:  return 15.0
    return 0.0


def _score_vol_surge(ratio: float) -> float:
    """ratio = best vol day in Stage 2 / 50d avg. Higher = more convincing breakout."""
    if ratio >= 2.5: return 100.0
    if ratio >= 2.0: return 85.0
    if ratio >= 1.5: return 65.0
    if ratio >= 1.2: return 40.0
    if ratio >= 1.0: return 20.0
    return 5.0


def _score_extension(ext_pct: float) -> float:
    """ext_pct = max(0, price - base_high) / base_high × 100. Lower = less chasing."""
    if ext_pct < 2:  return 100.0
    if ext_pct < 5:  return 75.0
    if ext_pct < 8:  return 40.0
    if ext_pct < 12: return 15.0
    return 0.0


# =============================================================================
# SETUP CLASSIFIERS
# =============================================================================

def _classify_path_a(
    vol_ratio:      float,
    extension_pct:  float,
    rs_leading:     bool,
    breakout_state: str = "",
) -> str:
    """
    vol_ratio has DIFFERENT semantics depending on breakout_state:
      AT_PIVOT / IN_BASE / EARLY → vol_ratio = recent 5d avg / 50d avg
                                    LOWER is better (volume dry-up is bullish pre-breakout)
      BREAKOUT / WEAK / FADING   → vol_ratio = breakout bar vol / 50d avg
                                    HIGHER is better (surge confirms institutions buying)
    """
    # Hard overrides — state takes priority
    if breakout_state == "EXTENDED" or extension_pct > 10:
        return "🔴 Extended — Too Late"
    if breakout_state == "FADING":
        return "🔴 Fading Breakout — Failed"      # above pivot, volume dried up = failed
    if breakout_state == "WEAK_BREAKOUT":
        return "🟡 Weak Breakout — Low Volume"    # above pivot, ~avg volume = suspect

    # Pre-breakout: vol_ratio is the DRY-UP ratio (lower = better)
    if breakout_state in ("AT_PIVOT", "IN_BASE", "EARLY"):
        dry = vol_ratio   # < 1.0 is good (drying up); > 1.0 is heavy volume = caution
        if dry <= 0.60 and rs_leading:
            return "🟢 At Pivot — Dry-Up + RS Leading"
        if dry <= 0.60:
            return "🟢 At Pivot — Volume Dry-Up"
        if dry <= 1.00 and rs_leading:
            return "🟢 Approaching Pivot — RS Leading"
        if dry <= 1.00:
            return "🔵 Approaching Pivot"
        return "🟡 Approaching Pivot — Heavy Volume"   # >1.0× avg near pivot = caution

    # Confirmed breakout: vol_ratio is the SURGE ratio (higher = better)
    surge = vol_ratio
    if surge >= 1.5 and extension_pct < 5 and rs_leading:
        return "🟢 Confirmed Breakout — RS Leading"
    if surge >= 1.5 and extension_pct < 5:
        return "🟢 Confirmed Breakout"
    if surge >= 1.2 and extension_pct < 8:
        return "🟡 Breakout Watch"
    return "🔵 Early Stage 2"


def _classify_path_b(
    pivot_dist:     float,
    vol_dry:        float,
    atr_contraction: float,
    state:          str,
    n_contractions: int,
) -> str:
    if state == "EXTENDED":
        return "🔴 Extended"
    if state in ("BREAKOUT",):
        return "🟢 Breaking Out"
    if state == "WEAK_BREAKOUT":
        return "🟡 Weak Breakout"
    if state == "FADING":
        return "🔴 Fading"
    if pivot_dist >= -5 and vol_dry <= 0.65 and n_contractions >= 2:
        return "🟢 Actionable VCP"
    if pivot_dist >= -8 and n_contractions >= 1 and atr_contraction < 0.85:
        return "🟡 Ready"
    if atr_contraction < 0.80:
        return "🔵 Coiling"
    return "🔵 Forming"


# =============================================================================
# UTILS
# =============================================================================

def _calc_atr_series(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """ATR as a rolling series. Uses 14 period or shorter for small windows."""
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    period = max(3, min(14, len(tr) // 3))
    return tr.rolling(period, min_periods=1).mean()


def _calc_rsi(close: pd.Series, period: int = 14) -> float:
    """
    Wilder's RSI(period).  Returns 50.0 if insufficient data.
    Uses EWM with alpha=1/period for Wilder smoothing (not simple rolling avg).
    """
    if len(close) < period + 1:
        return 50.0
    delta    = close.diff().dropna()
    gain     = delta.clip(lower=0)
    loss     = (-delta.clip(upper=0))
    avg_gain = float(gain.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])
    avg_loss = float(loss.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 1)


def _rsi_sepa_modifier(rsi: float, state: str) -> float:
    """
    Entry-timing RSI modifier for SEPA score.  Returns a multiplier (≤ 1.05).

    Philosophy:
      RSI adjusts TIMING quality only — it does not change the pattern quality
      (VCP, RS line, vol dry-up). A perfect VCP with RSI 88 is still a great
      pattern but the entry is extended; a penalty reflects that risk.

    For pre-breakout states (AT_PIVOT / IN_BASE):
      Stock is consolidating — RSI naturally 40-60. Only penalize extremes.
    For breakout states (BREAKOUT / WEAK_BREAKOUT):
      Momentum must be confirmed. Ideal 50-65. Penalize extension (>82) and
      absent momentum (<40) meaningfully.
    """
    if state in ("AT_PIVOT", "IN_BASE"):
        if rsi > 80:  return 0.88   # already extended before the break — caution
        if rsi < 30:  return 0.75   # structural breakdown, not consolidation
        return 1.00                  # normal consolidation range — no adjustment

    # BREAKOUT / WEAK_BREAKOUT / FADING / EXTENDED / other
    if 50 <= rsi <= 65:   return 1.05   # ideal: fresh momentum, room to run
    if 65 < rsi <= 75:    return 1.00   # strong, acceptable
    if 75 < rsi <= 82:    return 0.93   # somewhat extended
    if rsi > 82:          return 0.82   # very extended — likely to pull back
    if 40 <= rsi < 50:    return 0.93   # fading momentum
    return 0.80                          # < 40 — momentum absent, not a breakout


def _copy_stage_fields(r: SEPAResult, s: StageAnalysisResult) -> None:
    r.stage           = s.stage
    r.stage_full      = s.stage_full
    r.stage_score_s2  = round(s.stage_score.get("s2", 0), 1)
    r.stage_duration  = s.stage_duration
    r.rs_vs_benchmark = s.rs_vs_benchmark
    r.rs_status       = s.rs_status
    r.mom_label       = s.mom_label
    r.roc_fast        = s.roc_fast
    r.avg_dollar_vol  = s.avg_dollar_vol
    r.ema_dist_fast   = s.ema_dist_fast
    r.ema_dist_medium = s.ema_dist_medium
    r.ema_dist_slow   = s.ema_dist_slow
    r.beta            = s.beta
    r.beta_label      = s.beta_label
    r.vol_conviction  = s.vol_conviction
    r.vol_ratio       = s.vol_ratio
    r.market_cap      = s.market_cap
    r.is_cheat_entry  = s.is_cheat_entry


def _default_weights() -> dict:
    try:
        from config import SEPA_WEIGHTS
        return SEPA_WEIGHTS
    except ImportError:
        return {
            "vcp_contractions":  0.25,
            "vol_character":     0.20,
            "atr_contraction":   0.15,
            "rs_leading":        0.15,
            "vol_dry_up":        0.10,
            "current_tightness": 0.10,
            "pivot_proximity":   0.05,
        }
