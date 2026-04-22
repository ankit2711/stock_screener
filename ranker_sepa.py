# =============================================================================
# RANKER — SEPA (Minervini / Weinstein / Livermore) edition  v3
# =============================================================================
#
# Pipeline per run:
#   1. get_market_regime(benchmark)     — compute regime_mult once for all tickers
#   2. Resample benchmark to weekly     — for weekly stage checks
#
# Pipeline per ticker:
#   3. run_stage_analysis(daily)        — gate: skip if not Stage 2
#   4. get_weekly_stage_weinstein(weekly_df) — canonical 30-week SMA (screeners/weekly_stage.py)
#   5. run_sepa_analysis(...)           — Path A or B scoring (see sepa.py)
#   6. Rank by SEPA Score, return top N
#
# Market regime:
#   Checks 5 Minervini trend-template conditions on the benchmark index.
#   Full bull (5/5) → ×1.0.  Bear (0/5) → ×0.0 (no longs returned).
#
# Weekly stage (per ticker):
#   Simplified Weinstein: price vs 30-week MA slope.
#   Stage 3/4 weekly → score 0.  Stage 1 weekly (transitioning) → cap at 50%.
#   Stage 2 weekly → full score.  Unknown (< 35 weekly bars) → no penalty.
#
# =============================================================================

import logging
import pandas as pd
import numpy as np
from datetime import datetime

from screeners.stage_analysis import StageAnalysisConfig, run_stage_analysis
from screeners.sepa import SEPAConfig, SEPAResult, run_sepa_analysis
from screeners.weekly_stage import (
    to_weekly as _to_weekly_shared,
    get_weekly_stage_weinstein,
    compute_thewrap_signal,
)
from config import (
    TOP_N_US, TOP_N_INDIA, TOP_N_AI, RS_RATING,
    MIN_AVG_DOLLAR_VOL_NSE, MIN_AVG_DOLLAR_VOL_BSE,
    MIN_AVG_DOLLAR_VOL_US, MIN_AVG_DOLLAR_VOL_AI,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_CFG = StageAnalysisConfig(
    sensitivity       = "Aggressive",
    ma_length         = 200,   # 200-day EMA = 40-week EMA — matches TradingView visual reference
    slope_lookback    = 10,    # 2-week slope on EMA200 — responsive but not noisy
    ema_fast          = 10,
    ema_medium        = 21,
    ema_slow          = 50,
    rs_ma_length      = 52,
    vol_avg_len       = 50,
    mom_fast          = 10,
    mom_slow          = 20,
    beta_length       = 52,
    pead_threshold    = 10.0,
    pead_window       = 5,
)

_SEPA_CFG = SEPAConfig()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_screens_sepa(
    ohlcv:     dict,
    metadata:  dict,
    benchmark: pd.DataFrame,
    market:    str = "india",
    stage_cfg: StageAnalysisConfig = None,
    sepa_cfg:  SEPAConfig = None,
) -> pd.DataFrame:
    """
    Run Stage Analysis + SEPA on every ticker.

    Returns top-N DataFrame sorted by SEPA Score descending.

    Args:
        ohlcv:     {ticker: OHLCV DataFrame}
        metadata:  {ticker: {name, sector, market_cap}}
        benchmark: Benchmark OHLCV
        market:    'india' or 'us'
        stage_cfg: StageAnalysisConfig override
        sepa_cfg:  SEPAConfig override
    """
    if stage_cfg is None:
        stage_cfg = _STAGE_CFG
    if sepa_cfg is None:
        sepa_cfg = _SEPA_CFG

    top_n = TOP_N_AI if market == "ai" else TOP_N_US if market == "us" else TOP_N_INDIA
    total = len(ohlcv)

    # ── Step 1: Market regime — computed once ─────────────────────────────────
    regime_mult, regime_label = get_market_regime(benchmark)
    logger.info(f"SEPA: Market regime: {regime_label} (multiplier ×{regime_mult:.1f})")
    # NOTE: never block output based on regime. The multiplier already reduces scores
    # heavily in weak/bear markets. The user decides whether to act on the results.
    # A bear market score of 20 is still useful context — it means "technically set up
    # but market headwind is strong."

    # ── Step 2: Resample benchmark to weekly (once) ───────────────────────────
    weekly_bench = _to_weekly(benchmark)

    rows   = []
    stage2 = 0

    logger.info(f"SEPA: Screening {total} {market.upper()} tickers...")
    logger.info(f"SEPA: Regime={regime_label}, mult={regime_mult:.2f} — scores will be scaled accordingly")

    for i, (ticker, df) in enumerate(ohlcv.items(), 1):
        if i % 100 == 0:
            logger.info(f"  {i}/{total} | Qualified: {stage2}")

        if len(df) < 60:
            continue

        meta = metadata.get(ticker, {})

        try:
            # ═══════════════════════════════════════════════════════════════════
            # STEP 3: MINERVINI EMA TEMPLATE  (primary filter — replaces
            # stage-analysis as the gate)
            #
            # Root cause of prior bad results: the 30-day MA used in stage
            # analysis flattens within 2-3 weeks of any consolidation, causing
            # a stock that ran +150% and is now in a clean 6-week base to be
            # classified as Stage 3 (S2 score=3) and filtered out entirely.
            # Genuine Minervini setups were being EXCLUDED.
            #
            # Minervini's own template uses EMA50/EMA150/EMA200 alignment —
            # these are long enough to stay bullish through a 4-13 week base.
            # ═══════════════════════════════════════════════════════════════════
            template_ok, n_template = _check_minervini_template(df)
            if not template_ok:
                continue

            # ── 52-week range gates ───────────────────────────────────────────
            n_hist = len(df)
            lk_52w = min(252, n_hist)
            hi_52w = float(df["high"].iloc[-lk_52w:].max())
            lo_52w = float(df["low"].iloc[-lk_52w:].min())
            cur_px = float(df["close"].iloc[-1])

            # Price within 25% of 52-week high — eliminates declining/lagging stocks
            if cur_px < hi_52w * 0.75:
                continue

            # Price at least 25% above 52-week low — requires a prior advance
            # (Note: 25% not 30% — 30% was over-filtering; allow early Stage 2)
            if cur_px < lo_52w * 1.25:
                continue

            # ── Stage Analysis  (informational — not used as hard filter) ─────
            # We still run this for: RS vs benchmark, momentum, beta, dollar vol,
            # EMA distances, and is_cheat_entry fields used in the output table.
            # Stage == 4 (clear downtrend) is still rejected as a hard gate.
            stage_result = run_stage_analysis(
                df           = df,
                benchmark_df = benchmark,
                ticker       = ticker,
                market       = market,
                cfg          = stage_cfg,
            )
            stage_result.market_cap = meta.get("market_cap", 0)

            # Hard reject: clear Stage 4 downtrend AND template barely passing (3/5)
            if stage_result.stage == 4 and n_template < 4:
                continue

            # RS gate: exclude chronic underperformers (lagged benchmark by >20%)
            if stage_result.rs_vs_benchmark < -20:
                continue

            # ── Liquidity filter: exchange-aware minimum avg dollar volume ──────
            min_adv = (
                MIN_AVG_DOLLAR_VOL_AI  if market == "ai"
                else MIN_AVG_DOLLAR_VOL_US  if market == "us"
                else MIN_AVG_DOLLAR_VOL_BSE if ticker.endswith(".BO")
                else MIN_AVG_DOLLAR_VOL_NSE
            )
            if stage_result.avg_dollar_vol < min_adv:
                continue

            stage2 += 1

            # ── Step 4: Weekly stage check (shared 30-week SMA implementation) ─
            # Uses the canonical screeners/weekly_stage.py — same function as
            # ranker_stage.py — so both sheets show identical weekly classifications.
            weekly_df        = _to_weekly_shared(df)
            w_stage, _, _, _, _ = get_weekly_stage_weinstein(weekly_df)
            weekly_stage     = w_stage   # int: 0=unknown 1=S1 2=S2 3=S3 4=S4

            # ── Step 4b: TheWrap signal (weekly 10W/20W/40W EMA) ─────────────
            # Hard gate: TW_EXIT and TW_EXIT_40W are excluded from SEPA output.
            # These stocks have broken structural support — no valid entry exists.
            # TW_CAUTIOUS is also excluded: price below 10W/20W EMA = not a setup.
            tw_code, tw_label, _, _, _ = compute_thewrap_signal(weekly_df)
            if tw_code in ("TW_EXIT", "TW_EXIT_40W", "TW_CAUTIOUS"):
                continue   # structure compromised — not a buy

            # ── Step 5: SEPA scoring ──────────────────────────────────────────
            sepa_result = run_sepa_analysis(
                df           = df,
                benchmark_df = benchmark,
                stage_result = stage_result,
                ticker       = ticker,
                market       = market,
                cfg          = sepa_cfg,
                regime_mult  = regime_mult,
                weekly_stage = weekly_stage,
            )
            # Store n_template so it is visible in output (admission criterion)
            sepa_result.n_template = n_template

            # ── Step 5b: TheWrap score adjustment ────────────────────────────
            # Apply bonus/penalty to raw SEPA score based on weekly EMA health:
            #   TW_BULLISH  → +15 pts (price above 10W, 10W rising — highest conviction)
            #   TW_MAINTAIN → +8 pts  (full bull stack, 40W rising — trend confirmed)
            #   TW_WAIT     → 0 pts   (neutral — no directional conviction yet)
            #   TW_FADING   → -10 pts (aging trend — reduce score, not a tier-A entry)
            # Store tw_code on result for trade ranker Tier A gate
            sepa_result.tw_code  = tw_code
            sepa_result.tw_label = tw_label
            tw_adj = {"TW_BULLISH": 15, "TW_MAINTAIN": 8, "TW_WAIT": 0, "TW_FADING": -10}.get(tw_code, 0)
            if tw_adj != 0:
                sepa_result.sepa_score = max(0.0, sepa_result.sepa_score + tw_adj)
                sepa_result.score      = sepa_result.sepa_score  # keep in sync

            rows.append(_result_to_row(sepa_result, meta, ticker, regime_label, tw_label))

        except Exception as e:
            logger.debug(f"SEPA failed for {ticker}: {e}")

    logger.info(f"SEPA: {stage2} Stage-2 stocks scored → returning top {top_n}")

    if not rows:
        logger.warning("SEPA: No results")
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values("SEPA Score", ascending=False).reset_index(drop=True)
    df_out.insert(0, "Rank", range(1, len(df_out) + 1))
    return df_out.head(top_n)


# =============================================================================
# MINERVINI EMA TEMPLATE CHECK  (per-stock primary filter)
# =============================================================================

def _check_minervini_template(df: pd.DataFrame) -> tuple:
    """
    Minervini's 5-condition EMA alignment template applied to an individual stock.

    Conditions:
      1. Price > EMA200
      2. EMA200 rising  (today vs 20 bars ago)
      3. EMA150 > EMA200
      4. EMA50  > EMA150
      5. Price  > EMA50

    These use EXPONENTIAL MAs which respond faster than simple MAs and are
    more appropriate for shorter histories.

    Returns (passes: bool, n_conditions_met: int)

    Pass threshold: 4 of 5 conditions.
    Short history (< 150 bars): use EMA50/EMA100 3-condition fallback (pass ≥ 2).

    Why this is the primary admission filter rather than stage analysis alone:
      A stock in a 6-12 week VCP base after a 50%+ advance will have:
        • price above all EMAs (they lag and stay bullish through the base)
        • all EMAs in bullish stack (take weeks/months to invert)
      Stage analysis (EMA200 slope) can show flat/declining slope during a base
      because a 200-day EMA stops rising when price consolidates — it scores as S3.
      The Minervini template is immune to this: EMA stack doesn't invert in 6 weeks.
      Both systems together give the strongest filter: structural trend (EMA200) +
      short/medium alignment (EMA50/150) + price position (above EMA50 and EMA200).
    """
    close = df["close"]
    n     = len(close)

    if n < 60:
        return False, 0

    ema50 = close.ewm(span=50, adjust=False).mean()

    if n < 150:
        # Not enough history for EMA150/EMA200 — use 3-condition short version
        ema100 = close.ewm(span=100, adjust=False).mean()
        px     = float(close.iloc[-1])
        conds  = [
            px > float(ema50.iloc[-1]),
            float(ema50.iloc[-1]) > float(ema100.iloc[-1]),
            float(ema50.iloc[-1]) > float(ema50.iloc[-min(20, n - 1)]),  # EMA50 rising
        ]
        n_met = sum(conds)
        return n_met >= 2, n_met

    ema150 = close.ewm(span=150, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    px      = float(close.iloc[-1])
    e50     = float(ema50.iloc[-1])
    e150    = float(ema150.iloc[-1])
    e200    = float(ema200.iloc[-1])
    e200_20 = float(ema200.iloc[-min(21, n - 1)])

    conds = [
        px   > e200,          # 1. price above EMA200
        e200 > e200_20,       # 2. EMA200 trending up
        e150 > e200,          # 3. EMA150 above EMA200
        e50  > e150,          # 4. EMA50  above EMA150
        px   > e50,           # 5. price above EMA50
    ]
    n_met = sum(conds)
    return n_met >= 4, n_met


# =============================================================================
# MARKET REGIME
# =============================================================================

def get_market_regime(benchmark_df: pd.DataFrame) -> tuple:
    """
    Minervini's 5-condition trend template applied to the benchmark index.

    Conditions:
      1. Price > 200 EMA
      2. 200 EMA rising (now vs 20 bars ago)
      3. 150 EMA > 200 EMA
      4. 50 EMA > 150 EMA
      5. Price > 50 EMA

    Regime multipliers: 5/5=1.0, 4/5=0.8, 3/5=0.5, 2/5=0.3, ≤1/5=0.15
    Minimum is 0.15 (not 0.0) — results always shown, scores heavily reduced.

    Data requirement:
      EMA200 needs ~200 bars to stabilise.  With HISTORY_DAYS=365 (~252 trading
      bars) we only have ~52 bars of reliable EMA200 data which can misfire.
      If fewer than 220 bars are available, skip EMA200/EMA150 conditions and
      only score the three shorter-term conditions (50 EMA based), returning a
      conservative 0.7 multiplier ceiling.
    """
    close = benchmark_df["close"]
    n_bars = len(close)

    if n_bars < 60:
        return 1.0, "Insufficient benchmark data (< 60 bars)"

    ema50 = close.ewm(span=50, adjust=False).mean()
    price = float(close.iloc[-1])

    if n_bars < 220:
        # Not enough history for reliable EMA200 — use 3-condition short version
        ema100 = close.ewm(span=100, adjust=False).mean()
        conds_short = [
            price > float(ema50.iloc[-1]),
            float(ema50.iloc[-1]) > float(ema100.iloc[-1]),
            float(ema50.iloc[-1]) > float(ema50.iloc[-20]) if n_bars >= 70 else True,
        ]
        n_met = sum(conds_short)
        mult  = [0.40, 0.60, 0.80, 0.90][n_met]   # cap at 0.9 — uncertainty penalty
        return mult, f"Short-history regime {n_met}/3 ({n_bars} bars)"

    ema150 = close.ewm(span=150, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    conds = [
        price > float(ema200.iloc[-1]),
        float(ema200.iloc[-1]) > float(ema200.iloc[-20]),
        float(ema150.iloc[-1]) > float(ema200.iloc[-1]),
        float(ema50.iloc[-1])  > float(ema150.iloc[-1]),
        price > float(ema50.iloc[-1]),
    ]
    n = sum(conds)

    if n == 5: return 1.00, "Bull (5/5)"
    if n == 4: return 0.80, "Mild Bull (4/5)"
    if n == 3: return 0.50, "Neutral (3/5)"
    if n == 2: return 0.30, "Caution (2/5)"
    return 0.15, "Bear (≤1/5)"


# =============================================================================
# WEEKLY STAGE CHECK  (Weinstein 30-week MA)
# =============================================================================

def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly (Friday close)."""
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


def _get_weekly_stage(weekly_df: pd.DataFrame) -> int:
    """
    Weinstein stage from weekly data using the 40-week EMA.

    Uses EMA40 on weekly bars (= 40-week EMA) which is the same concept as
    EMA200 on daily bars. Both are the institutional structural trend reference.
    Using EMA (not SMA) matches TradingView's "40 Week EMA" indicator exactly.

    Slope is measured over 6 weekly bars (= 6 weeks) — enough to confirm
    direction without being fooled by a single week's move.

    Returns:
      0  — insufficient data (< 45 weekly bars) — no penalty applied
      1  — price below EMA40 but EMA40 turning up  (Stage 1 / accumulation)
      2  — price above rising 40-week EMA           (Stage 2 — ideal)
      3  — price above flat/falling 40-week EMA     (Stage 3 — distribution)
      4  — price below falling 40-week EMA           (Stage 4 — decline)
    """
    if weekly_df is None or len(weekly_df) < 45:
        return 0

    close_w = weekly_df["close"]
    ma40    = close_w.ewm(span=40, adjust=False).mean()

    price     = float(close_w.iloc[-1])
    ma_now    = float(ma40.iloc[-1])
    # Slope: compare EMA now vs 6 weeks ago (avoids single-week noise)
    ma_prev   = float(ma40.iloc[-6]) if len(ma40) >= 6 else ma_now
    ma_rising = ma_now > ma_prev

    above_ma = price > ma_now

    if above_ma and ma_rising:
        return 2
    if above_ma and not ma_rising:
        return 3   # price still above but EMA40 rolling over — distribution
    if not above_ma and ma_rising:
        return 1   # below EMA40 but MA still rising — accumulation / transition
    return 4       # price below falling EMA40 — decline


# =============================================================================
# ROW BUILDER
# =============================================================================

def _result_to_row(r: SEPAResult, meta: dict, raw_ticker: str, regime_label: str = "", tw_label: str = "—") -> dict:
    """Convert SEPAResult → flat dict for DataFrame output."""
    ticker_display = raw_ticker.replace(".NS", "").replace(".BO", "")

    if ".NS" in raw_ticker:
        tv_symbol = f"NSE:{ticker_display}"
    elif ".BO" in raw_ticker:
        tv_symbol = f"BSE:{ticker_display}"
    else:
        tv_symbol = ticker_display

    is_path_a = r.scoring_path.startswith("A")

    # ── Vol Signal label — dual semantics ─────────────────────────────────────
    # AT_PIVOT/IN_BASE: vol_surge_ratio = 5d avg / 50d avg  →  lower = drying up (bullish)
    # BREAKOUT/FADING:  vol_surge_ratio = breakout bar vol / 50d avg → higher = surge (bullish)
    is_prebreakout = r.breakout_state in ("AT_PIVOT", "IN_BASE", "EARLY")
    if is_path_a:
        vol_signal = (
            f"dry {r.vol_surge_ratio:.2f}×"
            if is_prebreakout
            else f"surge {r.vol_surge_ratio:.2f}×"
        )
    else:
        vol_signal = "—"

    # ── Weekly stage label (human-readable) ──────────────────────────────────
    weekly_stage_labels = {
        0: "Unknown",
        1: "W-S1 Accum",    # price below but MA turning — transitioning
        2: "W-S2 ✓",        # price above rising 30-week MA — Weinstein Stage 2
        3: "W-S3 Dist",     # price above but MA flattening/falling — distribution
        4: "W-S4 Decline",  # price below falling MA — avoid
    }
    weekly_stage_label = weekly_stage_labels.get(r.weekly_stage, "Unknown")

    # ── Template label — how many of Minervini's 5 EMA conditions are met ────
    template_label = f"{r.n_template}/5"

    # ── Daily MA stage — 30-day SMA on daily data (NOT Weinstein's original) ─
    # This diverges from the weekly stage during any consolidation base because
    # the 30-day daily SMA (~6 weeks) flattens immediately, while the 30-week
    # SMA (~150 trading days) stays trending for months.
    # Label it explicitly to avoid confusion.
    daily_stage_label = r.stage_full.replace(
        "S1 (Accumulation)", "Daily S1"
    ).replace(
        "S2 (Advancing)", "Daily S2"
    ).replace(
        "S3 (Distribution)", "Daily S3 ⚠"   # ⚠ because often false during base
    ).replace(
        "S4 (Declining)", "Daily S4 ✗"
    )

    return {
        # ══════════════════════════════════════════════════════════════════
        # COLUMN ORDER — left to right = decision priority for execution
        #
        # Columns 1-9   → read every row — they tell you what to DO
        # Columns 10-17 → read shortlisted rows — they build conviction
        # Columns 18+   → research / context — read when curious or sizing
        # ══════════════════════════════════════════════════════════════════

        # ── 1. IDENTITY ──────────────────────────────────────────────────
        "Ticker":         ticker_display,
        "Company":        meta.get("name", raw_ticker),

        # ── 2. SCORE + VERDICT ───────────────────────────────────────────
        # SEPA Score: already sorted descending. Primary ranking metric.
        # Setup: emoji tells you the action immediately — no reading required.
        #   🟢 At Pivot — Dry-Up + RS Leading  → highest priority, set alert now
        #   🟢 Confirmed Breakout — RS Leading  → enter if in buy zone
        #   🟢 Actionable VCP                  → base ready to break, buy-stop
        #   🟡 Weak / Watch / Ready             → monitor, not yet actionable
        #   🔵 Approaching / Forming / Coiling  → early, for watchlist only
        #   🔴 Fading / Extended / Distribution → skip or exit
        "SEPA Score":     r.sepa_score,
        "Setup":          r.setup_stage,

        # ── 3. PATH — tells you which columns to focus on ────────────────
        # A = Fresh Breakout → look at Vol Signal, Extension%, S1 Base CV%
        # B = VCP Base       → look at VCP Count, ATR Contract, Vol Dry%, etc.
        # C = Trending       → no setup, bottom of list, ignore
        "Path":           r.scoring_path,

        # ── 4. ENTRY STATE — can I enter today? ──────────────────────────
        # Breakout State:
        #   AT_PIVOT   (-5% to 0%)   → approaching entry, set buy-stop alert
        #   IN_BASE    (-15% to -5%) → building, not ready yet
        #   BREAKOUT   (0% to +10%, vol ≥ 1.4×) → active entry window NOW
        #   WEAK_BREAKOUT            → borderline, half-size or wait
        #   FADING                   → SKIP — above pivot on dying volume
        #   EXTENDED   (>10%)        → SKIP — chasing, risk/reward broken
        "Breakout State": r.breakout_state,

        # ── 5. ENTRY PRICE CONTEXT — how close, how much risk? ──────────
        # Pivot Dist %: signed distance from base_high (the buy-stop pivot)
        #   < -5%  = inside base, early          → set alert only
        #   -5% to 0% = approaching pivot         → ready to alert
        #   0% to +5% = buy zone                  → can enter
        #   > +5%  = extended, entering late       → caution / skip
        #
        # Stop Dist %: (price − natural stop) / price
        #   This IS your position size input.
        #   Risk 1% of account per trade → position = 1% / stop_dist
        #   Example: stop 6% → position size = 1/0.06 = 16.7% of capital
        #   > 8% stop = oversized risk → either reduce size or skip
        "Pivot Dist %":   f"{r.pivot_dist_pct:+.1f}%",
        "Stop Dist %":    f"{r.stop_dist_pct:.1f}%",
        "RSI(14)":        round(r.rsi_14, 0),

        # ── 6. VOLUME SIGNAL — the confirmation ──────────────────────────
        # Two different signals depending on state (see sepa.py _score_path_a):
        #   Pre-breakout (AT_PIVOT/IN_BASE): "dry 0.42×"
        #     = 5-day avg vol is 42% of 50-day avg → sellers dried up → BULLISH
        #     Lower is better.  ≤ 0.6× = ideal setup.  > 1.0× = distribution risk.
        #   Post-breakout (BREAKOUT/FADING): "surge 2.1×"
        #     = breakout bar vol was 2.1× the 50-day avg → institutions buying → CONFIRMED
        #     Higher is better.  ≥ 1.5× = strong.  < 1.0× = failed breakout.
        "Vol Signal":     vol_signal,

        # ── 7. RS LINE — leading indicator ───────────────────────────────
        # RS Leading ✓ = RS line hit a NEW HIGH before price broke out.
        # This is Minervini's most reliable signal. Institutions accumulate
        # before price moves — RS rises first.
        # RS vs Bench: stock % gain vs Nifty/SPX over the Stage 2 advance.
        #   > +20% = strongly outperforming → ideal
        #   0–20%  = outperforming → acceptable
        #   < 0%   = lagging the index → reduce conviction significantly
        "RS Leading":     "✓" if r.rs_leading else "·",
        "RS vs Bench %":  f"{r.rs_vs_benchmark:+.1f}%",

        # ── 8. STAGE ALIGNMENT — is the trend structure intact? ──────────
        # Weekly Stage (Weinstein 30-WEEK SMA — the authoritative stage):
        #   W-S2 ✓     = price above rising 30-week MA → ideal, full score
        #   W-S1 Accum = transitioning, score capped at 60%
        #   W-S3 Dist  = rolling over → score = 0 (filtered out normally)
        #   W-S4 Decline = downtrend → score = 0 (filtered out)
        #   Unknown    = < 35 weekly bars of history → no penalty applied
        #
        # Template (Minervini EMA50/150/200):
        #   5/5 = perfect alignment  (actual admission criterion: ≥ 4/5)
        #   4/5 = passed — one EMA out of alignment (typical during base)
        #   < 4/5 = would not appear in results (filtered out in ranker)
        #
        # Daily MA Stage (30-day SMA on daily — NOT Weinstein's original):
        #   Often shows S3 during a valid VCP base (30-day SMA flattens in 2 weeks).
        #   ⚠ label means: stock is likely in consolidation, NOT actually distributing.
        #   Use Weekly Stage + Template for real assessment. This is diagnostic only.
        "Weekly Stage":   weekly_stage_label,
        "TheWrap":        tw_label,           # weekly 10W/20W/40W EMA signal
        "Template":       template_label,
        "Daily MA":       daily_stage_label,

        # ── 9. EXECUTION PRICES — absolute ₹ values for order placement ────
        # Price:       current close
        # Entry:       buy-stop level (0.5% above pivot for AT_PIVOT;
        #              current price for BREAKOUT — already cleared pivot)
        # Stop:        natural stop (max of 20d low and EMA50) — place hard stop here
        # Raw Score:   SEPA quality score BEFORE regime multiplier — stock's own merit
        "Price ₹":        round(r.price, 2),
        "Entry ₹":        round(r.entry_price, 2),
        "Stop ₹":         round(r.stop_price, 2),
        "Raw Score":      r.raw_sepa_score,

        # ── 10. TRADEVIEW LINK — click to chart immediately ───────────────
        "TradingView":    f"https://www.tradingview.com/chart/?symbol={tv_symbol}",

        # ════════════════════════════════════════════════════════════════════
        # CONVICTION BUILDERS — read these for shortlisted stocks
        # ════════════════════════════════════════════════════════════════════

        # ── 10. BASE / PATTERN QUALITY (Path B — VCP) ────────────────────
        # All show "—" for Path A stocks.
        "VCP Count":      r.num_contractions,        # ≥ 2 = proper VCP
        "Last Tightest":  "✓" if r.last_is_tightest else "·",
        "Base Bars":      r.base_length_bars if r.base_valid else "—",
        "Base Depth %":   f"{r.base_depth_pct:.1f}%" if r.base_valid else "—",
        "ATR Contract":   f"{r.atr_contraction:.2f}×",   # < 0.70 = strong compression
        "Vol Dry %":      f"{r.vol_dry_ratio:.0%}",       # < 65% = proper dry-up
        "CV Tight %":     f"{r.current_cv_pct:.1f}%",    # late-base variability (lower = tighter)
        "Accum Ratio":    f"{r.accumulation_ratio:.2f}×", # > 1.0 = more up-vol than down-vol
        "Churn Bars":     r.churn_count,                  # 0 = clean; ≥ 2 = distribution warning
        "Base Count":     r.base_count,                   # 1st base = best odds
        "Base ×Mult":     f"×{r.base_count_mult:.2f}",   # penalty already in SEPA Score

        # ── 11. BASE / PATTERN QUALITY (Path A — Fresh Breakout) ─────────
        # All show "—" for Path B stocks.
        "S1 Base CV%":    f"{r.s1_cv_pct:.1f}%" if is_path_a else "—",
        "Extension %":    f"{r.extension_pct:+.1f}%" if is_path_a else "—",

        # ════════════════════════════════════════════════════════════════════
        # CONTEXT / RESEARCH — read when sizing, comparing, or curious
        # ════════════════════════════════════════════════════════════════════

        # ── 12. STAGE DETAIL ─────────────────────────────────────────────
        "S2 Score":       r.stage_score_s2,    # raw daily S2 scoring pts (max ~10)
        "Stage Days":     r.stage_duration,    # bars since Stage 2 estimated to have begun
        "Cheat Entry":    "✓" if r.is_cheat_entry else "·",

        # ── 13. MOMENTUM ─────────────────────────────────────────────────
        "RS Status":      r.rs_status,
        "Momentum":       r.mom_label,
        "ROC 10d %":      f"{r.roc_fast:+.1f}%",

        # ── 14. EMA DISTANCES (Minervini template detail) ─────────────────
        # How far price is stretched above each EMA.
        # Very high values = extended, mean-reversion risk.
        "EMA vs 50":      f"{r.ema_dist_fast:+.1f}%",    # price vs EMA50
        "EMA vs 150":     f"{r.ema_dist_medium:+.1f}%",  # price vs EMA150
        "EMA vs 200":     f"{r.ema_dist_slow:+.1f}%",    # price vs EMA200

        # ── 15. RISK PROFILE ─────────────────────────────────────────────
        "Beta":           r.beta,
        "Beta Label":     r.beta_label,
        "Vol Conv":       r.vol_conviction,

        # ── 16. SIZING CONTEXT ────────────────────────────────────────────
        "Avg $ Vol":      _fmt_dollar_vol(r.avg_dollar_vol),
        "Market Cap":     _fmt_mcap(r.market_cap),
        "Sector":         meta.get("sector", "Unknown"),
        "Market Regime":  regime_label,   # same for all rows — purely informational here
        "Last Updated":   datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# =============================================================================
# FORMATTERS
# =============================================================================

def _fmt_dollar_vol(adv: float) -> str:
    if adv >= 1e9: return f"${adv / 1e9:.2f}B"
    if adv >= 1e6: return f"${adv / 1e6:.2f}M"
    if adv >= 1e7: return f"₹{adv / 1e7:.2f}Cr"
    return f"${adv:,.0f}"


def _fmt_mcap(cap: float) -> str:
    if not cap:     return "N/A"
    if cap >= 1e12: return f"${cap / 1e12:.1f}T"
    if cap >= 1e9:  return f"${cap / 1e9:.1f}B"
    if cap >= 1e7:  return f"₹{cap / 1e7:.0f}Cr"
    return f"${cap:,.0f}"
