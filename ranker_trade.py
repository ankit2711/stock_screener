# =============================================================================
# TRADE RANKER — Unified 3-Lens Pipeline
# =============================================================================
#
# PURPOSE:
#   Single entry point for the daily "what do I trade today?" decision.
#   Runs Stage, SEPA, and RS Leaders scans and returns ALL results plus
#   a unified top-20 trade candidate list.
#
# RETURN VALUE:
#   dict with four DataFrames:
#     "stage"  → top-30 Stage-2 stocks (structural trend quality)
#     "sepa"   → top-30 SEPA entry setups (entry quality, RSI timing)
#     "rs"     → top-30 RS Leaders (institutional holding during correction)
#     "trade"  → top-20 unified trade candidates (single scoring function)
#
# UNIFIED SCORING (_score_candidate):
#   One function for every candidate regardless of which screener found it.
#   Takes the union of Stage + SEPA + RS pools and scores each stock once.
#
#   HARD ELIMINATORS (return None):
#     • Weekly stage = W-S1 Accum or W-S4 Decline (price below weekly EMA)
#     • TheWrap = TW_EXIT (multi-month structure broken)
#     • Fewer than 30 bars of OHLCV
#     • Score < 40 after all components (below quality floor)
#
#   SCORE COMPONENTS (0–100 before bonuses):
#     A. Signal breadth   0–30   (Stage + SEPA + RS lens scores)
#     B. Entry signal     0–25   (Cheat=25, BREAKOUT<3%=22, etc.)
#     C. Risk quality     0–20   (stop tightness + pivot proximity)
#     D. Volume           0–15   (Very High=15, High=10, Normal=5, Low=1)
#     E. Sector strength  0–10   (LEADING=10, NEUTRAL=6, LAGGING=2)
#
#   BONUSES (can push above 100, capped at 110):
#     W-S2 ✓ confirmed +5, Bullish TheWrap +4, Maintain TheWrap +2,
#     RS Leads Price (🌟) +5, Conviction streak +2/+4/+6
#
#   PENALTIES (multiplied after bonuses):
#     RSI > 82 × 0.88, RSI 75–82 × 0.95,
#     Stage duration < 15 bars × 0.85, TW_FADING × 0.82
#
# REGIME:
#   Regime now only affects position sizing (handled by user) — NOT list size.
#   All stocks scoring ≥ 40 are shown (up to 20). In a bear market, fewer
#   stocks naturally clear the threshold — the list shrinks organically.
# =============================================================================

import logging
import pandas as pd
import numpy as np
from datetime import datetime

from first_seen import annotate_df
from persistence import annotate_streak_df
from screeners.stage_analysis import StageAnalysisConfig
from screeners.sepa import SEPAConfig, detect_base
from screeners.weekly_stage import compute_thewrap_signal as _compute_thewrap, to_weekly as _to_weekly_ohlcv
from screeners.sector_rotation import (
    run_sector_rotation,
    get_sector_for_ticker,
    get_regime_from_sectors,
    SectorResult,
)
from ranker_sepa       import run_screens_sepa, get_market_regime, _STAGE_CFG as SEPA_STAGE_CFG, _SEPA_CFG
from ranker_stage      import run_screens_stage, DEFAULT_CFG as STAGE_CFG
from ranker_rs         import run_screens_rs
from ranker_conviction import run_conviction_scan
from data_quality      import run_data_quality_scan
from persistence       import append_screener_exits, get_data_as_of
from config import (
    TOP_N_INDIA, TOP_N_US, TOP_N_AI,
    POOL_N_INDIA, POOL_N_US, POOL_N_AI,
)

logger = logging.getLogger(__name__)

MAX_TRADE_CANDIDATES = 20   # upper bound (actual cap is regime-aware, see Step 7)

# =============================================================================
# REGIME WEIGHT TABLE
# =============================================================================

_REGIME_WEIGHTS = {
    # regime_mult threshold → weight dict
    # Components: sepa, stage, rs, state, stop
    "bull":      {"sepa": 0.35, "stage": 0.25, "rs": 0.15, "state": 0.15, "stop": 0.10},
    "mild_bull": {"sepa": 0.30, "stage": 0.25, "rs": 0.20, "state": 0.15, "stop": 0.10},
    "neutral":   {"sepa": 0.25, "stage": 0.22, "rs": 0.28, "state": 0.15, "stop": 0.10},
    "caution":   {"sepa": 0.18, "stage": 0.20, "rs": 0.37, "state": 0.15, "stop": 0.10},
    "bear":      {"sepa": 0.12, "stage": 0.15, "rs": 0.48, "state": 0.15, "stop": 0.10},
}

# _REGIME_WEIGHTS kept as reference documentation for the market regime framework.
# They are NO LONGER used for scoring (regime only affects the warning label shown
# to the user — position sizing is manual).  Do not delete — they document the
# intended weighting rationale for each market condition.


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_trade_scan(
    ohlcv:     dict,
    metadata:  dict,
    benchmark: pd.DataFrame,
    market:    str = "india",
) -> dict:
    """
    Run Stage + SEPA + RS Leaders scans and produce unified trade candidates.

    Returns:
        dict with keys: "stage", "sepa", "rs", "trade"
        Each value is a pd.DataFrame (empty DataFrame if no results).
    """
    logger.info("TRADE SCAN ▶ Starting unified 3-lens pipeline...")

    # ── Step 0: Data-as-of — compute FIRST before any annotations ─────────────
    # All streak / first_seen stamps must use the OHLCV data date, not run date.
    # This ensures re-running with the same benchmark data (e.g. on a weekend or
    # after a re-fetch that returned the same last bar) never changes streak counts.
    data_as_of = get_data_as_of(benchmark)
    logger.info(f"TRADE SCAN: Data as of {data_as_of}")

    # ── Step 1: Market regime ─────────────────────────────────────────────────
    regime_mult, regime_label = get_market_regime(benchmark)
    weights = _get_regime_weights(regime_mult)
    logger.info(f"TRADE SCAN: Regime={regime_label} (×{regime_mult:.2f}) "
                f"→ RS weight={weights['rs']:.0%}, SEPA weight={weights['sepa']:.0%}")

    # Pool = large internal set fed into Trade Candidates scoring (see pool/display split below).
    # Display = what goes to each screener tab (TOP_N).
    pool_n = POOL_N_AI if market == "ai" else POOL_N_US if market == "us" else POOL_N_INDIA
    disp_n = TOP_N_AI  if market == "ai" else TOP_N_US  if market == "us" else TOP_N_INDIA

    # ── Step 1b: Sector Rotation ──────────────────────────────────────────────
    # Run before the 3 scans so sector multipliers are ready when scoring starts.
    # Graceful: if sector rotation fails for any reason, sector_results = {}
    # and all multipliers default to 1.0 (no impact on existing behaviour).
    logger.info("TRADE SCAN: Running Sector Rotation scan...")
    sector_results: dict[str, SectorResult] = {}
    try:
        sector_results = run_sector_rotation(ohlcv, metadata, benchmark, market=market)
        sector_regime, _ = get_regime_from_sectors(sector_results)
        lead_n  = sum(1 for r in sector_results.values() if r.sector_label in ("LEADING", "IMPROVING"))
        lag_n   = sum(1 for r in sector_results.values() if r.sector_label in ("WEAKENING", "LAGGING"))
        logger.info(f"TRADE SCAN: Sectors={len(sector_results)} "
                    f"(Leading/Improving={lead_n}, Weakening/Lagging={lag_n}) "
                    f"→ {sector_regime}")
    except Exception as _se:
        import traceback
        logger.warning(
            f"TRADE SCAN: Sector rotation failed (non-fatal) — "
            f"sector tabs will show 'No data'. Error: {_se}\n"
            + traceback.format_exc()
        )

    # ── Step 2: Run all 3 scans — use LARGER POOL for Trade Candidate scoring ─
    # Each screener is run with top_n=pool_n (80 India / 60 US) so we see
    # 2–3× more candidates when building cross-screener maps.  The display
    # tabs (Stage / SEPA / RS Leaders) are truncated to TOP_N after the maps
    # are built — users still see a clean 30-stock tab.
    logger.info(f"TRADE SCAN: Running Stage scan (pool={pool_n}, display={disp_n})...")
    stage_pool = _safe_run(run_screens_stage,
                           ohlcv=ohlcv, metadata=metadata, benchmark=benchmark,
                           market=market, cfg=STAGE_CFG, top_n=pool_n)

    logger.info(f"TRADE SCAN: Running SEPA scan (pool={pool_n})...")
    sepa_pool = _safe_run(run_screens_sepa,
                          ohlcv=ohlcv, metadata=metadata, benchmark=benchmark,
                          market=market, stage_cfg=SEPA_STAGE_CFG, sepa_cfg=_SEPA_CFG,
                          top_n=pool_n)

    logger.info(f"TRADE SCAN: Running RS Leaders scan (pool={pool_n})...")
    rs_pool = _safe_run(run_screens_rs,
                        ohlcv=ohlcv, metadata=metadata, benchmark=benchmark,
                        market=market, top_n=pool_n)

    # Display DataFrames: truncated to TOP_N for screener tab output
    stage_df = stage_pool.head(disp_n) if not stage_pool.empty else stage_pool
    sepa_df  = sepa_pool.head(disp_n)  if not sepa_pool.empty  else sepa_pool
    rs_df    = rs_pool.head(disp_n)    if not rs_pool.empty    else rs_pool

    logger.info(
        f"TRADE SCAN: Pool  Stage={len(stage_pool)}, SEPA={len(sepa_pool)}, RS={len(rs_pool)} | "
        f"Display Stage={len(stage_df)}, SEPA={len(sepa_df)}, RS={len(rs_df)}"
    )

    # ── Step 3: Build lookup maps from FULL POOL ──────────────────────────────
    # Maps cover pool_n stocks — Trade Candidate scoring sees all of them,
    # not just the 30 that appear in the display tab.
    stage_map = _df_to_map(stage_pool, "Ticker")
    sepa_map  = _df_to_map(sepa_pool,  "Ticker")
    rs_map    = _df_to_map(rs_pool,    "Ticker")

    # ── Step 4: Combined candidate pool — union of all three screener pools ──
    all_tickers = (
        {str(r.get("Ticker", "")) for _, r in stage_pool.iterrows()} |
        {str(r.get("Ticker", "")) for _, r in sepa_pool.iterrows()}  |
        {str(r.get("Ticker", "")) for _, r in rs_pool.iterrows()}
    ) - {"", "nan"}
    logger.info(f"TRADE SCAN: Combined pool = {len(all_tickers)} unique tickers to score")

    # ── Step 5: Score every candidate through the unified function ────────────
    # Each ticker is wrapped in try/except — a single bad ticker (missing column,
    # zero-division, bad cast) must never abort the entire trade scan.
    candidates: list[dict] = []
    for ticker in all_tickers:
        try:
            result = _score_candidate(
                ticker         = ticker,
                stage_row      = stage_map.get(ticker, {}),
                sepa_row       = sepa_map.get(ticker, {}),
                rs_row         = rs_map.get(ticker,   {}),
                ohlcv          = ohlcv,
                regime_label   = regime_label,
                sector_results = sector_results,
                metadata       = metadata,
                market         = market,
            )
            if result is not None:
                candidates.append(result)
        except Exception as _score_err:
            logger.debug(f"_score_candidate failed for {ticker}: {_score_err}")

    logger.info(f"TRADE SCAN: Unified scoring → {len(candidates)} candidates above quality floor (score ≥ 40)")

    # ── Step 6: Conviction streak score boost ────────────────────────────────
    # (same as before — streak from previous run's conviction scan)
    try:
        from persistence import get_all_active as _get_conv_active
        _conv_streaks = {t: r.get("streak", 0)
                         for t, r in _get_conv_active("conviction").items()}
        _boosted = 0
        for _cand in candidates:
            _streak = _conv_streaks.get(_cand["_ticker"], 0)
            if   _streak >= 15: _cand["_score"] = round(_cand["_score"] + 6.0, 1); _boosted += 1
            elif _streak >= 7:  _cand["_score"] = round(_cand["_score"] + 4.0, 1); _boosted += 1
            elif _streak >= 3:  _cand["_score"] = round(_cand["_score"] + 2.0, 1); _boosted += 1
        logger.info(f"TRADE SCAN: Conviction streak boost applied to {_boosted} candidates")
    except Exception as _cse:
        logger.debug(f"TRADE SCAN: Conviction streak boost skipped: {_cse}")

    # ── Step 7: Holdings Alert — TheWrap signals for held positions only ────────
    # Loads Om-Holdings and scans ONLY those positions (fast — 40-60 stocks vs 1500+).
    # Output: one row per holding sorted by urgency — the morning "what to do" view.
    logger.info("TRADE SCAN: Running Holdings Alert (TheWrap scan)...")
    from ranker_stage import run_exit_monitor
    from screeners.holdings_reader import load_holdings
    from sheets_writer import get_client as _get_sheets_client

    holdings = {}
    try:
        sheets_client = _get_sheets_client()
        holdings      = load_holdings(sheets_client, market)
        logger.info(f"TRADE SCAN: Loaded {len(holdings)} {market.upper()} holdings")
    except Exception as e:
        logger.warning(f"TRADE SCAN: Could not load holdings: {e}")

    holdings_alert_df = _safe_run(
        run_exit_monitor,
        ohlcv=ohlcv, metadata=metadata, benchmark=benchmark,
        market=market, cfg=STAGE_CFG, holdings=holdings,
    )
    # _safe_run returns empty DataFrame on error; run_exit_monitor now returns
    # a single DataFrame (holdings only — no full-universe tuple anymore)
    if isinstance(holdings_alert_df, tuple):
        holdings_alert_df = holdings_alert_df[0]   # backward compat guard

    logger.info(f"TRADE SCAN: Holdings Alert — {len(holdings_alert_df)} positions analysed")

    # ── Step 8: Quality threshold → sort → top 20 ────────────────────────────
    # Regime now only affects position sizing (handled by user) — NOT list size.
    # All stocks scoring ≥ 40 are shown (up to 20). In a bear market, fewer
    # stocks naturally clear the threshold — the list shrinks organically.
    # Build full pool df for reentry_pool before slicing (prevents false exits
    # for stocks that scored ≥ 40 but fell just outside the top-20 display).
    trade_full_pool_df = pd.DataFrame(
        [{"Ticker": c["_ticker"]} for c in candidates]
    ) if candidates else pd.DataFrame(columns=["Ticker"])

    top_candidates = sorted(candidates, key=lambda c: c["_score"], reverse=True)[:20]
    trade_df = _build_trade_output(top_candidates, market, regime_mult)
    trade_df = annotate_df(trade_df, "trade", data_as_of=data_as_of)
    trade_df = annotate_streak_df(trade_df, f"streak_trade_{market}", data_as_of=data_as_of)

    logger.info(f"TRADE SCAN ✓ {len(top_candidates)} trade candidates (regime: {regime_label})")

    # ── Step 9: Daily BUY Conviction — 3 signals on full universe ────────────
    # Computes Stage2 / RS / SEPA signals on EVERY stock (not just screener top-30).
    # Top-20 by conviction score with streak tracking.
    logger.info("TRADE SCAN: Running Daily BUY Conviction scan...")
    conviction_df = pd.DataFrame()
    try:
        conviction_df = run_conviction_scan(
            ohlcv     = ohlcv,
            metadata  = metadata,
            benchmark = benchmark,
            market    = market,
            stage_df  = stage_pool,   # full pool (not display-30) so rank-35 stocks get enrichment
            sepa_df   = sepa_pool,
            rs_df     = rs_pool,      # critical: "🌟 RS Leads" annotation needs the full pool
        )
        n3 = int((conviction_df["# Signals"] == 3).sum()) if not conviction_df.empty else 0
        logger.info(
            f"TRADE SCAN: Daily BUY Conviction = {len(conviction_df)} stocks "
            f"({n3} triple-signal)"
            if not conviction_df.empty else "TRADE SCAN: Daily BUY Conviction = 0 stocks"
        )
    except Exception as _ce:
        import traceback
        logger.warning(
            f"TRADE SCAN: Conviction scan failed (non-fatal): {_ce}\n"
            + traceback.format_exc()
        )

    # ── Step 10: Data Quality — flag tickers with NaN price/volume or stale data ─
    logger.info("TRADE SCAN: Running Data Quality scan...")
    data_quality_df = pd.DataFrame()
    try:
        data_quality_df = run_data_quality_scan(ohlcv, metadata, market=market)
        logger.info(f"TRADE SCAN: Data Issues = {len(data_quality_df)} bad tickers")
    except Exception as _dqe:
        logger.warning(f"TRADE SCAN: Data Quality scan failed (non-fatal): {_dqe}")

    # ── Step 11: Append 14-day exit history to screener tabs ─────────────────
    # Stocks that drop out of each screener are kept at the bottom for 14 days
    # with an "Exit Date" column showing when they left. The tabs are fixed (no
    # daily dated tabs) so this provides rolling history without tab proliferation.
    logger.info("TRADE SCAN: Appending 14-day exit history to screener tabs...")
    try:
        # Use FULL POOLS (not display-frame top-N) for trade exit reason lookup.
        # A stock ranked 35 in Stage is still an active Stage-2 candidate — using
        # stage_df (top-30 display) would incorrectly report "Stage 2 lost" for it.
        _active_stage = stage_pool.copy() if not stage_pool.empty else stage_pool
        _active_sepa  = sepa_pool.copy()  if not sepa_pool.empty  else sepa_pool
        _active_rs    = rs_pool.copy()    if not rs_pool.empty    else rs_pool

        stage_df = append_screener_exits(
            stage_df, bucket=f"stage_{market}",
            exit_reason=lambda t, row: _stage_exit_reason(t, row, ohlcv),
            reentry_pool=stage_pool,   # full pool so rank 31–80 stocks don't false-exit
            data_as_of=data_as_of,
        )
        sepa_df  = append_screener_exits(
            sepa_df,  bucket=f"sepa_{market}",
            exit_reason=lambda t, row: _sepa_exit_reason(t, row, ohlcv),
            reentry_pool=sepa_pool,    # same — prevent false-exit for display-cutoff stocks
            data_as_of=data_as_of,
        )
        rs_df    = append_screener_exits(
            rs_df,    bucket=f"rs_{market}",
            exit_reason=lambda t, row: _rs_exit_reason(t, row, ohlcv, benchmark),
            reentry_pool=rs_pool,      # same — RS Leaders pool is wider than display
            data_as_of=data_as_of,
        )
        trade_df = append_screener_exits(
            trade_df, bucket=f"trade_{market}",
            exit_reason=lambda t, row: _trade_exit_reason(
                t, row, ohlcv, _active_stage, _active_sepa, _active_rs
            ),
            reentry_pool=trade_full_pool_df,   # prevent false-exit for regime-slot-capped stocks
            data_as_of=data_as_of,
        )
    except Exception as _ee:
        logger.warning(f"TRADE SCAN: Exit history append failed (non-fatal): {_ee}")

    # data_as_of was computed at Step 0 (top of run_trade_scan) and used throughout.
    # It is also returned so sheets_writer can write it to the Run Log.
    return {
        "stage":          stage_df,
        "sepa":           sepa_df,
        "rs":             rs_df,
        "trade":          trade_df,
        "holdings_alert": holdings_alert_df,
        "sectors":        sector_results,      # dict[str, SectorResult] — for display + JSON export
        "conviction":     conviction_df,       # Daily BUY — 2+ signal stocks with streak
        "data_quality":   data_quality_df,     # Data Issues — NaN / stale / spike tickers
        "data_as_of":     data_as_of,          # last trading bar date in the benchmark feed
    }


# =============================================================================
# UNIFIED CANDIDATE SCORER
# =============================================================================

def _score_candidate(
    ticker:         str,
    stage_row:      dict,
    sepa_row:       dict,
    rs_row:         dict,
    ohlcv:          dict,
    regime_label:   str,
    sector_results: dict,
    metadata:       dict,
    market:         str,
) -> "dict | None":
    """
    Unified scoring — one function for every candidate regardless of which
    screener found it.  Replaces the old 4-path Tier A/B system.

    HARD ELIMINATORS (return None):
      • Weekly stage = W-S1 Accum or W-S4 Decline (price below weekly EMA)
      • TheWrap = TW_EXIT (multi-month structure broken — don't fight it)
      • Fewer than 30 bars of OHLCV
      • Stop > 11% (can't size it)
      • Score < 40 after all components (below quality floor)

    SCORE COMPONENTS (0–100 before bonuses):
      A. Signal breadth   0–30   (Stage lens + SEPA lens + RS lens, each 0–10)
      B. Entry signal     0–25   (Cheat=25, BREAKOUT<3%=22, BREAKOUT3-5%=16,
                                  AT_PIVOT=18, WEAK_BREAKOUT=12, IN_BASE=6, none=2)
      C. Risk quality     0–20   (stop tightness 0–10 + pivot proximity 0–10)
      D. Volume           0–15   (Very High=15, High=10, Normal=5, Low=1)
      E. Sector strength  0–10   (LEADING=10, NEUTRAL=6, LAGGING=2)

    BONUSES (added after subtotal, can push above 100, capped at 110):
      W-S2 ✓ confirmed          +5
      Bullish TheWrap            +4
      Maintain TheWrap           +2
      RS Leads Price (🌟)        +5
      Streak handled separately in run_trade_scan

    PENALTIES (multiplied after bonuses):
      RSI > 82                   × 0.88
      RSI 75–82                  × 0.95
      Stage duration < 15 bars   × 0.85
      TW_FADING                  × 0.82   (demote, not eliminate)
    """
    # ── Resolve raw ticker once — used for all OHLCV lookups below ───────────
    # BUG FIX: _restore_ticker was called twice (TheWrap block + OHLCV block),
    # iterating the suffix list twice per candidate.  Cache it here.
    raw_ticker = _restore_ticker(ticker, ohlcv)

    # ── Hard gate 1: Weekly stage ─────────────────────────────────────────────
    # BUG FIX: old code used a single ternary — if sepa_row is non-empty but
    # lacks "Weekly Stage" (e.g. SEPA computation failed for this ticker), it
    # returned "" → "Unknown" and never fell through to stage_row.  A Stage-only
    # stock with W-S4 would slip through the hard gate.
    # Fix: explicit fallback chain — sepa_row → stage_row → "Unknown".
    weekly_stage_str = (
        str(sepa_row.get("Weekly Stage", "") or "")  if sepa_row  else ""
    ) or (
        str(stage_row.get("Weekly Stage", "") or "") if stage_row else ""
    ) or "Unknown"
    if weekly_stage_str in ("W-S1 Accum", "W-S4 Decline"):
        return None

    # ── TheWrap ───────────────────────────────────────────────────────────────
    # BUG FIX: same fallback-chain issue as weekly_stage above.
    # Priority: sepa_row → stage_row → recompute from OHLCV.
    # Recomputing from OHLCV is the last resort — stage_row already has it
    # pre-computed; using it avoids a silent failure on the OHLCV path.
    tw_str = (
        str(sepa_row.get("TheWrap",  "") or "") if sepa_row  else ""
    ) or (
        str(stage_row.get("TheWrap", "") or "") if stage_row else ""
    )
    if not tw_str or tw_str in ("nan", "None", "⚪ No Data"):
        # Recompute from OHLCV as last resort (stocks that only appear in RS pool)
        _raw = ohlcv.get(raw_ticker, pd.DataFrame())
        if not _raw.empty:
            try:
                _, _tw_lbl, *_ = _compute_thewrap(_to_weekly_ohlcv(_raw))
                tw_str = _tw_lbl
            except Exception:
                pass
    if not tw_str:
        tw_str = "—"
    # Hard gate 2: TW_EXIT — multi-month structure broken, don't fight it
    if any(x in tw_str for x in ("TW_EXIT", "TW: Exit")):
        return None

    # ── OHLCV ─────────────────────────────────────────────────────────────────
    raw_df = ohlcv.get(raw_ticker, pd.DataFrame())
    if raw_df.empty or len(raw_df) < 30:
        return None
    close       = float(raw_df["close"].iloc[-1])
    close_s     = raw_df["close"]
    ema21       = float(close_s.ewm(span=21, adjust=False).mean().iloc[-1])
    rsi         = _quick_rsi(raw_df)

    # ── Entry signal detection ────────────────────────────────────────────────
    entry_signal_stage = str(stage_row.get("Entry Signal", "")) if stage_row else ""
    is_cheat           = "Cheat Entry" in entry_signal_stage

    # ── Helper: safe float parse for formatted SEPA percentage strings ─────────
    def _safe_pct(val, default: float) -> float:
        """Parse '±X.X%' strings; return default on any error."""
        try:
            return float(str(val).replace("%", "").replace("+", "").strip() or default)
        except (ValueError, TypeError):
            return default

    # Prefer SEPA's computed prices when available (SEPA already ran detect_base)
    try:
        _sepa_entry_val = float(sepa_row.get("Entry ₹") or 0) if sepa_row else 0.0
    except (ValueError, TypeError):
        _sepa_entry_val = 0.0

    if sepa_row and _sepa_entry_val > 0:
        state         = str(sepa_row.get("Breakout State", ""))
        entry_price   = _sepa_entry_val
        stop_price    = float(sepa_row.get("Stop ₹", ema21 * 0.97) or (ema21 * 0.97))
        pivot_dist    = _safe_pct(sepa_row.get("Pivot Dist %", "0"), 0.0)
        # BUG FIX: clamp stop_dist_pct — a data error returning 0 or negative
        # would give stop_norm = 1.0 (perfect score) which inflates score_c.
        stop_dist_pct = max(0.5, min(_safe_pct(sepa_row.get("Stop Dist %", "7"), 7.0), 15.0))
        # Override state with Cheat Entry if Stage confirms it (highest conviction)
        if is_cheat:
            state         = "AT_PIVOT"
            entry_price   = round(close * 1.001, 2)
            stop_price    = round(ema21 * 0.97,  2)
            pivot_dist    = round((close - ema21) / ema21 * 100, 1) if ema21 > 0 else 0.0
            stop_dist_pct = max(0.5, min((entry_price - stop_price) / entry_price * 100, 15.0))
    elif is_cheat:
        # Cheat Entry without a SEPA row — compute from OHLCV
        state         = "AT_PIVOT"
        entry_price   = round(close * 1.001, 2)
        stop_price    = round(ema21 * 0.97,  2)
        pivot_dist    = round((close - ema21) / ema21 * 100, 1) if ema21 > 0 else 0.0
        stop_dist_pct = max(0.5, min((entry_price - stop_price) / entry_price * 100, 15.0))
    else:
        # No SEPA row and not a cheat entry — detect from OHLCV directly
        _sepa_cfg   = SEPAConfig()
        _base       = detect_base(raw_df["high"], raw_df["low"], raw_df["close"], raw_df["volume"], _sepa_cfg)
        pivot_high  = _base.base_high if _base.valid else float(raw_df["high"].iloc[-20:].max())
        pivot_dist  = round((close - pivot_high) / pivot_high * 100, 1) if pivot_high > 0 else 0.0
        stop_price  = round(ema21 * 0.97, 2)
        stop_dist_pct = max(0.5, (close - stop_price) / close * 100) if close > stop_price else 7.0

        if pivot_dist > 5.0 or pivot_dist < -8.0:
            # Too extended or too far below pivot — build base before acting
            state       = "IN_BASE"
            entry_price = round(pivot_high * 1.002, 2)
        elif pivot_dist > 0:
            state       = "BREAKOUT"
            entry_price = round(close * 1.001, 2)
        elif pivot_dist > -3.0:
            state       = "AT_PIVOT"
            entry_price = round(pivot_high * 1.002, 2)
        else:
            # 3–8% below pivot — pulled back into the base after a breakout attempt
            state       = "WEAK_BREAKOUT"
            entry_price = round(pivot_high * 1.002, 2)

    # Hard gate 3: stop too wide to size — ELIMINATE, do not fake a tighter stop.
    # BUG FIX: old code set stop_price = entry * 0.92 and stop_dist_pct = 8.0,
    # flowing through with an inflated stop_norm (good risk score for a bad setup).
    # A >11% stop means the stock is too volatile or too extended for a proper entry.
    # Minervini rule: if you can't define a 7-8% stop, skip the trade.
    if stop_dist_pct > 11.0:
        return None

    # Risk % for display
    risk_pct = round((entry_price - stop_price) / entry_price * 100, 1) if entry_price > stop_price > 0 else stop_dist_pct

    # ── A. Signal Breadth (0–30) ──────────────────────────────────────────────
    s2_pts     = float(stage_row.get("Stage Score S2", 0)) if stage_row else 0.0
    sepa_raw   = float(sepa_row.get("Raw Score",       0)) if sepa_row else 0.0
    rs_pts     = float(rs_row.get("RS Score",          0)) if rs_row   else 0.0
    stage_lens = min(s2_pts  / 10.0,  1.0) * 10.0 if stage_row else 0.0
    sepa_lens  = min(sepa_raw / 100.0, 1.0) * 10.0 if sepa_row  else 0.0
    rs_lens    = min(rs_pts  / 100.0, 1.0) * 10.0 if rs_row    else 0.0
    score_a    = stage_lens + sepa_lens + rs_lens  # 0–30

    # ── B. Entry Signal Quality (0–25) ────────────────────────────────────────
    if is_cheat:
        score_b       = 25.0
        entry_quality = "🟢 Cheat Entry — EMA21 pullback"
    elif state == "BREAKOUT" and abs(pivot_dist) <= 3.0:
        score_b       = 22.0
        entry_quality = "🟢 Fresh Breakout"
    elif state == "BREAKOUT":
        score_b       = 16.0
        entry_quality = "🟡 Breakout — slight extension"
    elif state == "AT_PIVOT":
        score_b       = 18.0
        entry_quality = "🔔 At Pivot — buy stop"
    elif state == "WEAK_BREAKOUT":
        score_b       = 12.0
        entry_quality = "🟡 Back in Base — wait"
    elif state == "IN_BASE":
        score_b       = 6.0
        entry_quality = "📋 Building Base — set alert"
    else:
        score_b       = 2.0
        entry_quality = "👁 Watchlist"

    # ── C. Risk Quality (0–20) ────────────────────────────────────────────────
    # stop_norm:  tight stop = high score.  3% stop → 1.0,  11% stop → 0.0
    # prox_norm:  close to pivot = high score.  0% away → 1.0,  8% away → 0.0
    #
    # BUG FIX: old _prox_dist had a redundant `is_cheat` branch that recomputed
    # `(close - ema21) / ema21 * 100` — identical to the already-assigned
    # `pivot_dist` for cheat entries.  Use abs(pivot_dist) unconditionally.
    stop_norm  = min(max(0.0, (11.0 - stop_dist_pct) / 8.0), 1.0)  # 3%→1.0, 11%→0.0
    prox_norm  = max(0.0, 1.0 - abs(pivot_dist) / 8.0)              # 0%→1.0, 8%→0.0
    score_c    = stop_norm * 10.0 + prox_norm * 10.0                # 0–20

    # ── D. Volume Conviction (0–15) ───────────────────────────────────────────
    # BUG FIX: old code used a single ternary — if sepa_row exists but lacks
    # "Vol Conv" (e.g. RS-only stock), the empty string caused "Normal" (5 pts)
    # even when stage_row had "Very High".  Fix: explicit fallback chain.
    vol_conv = (
        str(sepa_row.get("Vol Conv",       "") or "") if sepa_row  else ""
    ) or (
        str(stage_row.get("Vol Conviction","") or "") if stage_row else ""
    )
    if "Very High" in vol_conv: score_d = 15.0
    elif "High"    in vol_conv: score_d = 10.0
    elif "Low"     in vol_conv: score_d =  1.0
    else:                       score_d =  5.0   # Normal / unknown

    # ── E. Sector Strength (0–10) ─────────────────────────────────────────────
    sector_res     = get_sector_for_ticker(ticker, metadata, sector_results, market)
    sector_mult_v  = sector_res.sector_mult   if sector_res else 1.0
    sector_label_v = sector_res.sector_label  if sector_res else "NEUTRAL"
    sector_score_v = sector_res.sector_score  if sector_res else 50.0
    if   sector_label_v == "LEADING":  score_e = 10.0
    elif sector_label_v == "LAGGING":  score_e =  2.0
    else:                              score_e =  6.0

    score = score_a + score_b + score_c + score_d + score_e  # 0–100

    # ── Bonuses ───────────────────────────────────────────────────────────────
    if weekly_stage_str == "W-S2 ✓":
        score += 5.0
    if any(x in tw_str for x in ("TW_BULLISH", "TW: Bullish")):
        score += 4.0
    elif any(x in tw_str for x in ("TW_MAINTAIN", "TW: Maintain")):
        score += 2.0

    # RS Leads Price
    rs_leads_price = str(rs_row.get("RS Leads Price", "")) if rs_row else ""
    rs_at_high     = str(rs_row.get("RS at 52w High", "")) if rs_row else ""
    if rs_leads_price == "🌟 Leads":
        rs_leading = "🌟 RS Leads"
        score     += 5.0
    elif rs_at_high == "✓":
        rs_leading = "✓"
    else:
        rs_leading = "·"

    # ── Penalties ─────────────────────────────────────────────────────────────
    if   rsi > 82: score *= 0.88
    elif rsi > 75: score *= 0.95
    stage_dur = float(stage_row.get("Duration (bars)", 60)) if stage_row else 60.0
    if stage_dur < 15:
        score *= 0.85
    if any(x in tw_str for x in ("TW_FADING", "TW: Fading")):
        score *= 0.82

    score = min(round(score, 1), 110.0)  # cap at 110

    # ── Quality floor ─────────────────────────────────────────────────────────
    if score < 40.0:
        return None

    # ── Action label ──────────────────────────────────────────────────────────
    # BUG FIX: old code had a gap — BREAKOUT/AT_PIVOT stocks scoring 55–71 fell
    # through to "👁 WATCHLIST" with no actionable signal.  A stock at the exact
    # pivot with decent conviction should get "NEAR PIVOT" (place buy stop).
    #
    # Four tiers:
    #   🟢 BUY NOW      score ≥ 72, at/near the trigger (Minervini: act immediately)
    #   🔔 NEAR PIVOT   score 55–71, at/near the trigger (buy stop order, manage size)
    #   📋 BASE BUILDING any score ≥ 40, stock building a base (set alert for breakout)
    #   👁 WATCHLIST    everything else (monitor, no immediate action)
    if score >= 72 and (is_cheat or state in ("BREAKOUT", "AT_PIVOT")):
        action = "🟢 BUY NOW"
    elif score >= 55 and (is_cheat or state in ("BREAKOUT", "AT_PIVOT")):
        action = "🔔 NEAR PIVOT"
    elif state in ("WEAK_BREAKOUT", "IN_BASE"):
        action = "📋 BASE BUILDING"
    else:
        action = "👁 WATCHLIST"

    # ── Weekly label ──────────────────────────────────────────────────────────
    if   weekly_stage_str == "W-S2 ✓":          weekly_label = "W-Confirmed"
    elif "W-S3" in weekly_stage_str:             weekly_label = "W-S3 Pending"
    else:                                        weekly_label = "W-Pending"

    # ── Lens confirmation string ──────────────────────────────────────────────
    lenses = []
    if stage_row: lenses.append("Stage")
    if sepa_row:  lenses.append("SEPA")
    if rs_row:    lenses.append("RS")
    lens_str = " + ".join(lenses) if lenses else "—"

    # ── Metadata ──────────────────────────────────────────────────────────────
    company  = (
        str(sepa_row.get("Company",    ticker)) if sepa_row  else
        str(stage_row.get("Company",   ticker)) if stage_row else
        str(rs_row.get("Company",      ticker)) if rs_row    else ticker
    )
    sector   = (
        str(sepa_row.get("Sector",     "Unknown")) if sepa_row  else
        str(stage_row.get("Sector",    "Unknown")) if stage_row else "Unknown"
    )
    tv_url   = (
        str(sepa_row.get("TradingView",  "")) if sepa_row  else
        str(stage_row.get("TradingView", "")) if stage_row else
        str(rs_row.get("TradingView",    "")) if rs_row    else ""
    )

    return {
        "_ticker":       ticker,
        "_score":        score,
        "_action":       action,
        "_entry_quality":entry_quality,
        "_state":        state,
        "_sepa_raw":     sepa_raw,
        "_s2_pts":       round(s2_pts,  1),
        "_rs_pts":       round(rs_pts,  1),
        "_rsi":          round(rsi,     0),
        "_stop_dist":    round(stop_dist_pct, 1),
        "_pivot_dist":   round(pivot_dist,    1),
        "_regime":       regime_label,
        "_price":        round(close,       2),
        "_entry":        entry_price,
        "_stop":         round(stop_price,  2),
        "_risk_pct":     risk_pct,
        "_company":      company,
        "_sector":       sector,
        "_tv":           tv_url,
        "_weekly_stage": weekly_stage_str,
        "_weekly_label": weekly_label,
        "_tw_label":     tw_str,
        "_rs_leading":   rs_leading,
        "_lens_str":     lens_str,
        "_is_cheat":     is_cheat,
        "_vcp":          sepa_row.get("VCP Count",  0) if sepa_row  else 0,
        "_base_count":   sepa_row.get("Base Count", 0) if sepa_row  else 0,
        "_sepa_score":   float(sepa_row.get("SEPA Score", 0)) if sepa_row else 0.0,
        "_regime_mult":  1.0,   # stored on candidate for reference only
        "_sector_label": sector_label_v,
        "_sector_score": round(sector_score_v, 1),
        "_sector_mult":  round(sector_mult_v,  2),
    }


# =============================================================================
# TRADE OUTPUT BUILDER
# =============================================================================

def _build_trade_output(candidates: list, market: str, regime_mult: float = 1.0) -> pd.DataFrame:
    """Convert unified candidate dicts to the clean trade output DataFrame."""
    if not candidates:
        return _empty_trade_result()

    rows = []
    for c in candidates:
        entry = c["_entry"]
        stop  = c["_stop"]
        state = c["_state"]

        # Signal Summary — what confirmed this stock
        parts = []
        rs_lead = c.get("_rs_leading", "·")
        if rs_lead == "🌟 RS Leads":    parts.append("🌟 RS Leads Price")
        elif rs_lead == "✓":            parts.append("RS Leading ✓")
        weekly_lbl = c.get("_weekly_label", "")
        if   weekly_lbl == "W-Confirmed":  parts.append("W-Confirmed ✓")
        elif weekly_lbl == "W-S3 Pending": parts.append("W-S3")
        tw_lbl = c.get("_tw_label", "—")
        if any(x in tw_lbl for x in ("TW_BULLISH", "TW: Bullish")):   parts.append("TW: Bullish ✓")
        elif any(x in tw_lbl for x in ("TW_MAINTAIN", "TW: Maintain")): parts.append("TW: Maintain")
        elif any(x in tw_lbl for x in ("TW_FADING",  "TW: Fading")):  parts.append("TW: Fading ⚠")
        vcp = c.get("_vcp", 0)
        if isinstance(vcp, (int, float)) and vcp >= 2:
            parts.append(f"VCP {int(vcp)}×")
        if c.get("_base_count", 0) == 1:
            parts.append("1st base")
        lens_str = c.get("_lens_str", "")
        if lens_str and lens_str != "—":
            parts.append(lens_str)
        signal = " | ".join(parts) if parts else "—"

        rows.append({
            "Ticker":         c["_ticker"],
            "Company":        c["_company"],
            "Action":         c["_action"],
            "Entry Quality":  c["_entry_quality"],
            "Entry ₹":        round(entry, 2) if entry > 0 else "—",
            "Stop ₹":         round(stop,  2) if stop  > 0 else "—",
            "Risk %":         round(c["_risk_pct"], 1) if c["_risk_pct"] > 0 else "—",
            "Trade Score":    c["_score"],
            "RS Score":       c["_rs_pts"],
            "SEPA Score":     c["_sepa_score"],
            "RSI(14)":        c["_rsi"],
            "Breakout State": state,
            "Signal Summary": signal,
            "Regime ⚠":       _regime_warning(c["_regime"]),
            "Sector":         c["_sector"],
            "Sector Label":   c.get("_sector_label", "NEUTRAL"),
            "TradingView":    c["_tv"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Trade Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


# =============================================================================
# HELPERS
# =============================================================================

def _get_regime_weights(regime_mult: float) -> dict:
    """
    Map regime_mult to the appropriate weight set.

    BUG FIX: thresholds now match the slot allocation thresholds and the discrete
    values returned by get_market_regime() — 1.00 / 0.80 / 0.50 / 0.30 / 0.15.
    Old thresholds (0.85 / 0.60) caused regime_mult=0.80 to get "bull" weights
    (highest SEPA weight) while getting only "Mild Bull" slot count — a mismatch.
    Now both functions use the same boundary set: 0.90 / 0.65 / 0.40 / 0.22.
    """
    if regime_mult >= 0.90:   return _REGIME_WEIGHTS["bull"]
    if regime_mult >= 0.65:   return _REGIME_WEIGHTS["mild_bull"]
    if regime_mult >= 0.40:   return _REGIME_WEIGHTS["neutral"]
    if regime_mult >= 0.22:   return _REGIME_WEIGHTS["caution"]
    return _REGIME_WEIGHTS["bear"]


def _regime_warning(regime_label: str) -> str:
    """Market regime label — position size advice is already baked into Pos Size % column."""
    # IMPORTANT: check "Mild" before "Bull" — "Mild Bull" contains "Bull" as a substring
    if "4/5" in regime_label or "Mild" in regime_label:
        return "🟡 Mild Bull"
    if "5/5" in regime_label or "Bull" in regime_label:
        return "✅ Bull Market"
    if "3/5" in regime_label or "Neutral" in regime_label:
        return "🟠 Neutral Market"
    if "2/5" in regime_label or "Caution" in regime_label:
        return "🔴 Caution"
    return "🚨 Bear — Paper Only"


def _safe_run(fn, **kwargs) -> pd.DataFrame:
    """Run a screener function and return empty DataFrame on failure."""
    try:
        result = fn(**kwargs)
        return result if result is not None and not (hasattr(result, "empty") and result.empty) else pd.DataFrame()
    except (KeyboardInterrupt, SystemExit):
        raise   # never swallow — Ctrl+C must stop the run immediately
    except Exception as e:
        logger.warning(f"Screener {fn.__name__} failed: {e}")
        return pd.DataFrame()


def _df_to_map(df: pd.DataFrame, key_col: str) -> dict:
    """Convert DataFrame rows to dict keyed by key_col value."""
    if df.empty or key_col not in df.columns:
        return {}
    return {str(row[key_col]): row.to_dict() for _, row in df.iterrows()}


def _restore_ticker(clean: str, ohlcv: dict) -> str:
    """Try to find the original ticker (with .NS/.BO suffix) from the clean display name."""
    if clean in ohlcv:
        return clean
    for suffix in (".NS", ".BO", ""):
        candidate = clean + suffix
        if candidate in ohlcv:
            return candidate
    return clean


def _quick_rsi(df: pd.DataFrame, period: int = 14) -> float:
    """Compute RSI(14) from an OHLCV df. Returns 50 if insufficient data."""
    if df.empty or "close" not in df.columns or len(df) < period + 1:
        return 50.0
    close = df["close"].dropna()
    if len(close) < period + 1:
        return 50.0
    delta    = close.diff().dropna()
    gain     = delta.clip(lower=0)
    loss     = (-delta.clip(upper=0))
    avg_gain = float(gain.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])
    avg_loss = float(loss.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])
    if avg_loss == 0:
        return 100.0
    return round(100 - 100 / (1 + avg_gain / avg_loss), 1)



# =============================================================================
# EXIT REASON DIAGNOSTICS
# Each function inspects current OHLCV (and cross-screener state for trade tab)
# to explain specifically WHY a ticker dropped out of that screener.
# These are passed as callables to append_screener_exits() so the reason is
# computed at the moment of first exit and stored permanently in persistence.json.
# =============================================================================

def _stage_exit_reason(ticker: str, saved_row: dict, ohlcv: dict) -> str:
    """
    Why did this stock leave Stage Leaders?

    Stage 2 bullish EMA stack: Price > EMA21 > EMA50 > EMA200, EMA200 slope > 0.
    EMA21 (fast, 1-month) sits ABOVE EMA50 (slow, 2.5-month) in an uptrend
    because recent prices are higher — fast-above-slow is bullish.
    When EMA21 drops below EMA50, the short-term trend has reversed through
    the medium-term trend: that is the bearish cross we flag.
    """
    raw_key = _restore_ticker(ticker, ohlcv)
    df = ohlcv.get(raw_key, pd.DataFrame())
    if df.empty or "close" not in df.columns:
        return ""   # blank = "not computed yet"; backfill will retry next run when data is available

    close = df["close"].dropna()
    if len(close) < 50:
        return ""   # same: insufficient history this run — retry next run

    try:
        ema21_s  = close.ewm(span=21,  adjust=False).mean()
        ema50_s  = close.ewm(span=50,  adjust=False).mean()
        ema200_s = close.ewm(span=200, adjust=False).mean()
        price    = float(close.iloc[-1])
        e21      = float(ema21_s.iloc[-1])
        e50      = float(ema50_s.iloc[-1])
        e200     = float(ema200_s.iloc[-1])
        slope    = float(ema200_s.pct_change(10).iloc[-1]) * 100

        parts = []

        # Gate 1 — price vs EMA200 (most important)
        if price < e200:
            pct = (e200 - price) / e200 * 100
            parts.append(f"Price below EMA200 ({pct:.1f}% under)")
        if slope < 0:
            parts.append(f"EMA200 slope negative ({slope:.2f}%)")

        # Gate 2 — EMA stack: healthy uptrend = EMA21 > EMA50 (fast above slow)
        if e21 < e50:
            # Short-term MA has crossed below medium-term MA — trend weakening
            pct = (e50 - e21) / e50 * 100
            parts.append(f"EMA21 below EMA50 ({pct:.1f}% gap — short-term trend broken)")

        # Gate 3 — price vs EMA21
        if price < e21:
            pct = (e21 - price) / e21 * 100
            parts.append(f"Price below EMA21 ({pct:.1f}% under)")

        # RS context from last saved row when EMA structure is still intact
        last_rs = str(saved_row.get("RS Status", ""))
        if "Weak" in last_rs and not parts:
            parts.append(f"RS weakened (was {last_rs})")

        if parts:
            return " · ".join(parts)

        # All EMA/RS checks still pass — the stock is still in Stage 2 but
        # scored lower than the other 30 stocks in the universe today.
        last_score = saved_row.get("Score", "") or saved_row.get("Stage Score S2", "")
        score_str  = f" — score was {float(last_score):.2f}" if last_score else ""
        return f"Ranked out of Stage Leaders top 30{score_str} (EMA structure intact)"
    except Exception:
        return "Stage 2 structure lost"


def _sepa_exit_reason(ticker: str, saved_row: dict, ohlcv: dict) -> str:
    """Why did this stock leave SEPA Setups? Check pivot distance and structure."""
    raw_key = _restore_ticker(ticker, ohlcv)
    df = ohlcv.get(raw_key, pd.DataFrame())
    if df.empty or "close" not in df.columns:
        return ""   # blank = retry next run; don't store "data unavailable" as permanent reason

    close = df["close"].dropna()
    if len(close) < 20:
        return ""   # same: insufficient history this run

    try:
        price = float(close.iloc[-1])
        # BUG FIX: old code used close.iloc[-20:].max() as the pivot proxy.
        # Closing prices understate the 20-bar high — a stock that peaked intra-day
        # and closed lower would show a smaller distance than its real pullback.
        # Use the actual high series (canonical Minervini/Weinstein pivot definition).
        if "high" in df.columns and len(df) >= 20:
            high_20 = float(df["high"].dropna().iloc[-20:].max())
        else:
            high_20 = float(close.iloc[-20:].max())
        dist    = (price - high_20) / high_20 * 100   # +ve = extended, -ve = below pivot

        # Stage break overrides everything
        if len(close) >= 200:
            e200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
            if price < e200:
                pct = (e200 - price) / e200 * 100
                return f"Stage 2 broken — {pct:.1f}% below EMA200"

        if dist < -12:
            return f"Base failed — {abs(dist):.1f}% below pivot"
        if dist < -5:
            return f"Pulled back {abs(dist):.1f}% below pivot"
        if dist > 8:
            return f"Extended {dist:+.1f}% past pivot — setup expired"

        last_state = str(saved_row.get("Breakout State", ""))
        last_score = saved_row.get("SEPA Score", "")
        if last_state:
            suffix = f" — score {last_score}" if last_score else ""
            return f"Pivot zone left (was {last_state}{suffix})"
        return "Pivot zone left — setup invalidated"
    except Exception:
        return "Setup invalidated"


def _rs_exit_reason(ticker: str, saved_row: dict, ohlcv: dict,
                    benchmark: pd.DataFrame) -> str:
    """Why did this stock leave RS Leaders? Measure RS gap from 52w high."""
    raw_key = _restore_ticker(ticker, ohlcv)
    df = ohlcv.get(raw_key, pd.DataFrame())
    if df.empty or "close" not in df.columns:
        return ""   # blank = retry next run; don't store "data unavailable" as permanent reason

    close = df["close"].dropna()
    bench = (benchmark["close"].dropna()
             if benchmark is not None and not benchmark.empty
             else pd.Series(dtype=float))

    try:
        price = float(close.iloc[-1])

        # Stage breakdown is the most important signal
        if len(close) >= 200:
            e200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
            if price < e200:
                pct = (e200 - price) / e200 * 100
                return f"Stage 2 broken ({pct:.1f}% below EMA200) + RS dropped"

        # RS line gap from 52-week RS high
        if len(close) >= 60 and len(bench) >= 60:
            aligned = pd.concat(
                [close.rename("c"), bench.rename("b")], axis=1
            ).dropna()
            if len(aligned) >= 60:
                c_a   = aligned["c"]
                b_a   = aligned["b"]
                rs_ln = (c_a / b_a) / (float(c_a.iloc[0]) / float(b_a.iloc[0])) * 100
                window = min(252, len(rs_ln))
                rs_52w = float(rs_ln.rolling(window, min_periods=60).max().iloc[-1])
                rs_now = float(rs_ln.iloc[-1])
                gap    = (rs_52w - rs_now) / rs_52w * 100   # % below 52w RS high
                last   = saved_row.get("RS Score", "")
                if gap > 15:
                    return f"RS fell {gap:.1f}% from 52w high (score was {last})"
                if gap > 5:
                    return f"RS weakening — {gap:.1f}% below 52w RS high"

        # RS score dropped below the screener minimum (20) without a clear
        # RS-line gap signal — show the actual number so it's actionable.
        last_score = saved_row.get("RS Score", "")
        if last_score:
            try:
                return f"RS score dropped to below 20 (was {float(last_score):.0f})"
            except (ValueError, TypeError):
                return f"RS score dropped below minimum (was {last_score})"
        return "RS score dropped below minimum (20)"
    except Exception:
        return "RS leadership lost"


def _trade_exit_reason(ticker: str, saved_row: dict, ohlcv: dict,
                       stage_df: pd.DataFrame, sepa_df: pd.DataFrame,
                       rs_df: pd.DataFrame) -> str:
    """
    Why did this stock leave Trade Candidates?

    Diagnostic priority (stops at first match):
      1. Hard gate hit     — EMA200 broken, stop hit, TheWrap EXIT, W-S4 Decline
      2. Entry invalidated — extended past entry, breakout failed, base undercut
      3. Lens membership   — which screeners this stock left (with EMA detail)
      4. Score penalty     — RSI extension, TW_FADING, over-extension from EMA21
      5. Score context     — what score it had and what state it was in
    """
    # ── Saved context from last active appearance ──────────────────────────────
    last_state  = str(saved_row.get("Breakout State", ""))
    last_signal = str(saved_row.get("Signal Summary", ""))
    last_action = str(saved_row.get("Action", ""))
    last_quality = str(saved_row.get("Entry Quality", ""))

    def _flt(val, default=0.0) -> float:
        """Safe float — handles '₹1,234', '7.5%', '—', None."""
        try:
            return float(str(val).replace("₹","").replace("$","").replace("%","")
                         .replace(",","").replace("+","").strip() or default)
        except (ValueError, TypeError):
            return default

    last_score  = _flt(saved_row.get("Trade Score",  0))
    last_entry  = _flt(saved_row.get("Entry ₹",      0))
    last_stop   = _flt(saved_row.get("Stop ₹",       0))
    last_rsi    = _flt(saved_row.get("RSI(14)",       0))
    score_ctx   = f" (was {last_action}, score {last_score:.0f})" if last_score else ""

    # ── Resolve OHLCV ──────────────────────────────────────────────────────────
    raw_key  = _restore_ticker(ticker, ohlcv)
    df_ohlcv = ohlcv.get(raw_key, pd.DataFrame())
    has_data = not df_ohlcv.empty and "close" in df_ohlcv.columns and len(df_ohlcv) >= 30

    if has_data:
        close_s = df_ohlcv["close"].dropna()
        price   = float(close_s.iloc[-1])
        ema21   = float(close_s.ewm(span=21,  adjust=False).mean().iloc[-1])
        ema50   = float(close_s.ewm(span=50,  adjust=False).mean().iloc[-1])
        ema200  = float(close_s.ewm(span=200, adjust=False).mean().iloc[-1]) if len(close_s) >= 200 else 0.0
        rsi_now = _quick_rsi(df_ohlcv)

        # ── 1a. Stage 2 broken — most serious structural failure ───────────────
        if ema200 > 0 and price < ema200:
            pct = (ema200 - price) / ema200 * 100
            return f"Stage 2 broken — price {pct:.1f}% below EMA200{score_ctx}"

        # ── 1b. Stop level hit — trade invalidated ────────────────────────────
        if last_stop > 0 and price < last_stop:
            loss_pct = (last_stop - price) / last_stop * 100
            entry_ref = f" (entry was ₹{last_entry:.0f})" if last_entry > 0 else ""
            return f"Stop hit — price ₹{price:.0f} is {loss_pct:.1f}% below stop ₹{last_stop:.0f}{entry_ref}"

        # ── 1c. TheWrap EXIT — weekly EMA structure broken ────────────────────
        try:
            tw_code, tw_lbl, *_ = _compute_thewrap(_to_weekly_ohlcv(df_ohlcv))
            if "TW_EXIT" in tw_code:
                return f"TheWrap EXIT — weekly EMAs broken{score_ctx}"
        except Exception:
            pass

        # ── 1d. W-S4 Decline — Weinstein weekly stage broken ─────────────────
        try:
            from screeners.weekly_stage import to_weekly as _tw_resample, get_weekly_stage_weinstein as _wsg
            _wdf = _tw_resample(df_ohlcv)
            _wstage, _wlbl, _wsma, *_ = _wsg(_wdf)
            if _wstage == 4:
                pct = (price - _wsma) / _wsma * 100 if _wsma > 0 else 0.0
                return f"W-S4 Decline — price {abs(pct):.1f}% below 30-week SMA ₹{_wsma:.0f}{score_ctx}"
        except Exception:
            pass

        # ── 2a. Extended past entry — risk/reward gone ────────────────────────
        if last_entry > 0 and price > last_entry:
            ext = (price - last_entry) / last_entry * 100
            if ext > 10:
                return f"Extended {ext:.1f}% past entry ₹{last_entry:.0f} — buy stop no longer valid"
            if ext > 5 and last_state in ("AT_PIVOT", "BREAKOUT"):
                return f"Moved {ext:.1f}% past trigger ₹{last_entry:.0f} — chasing at poor risk/reward"

        # ── 2b. Breakout failed — price fell back through entry ───────────────
        if last_entry > 0 and last_state in ("BREAKOUT", "AT_PIVOT") and price < last_entry:
            loss = (last_entry - price) / last_entry * 100
            return f"Breakout failed — {loss:.1f}% below entry ₹{last_entry:.0f} (stop at ₹{last_stop:.0f})"

        # ── 2c. Base undercut — price >7% below pivot ─────────────────────────
        if "high" in df_ohlcv.columns and len(df_ohlcv) >= 20:
            pivot = float(df_ohlcv["high"].dropna().iloc[-20:].max())
            dist  = (price - pivot) / pivot * 100
            if dist < -12:
                return f"Base failed — {abs(dist):.1f}% below 20-day pivot ₹{pivot:.0f}"
            if dist < -7 and last_state in ("BREAKOUT", "AT_PIVOT"):
                return f"Breakout reversed — {abs(dist):.1f}% below trigger zone ₹{pivot:.0f}"

    # ── 3. Lens membership loss — which screener dropped it ───────────────────
    def _has(df, t):
        if df is None or df.empty or "Ticker" not in df.columns:
            return False
        return t in df["Ticker"].astype(str).values

    in_stage = _has(stage_df, ticker)
    in_sepa  = _has(sepa_df,  ticker)
    in_rs    = _has(rs_df,    ticker)

    had_stage = "Stage" in last_signal
    had_sepa  = "SEPA"  in last_signal
    had_rs    = "RS"    in last_signal

    lost = []
    if had_stage and not in_stage:
        # Try to explain WHY Stage 2 was lost
        if has_data:
            parts = []
            if ema200 > 0 and ema21 < ema200: parts.append(f"EMA21 ₹{ema21:.0f} below EMA200 ₹{ema200:.0f}")
            elif ema21 < ema50:               parts.append(f"EMA21 ₹{ema21:.0f} crossed below EMA50 ₹{ema50:.0f}")
            detail = f" ({'; '.join(parts)})" if parts else ""
            lost.append(f"Stage 2{detail}")
        else:
            lost.append("Stage 2")

    if had_sepa and not in_sepa:
        if has_data and last_entry > 0:
            gap = (price - last_entry) / last_entry * 100
            if   gap < -5:   lost.append(f"SEPA setup ({abs(gap):.1f}% below pivot)")
            elif gap > 8:    lost.append(f"SEPA setup (extended {gap:.1f}% past pivot)")
            else:            lost.append("SEPA setup (setup invalidated)")
        else:
            lost.append("SEPA setup")

    if had_rs and not in_rs:
        rs_score = _flt(saved_row.get("RS Score", 0))
        rs_str = f" (RS score was {rs_score:.0f})" if rs_score else ""
        lost.append(f"RS leadership{rs_str}")

    if lost:
        still = []
        if in_stage: still.append("Stage ✓")
        if in_sepa:  still.append("SEPA ✓")
        if in_rs:    still.append("RS ✓")
        still_str = f" | still holds: {', '.join(still)}" if still else " | all lenses lost"
        return f"{'  +  '.join(lost)} lost{still_str}"

    # ── 4. Score penalty diagnosis — stock still in pools but scored out ───────
    # Reached here = all lens memberships intact, no structural break found.
    # Explain why the score dropped (RSI, TW_FADING, over-extension, sector).
    if has_data:
        penalties = []
        if rsi_now > 82:
            penalties.append(f"RSI {rsi_now:.0f} → ×0.88 penalty")
        elif rsi_now > 75:
            penalties.append(f"RSI {rsi_now:.0f} → ×0.95 penalty")

        dist_ema21 = (price - ema21) / ema21 * 100 if ema21 > 0 else 0.0
        if dist_ema21 > 12:
            penalties.append(f"{dist_ema21:.1f}% above EMA21 (extended — wide stop)")
        elif dist_ema21 > 7 and last_stop > 0:
            stop_gap = (price - last_stop) / price * 100
            if stop_gap > 11:
                penalties.append(f"Stop now {stop_gap:.1f}% away (too wide to size)")

        tw_str = str(saved_row.get("Signal Summary", ""))
        if "TW: Fading" in tw_str:
            penalties.append("TW Fading ×0.82 penalty active")

        if penalties:
            return (f"Score penalties: {' · '.join(penalties)}"
                    f" → score dropped from {last_score:.0f}"
                    f" | was {last_quality}")

    # ── 5. Catch-all with maximum context ─────────────────────────────────────
    state_ctx = f", was {last_state}" if last_state else ""
    quality_ctx = f" ({last_quality})" if last_quality and last_quality not in ("—","") else ""
    return (f"Scored below top {MAX_TRADE_CANDIDATES} threshold"
            f" — score {last_score:.0f}{state_ctx}{quality_ctx}")


def _empty_trade_result() -> pd.DataFrame:
    return pd.DataFrame([{
        "Rank": 1, "Ticker": "—",
        "Company": "No actionable setups found. Review again tomorrow.",
        "Action": "WAIT", "Entry Quality": "—",
        "Entry ₹": "—", "Stop ₹": "—", "Risk %": "—",
        "Trade Score": 0, "RS Score": 0, "SEPA Score": 0, "RSI(14)": "—",
        "Signal Summary": "No candidates passed all filters.",
        "Breakout State": "—", "Regime ⚠": "—",
        "Sector": "—", "Sector Label": "—",
        "TradingView": "",
    }])
