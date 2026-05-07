# =============================================================================
# TRADE RANKER — Unified 3-Lens Pipeline
# =============================================================================
#
# PURPOSE:
#   Single entry point for the daily "what do I trade today?" decision.
#   Runs Stage, SEPA, and RS Leaders scans and returns ALL results plus
#   a unified top-15 trade candidate list.
#
# RETURN VALUE:
#   dict with four DataFrames:
#     "stage"  → top-30 Stage-2 stocks (structural trend quality)
#     "sepa"   → top-30 SEPA entry setups (entry quality, RSI timing)
#     "rs"     → top-30 RS Leaders (institutional holding during correction)
#     "trade"  → top-15 unified trade candidates (Tier A + Tier B)
#
# TIER A — Trade Now  (two paths):
#
#   Path 1 — SEPA Entry (original):
#     Active entry signal (BREAKOUT / AT_PIVOT / WEAK_BREAKOUT) from SEPA scan
#     Passes hard gates (stop ≤ 9%, pivot dist -8% to +5%)
#     Weekly gate: weekly close must be > weekly EMA200 (W-S2 or W-S3 only)
#       W-S1 Accum / W-S4 Decline → demoted to Tier B (price below weekly EMA)
#     Score boost: +10 pts if weekly EMA200 slope is also rising (W-S2 ✓)
#     Label: "W-Confirmed" (W-S2) | "W-Pending" (W-S3 or Unknown)
#
#   Path 2 — Stage2 + RS Leader (no SEPA base required):
#     Catches genuine movers that SEPA misses. SEPA routes trending stocks to
#     "C: Trending (No Setup)" with score=2.0 because detect_base() fails when
#     the highest high is very recent — they never make SEPA top-30 and are
#     absent from sepa_df entirely.
#
#     Two sub-paths:
#       Cheat Entry  (Entry Signal = "🟢 Cheat Entry" in Stage output):
#         Running leader pulling back to EMA21. Entry = current price,
#         Stop = EMA21 × 0.97 (3% tight structural stop). No RSI gate.
#         No 20-bar high dist gate. Score bonus: +12 pts.
#       Standard Stage2+RS:
#         Confirmed by Stage + RS, near 20-bar breakout high.
#         Stop = EMA21 × 0.97. RSI hard gate raised to >85 (was >80).
#     Reason label: "Stage2 + RS Leader" / "Stage2 + RS Leader (Cheat)"
#
#   Both paths feed into the same MAX_TIER_A cap, ranked by unified score.
#
# TIER B — Watchlist:
#   RS Leader score ≥ 65 AND Stage 2 AND no active entry signal
#   "Set price alert at pivot" — these are your post-FTD buys
#
# REGIME-AWARE WEIGHTS:
#   In a bull market, SEPA entry quality matters most.
#   In a correction/bear, RS Leadership becomes primary.
#   Weights shift automatically based on benchmark health.
#
#   Regime        SEPA   Stage   RS     State   Stop
#   Bull (5/5)    35%    25%     15%    15%     10%
#   Mild (4/5)    30%    25%     20%    15%     10%
#   Neutral(3/5)  25%    22%     28%    15%     10%
#   Caution(2/5)  18%    20%     37%    15%     10%
#   Bear  (1/5)   12%    15%     48%    15%     10%
# =============================================================================

import logging
import pandas as pd
import numpy as np
from datetime import datetime

from first_seen import annotate_df
from screeners.stage_analysis import StageAnalysisConfig
from screeners.sepa import SEPAConfig, detect_base
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
from persistence       import append_screener_exits

logger = logging.getLogger(__name__)

MAX_TRADE_CANDIDATES = 15
MAX_TIER_A = 8   # max "Trade Now" slots
MAX_TIER_B = 7   # max "Watchlist" slots

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

# Hard gate thresholds — any failure eliminates Tier A candidates
_GATE = {
    "allowed_states":     {"BREAKOUT", "AT_PIVOT", "WEAK_BREAKOUT", "IN_BASE"},
    "max_pivot_dist_pct":  5.0,
    "min_pivot_dist_pct": -8.0,   # IN_BASE stocks >8% below pivot → Tier B only
    "max_stop_dist_pct":  11.0,   # raised from 9% — valid VCP bases can have wider stops
}


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

    # ── Step 1: Market regime ─────────────────────────────────────────────────
    regime_mult, regime_label = get_market_regime(benchmark)
    weights = _get_regime_weights(regime_mult)
    logger.info(f"TRADE SCAN: Regime={regime_label} (×{regime_mult:.2f}) "
                f"→ RS weight={weights['rs']:.0%}, SEPA weight={weights['sepa']:.0%}")

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

    # ── Step 2: Run all 3 scans ───────────────────────────────────────────────
    logger.info("TRADE SCAN: Running Stage scan...")
    stage_df = _safe_run(run_screens_stage,
                         ohlcv=ohlcv, metadata=metadata, benchmark=benchmark,
                         market=market, cfg=STAGE_CFG)

    logger.info("TRADE SCAN: Running SEPA scan...")
    sepa_df = _safe_run(run_screens_sepa,
                        ohlcv=ohlcv, metadata=metadata, benchmark=benchmark,
                        market=market, stage_cfg=SEPA_STAGE_CFG, sepa_cfg=_SEPA_CFG)

    logger.info("TRADE SCAN: Running RS Leaders scan...")
    rs_df = _safe_run(run_screens_rs,
                      ohlcv=ohlcv, metadata=metadata, benchmark=benchmark,
                      market=market)

    logger.info(f"TRADE SCAN: Stage={len(stage_df)}, SEPA={len(sepa_df)}, RS={len(rs_df)}")

    # ── Step 3: Build lookup maps keyed by clean ticker ──────────────────────
    stage_map = _df_to_map(stage_df, "Ticker")
    sepa_map  = _df_to_map(sepa_df,  "Ticker")
    rs_map    = _df_to_map(rs_df,    "Ticker")

    # ── Step 4: Build Tier A — Path 1: SEPA entries ──────────────────────────
    tier_a = _build_tier_a(
        sepa_df, stage_map, rs_map, weights, regime_label, regime_mult,
        sector_results=sector_results, metadata=metadata, market=market,
    )
    logger.info(f"TRADE SCAN: Tier A Path1 (SEPA)         = {len(tier_a)} candidates")

    # ── Step 4b: Tier A — Path 2: Stage2 + RS Leader (no SEPA base needed) ──
    tier_a_tickers = {r["_ticker"] for r in tier_a}
    tier_a_sr = _build_tier_a_stage_rs(
        stage_df, rs_df, sepa_map, ohlcv,
        weights, regime_label, regime_mult,
        exclude=tier_a_tickers,
        sector_results=sector_results, metadata=metadata, market=market,
    )
    tier_a.extend(tier_a_sr)
    logger.info(f"TRADE SCAN: Tier A Path2 (Stage2+RS)    = {len(tier_a_sr)} candidates")
    logger.info(f"TRADE SCAN: Tier A total                = {len(tier_a)} candidates")

    # ── Step 5: Build Tier B — RS Leaders in Stage 2 waiting for FTD ─────────
    tier_a_tickers = {r["_ticker"] for r in tier_a}   # refresh after path 2
    tier_b = _build_tier_b(rs_df, stage_map, sepa_map, ohlcv, weights,
                           regime_label, exclude=tier_a_tickers, regime_mult=regime_mult,
                           sector_results=sector_results, metadata=metadata, market=market)
    logger.info(f"TRADE SCAN: Tier B (Watchlist)          = {len(tier_b)} candidates")

    # ── Step 5b: Tier B supplement — Stage2 fast movers not in RS Leaders ─────
    # Catches stocks that just broke into Stage2 and are running but haven't
    # built enough RS history (13–26 weeks) to qualify for the RS Leaders list.
    # Without this step, these stocks are invisible in ALL candidate paths.
    all_tier_ab_tickers = {r["_ticker"] for r in tier_a} | {r["_ticker"] for r in tier_b}
    tier_b_stage = _build_tier_b_stage(
        stage_df, exclude=all_tier_ab_tickers,
        weights=weights, regime_label=regime_label, regime_mult=regime_mult,
        sector_results=sector_results, metadata=metadata, market=market,
    )
    tier_b.extend(tier_b_stage)
    logger.info(f"TRADE SCAN: Tier B Momentum supplement  = {len(tier_b_stage)} candidates")

    # ── Step 6: Holdings Alert — TheWrap signals for held positions only ────────
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

    # ── Step 7: Combine, cap, rank ────────────────────────────────────────────
    all_candidates = (
        sorted(tier_a, key=lambda r: r["_score"], reverse=True)[:MAX_TIER_A] +
        sorted(tier_b, key=lambda r: r["_score"], reverse=True)[:MAX_TIER_B]
    )
    trade_df = _build_trade_output(all_candidates, market, regime_mult)
    trade_df = annotate_df(trade_df, "trade")

    logger.info(f"TRADE SCAN ✓ Returning {len(trade_df)} trade candidates")

    # ── Step 8: Daily BUY Conviction — 3 signals on full universe ────────────
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
            stage_df  = stage_df,   # optional — for RS Signal / Weekly Stage enrichment
            sepa_df   = sepa_df,
            rs_df     = rs_df,
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

    # ── Step 9: Data Quality — flag tickers with NaN price/volume or stale data ─
    logger.info("TRADE SCAN: Running Data Quality scan...")
    data_quality_df = pd.DataFrame()
    try:
        data_quality_df = run_data_quality_scan(ohlcv, metadata, market=market)
        logger.info(f"TRADE SCAN: Data Issues = {len(data_quality_df)} bad tickers")
    except Exception as _dqe:
        logger.warning(f"TRADE SCAN: Data Quality scan failed (non-fatal): {_dqe}")

    # ── Step 10: Append 14-day exit history to screener tabs ─────────────────
    # Stocks that drop out of each screener are kept at the bottom for 14 days
    # with an "Exit Date" column showing when they left. The tabs are fixed (no
    # daily dated tabs) so this provides rolling history without tab proliferation.
    logger.info("TRADE SCAN: Appending 14-day exit history to screener tabs...")
    try:
        stage_df  = append_screener_exits(stage_df,  bucket=f"stage_{market}")
        sepa_df   = append_screener_exits(sepa_df,   bucket=f"sepa_{market}")
        rs_df     = append_screener_exits(rs_df,     bucket=f"rs_{market}")
        trade_df  = append_screener_exits(trade_df,  bucket=f"trade_{market}")
    except Exception as _ee:
        logger.warning(f"TRADE SCAN: Exit history append failed (non-fatal): {_ee}")

    return {
        "stage":          stage_df,
        "sepa":           sepa_df,
        "rs":             rs_df,
        "trade":          trade_df,
        "holdings_alert": holdings_alert_df,
        "sectors":        sector_results,      # dict[str, SectorResult] — for display + JSON export
        "conviction":     conviction_df,       # Daily BUY — 2+ signal stocks with streak
        "data_quality":   data_quality_df,     # Data Issues — NaN / stale / spike tickers
    }


# =============================================================================
# TIER A — ACTIVE ENTRY CANDIDATES (from SEPA)
# =============================================================================

def _sort_sepa_for_trade(sepa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-prioritize SEPA output for trade use: actionable (near-pivot) stocks first.

    WHY THIS IS NEEDED:
      SEPA sorts by base quality (VCP, RS, vol dry-up). A deep-in-base stock with a
      perfect VCP scores 95 and ranks #1. An AT_PIVOT stock with an average VCP scores
      45 and ranks #25 or falls off the top-30 entirely. The trade ranker's pivot-
      distance gate then rejects the perfect-VCP deep-in-base stock, and the AT_PIVOT
      stock was never seen. Result: Tier A Path 1 produces nothing.

      This sort puts actionable stocks first so the trade ranker sees them regardless
      of where they ranked in the base-quality sort. Within each tier, SEPA Score is
      still the tiebreaker so better setups rank ahead of weaker ones.

    Tiers (ascending sort key → lower = higher priority):
      0  BREAKOUT / AT_PIVOT / WEAK_BREAKOUT within −5% to +5% of pivot — enter today
      1  IN_BASE within −8% to −5% of pivot — near entry, set alert
      2  Everything else (deep in base, extended, trending) — rarely qualify
    """
    if sepa_df.empty:
        return sepa_df

    df   = sepa_df.copy()
    idx  = df.index

    states = df["Breakout State"].astype(str) if "Breakout State" in df.columns else pd.Series([""] * len(df), index=idx)
    pivots = df["Pivot Dist %"].apply(_pct_val)  if "Pivot Dist %"  in df.columns else pd.Series([0.0]  * len(df), index=idx)

    t0 = states.isin({"BREAKOUT", "AT_PIVOT", "WEAK_BREAKOUT"}) & (pivots >= -5.0) & (pivots <= 5.0)
    t1 = states.isin({"IN_BASE"})                                & (pivots >= -8.0) & (pivots <  -5.0)

    tier = pd.Series(2, index=idx)
    tier[t1] = 1
    tier[t0] = 0

    df["_trade_tier"] = tier
    df = (df.sort_values(["_trade_tier", "SEPA Score"], ascending=[True, False])
            .drop(columns=["_trade_tier"])
            .reset_index(drop=True))
    return df


def _build_tier_a(sepa_df: pd.DataFrame, stage_map: dict, rs_map: dict,
                  weights: dict, regime_label: str, regime_mult: float = 1.0,
                  sector_results: dict = None, metadata: dict = None,
                  market: str = "india") -> list:
    """
    Filter SEPA results to stocks with active entry signals that pass hard gates.
    Score each with regime-aware unified score + sector multiplier.
    """
    # Re-sort: actionable (near-pivot) stocks float to the top so the trade ranker
    # sees them even if base-quality scoring pushed them to rank #25–30 in sepa_df.
    sepa_df = _sort_sepa_for_trade(sepa_df)

    candidates = []
    g = _GATE
    sector_results = sector_results or {}
    metadata       = metadata or {}

    for _, row in sepa_df.iterrows():
        state      = str(row.get("Breakout State", ""))
        pivot_dist = _pct_val(row.get("Pivot Dist %", "0%"))
        stop_dist  = _pct_val(row.get("Stop Dist %",  "9%"))
        setup      = str(row.get("Setup", ""))

        # Hard gate: state + distances + not fading
        if state not in g["allowed_states"]:
            continue
        if not (g["min_pivot_dist_pct"] <= pivot_dist <= g["max_pivot_dist_pct"]):
            continue
        if stop_dist > g["max_stop_dist_pct"]:
            continue
        if setup.startswith("🔴"):
            continue

        # Hard gate: weekly price must be ABOVE weekly EMA200
        # Logic: W-S2 (price > rising EMA) and W-S3 (price > flat EMA) both pass.
        # W-S1 (price < EMA, transitioning) and W-S4 (price < falling EMA) fail →
        # these stocks are demoted to Tier B — they have not cleared the weekly level.
        # Unknown (< 45 weekly bars) is allowed through with no penalty.
        weekly_stage_str = str(row.get("Weekly Stage", "Unknown"))
        if weekly_stage_str in ("W-S1 Accum", "W-S4 Decline"):
            continue  # fails weekly price-above-EMA gate → Tier B only

        # Weekly label and score boost
        if weekly_stage_str == "W-S2 ✓":
            weekly_label  = "W-Confirmed"   # price above AND EMA rising → full confirmation
            weekly_boost  = 10.0            # +10 pts for full weekly S2
        elif weekly_stage_str == "W-S3 Dist":
            weekly_label  = "W-S3 Pending"  # price above but EMA flattening → caution
            weekly_boost  = 0.0
        else:
            weekly_label  = "W-Pending"     # Unknown — insufficient history, no penalty
            weekly_boost  = 0.0

        # ── TheWrap gate: TW_FADING → Tier B only (demote, not hard exclude) ─
        # TW_EXIT / TW_EXIT_40W were already excluded in SEPA ranker.
        # TW_CAUTIOUS is now a score penalty in SEPA ranker (not hard excluded).
        # TW_FADING is in SEPA output but not a Tier A buy — aging trend.
        # TW_BULLISH / TW_MAINTAIN → additional weekly_boost
        tw_str = str(row.get("TheWrap", "—"))
        if "TW_FADING" in tw_str or "TW: Fading" in tw_str:
            continue   # demote to Tier B (excluded from Tier A)
        if "TW_BULLISH" in tw_str or "TW: Bullish" in tw_str:
            weekly_boost += 8.0   # extra conviction — short-term + long-term aligned
        elif "TW_MAINTAIN" in tw_str or "TW: Maintain" in tw_str:
            weekly_boost += 5.0   # full bull stack confirmed

        ticker = str(row.get("Ticker", ""))
        stage_row = stage_map.get(ticker, {})
        rs_row    = rs_map.get(ticker, {})

        # Which of the 3 lenses confirmed this stock?
        in_stage = bool(stage_row)
        in_rs    = bool(rs_row)
        if in_stage and in_rs:  reason = "Stage2 + SEPA + RS"
        elif in_stage:          reason = "Stage2 + SEPA"
        elif in_rs:             reason = "SEPA + RS"
        else:                   reason = "SEPA"

        # Component scores (all normalised 0–1)
        sepa_raw  = float(row.get("Raw Score", 0))
        sepa_norm = min(sepa_raw / 100.0, 1.0)   # Fix 5: cap at 100 not 80 (score range is 0-100)

        s2_pts    = float(stage_row.get("Stage Score S2", row.get("S2 Score", 0)))
        stage_norm = min(s2_pts / 10.0, 1.0)

        rs_pts    = float(rs_row.get("RS Score", 0))
        rs_norm   = rs_pts / 100.0

        state_norm = {"BREAKOUT": 1.0, "AT_PIVOT": 0.90, "WEAK_BREAKOUT": 0.55}.get(state, 0.3)
        stop_norm  = min(max(0.0, (9.0 - stop_dist) / 6.0), 1.0)   # capped 0–1

        score = (
            sepa_norm  * weights["sepa"]  +
            stage_norm * weights["stage"] +
            rs_norm    * weights["rs"]    +
            state_norm * weights["state"] +
            stop_norm  * weights["stop"]
        ) * 100 + weekly_boost  # +10 if weekly EMA200 also rising (W-S2 ✓)

        # RSI extension penalty (very extended = reduce score further)
        rsi = float(row.get("RSI(14)", 50))
        if rsi > 82:
            score *= 0.88
        elif rsi > 75:
            score *= 0.95

        # ── Base duration haircut (Fix 4) ─────────────────────────────────────
        # Weinstein requires ≥ 15 weekly bars (~75 daily) of base formation before
        # a valid breakout. We use a soft threshold of 30 daily bars (6 weeks) as
        # the minimum — anything shorter is a temporary pause, not a real base.
        # Penalty: 15% score reduction for bases younger than 6 weeks.
        # Reading from Stage output ("Duration (bars)" = consecutive daily bars
        # where the EMA200 slope stayed in the same direction).
        stage_dur = float(stage_row.get("Duration (bars)", 60)) if stage_row else 60.0
        if stage_dur < 15:   # < 3 weeks — still establishing structure, genuine noise risk
            score *= 0.85    # 15% haircut — still tradeable, just discounted

        # ── Sector rotation multiplier ─────────────────────────────────────────
        # get_sector_for_ticker() looks up the sector of `ticker` in metadata,
        # maps it to the matching SectorResult, and returns it (or None if unknown).
        # Multiplier range: 0.75 (LAGGING) → 1.00 (NEUTRAL) → 1.25 (LEADING).
        sector_res     = get_sector_for_ticker(ticker, metadata, sector_results, market)
        sector_mult_v  = sector_res.sector_mult   if sector_res else 1.0
        sector_label_v = sector_res.sector_label  if sector_res else "NEUTRAL"
        sector_score_v = sector_res.sector_score  if sector_res else 50.0

        # Soft gate: LAGGING sector → demote to Tier B (👁 Watchlist) instead of
        # hard-excluding the stock. The user still sees it — they can override.
        if sector_label_v == "LAGGING":
            tier_label    = "👁 Watchlist"
            demote_reason = f" | Sector LAGGING ({sector_score_v:.0f})"
        else:
            tier_label    = "🟢 Trade Now"
            demote_reason = ""

        score = round(score * sector_mult_v, 1)

        candidates.append({
            "_ticker":       ticker,
            "_score":        score,
            "_tier":         tier_label,
            "_reason":       reason + demote_reason,
            "_state":        state,
            "_sepa_raw":     sepa_raw,
            "_s2_pts":       round(s2_pts, 1),
            "_rs_pts":       round(rs_pts, 1),
            "_rsi":          round(rsi, 0),
            "_stop_dist":    stop_dist,
            "_pivot_dist":   pivot_dist,
            "_regime":       regime_label,
            # Execution prices
            "_price":        float(row.get("Price ₹", 0)),
            "_entry":        float(row.get("Entry ₹",  0)),
            "_stop":         float(row.get("Stop ₹",   0)),
            # Metadata
            "_company":      str(row.get("Company", ticker)),
            "_sector":       str(row.get("Sector",  "Unknown")),
            "_tv":           str(row.get("TradingView", "")),
            "_weekly_stage": weekly_stage_str,
            "_weekly_label": weekly_label,
            "_tw_label":     tw_str,
            "_rs_leading":   str(row.get("RS Leading",   "·")),
            "_setup":        str(row.get("Setup",        "")),
            "_vcp":          row.get("VCP Count", 0),
            "_base_count":   row.get("Base Count", 1),
            "_sepa_score":   float(row.get("SEPA Score", 0)),
            "_path":         str(row.get("Path", "")),
            "_regime_mult":  regime_mult,
            # Sector
            "_sector_label": sector_label_v,
            "_sector_score": round(sector_score_v, 1),
            "_sector_mult":  round(sector_mult_v, 2),
        })

    return candidates


# =============================================================================
# TIER A — PATH 2: STAGE2 + RS LEADER (no SEPA base required)
# =============================================================================

def _build_tier_a_stage_rs(
    stage_df:      pd.DataFrame,
    rs_df:         pd.DataFrame,
    sepa_map:      dict,
    ohlcv:         dict,
    weights:       dict,
    regime_label:  str,
    regime_mult:   float,
    exclude:       set,
    sector_results: dict = None,
    metadata:       dict = None,
    market:         str  = "india",
) -> list:
    """
    Tier A path 2 — catches genuine running leaders that SEPA misses.

    WHY SEPA MISSES RUNNING STOCKS:
      sepa.py routes trending stocks to "C: Trending (No Setup)" with score=2.0
      because detect_base() fails when the highest high is very recent (pivot too
      recent → base_length < min_bars).  They never make the SEPA top-30 and are
      therefore absent from sepa_df entirely — invisible to Tier A Path 1.

    TWO SUB-PATHS:

      Cheat Entry  (🟢 Cheat Entry in stage_df "Entry Signal"):
        Stock is an established Stage 2 leader pulling back to its EMA21.
        This is Minervini's highest-conviction follow-on entry on a proven winner.
        • Entry  = current price (buy the EMA21 test)
        • Stop   = EMA21 × 0.97  (3% below the MA being tested — tight structural)
        • RSI gate is REMOVED — RSI after a pullback is typically 50–70, irrelevant
        • dist_pct gate vs 20-bar high is BYPASSED — the relevant reference is EMA21
        • Score bonus: +12 pts (highest-conviction running-leader entry)

      Standard Stage2+RS  (no cheat entry):
        Stock confirmed by both Stage and RS Leaders screeners, trading near its
        20-bar breakout high with a position-sizeable stop.
        • Stop computed from EMA21 × 0.97 (structural stop, tighter than 15-bar low)
        • RSI hard gate raised to > 85 (was 80); penalty applied at 78–85
        • Covers: at-pivot breakouts and new-high momentum stocks

    SHARED GATES (both sub-paths):
      • In Stage output AND RS output        (two-lens confirmation)
      • Weekly stage not W-S1 or W-S4        (price above weekly EMA)
      • TheWrap not TW_FADING / TW_EXIT*     (structure not compromised)
      • Not already in Tier A via SEPA        (exclude set)
      • Stop ≤ 9%                             (position-sizeable)
    """
    if stage_df.empty or rs_df.empty:
        return []

    sector_results = sector_results or {}
    metadata       = metadata or {}

    rs_map_local = _df_to_map(rs_df, "Ticker")
    candidates   = []

    # Redistribute sepa weight to stage+rs proportionally so weights still sum to 1
    sepa_w   = weights["sepa"]
    sr_total = weights["stage"] + weights["rs"]
    w_stage  = weights["stage"] + sepa_w * (weights["stage"] / sr_total)
    w_rs     = weights["rs"]    + sepa_w * (weights["rs"]    / sr_total)
    w_state  = weights["state"]
    w_stop   = weights["stop"]

    for _, row in stage_df.iterrows():
        ticker = str(row.get("Ticker", ""))

        # ── Gate 1: must not already be a Tier A SEPA candidate ──────────────
        if ticker in exclude:
            continue

        # ── Gate 2: RS confirmation ───────────────────────────────────────────
        # Primary: stock appears in RS Leaders top-30 (two-lens confirmation).
        # Fallback: Stage2 fast mover — RS is RISING in Stage output but the
        #   stock hasn't yet earned RS Leader status (takes 13–26 weeks to build).
        #   These are the "moved fast into Stage 2 and running" stocks.
        #   Accepted only if Stage output shows RS Strong + clean entry + momentum.
        rs_row       = rs_map_local.get(ticker)
        is_rs_leader = rs_row is not None
        rs_pts       = 0.0

        if is_rs_leader:
            rs_pts = float(rs_row.get("RS Score", 0))
        else:
            _rs_status = str(row.get("RS Status",    ""))
            _momentum  = str(row.get("Momentum",     ""))
            _has_rs    = "RS Strong" in _rs_status
            _has_mom   = "↑" in _momentum      # ↑↑ Strong or ↑ Rising
            # _has_entry gate REMOVED: it required "Near Pivot" or "Cheat Entry"
            # which excluded stocks that ARE breaking out (entry = "⚪ Extended"
            # because they just cleared the 4-week high). The pivot-distance gate
            # below (dist_pct check) is the correct risk control — if the stock is
            # too extended past its structural pivot it fails there.
            if not (_has_rs and _has_mom):
                continue   # no RS evidence or no momentum → skip
            # Synthetic RS score: below RS Leader floor (65) — scored conservatively
            rs_pts = 55.0 if "↑↑" in _rs_status else 42.0

        # ── Gate 3: weekly stage — demote if price is below weekly EMA ───────
        weekly_stage_str = str(row.get("Weekly Stage", "Unknown"))
        if weekly_stage_str in ("W-S1 Accum", "W-S4 Decline"):
            continue

        # ── Gate 4: TheWrap exit gates ────────────────────────────────────────
        sepa_row = sepa_map.get(ticker, {})
        tw_str   = str(sepa_row.get("TheWrap", "—"))
        if any(x in tw_str for x in ("TW_FADING", "TW: Fading", "TW_EXIT", "TW: Exit")):
            continue

        # ── Gate 5: OHLCV ─────────────────────────────────────────────────────
        raw_df = ohlcv.get(_restore_ticker(ticker, ohlcv), pd.DataFrame())
        if raw_df.empty or len(raw_df) < 30:
            continue

        close        = float(raw_df["close"].iloc[-1])
        close_series = raw_df["close"]
        rsi          = _quick_rsi(raw_df)

        # EMA21 — structural reference for both cheat entry and stop
        ema21 = float(close_series.ewm(span=21, adjust=False).mean().iloc[-1])

        # ── Detect which sub-path applies ─────────────────────────────────────
        entry_signal = str(row.get("Entry Signal", ""))
        is_cheat     = "Cheat Entry" in entry_signal

        if is_cheat:
            # ══════════════════════════════════════════════════════════════════
            # SUB-PATH A: Cheat Entry — running leader at EMA21 pullback
            # ══════════════════════════════════════════════════════════════════
            # The stage ranker already confirmed: price within 2.5% of EMA21
            # AND volume drying up.  No additional distance gate needed here.
            # EMA21 IS the pivot in this context, not the 20-bar high.
            # ══════════════════════════════════════════════════════════════════
            state    = "AT_PIVOT"
            entry    = round(close * 1.001, 2)       # buy at market / EMA21 test
            stop     = round(ema21 * 0.97, 2)        # 3% below EMA21 — tight structural
            stop_pct = max(0.5, (entry - stop) / entry * 100)
            dist_pct = (close - ema21) / ema21 * 100  # distance from EMA21

            # No RSI gate for cheat entries — pullback naturally resets RSI to 50–70
            # (if RSI is still 85+ during a pullback, it's a shallow one = bullish)

        else:
            # ══════════════════════════════════════════════════════════════════
            # SUB-PATH B: Standard Stage2+RS — at base pivot (detect_base) or
            #             20-bar high fallback when no formal base is detected.
            # Fix 7: use detect_base() so the pivot matches the VCP structure
            # rather than a naive 20-bar rolling max that picks up intra-base
            # noise and pushes the pivot reference too high, causing stocks to
            # look "too extended" when they are actually just above a real base.
            # ══════════════════════════════════════════════════════════════════
            _sepa_cfg   = SEPAConfig()
            base_result = detect_base(
                raw_df["high"], raw_df["low"], raw_df["close"], raw_df["volume"], _sepa_cfg
            )
            if base_result.valid:
                pivot_high = base_result.base_high
            else:
                pivot_high = float(raw_df["high"].iloc[-20:].max())   # fallback

            dist_pct = (close - pivot_high) / pivot_high * 100

            if dist_pct > _GATE["max_pivot_dist_pct"]:    # too extended past pivot
                continue
            if dist_pct < _GATE["min_pivot_dist_pct"]:    # too far below pivot
                continue

            if dist_pct > 0:
                state = "BREAKOUT"
                entry = round(close * 1.001, 2)
            elif dist_pct > -3.0:
                state = "AT_PIVOT"
                entry = round(pivot_high * 1.002, 2)
            else:
                state = "WEAK_BREAKOUT"
                entry = round(pivot_high * 1.002, 2)

            # Stop: EMA21 × 0.97 is a cleaner structural stop than 15-bar swing low
            # (the 15-bar low picks up normal market noise; EMA21 is a deliberate MA)
            stop     = round(ema21 * 0.97, 2)
            stop_pct = (entry - stop) / entry * 100
            if stop_pct > _GATE["max_stop_dist_pct"]:
                stop     = round(entry * 0.92, 2)
                stop_pct = 8.0

            # RSI gate — hard exclude only at extreme overbought (>85)
            # At 78–85: apply score penalty below (not a hard exclude)
            if rsi > 85:
                continue

        # ── Shared stop gate ──────────────────────────────────────────────────
        if stop_pct > _GATE["max_stop_dist_pct"]:
            continue

        # ── Score ─────────────────────────────────────────────────────────────
        s2_pts     = float(row.get("Stage Score S2", 0))
        stage_norm = min(s2_pts / 10.0, 1.0)
        rs_norm    = rs_pts / 100.0
        state_norm = {"BREAKOUT": 1.0, "AT_PIVOT": 0.90, "WEAK_BREAKOUT": 0.55}[state]
        stop_norm  = min(max(0.0, (9.0 - stop_pct) / 6.0), 1.0)

        score = (
            stage_norm * w_stage +
            rs_norm    * w_rs    +
            state_norm * w_state +
            stop_norm  * w_stop
        ) * 100

        # Cheat entry bonus — highest-conviction running-leader signal
        if is_cheat:
            score += 12.0

        # Weekly and TheWrap boosts
        if weekly_stage_str == "W-S2 ✓":
            score += 10.0
        if any(x in tw_str for x in ("TW_BULLISH", "TW: Bullish")):
            score += 8.0
        elif any(x in tw_str for x in ("TW_MAINTAIN", "TW: Maintain")):
            score += 5.0

        # RSI penalties (graduated — not hard excludes for cheat entries)
        if rsi > 85:
            score *= 0.85
        elif rsi > 78:
            score *= 0.93

        # ── RS leading signal ─────────────────────────────────────────────────
        rs_leads_price = str(rs_row.get("RS Leads Price", "")) if rs_row else ""
        rs_at_high     = str(rs_row.get("RS at 52w High", "")) if rs_row else ""
        if rs_leads_price == "🌟 Leads":
            rs_leading = "🌟 RS Leads"
            score     += 6.0
        elif rs_at_high == "✓":
            rs_leading = "✓"
        else:
            rs_leading = "·"

        # ── Reason and setup string ───────────────────────────────────────────
        if is_rs_leader:
            reason    = "Stage2 + RS Leader (Cheat)" if is_cheat else "Stage2 + RS Leader"
            setup_str = entry_signal if is_cheat else f"RS Leader | Stage2 | {entry_signal}"
            path_str  = "Stage2+RS+Cheat" if is_cheat else "Stage2+RS"
        else:
            # Fast-mover path: Stage2 + RS rising but not yet in RS Leaders top-30
            reason    = "Stage2 + Momentum (Cheat)" if is_cheat else "Stage2 + Momentum"
            setup_str = entry_signal if is_cheat else f"Momentum | Stage2 | {entry_signal}"
            path_str  = "Stage2+Momentum"
            score     = min(score, 68.0)   # cap: fresh movers rank below established RS Leaders

        weekly_label = (
            "W-Confirmed" if weekly_stage_str == "W-S2 ✓"
            else "W-S3 Pending" if "W-S3" in weekly_stage_str
            else "W-Pending"
        )

        # ── Sector rotation multiplier ─────────────────────────────────────────
        sector_res     = get_sector_for_ticker(ticker, metadata, sector_results, market)
        sector_mult_v  = sector_res.sector_mult   if sector_res else 1.0
        sector_label_v = sector_res.sector_label  if sector_res else "NEUTRAL"
        sector_score_v = sector_res.sector_score  if sector_res else 50.0

        if sector_label_v == "LAGGING":
            tier_label    = "👁 Watchlist"
            demote_reason = f" | Sector LAGGING ({sector_score_v:.0f})"
        else:
            tier_label    = "🟢 Trade Now"
            demote_reason = ""

        score = round(score * sector_mult_v, 1)

        candidates.append({
            "_ticker":       ticker,
            "_score":        score,
            "_tier":         tier_label,
            "_reason":       reason + demote_reason,
            "_state":        state,
            "_sepa_raw":     0.0,
            "_s2_pts":       round(s2_pts, 1),
            "_rs_pts":       round(rs_pts, 1),
            "_rsi":          round(rsi, 0),
            "_stop_dist":    round(stop_pct, 1),
            "_pivot_dist":   round(dist_pct, 1),
            "_regime":       regime_label,
            "_price":        round(close, 2),
            "_entry":        entry,
            "_stop":         stop,
            "_company":      str(row.get("Company", ticker)),
            "_sector":       str(row.get("Sector", "Unknown")),
            "_tv":           str(row.get("TradingView", "")),
            "_weekly_stage": weekly_stage_str,
            "_weekly_label": weekly_label,
            "_tw_label":     tw_str,
            "_rs_leading":   rs_leading,
            "_setup":        setup_str,
            "_vcp":          0,
            "_base_count":   0,
            "_sepa_score":   0.0,
            "_path":         path_str,
            "_regime_mult":  regime_mult,
            # Sector
            "_sector_label": sector_label_v,
            "_sector_score": round(sector_score_v, 1),
            "_sector_mult":  round(sector_mult_v, 2),
        })

    return candidates


# =============================================================================
# TIER B — WATCHLIST (RS Leaders in Stage 2 awaiting FTD)
# =============================================================================

def _build_tier_b(rs_df: pd.DataFrame, stage_map: dict, sepa_map: dict,
                  ohlcv: dict, weights: dict, regime_label: str,
                  exclude: set, regime_mult: float = 1.0,
                  sector_results: dict = None, metadata: dict = None,
                  market: str = "india") -> list:
    """
    RS Leaders that are Stage 2 but have no active SEPA entry signal.
    These are your post-FTD buys — set price alerts at the pivot.
    """
    sector_results = sector_results or {}
    metadata       = metadata or {}
    candidates     = []

    for _, row in rs_df.iterrows():
        ticker = str(row.get("Ticker", ""))
        if ticker in exclude:
            continue

        rs_pts    = float(row.get("RS Score", 0))
        stage_str = str(row.get("Stage", ""))

        # Must be RS Leader (score ≥ 65) and in Stage 2 structure
        if rs_pts < 65:
            continue
        if "Stage 2" not in stage_str and "Stage 1" not in stage_str:
            continue

        stage_row  = stage_map.get(ticker, {})
        in_stage   = bool(stage_row)
        reason     = "Stage2 + RS" if in_stage else "RS Leader"

        s2_pts     = float(stage_row.get("Stage Score S2", 0))
        stage_norm = min(s2_pts / 10.0, 1.0)
        rs_norm    = rs_pts / 100.0

        # No SEPA entry — sepa component is 0
        score = (
            stage_norm * weights["stage"] +
            rs_norm    * weights["rs"]    +
            0.5        * weights["state"]   # neutral state bonus
        ) * 100

        # RSI from ohlcv if available
        rsi = _quick_rsi(ohlcv.get(_restore_ticker(ticker, ohlcv), pd.DataFrame()))

        # Price and pivot from RS output or stage output
        price = float(stage_row.get("Price ₹", 0)) if "Price ₹" in stage_row else 0.0
        pivot = _estimate_pivot(ohlcv.get(_restore_ticker(ticker, ohlcv), pd.DataFrame()), price)

        # Sector info — applied to score for ranking; no soft gate (already Watchlist)
        sector_res     = get_sector_for_ticker(ticker, metadata, sector_results, market)
        sector_mult_v  = sector_res.sector_mult   if sector_res else 1.0
        sector_label_v = sector_res.sector_label  if sector_res else "NEUTRAL"
        sector_score_v = sector_res.sector_score  if sector_res else 50.0
        score          = round(score * sector_mult_v, 1)

        candidates.append({
            "_ticker":       ticker,
            "_score":        score,
            "_tier":         "👁 Watchlist",
            "_reason":       reason,
            "_state":        "WATCHLIST",
            "_sepa_raw":     0.0,
            "_s2_pts":       round(s2_pts, 1),
            "_rs_pts":       round(rs_pts, 1),
            "_rsi":          round(rsi, 0),
            "_stop_dist":    0.0,
            "_pivot_dist":   0.0,
            "_regime":       regime_label,
            "_price":        round(price, 2),
            "_entry":        round(pivot, 2),
            "_stop":         0.0,
            "_company":      str(row.get("Company", ticker)),
            "_sector":       str(row.get("Sector",  "Unknown")),
            "_tv":           str(row.get("TradingView", "")),
            "_weekly_stage": "—",
            # RS Leads Price: 🌟 = RS at new high while price still in base (highest conviction)
            #                 ✓  = RS at new high with price also near high (confirming)
            "_rs_leading":   (
                "🌟 RS Leads" if row.get("RS Leads Price") == "🌟 Leads"
                else "✓" if row.get("RS at 52w High") == "✓"
                else "·"
            ),
            "_setup":        "📋 RS Leader — Await FTD",
            "_vcp":          0,
            "_base_count":   0,
            "_sepa_score":   0.0,
            "_path":         "RS",
            "_regime_mult":  regime_mult,
            # Sector
            "_sector_label": sector_label_v,
            "_sector_score": round(sector_score_v, 1),
            "_sector_mult":  round(sector_mult_v, 2),
        })

    return candidates


# =============================================================================
# TIER B SUPPLEMENT — STAGE2 MOMENTUM (no RS Leader required)
# =============================================================================

def _build_tier_b_stage(
    stage_df:       pd.DataFrame,
    exclude:        set,
    weights:        dict,
    regime_label:   str,
    regime_mult:    float = 1.0,
    sector_results: dict  = None,
    metadata:       dict  = None,
    market:         str   = "india",
) -> list:
    """
    Watchlist supplement — Stage2 fast movers not yet in RS Leaders top-30.

    WHY THIS IS NEEDED:
      RS Leader status requires 13–26 weeks of outperformance history to build.
      A stock that just burst into Stage 2 won't have that history yet, so it
      scores below the RS Leaders floor and is absent from rs_df entirely.
      Without this path, such stocks are invisible in ALL trade candidate paths.

    GATES:
      • Stage 2 confirmed (from stage_df)
      • RS Status = "RS Strong ↑↑" or "RS Strong ↑" (rising even if not Leader)
      • Momentum = "↑↑ Strong" or "↑ Rising"  (the stock is actually moving)
      • Vol Conviction ≠ "Low"                 (not a quiet drift)
      • Weekly stage not W-S1 / W-S4           (price above weekly EMA)
      • Not already in Tier A or Tier B (exclude set)

    SCORE:
      Stage S2 quality 60% + momentum 25% + volume 15% → max ~65.
      Stays below Tier A Path 2 RS Leaders to preserve ranking hierarchy.

    OUTPUT TIER: 👁 Watchlist — "set pivot alert" or monitor for entry.
    """
    if stage_df.empty:
        return []

    sector_results = sector_results or {}
    metadata       = metadata or {}
    candidates     = []

    for _, row in stage_df.iterrows():
        ticker = str(row.get("Ticker", ""))
        if ticker in exclude:
            continue

        # Must show rising RS strength in Stage output
        rs_status = str(row.get("RS Status", ""))
        if "RS Strong" not in rs_status:
            continue

        # Must show upward momentum
        momentum = str(row.get("Momentum", ""))
        if "↑" not in momentum:
            continue

        # Require at least Normal volume conviction
        vol_conv = str(row.get("Vol Conviction", ""))
        if vol_conv == "Low":
            continue

        # Weekly gate: price must be above weekly EMA (not basing below it)
        weekly_stage_str = str(row.get("Weekly Stage", "Unknown"))
        if weekly_stage_str in ("W-S1 Accum", "W-S4 Decline"):
            continue

        # Score: stage quality + momentum + volume (no SEPA/RS Leader component)
        s2_pts     = float(row.get("Stage Score S2", 0))
        stage_norm = min(s2_pts / 10.0, 1.0)
        mom_norm   = 1.0 if "↑↑" in momentum  else 0.6
        vol_norm   = 1.0 if "Very High" in vol_conv else (0.7 if "High" in vol_conv else 0.4)

        score = (stage_norm * 0.60 + mom_norm * 0.25 + vol_norm * 0.15) * 60.0
        if "↑↑" in rs_status:
            score += 5.0    # RS Strong ↑↑ gets a small boost over RS Strong ↑

        # Sector multiplier
        sector_res     = get_sector_for_ticker(ticker, metadata, sector_results, market)
        sector_mult_v  = sector_res.sector_mult   if sector_res else 1.0
        sector_label_v = sector_res.sector_label  if sector_res else "NEUTRAL"
        sector_score_v = sector_res.sector_score  if sector_res else 50.0
        score          = round(score * sector_mult_v, 1)

        entry_signal = str(row.get("Entry Signal", ""))
        candidates.append({
            "_ticker":       ticker,
            "_score":        score,
            "_tier":         "👁 Watchlist",
            "_reason":       "Stage2 + Momentum",
            "_state":        "WATCHLIST",
            "_sepa_raw":     0.0,
            "_s2_pts":       round(s2_pts, 1),
            "_rs_pts":       0.0,
            "_rsi":          50.0,
            "_stop_dist":    0.0,
            "_pivot_dist":   0.0,
            "_regime":       regime_label,
            "_price":        0.0,
            "_entry":        0.0,
            "_stop":         0.0,
            "_company":      str(row.get("Company", ticker)),
            "_sector":       str(row.get("Sector",  "Unknown")),
            "_tv":           str(row.get("TradingView", "")),
            "_weekly_stage": weekly_stage_str,
            "_weekly_label": "W-Confirmed" if weekly_stage_str == "W-S2 ✓" else "W-Pending",
            "_tw_label":     "—",
            "_rs_leading":   "·",
            "_setup":        f"📋 Stage2 Momentum — {entry_signal}",
            "_vcp":          0,
            "_base_count":   0,
            "_sepa_score":   0.0,
            "_path":         "Stage2+Momentum",
            "_regime_mult":  regime_mult,
            "_sector_label": sector_label_v,
            "_sector_score": round(sector_score_v, 1),
            "_sector_mult":  round(sector_mult_v, 2),
        })

    return candidates


# =============================================================================
# TRADE OUTPUT BUILDER
# =============================================================================

def _build_trade_output(candidates: list, market: str, regime_mult: float = 1.0) -> pd.DataFrame:
    """Convert internal candidate dicts to the clean trade output DataFrame."""
    if not candidates:
        return _empty_trade_result()

    rows = []
    for c in candidates:
        entry    = c["_entry"]
        stop     = c["_stop"]
        price    = c["_price"]
        state    = c["_state"]
        tier     = c["_tier"]

        # Risk % from entry to stop
        if entry > 0 and stop > 0 and entry > stop:
            risk_pct = (entry - stop) / entry * 100
        else:
            risk_pct = c["_stop_dist"] if c["_stop_dist"] > 0 else 7.0

        # Position size: 1% risk rule, capped at 8%, then scaled by market regime
        # Raw:  1% portfolio risk / stop% = position size (e.g. 5% stop → 20% raw → capped 8%)
        # Then: multiplied by regime factor so the displayed number is already market-adjusted.
        #   Regime     Factor   Example (8% raw)
        #   Bull        1.00    8.0%  — full allocation
        #   Mild Bull   0.75    6.0%  — slightly reduced
        #   Neutral     0.50    4.0%  — half size (this is the "50% size" that confused users)
        #   Caution     0.25    2.0%  — token position / monitoring only
        #   Bear        0.00    0.0%  — paper trade only
        raw_pos = min(1.0 / (risk_pct / 100), 0.08) * 100 if risk_pct > 0 else 5.0
        c_regime_mult = c.get("_regime_mult", regime_mult)
        if c_regime_mult >= 0.85:   regime_factor = 1.00
        elif c_regime_mult >= 0.60: regime_factor = 0.75
        elif c_regime_mult >= 0.40: regime_factor = 0.50
        elif c_regime_mult >= 0.22: regime_factor = 0.25
        else:                       regime_factor = 0.00
        pos_size = round(raw_pos * regime_factor, 1)

        # Action label
        rsi = c["_rsi"]
        if tier == "👁 Watchlist":
            action = f"📋 ALERT → ₹{entry:,.0f}" if entry > 0 else "📋 ALERT — set pivot alert"
        elif state == "BREAKOUT":
            if rsi > 82:
                action = "⚠ EXTENDED (RSI high) — half size"
            elif c["_pivot_dist"] <= 3.0:
                action = "🟢 BUY NOW"
            else:
                action = "🟡 BUY — confirm vol"
        elif state == "AT_PIVOT":
            action = "🔔 BUY STOP order"
        elif state == "IN_BASE":
            action = "📋 SET ALERT — buy stop at pivot"
        elif state == "WEAK_BREAKOUT":
            action = "🟡 CONFIRM VOL — watch"
        else:
            action = "📋 ALERT only"

        # Signal summary
        parts = []
        rs_lead = c.get("_rs_leading", "·")
        if rs_lead == "🌟 RS Leads":   parts.append("🌟 RS Leads Price")   # pre-breakout divergence
        elif rs_lead == "✓":           parts.append("RS Leading ✓")
        weekly_lbl = c.get("_weekly_label", "")
        if weekly_lbl == "W-Confirmed":   parts.append("W-Confirmed ✓")
        elif weekly_lbl == "W-S3 Pending": parts.append("W-S3 (EMA flat)")
        elif weekly_lbl == "W-Pending":    parts.append("W-Pending")
        # TheWrap signal in summary
        tw_lbl = c.get("_tw_label", "—")
        if "BULLISH" in tw_lbl or "Bullish" in tw_lbl:
            parts.append("TW: Bullish ✓")
        elif "MAINTAIN" in tw_lbl or "Maintain" in tw_lbl:
            parts.append("TW: Maintain ✓")
        vcp = c.get("_vcp", 0)
        if isinstance(vcp, (int, float)) and vcp >= 2:
            parts.append(f"VCP {int(vcp)}×")
        if c["_base_count"] == 1:      parts.append("1st base")
        setup_short = c["_setup"].split("—")[0].strip()
        if setup_short:                parts.append(setup_short)
        signal = " | ".join(parts) if parts else c["_setup"]

        rows.append({
            "Tier":          tier,
            "Reason":        c.get("_reason", "—"),
            "Ticker":        c["_ticker"],
            "Company":       c["_company"],
            "Action":        action,
            "Entry ₹":       round(entry, 2) if entry > 0 else "—",
            "Stop ₹":        round(stop,  2) if stop  > 0 else "—",
            "Risk %":        round(risk_pct, 1) if stop > 0 else "—",
            "Pos Size %":    round(pos_size, 1) if stop > 0 else "—",
            "Trade Score":   c["_score"],
            "Stage S2":      c["_s2_pts"],
            "RS Score":      c["_rs_pts"],
            "SEPA Score":    c["_sepa_score"],
            "RSI(14)":       c["_rsi"],
            "Signal Summary": signal,
            "Breakout State": state,
            "Regime ⚠":      _regime_warning(c["_regime"]),
            "Sector":        c["_sector"],
            "Sector Label":  c.get("_sector_label", "NEUTRAL"),
            "Sector ×":      c.get("_sector_mult",  1.0),
            "TradingView":   c["_tv"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["Tier", "Trade Score"],
        ascending=[True, False],   # Tier A before Tier B (🟢 < 👁 alphabetically)
        key=lambda col: col if col.name != "Tier" else col.map({"🟢 Trade Now": 0, "👁 Watchlist": 1})
    ).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


# =============================================================================
# HELPERS
# =============================================================================

def _get_regime_weights(regime_mult: float) -> dict:
    """Map regime_mult to the appropriate weight set."""
    if regime_mult >= 0.85:   return _REGIME_WEIGHTS["bull"]
    if regime_mult >= 0.60:   return _REGIME_WEIGHTS["mild_bull"]
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


def _estimate_pivot(df: pd.DataFrame, price: float) -> float:
    """Estimate the nearest resistance pivot from recent 20-bar high."""
    if df.empty or "high" not in df.columns:
        return price * 1.005
    recent_high = float(df["high"].iloc[-20:].max()) if len(df) >= 20 else price
    return round(recent_high * 1.005, 2)


def _pct_val(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace("%", "").replace("+", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def _empty_trade_result() -> pd.DataFrame:
    return pd.DataFrame([{
        "Rank": 1, "Tier": "—", "Reason": "—", "Ticker": "—",
        "Company": "No actionable setups found. Review again tomorrow.",
        "Action": "WAIT", "Entry ₹": "—", "Stop ₹": "—",
        "Risk %": "—", "Pos Size %": "—", "Trade Score": 0,
        "Stage S2": 0, "RS Score": 0, "SEPA Score": 0, "RSI(14)": "—",
        "Signal Summary": "No candidates passed all filters.",
        "Breakout State": "—", "Regime ⚠": "—", "Sector": "—", "TradingView": "",
    }])
