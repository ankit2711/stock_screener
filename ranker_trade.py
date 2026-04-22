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
# TIER A — Trade Now:
#   Active entry signal (BREAKOUT / AT_PIVOT / WEAK_BREAKOUT)
#   Passes hard gates (stop ≤ 9%, pivot dist -8% to +5%)
#   Weekly gate: weekly close must be > weekly EMA200 (W-S2 or W-S3 only)
#     W-S1 Accum / W-S4 Decline → demoted to Tier B (price below weekly EMA)
#   Score boost: +10 pts if weekly EMA200 slope is also rising (W-S2 ✓)
#   Label: "W-Confirmed" (W-S2) | "W-Pending" (W-S3 or Unknown)
#   Ranked by regime-aware unified score
#
# TIER B — Watchlist:
#   RS Leader score ≥ 45 AND Stage 2 AND no active SEPA entry
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

from screeners.stage_analysis import StageAnalysisConfig
from screeners.sepa import SEPAConfig
from ranker_sepa  import run_screens_sepa, get_market_regime, _STAGE_CFG as SEPA_STAGE_CFG, _SEPA_CFG
from ranker_stage import run_screens_stage, DEFAULT_CFG as STAGE_CFG
from ranker_rs    import run_screens_rs

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
    "allowed_states":     {"BREAKOUT", "AT_PIVOT", "WEAK_BREAKOUT"},
    "max_pivot_dist_pct":  5.0,
    "min_pivot_dist_pct": -8.0,
    "max_stop_dist_pct":   9.0,
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

    # ── Step 4: Build Tier A — active entries from SEPA that pass hard gates ──
    tier_a = _build_tier_a(sepa_df, stage_map, rs_map, weights, regime_label, regime_mult)
    logger.info(f"TRADE SCAN: Tier A (Trade Now) = {len(tier_a)} candidates")

    # ── Step 5: Build Tier B — RS Leaders in Stage 2 waiting for FTD ─────────
    tier_a_tickers = {r["_ticker"] for r in tier_a}
    tier_b = _build_tier_b(rs_df, stage_map, sepa_map, ohlcv, weights,
                           regime_label, exclude=tier_a_tickers, regime_mult=regime_mult)
    logger.info(f"TRADE SCAN: Tier B (Watchlist)  = {len(tier_b)} candidates")

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

    logger.info(f"TRADE SCAN ✓ Returning {len(trade_df)} trade candidates")

    return {
        "stage":          stage_df,
        "sepa":           sepa_df,
        "rs":             rs_df,
        "trade":          trade_df,
        "holdings_alert": holdings_alert_df,   # only tab written for exit monitoring
    }


# =============================================================================
# TIER A — ACTIVE ENTRY CANDIDATES (from SEPA)
# =============================================================================

def _build_tier_a(sepa_df: pd.DataFrame, stage_map: dict, rs_map: dict,
                  weights: dict, regime_label: str, regime_mult: float = 1.0) -> list:
    """
    Filter SEPA results to stocks with active entry signals that pass hard gates.
    Score each with regime-aware unified score.
    """
    candidates = []
    g = _GATE

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
        # TW_EXIT / TW_EXIT_40W / TW_CAUTIOUS were already excluded in SEPA ranker.
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
        sepa_norm = min(sepa_raw / 80.0, 1.0)

        s2_pts    = float(stage_row.get("Stage Score S2", row.get("S2 Score", 0)))
        stage_norm = min(s2_pts / 10.0, 1.0)

        rs_pts    = float(rs_row.get("RS Score", 0))
        rs_norm   = rs_pts / 100.0

        state_norm = {"BREAKOUT": 1.0, "AT_PIVOT": 0.90, "WEAK_BREAKOUT": 0.55}.get(state, 0.3)
        stop_norm  = max(0, (9.0 - stop_dist) / 6.0)

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
        if stage_dur < 30:   # < 6 weeks of daily bars = too fresh, high failure rate
            score *= 0.85    # 15% haircut — still tradeable, just discounted

        candidates.append({
            "_ticker":       ticker,
            "_score":        round(score, 1),
            "_tier":         "🟢 Trade Now",
            "_reason":       reason,
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
        })

    return candidates


# =============================================================================
# TIER B — WATCHLIST (RS Leaders in Stage 2 awaiting FTD)
# =============================================================================

def _build_tier_b(rs_df: pd.DataFrame, stage_map: dict, sepa_map: dict,
                  ohlcv: dict, weights: dict, regime_label: str,
                  exclude: set, regime_mult: float = 1.0) -> list:
    """
    RS Leaders that are Stage 2 but have no active SEPA entry signal.
    These are your post-FTD buys — set price alerts at the pivot.
    """
    candidates = []

    for _, row in rs_df.iterrows():
        ticker = str(row.get("Ticker", ""))
        if ticker in exclude:
            continue

        rs_pts    = float(row.get("RS Score", 0))
        stage_str = str(row.get("Stage", ""))

        # Must be RS Leader (score ≥ 45) and in Stage 2 structure
        if rs_pts < 45:
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

        candidates.append({
            "_ticker":       ticker,
            "_score":        round(score, 1),
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
