# =============================================================================
# RANKER — Daily BUY Conviction
# =============================================================================
#
# PURPOSE:
#   "Which stocks are BUY candidates appearing in multiple screener rules?
#    Show consecutive days each stock has been in the list.
#    When it stops appearing, capture the last date it was visible."
#
# WHY THE OLD INTERSECTION APPROACH GAVE ONLY 3 STOCKS:
#   Each screener returns its top-30. With a 1,500-stock universe the expected
#   overlap between any two top-30 lists is 30×30/1,500 ≈ 0.6 stocks by chance.
#   No amount of tuning fixes that — the math is wrong by design.
#
# THE FIX — COMPUTE ON THE FULL UNIVERSE:
#   Run three independent signals on every stock in the universe, just like the
#   backtest does. Show the top-20 by combined conviction score. No intersection
#   of pre-filtered lists. The 3 screeners (Stage/SEPA/RS) still feed their own
#   tabs; this ranker works directly on OHLCV.
#
# THREE SIGNALS (all three are computed for every stock):
#
#   Stage2  — structural trend quality
#     price > EMA200  AND  EMA200 10-day slope > 0  AND  price > EMA50 > EMA21
#     Score: weighted by slope magnitude + EMA stack tightness (0–100)
#
#   RS      — relative strength vs benchmark
#     RS line (close/bench) within 10% of its 52-week RS line high
#     Score: 100 − gap_to_52w_high × 10, floor 0 (0–100)
#
#   SEPA    — near-pivot zone (actionable entry)
#     price within 10% below  OR  within 3% above its 20-bar high
#     Score: 100 × (1 − |dist_pct| / 10), floor 0 (0–100)
#
# CONVICTION = 2+ signals firing simultaneously (score > 0).
# CONVICTION SCORE = weighted average of the signals that fire:
#   Stage2 30% + RS 40% + SEPA 30%
#   3/3 → ×1.10 bonus (capped 100).
#
# HARD EXCLUSIONS (minimal):
#   • Fewer than 100 bars of OHLCV history
#   • Below daily liquidity floor (same as other screeners)
#   • EMA200 slope clearly negative AND price below EMA200 (Stage 3/4)
#
# STREAK:
#   Each daily run increments a per-ticker streak counter if the stock is still
#   in the top-20. Gap ≤ 3 calendar days = consecutive (covers weekends).
#   Exit date is stamped when the streak breaks.
#
# OUTPUT: top-20 stocks, 15 clean columns. Written to "Daily BUY" tab.
# =============================================================================

import logging
import pandas as pd
import numpy as np
from datetime import datetime

from config import (
    MIN_AVG_DOLLAR_VOL_NSE,
    MIN_AVG_DOLLAR_VOL_BSE,
    MIN_AVG_DOLLAR_VOL_US,
    MIN_AVG_DOLLAR_VOL_AI,
)
from persistence import annotate_conviction_df, get_recent_exits, get_data_as_of

logger = logging.getLogger(__name__)

# ── Tuning knobs ───────────────────────────────────────────────────────────────
TOP_N_CONVICTION   = 20     # stocks shown in Daily BUY tab
MIN_SIGNALS        = 2      # stocks must fire at least this many signals
MIN_BARS           = 100    # minimum OHLCV history
RS_HIGH_THRESHOLD  = 0.90   # RS line must be ≥ 90% of its 52-week RS high
SEPA_LOWER_BOUND   = 0.90   # price ≥ 90% of 20-bar high (within 10% below)
SEPA_UPPER_BOUND   = 1.03   # price ≤ 103% of 20-bar high (not too extended)
TRIPLE_BONUS       = 1.10   # ×1.10 score for stocks with 3/3 signals

# Scoring weights (must sum to 1.0)
_W = {"stage2": 0.30, "rs": 0.40, "sepa": 0.30}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_conviction_scan(
    ohlcv:     dict,
    metadata:  dict,
    benchmark: pd.DataFrame,
    market:    str = "india",
    # Optional: screener outputs for rank enrichment (not required for scoring)
    stage_df:  pd.DataFrame = None,
    sepa_df:   pd.DataFrame = None,
    rs_df:     pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Compute conviction signals on the full universe and return top-20 stocks.

    Args:
        ohlcv:     {ticker: OHLCV DataFrame} — full daily history
        metadata:  {ticker: {name, sector, market_cap, ...}}
        benchmark: benchmark OHLCV DataFrame (^NSEI / ^GSPC)
        market:    "india" | "us" | "ai"
        stage_df, sepa_df, rs_df: optional — used only to annotate which
            screeners also flagged the stock (no scoring impact)

    Returns:
        DataFrame of top-20 conviction stocks, sorted by conviction score DESC,
        annotated with streak data. Empty DataFrame if < MIN_SIGNALS stocks pass.
    """
    if benchmark is None or benchmark.empty:
        logger.warning("Conviction: no benchmark data — RS signal disabled")

    bench_close = (
        benchmark["close"].dropna()
        if benchmark is not None and not benchmark.empty
        else pd.Series(dtype=float)
    )

    # Build lookup maps from optional screener DFs (for rank annotation only)
    stage_map = _to_map(stage_df, "Ticker")
    sepa_map  = _to_map(sepa_df,  "Ticker")
    rs_map    = _to_map(rs_df,    "Ticker")

    total   = len(ohlcv)
    candidates = []

    for i, (ticker, df) in enumerate(ohlcv.items(), 1):
        if i % 200 == 0:
            logger.debug(f"Conviction: {i}/{total}")

        if len(df) < MIN_BARS:
            continue

        # ── Liquidity filter ──────────────────────────────────────────────────
        adv = _avg_dollar_vol(df)
        min_adv = (
            MIN_AVG_DOLLAR_VOL_US  if market == "us"
            else MIN_AVG_DOLLAR_VOL_AI if market == "ai"
            else MIN_AVG_DOLLAR_VOL_BSE if ticker.endswith(".BO")
            else MIN_AVG_DOLLAR_VOL_NSE
        )
        if adv < min_adv:
            continue

        try:
            row = _score_stock(ticker, df, bench_close, metadata, market,
                               stage_map, sepa_map, rs_map)
        except Exception as e:
            logger.debug(f"Conviction: {ticker} failed — {e}")
            continue

        if row is None:
            continue

        candidates.append(row)

    logger.info(
        f"Conviction: {len(candidates)} stocks fired 2+ signals "
        f"(universe={total}) → returning top {TOP_N_CONVICTION}"
    )

    if not candidates:
        return pd.DataFrame()

    # Capture ALL tickers that fired ≥ MIN_SIGNALS BEFORE the top-20 cut.
    # Used below to distinguish "ranked out" (still qualifies) from "signals
    # dropped" (no longer fires 2+ signals) in the exit reason column.
    all_qualifier_tickers = {row["Ticker"] for row in candidates}

    df_out = pd.DataFrame(candidates)
    # Sort by Conviction DESC, then ROC 5D % DESC as tiebreaker.
    # On strong market days many stocks score conviction=100; without a tiebreaker
    # head(20) is arbitrary and big-move stocks can be unfairly cut.
    df_out = df_out.sort_values(
        ["Conviction", "ROC 5D %"], ascending=[False, False]
    ).head(TOP_N_CONVICTION)
    df_out = df_out.reset_index(drop=True)

    # Persistence bucket is scoped per market so India / US / AI streaks don't mix.
    _bucket = f"conviction_{market}"   # e.g. "conviction_india", "conviction_us"

    # Annotate with streak (updates persistence.json, adds Streak / First Seen / Days Here / Left On)
    df_out = annotate_conviction_df(df_out, bucket=_bucket, data_as_of=get_data_as_of(benchmark))

    # Re-sort: streak DESC within same conviction band, then conviction DESC
    # Rationale: a stock at streak=7 has been confirmed for 7 days running —
    # more reliable than a new signal at slightly higher score.
    df_out = (
        df_out
        .sort_values(["Streak", "Conviction"], ascending=[False, False])
        .reset_index(drop=True)
    )

    # ── Append exited stocks at the bottom ────────────────────────────────────
    # Show stocks that recently dropped out of the conviction list so the user
    # can see when they exited. Sorted by exit_date DESC (most recent first).
    active_tickers = set(df_out["Ticker"].tolist())
    exits = get_recent_exits(bucket=_bucket, days=30, data_as_of=get_data_as_of(benchmark))
    exited_rows = []
    for e in exits:
        t = e["ticker"].replace(".NS", "").replace(".BO", "")
        if t not in active_tickers:
            # Build TradingView URL from ticker
            if ".NS" in e["ticker"]:
                tv_sym = f"NSE:{t}"
            elif ".BO" in e["ticker"]:
                tv_sym = f"BSE:{t}"
            else:
                tv_sym = t

            # ── Live price lookup ─────────────────────────────────────────────
            # Prefer today's close from ohlcv over the stale persistence price.
            # Try the raw key first, then with common suffixes (.NS, .BO).
            price_str = e["last_price"]   # fallback: last seen price in persistence
            raw_key   = e["ticker"]
            if raw_key not in ohlcv:
                for suffix in (".NS", ".BO", ""):
                    candidate = t + suffix
                    if candidate in ohlcv:
                        raw_key = candidate
                        break
            if raw_key in ohlcv:
                try:
                    live_close = float(ohlcv[raw_key]["close"].dropna().iloc[-1])
                    if live_close > 0:
                        market_sym = "₹" if market in ("india",) else "$"
                        price_str = f"{market_sym}{live_close:,.2f}"
                except Exception:
                    pass   # stick with persistence fallback

            # ── Exit reason ───────────────────────────────────────────────────
            # "Ranked out" = still fires 2+ signals but didn't make the top-20.
            # Otherwise diagnose which signals are still active vs dropped.
            if t in all_qualifier_tickers:
                reason_str = "Ranked out of top 20"
            else:
                reason_str = _conviction_exit_reason(e["ticker"], ohlcv, bench_close)

            exited_rows.append({
                "Ticker":       t,
                "Company":      e["company"],
                "# Signals":    "—",
                "Signals":      "—",
                "Conviction":   "—",
                "Action":       "⚪ Exited",
                "RS Signal":    "—",
                "Price ₹":      price_str,
                "ROC 5D %":     "—",        # no live ROC for exited rows
                "Pivot Dist %": "—",
                "Weekly Stage": "—",
                "Sector":       e["sector"],
                "TradingView":  f"https://www.tradingview.com/chart/?symbol={tv_sym}",
                "Streak":       e["streak"],
                "First Seen":   e["first_seen"],
                "Days Here":    e["days_here"],
                "Left On":      e["exit_date"],
                "Exit Reason":  reason_str,
            })

    # BUG FIX: insert Rank BEFORE appending exited/separator rows so only
    # active stocks get sequential rank numbers.  Exited rows and the separator
    # get "—" to make it visually clear they are historical, not ranked.
    df_out.insert(0, "Rank", range(1, len(df_out) + 1))

    if exited_rows:
        df_exited = pd.DataFrame(exited_rows)
        # Sort exited by exit_date DESC so most recent exits appear first
        df_exited = df_exited.sort_values("Left On", ascending=False).reset_index(drop=True)

        # Visual separator between the live conviction list and the exited section
        sep = {col: "" for col in df_out.columns}
        sep["Ticker"] = "─── Exited — last 30 days ───"
        sep["Rank"]   = "—"
        df_sep = pd.DataFrame([sep])

        # Exited rows get "—" rank, not a sequential number
        if "Rank" not in df_exited.columns:
            df_exited.insert(0, "Rank", "—")
        else:
            df_exited["Rank"] = "—"

        df_out = pd.concat([df_out, df_sep, df_exited], ignore_index=True)

    # Replace any remaining NaN with "" so JSON serialisation (gspread) never fails.
    # NaN can appear in columns that exist on active rows but are absent from
    # exited/separator dicts after pd.concat (e.g. ROC 5D % before fix, or future
    # columns added later).
    df_out = df_out.where(pd.notna(df_out), other="")

    triple = int((df_out["# Signals"] == 3).sum())
    double = int((df_out["# Signals"] == 2).sum())
    logger.info(
        f"Conviction ✓  {len(df_out)} stocks | "
        f"triple={triple} double={double} | "
        f"top: {df_out['Ticker'].head(5).tolist()}"
    )

    return df_out


# =============================================================================
# PER-STOCK SIGNAL ENGINE
# =============================================================================

def _score_stock(
    ticker:    str,
    df:        pd.DataFrame,
    bench_cls: pd.Series,
    metadata:  dict,
    market:    str,
    stage_map: dict,
    sepa_map:  dict,
    rs_map:    dict,
) -> dict | None:
    """
    Compute the 3 conviction signals for one stock.
    Returns a row dict or None if < MIN_SIGNALS fire.
    """
    close = df["close"].dropna()
    if len(close) < MIN_BARS:
        return None

    # ── EMAs ──────────────────────────────────────────────────────────────────
    ema21  = close.ewm(span=21,  adjust=False).mean()
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    price       = float(close.iloc[-1])
    e21         = float(ema21.iloc[-1])
    e50         = float(ema50.iloc[-1])
    e200        = float(ema200.iloc[-1])
    slope_10d   = float(ema200.pct_change(10).iloc[-1]) * 100   # 10-day % change

    # Hard exclusion: confirmed downtrend (Stage 3/4)
    if price < e200 and slope_10d < -0.5:
        return None

    # ── Signal 1: Stage2 ──────────────────────────────────────────────────────
    above_ema200    = price > e200
    slope_positive  = slope_10d > 0
    # Bullish EMA stack: Price > EMA21 > EMA50 (fast above slow).
    # EMA21 (1-month, fast) sits above EMA50 (2.5-month, slow) in an uptrend
    # because recent prices are higher than older ones. Fast-above-slow = bullish.
    ema_stack       = price > e21 > e50

    stage2_on = above_ema200 and slope_positive and ema_stack
    if stage2_on:
        # Score: slope strength (0-50 pts) + stack tightness (0-50 pts)
        slope_pts = min(slope_10d / 0.5 * 25, 50.0)   # full 50 pts at slope ≥ 1%
        vs200_pct = (price - e200) / e200 * 100
        stack_pts = max(0.0, 50.0 - vs200_pct * 2)    # tighter = better (< 25% extension)
        stage2_score = round(min(slope_pts + stack_pts, 100.0), 1)
    else:
        stage2_score = 0.0

    # ── Signal 2: RS vs benchmark ─────────────────────────────────────────────
    rs_score = 0.0
    if len(bench_cls) >= 60:
        aligned = pd.concat(
            [close.rename("c"), bench_cls.rename("b")], axis=1
        ).dropna()
        if len(aligned) >= 60:
            c_a   = aligned["c"]
            b_a   = aligned["b"]
            # Normalise RS line so first bar = 100 (removes level effect)
            rs_ln = (c_a / b_a) / (float(c_a.iloc[0]) / float(b_a.iloc[0])) * 100
            window = min(252, len(rs_ln))
            rs_52w = float(rs_ln.rolling(window, min_periods=60).max().iloc[-1])
            rs_now = float(rs_ln.iloc[-1])
            gap    = rs_now / rs_52w   # 1.0 = at high, 0.90 = 10% below
            if gap >= RS_HIGH_THRESHOLD:
                rs_score = round(min((gap - RS_HIGH_THRESHOLD) / (1.0 - RS_HIGH_THRESHOLD) * 100, 100.0), 1)

    # ── Signal 3: SEPA — near pivot ───────────────────────────────────────────
    high_20     = float(close.iloc[-20:].max()) if len(close) >= 20 else price
    ratio       = price / high_20
    sepa_on     = SEPA_LOWER_BOUND <= ratio <= SEPA_UPPER_BOUND
    if sepa_on:
        dist_pct   = (ratio - 1.0) * 100       # +ve = past pivot, -ve = approaching
        # score peaks at ratio=1.0 (exactly at pivot) and decays toward the bounds
        sepa_score = round(max(0.0, 100.0 - abs(dist_pct) * 10), 1)
    else:
        sepa_score = 0.0
        dist_pct   = (ratio - 1.0) * 100

    # ── Count signals ─────────────────────────────────────────────────────────
    n_signals = int(stage2_on) + int(rs_score > 0) + int(sepa_on)
    if n_signals < MIN_SIGNALS:
        return None

    # ── ROC 5D % — tiebreaker when conviction scores cluster ─────────────────
    # Used to rank stocks when many hit conviction=100 on strong market days.
    roc_5d = 0.0
    if len(close) >= 6:
        try:
            roc_5d = round(
                (float(close.iloc[-1]) - float(close.iloc[-6])) / float(close.iloc[-6]) * 100, 2
            )
        except Exception:
            roc_5d = 0.0

    # ── Weighted conviction score ─────────────────────────────────────────────
    w_sum  = 0.0
    sc_sum = 0.0
    if stage2_on:   sc_sum += stage2_score * _W["stage2"]; w_sum += _W["stage2"]
    if rs_score > 0: sc_sum += rs_score    * _W["rs"];     w_sum += _W["rs"]
    if sepa_on:     sc_sum += sepa_score   * _W["sepa"];   w_sum += _W["sepa"]

    conviction = round(sc_sum / w_sum, 1) if w_sum > 0 else 0.0
    if n_signals == 3:
        conviction = round(min(conviction * TRIPLE_BONUS, 100.0), 1)

    # ── Action label ──────────────────────────────────────────────────────────
    if ratio > 1.0:          action = "🟢 BUY NOW"
    elif ratio >= 0.97:      action = "🔔 BUY STOP"
    elif ratio >= 0.92:      action = "📋 SET ALERT"
    else:                    action = "👁 WATCHLIST"

    # ── Signal label (compact 3-column) ──────────────────────────────────────
    signals_str = " | ".join([
        "Stage ✓" if stage2_on  else "Stage ·",
        "RS ✓"    if rs_score>0 else "RS ·",
        "SEPA ✓"  if sepa_on    else "SEPA ·",
    ])

    # ── RS leading signal (from RS screener output if available) ─────────────
    ticker_display = ticker.replace(".NS", "").replace(".BO", "")
    rs_row      = rs_map.get(ticker_display, {})
    rs_leads    = str(rs_row.get("RS Leads Price", ""))
    rs_at_high  = str(rs_row.get("RS at 52w High", ""))
    if rs_leads == "🌟 Leads":
        rs_signal = "🌟 RS Leads"
    elif rs_at_high == "✓" or rs_score >= 90:
        rs_signal = "✓ RS High"
    else:
        rs_signal = "·"

    # ── Weekly Stage (from SEPA output if available) ──────────────────────────
    sepa_row     = sepa_map.get(ticker_display, {})
    weekly_stage = str(sepa_row.get("Weekly Stage", "—"))

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta    = metadata.get(ticker, {})
    company = meta.get("name", ticker_display)
    sector  = meta.get("sector", "Unknown")

    # ── TradingView URL ───────────────────────────────────────────────────────
    if ".NS" in ticker:
        tv_sym = f"NSE:{ticker_display}"
    elif ".BO" in ticker:
        tv_sym = f"BSE:{ticker_display}"
    else:
        tv_sym = ticker_display
    tv_url = f"https://www.tradingview.com/chart/?symbol={tv_sym}"

    return {
        "Ticker":        ticker_display,
        "Company":       company,
        "# Signals":     n_signals,
        "Signals":       signals_str,
        "Conviction":    conviction,
        "ROC 5D %":      roc_5d,
        "Action":        action,
        "RS Signal":     rs_signal,
        "Price ₹":       f"₹{price:,.2f}" if price and price > 0 else "—",
        "Pivot Dist %":  f"{dist_pct:+.1f}%",
        "Weekly Stage":  weekly_stage,
        "Sector":        sector,
        "TradingView":   tv_url,
    }


# =============================================================================
# HELPERS
# =============================================================================

def _conviction_exit_reason(raw_ticker: str, ohlcv: dict,
                             bench_close: pd.Series) -> str:
    """
    Diagnose why a stock no longer fires 2+ conviction signals.

    Re-runs the three signal checks (Stage2 / RS / SEPA) on current ohlcv
    and returns a compact reason string, e.g.:
      "RS + SEPA lost (1/3 remain)"
      "All signals off"
      "Stage lost (2/3 remain)"
    Falls back to "Signals dropped" if data is unavailable.
    """
    t = raw_ticker.replace(".NS", "").replace(".BO", "")

    # Locate the ohlcv key (try raw, then with suffix variants)
    raw_key = raw_ticker
    if raw_key not in ohlcv:
        for suffix in (".NS", ".BO", ""):
            candidate = t + suffix
            if candidate in ohlcv:
                raw_key = candidate
                break
    if raw_key not in ohlcv:
        return "No data"

    df = ohlcv[raw_key]
    close = df["close"].dropna()
    if len(close) < MIN_BARS:
        return "Insufficient history"

    try:
        ema21  = close.ewm(span=21,  adjust=False).mean()
        ema50  = close.ewm(span=50,  adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        price  = float(close.iloc[-1])
        e21    = float(ema21.iloc[-1])
        e50    = float(ema50.iloc[-1])
        e200   = float(ema200.iloc[-1])
        slope  = float(ema200.pct_change(10).iloc[-1]) * 100

        stage2 = price > e200 and slope > 0 and price > e21 > e50

        rs_on = False
        if len(bench_close) >= 60:
            aligned = pd.concat(
                [close.rename("c"), bench_close.rename("b")], axis=1
            ).dropna()
            if len(aligned) >= 60:
                c_a   = aligned["c"]
                b_a   = aligned["b"]
                rs_ln = (c_a / b_a) / (float(c_a.iloc[0]) / float(b_a.iloc[0])) * 100
                window = min(252, len(rs_ln))
                rs_52w = float(rs_ln.rolling(window, min_periods=60).max().iloc[-1])
                rs_now = float(rs_ln.iloc[-1])
                rs_on  = (rs_now / rs_52w) >= RS_HIGH_THRESHOLD

        high_20 = float(close.iloc[-20:].max()) if len(close) >= 20 else price
        ratio   = price / high_20
        sepa_on = SEPA_LOWER_BOUND <= ratio <= SEPA_UPPER_BOUND

        n_active = int(stage2) + int(rs_on) + int(sepa_on)

        # Should not happen (would be in all_qualifier_tickers), but guard anyway
        if n_active >= MIN_SIGNALS:
            return "Ranked out of top 20"

        lost = [name for name, on in
                [("Stage", stage2), ("RS", rs_on), ("SEPA", sepa_on)] if not on]

        if n_active == 0:
            return "All signals off"
        return f"{' + '.join(lost)} lost ({n_active}/3 remain)"

    except Exception:
        return "Signals dropped"


def _avg_dollar_vol(df: pd.DataFrame, n: int = 20) -> float:
    """Average daily traded value over last n bars."""
    try:
        close  = df["close"].dropna().iloc[-n:]
        volume = df["volume"].dropna().iloc[-n:]
        return float((close * volume).mean())
    except Exception:
        return 0.0


def _to_map(df: pd.DataFrame | None, key_col: str) -> dict:
    """Convert screener DataFrame to {ticker: row_dict} map."""
    if df is None or (hasattr(df, "empty") and df.empty) or key_col not in df.columns:
        return {}
    return {str(row[key_col]).strip(): row.to_dict() for _, row in df.iterrows()}
