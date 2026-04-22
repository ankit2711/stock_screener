# =============================================================================
# RANKER — Stage Analysis edition  (rewritten)
# =============================================================================
#
# What changed vs the original:
#   1. Hard filter: only Stage-2 stocks (or confirmed cheat-entry) reach the
#      output. S1 / S3 / S4 are discarded before ranking.
#   2. Entry signal is now a scored component, not just a bonus flag.
#      Every Stage-2 stock is evaluated for how close it is to a valid
#      low-risk entry (cheat entry, pullback to 21 EMA, near 4-week high).
#   3. Composite score weights reflect: trend quality + RS + entry proximity
#      + momentum + volume. Equal emphasis on trend AND opportunity.
#
# Composite score breakdown (all components 0-1):
#   Stage-2 quality   45%   s2_score / 10 (max possible Stage-2 score)
#   RS strength       20%   strong (above RS-MA): 20%  (flat — rs_rising not a separate component)
#   Momentum          25%   Strong↑↑: 25% | Rising↑: 12%
#   Entry signal       6%   cheat entry: 6% | EMA pullback: ~4% | near 4wk high: 2.4%
#   Volume conviction  4%   Very High: 4% | High: 2%
#
# Rationale for weights:
#   The Stage screener is a WATCHLIST screener (find the best Stage-2 trends),
#   not an execution screener (that's the SEPA/Trade screener's job).
#   Entry proximity was over-weighted at 25%: a strong trending stock that is
#   "Extended" from its pivot got 0.00 entry bonus, penalising it equally to a
#   stock with mediocre trend but currently near a pivot. Strong Stage-2 stocks
#   with high RS and accelerating momentum belong on the watchlist regardless of
#   whether they're offering an immediate entry point.
# =============================================================================

import logging
import pandas as pd
import numpy as np
from datetime import datetime

from screeners.stage_analysis import (
    StageAnalysisConfig, StageAnalysisResult, run_stage_analysis
)
from screeners.weekly_stage import to_weekly, get_weekly_stage_weinstein
from screeners.exit_analysis import analyse_exit, slope_label
from config import (
    TOP_N_US, TOP_N_INDIA, TOP_N_AI, RS_RATING,
    MIN_AVG_DOLLAR_VOL_NSE, MIN_AVG_DOLLAR_VOL_BSE,
    MIN_AVG_DOLLAR_VOL_US, MIN_AVG_DOLLAR_VOL_AI,
)

logger = logging.getLogger(__name__)


# =============================================================================
# WEEKLY STAGE ANALYSIS — canonical implementation lives in screeners/weekly_stage.py
# =============================================================================
#
# Imports: to_weekly, get_weekly_stage_weinstein
#   Both are now shared across ranker_stage, ranker_sepa, and ranker_trade
#   so that all three sheets classify weekly stages identically.
#
# Weinstein's method (1988 + 2018-2024 seminars):
#   Weekly bars, 30-week SMA (not EMA40), slope measured over 4 weekly bars.
#   Volume dry-up: right 13w avg < prior 13w avg × 0.85 (sellers exhausted).
#   Breakout volume: latest week ≥ 90% of 13-week volume high (institutional).
#
# USING BOTH TIMEFRAMES:
#   Weekly  → admission gate (hard): is the primary trend healthy?
#   Daily   → entry timing (soft):   is there an actionable setup right now?
# =============================================================================
# (to_weekly, get_weekly_stage_weinstein imported at top of file)


DEFAULT_CFG = StageAnalysisConfig(
    sensitivity      = "Aggressive",
    ma_length        = 200,   # 200-day EMA = 40-week EMA — Weinstein structural MA.
                              # MUST match ranker_sepa.py. Both stage sheets must use
                              # the same MA so S2/S4 classifications are consistent.
                              # DO NOT use 30 — a 30-day SMA is only 6 weeks and
                              # fires on short-term bounces, producing false Stage 2.
    slope_lookback   = 10,    # 10-day slope on EMA200 = 2-week direction signal
    ema_fast         = 10,
    ema_medium       = 21,
    ema_slow         = 50,
    rs_ma_length     = 52,
    vol_avg_len      = 50,
    mom_fast         = 10,
    mom_slow         = 20,
    beta_length      = 52,
    pead_threshold   = 10.0,
    pead_window      = 5,
)


def run_screens_stage(
    ohlcv:     dict,
    metadata:  dict,
    benchmark: pd.DataFrame,
    market:    str = "india",
    cfg:       StageAnalysisConfig = None,
) -> pd.DataFrame:
    """
    Run stage analysis → filter Stage 2 only → rank by trend + entry quality.

    Returns top-N DataFrame sorted by composite Score descending.
    """
    if cfg is None:
        cfg = DEFAULT_CFG

    top_n  = TOP_N_AI if market == "ai" else TOP_N_US if market == "us" else TOP_N_INDIA
    rows   = []
    total  = len(ohlcv)
    s2_count = 0

    logger.info(f"Stage screener: analysing {total} {market.upper()} tickers...")

    for i, (ticker, df) in enumerate(ohlcv.items(), 1):
        if i % 100 == 0:
            logger.info(f"  {i}/{total} | Stage-2 so far: {s2_count}")

        if len(df) < 60:
            continue

        meta = metadata.get(ticker, {})

        try:
            # ══════════════════════════════════════════════════════════════════
            # STEP A: WEEKLY STAGE — Weinstein primary classification
            # Run on weekly bars first. This is the authoritative trend view.
            # Daily analysis is secondary (entry timing).
            # ══════════════════════════════════════════════════════════════════
            weekly_df = to_weekly(df)
            w_stage, w_label, w_sma30, w_vol_dry, w_breakout_vol = \
                get_weekly_stage_weinstein(weekly_df)

            # Hard gate: weekly Stage 3 (distribution) or Stage 4 (decline)
            # means the primary trend is deteriorating or already broken.
            # No entry makes sense here regardless of how the daily looks.
            if w_stage in (3, 4):
                continue

            # ══════════════════════════════════════════════════════════════════
            # STEP B: DAILY STAGE — entry timing + detailed scoring
            # ══════════════════════════════════════════════════════════════════
            result = run_stage_analysis(
                df           = df,
                benchmark_df = benchmark,
                ticker       = ticker,
                market       = market,
                cfg          = cfg,
            )
            result.market_cap = meta.get("market_cap", 0)

            # ── Liquidity filter: exchange-aware minimum avg dollar volume ──
            # BSE-only stocks (.BO) have wider spreads — require higher threshold.
            # US stocks use their own USD threshold.
            min_adv = (
                MIN_AVG_DOLLAR_VOL_AI  if market == "ai"
                else MIN_AVG_DOLLAR_VOL_US  if market == "us"
                else MIN_AVG_DOLLAR_VOL_BSE if ticker.endswith(".BO")
                else MIN_AVG_DOLLAR_VOL_NSE
            )
            if result.avg_dollar_vol < min_adv:
                continue

            # Attach weekly data to result for scoring + output
            result.weekly_stage       = w_stage
            result.weekly_stage_label = w_label
            result.weekly_sma30       = round(w_sma30, 2)
            result.weekly_vol_dry     = w_vol_dry

            # ── Hard filter: Stage 2 only (cheat entry is always Stage 2) ──
            # Note: weekly gate already eliminated S3/S4 weekly stocks.
            # Daily stage must still confirm S2 (or cheat entry in S2).
            if result.stage != 2 and not result.is_cheat_entry:
                continue
            s2_count += 1

            # ── Entry signal quality (needs raw OHLCV) ──────────────────────
            entry_score, entry_label = _entry_signal(df, result)

            # Pass weekly breakout vol flag into entry scoring
            if w_breakout_vol and entry_label == "⚪ Extended":
                # Heavy weekly volume on an extended stock = institutional breakout
                # downgrade "Extended" signal rather than zero it completely
                entry_score = max(entry_score, 0.20)
                entry_label = "🔵 Vol Breakout (extended)"

            # ── Composite score ─────────────────────────────────────────────
            score = _composite_score(result, entry_score)

            row = _result_to_row(result, meta, ticker, entry_score, entry_label,
                                 score, w_breakout_vol)
            rows.append(row)

        except Exception as e:
            logger.debug(f"Stage analysis failed for {ticker}: {e}")

    logger.info(f"Stage screener: {s2_count} Stage-2 stocks → top {top_n} returned")

    if not rows:
        logger.warning("Stage screener: no Stage-2 results")
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values("Score", ascending=False).reset_index(drop=True)
    df_out.insert(0, "Rank", range(1, len(df_out) + 1))
    return df_out.head(top_n)


# =============================================================================
# EXIT MONITOR — TheWrap signals for HELD positions only
# =============================================================================

def run_exit_monitor(
    ohlcv:    dict,
    metadata: dict,
    benchmark: pd.DataFrame,      # kept for API compat, not used
    market:   str = "india",
    cfg:      StageAnalysisConfig = None,
    holdings: dict = None,
) -> pd.DataFrame:
    """
    Scan ONLY held positions (from Om-Holdings) for TheWrap exit signals.

    Does NOT scan the full universe — only positions the user actually holds.
    This makes each run fast (43 India + 12 US positions vs 1,500+ tickers).

    TheWrap signal codes (urgency order):
      TW_EXIT     (90)  — below all weekly EMAs → EXIT FULL POSITION
      TW_EXIT_40W (75)  — below 20W + 40W rolling over → REDUCE 50-75%
      TW_CAUTIOUS (55)  — below 10W/20W EMA → TIGHTEN STOP
      TW_FADING   (40)  — bull stack but 40W flattening → RAISE STOP
      TW_WAIT     (20)  — testing 20W support → WATCH
      TW_BULLISH  ( 8)  — above 10W, rising → HOLD
      TW_MAINTAIN ( 5)  — full bull stack + 40W rising → HOLD (ideal)

    Returns:
        pd.DataFrame — one row per held position, sorted by urgency (highest first).
        Returns empty DataFrame if no holdings or no OHLCV data found.
    """
    if not holdings:
        logger.info("Exit Monitor: no holdings loaded — skipping")
        return pd.DataFrame()

    # Currency symbol for EMA display
    ccy = "₹" if market == "india" else "$"
    # Decimal places: INR rounds to integer, USD shows 2dp
    ema_fmt = lambda v: f"{ccy}{v:.0f}" if market == "india" else f"{ccy}{v:.2f}"

    rows = []
    logger.info(
        f"Exit Monitor: scanning {len(holdings)} {market.upper()} holdings "
        f"(TheWrap — weekly 10W/20W/40W EMA)"
    )

    for clean_ticker, held_info in holdings.items():
        # ── Find OHLCV data for this holding ─────────────────────────────────
        df = None
        found_key = None
        suffixes = [".NS", ".BO"] if market == "india" else [""]
        for sfx in suffixes:
            key = clean_ticker + sfx
            if key in ohlcv:
                df        = ohlcv[key]
                found_key = key
                break
        if df is None and clean_ticker in ohlcv:     # US — plain ticker
            df        = ohlcv[clean_ticker]
            found_key = clean_ticker

        if df is None:
            logger.debug(f"Exit Monitor: no OHLCV for holding '{clean_ticker}' — skipped")
            continue

        if len(df) < 60:
            continue

        try:
            # ── Weekly data + Weinstein stage (context only) ──────────────────
            weekly_df = to_weekly(df)
            _, w_label, _, _, _ = get_weekly_stage_weinstein(weekly_df)

            # ── TheWrap exit analysis ──────────────────────────────────────────
            er = analyse_exit(weekly_df=weekly_df, holdings_info=held_info)

            # ── Metadata ───────────────────────────────────────────────────────
            meta         = metadata.get(found_key, {})
            company_name = (held_info.get("name", "")
                            or meta.get("name", "")
                            or clean_ticker)
            if company_name.endswith(".NS") or company_name.endswith(".BO"):
                company_name = clean_ticker

            # TradingView link
            if market == "india":
                tv_sym = (f"NSE:{clean_ticker}" if found_key.endswith(".NS")
                          else f"BSE:{clean_ticker}")
            else:
                tv_sym = clean_ticker

            gain_str = f"{er.gain_pct:+.0f} %" if er.has_buy_price else "—"

            rows.append({
                "Ticker":      clean_ticker,
                "Company":     company_name,
                "Portfolio":   held_info.get("portfolio", ""),
                "Gain %":      gain_str,
                "Action":      er.action,
                "Urgency":     er.urgency_score,
                "TheWrap":     er.signal_label,
                "Signal Code": er.signal_code,
                # EMA values — concrete stop/support reference levels
                "10W EMA":     ema_fmt(er.ema10w)  if er.ema10w > 0 else "—",
                "20W EMA":     ema_fmt(er.ema20w)  if er.ema20w > 0 else "—",
                "40W EMA":     ema_fmt(er.ema40w)  if er.ema40w > 0 else "—",
                # % distances (positive = above EMA, negative = below)
                "vs 10W %":    f"{er.dist_10w_pct:+.1f} %" if er.ema10w > 0 else "—",
                "vs 20W %":    f"{er.dist_20w_pct:+.1f} %" if er.ema20w > 0 else "—",
                "vs 40W %":    f"{er.dist_40w_pct:+.1f} %" if er.ema40w > 0 else "—",
                # 40W slope velocity — rising/flat/falling
                "40W Slope":   slope_label(er.ema40w_slope),
                "Weekly Stage": w_label,    # Weinstein 30w SMA for context
                "TradingView": f"https://www.tradingview.com/chart/?symbol={tv_sym}",
                "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

        except Exception as e:
            logger.debug(f"Exit Monitor: analysis failed for '{clean_ticker}': {e}")

    if not rows:
        logger.info("Exit Monitor: no data found for any holding")
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    # Sort by urgency descending — most urgent action at the top
    df_out = df_out.sort_values("Urgency", ascending=False).reset_index(drop=True)
    df_out.insert(0, "Rank", range(1, len(df_out) + 1))

    logger.info(
        f"Exit Monitor ✓  {len(df_out)} holdings analysed "
        f"| most urgent: {df_out.iloc[0]['Signal Code']} ({df_out.iloc[0]['Ticker']})"
    )
    return df_out


# =============================================================================
# ENTRY SIGNAL SCORER
# =============================================================================

def _entry_signal(df: pd.DataFrame, result: StageAnalysisResult):
    """
    Score the entry quality of a Stage-2 stock.

    Priority order (only the best signal is used):
      1. Cheat entry (VCP-style pullback to 21 EMA)          → score 1.00
      2. EMA pullback (price within 5% of 21 EMA, vol quiet) → score 0.72
      3. Near 4-week high (within 3% of 20-bar high)         → score 0.40
      4. Extended — no near-term signal                       → score 0.00

    Returns (entry_score 0–1, entry_label str)
    """
    close  = df["close"]
    high   = df["high"]
    volume = df["volume"]

    price   = float(close.iloc[-1])
    ema21   = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    avg_vol = float(volume.rolling(50).mean().iloc[-1]) if len(volume) >= 50 else float(volume.mean())
    vol_10  = float(volume.iloc[-10:].mean())
    vol_dry = vol_10 < avg_vol * 0.75   # volume 25% below baseline = quiet

    dist_from_ema21 = abs(price - ema21) / ema21 * 100

    # 1. Cheat entry (already detected by stage analysis)
    if result.is_cheat_entry:
        return 1.00, "🟢 Cheat Entry"

    # 2. EMA pullback: within 5% of 21 EMA with quiet volume
    if dist_from_ema21 <= 5.0 and vol_dry:
        # Score scales with tightness: 4% away = 0.65, 0% away = 0.72
        closeness = max(0, (5.0 - dist_from_ema21) / 5.0) * 0.12 + 0.60
        return round(closeness, 3), "🟡 EMA Pullback"

    # 3. Near 4-week high (potential breakout setup)
    high_4w = float(high.iloc[-21:-1].max()) if len(high) > 21 else float(high.max())
    dist_from_4wh = (price - high_4w) / high_4w * 100   # negative = below, positive = above

    if -3.0 <= dist_from_4wh <= 1.0:
        return 0.40, "🔵 Near Pivot"

    # 4. Extended / no signal
    return 0.00, "⚪ Extended"


# =============================================================================
# COMPOSITE SCORE
# =============================================================================

def _composite_score(result: StageAnalysisResult, entry_score: float) -> float:
    """
    Combined score for ranking Stage-2 stocks.

    Components (sum = 100%):
        Stage-2 quality   40%   s2_score / 10 (max possible Stage-2 score)
        RS strength       18%   Mansfield RS > 0 (outperforming benchmark)
        Momentum          22%   Strong↑↑: 22% | Rising↑: 11%
        Weekly confirm     8%   W-S2 = +8% | W-S1 transitioning = +3% | W-Unknown = +2%
        Entry signal       6%   cheat entry: 6% | EMA pullback: ~4% | near 4wk high: 2.4%
        Vol dry in base    3%   Weinstein's volume pattern in base — accumulation signal
        Volume conviction  3%   Very High: 3% | High: 1.5%

    Weekly confirmation replaces part of prior RS/Momentum allocation.
    Weinstein's primary classification is weekly — stocks confirmed on both
    weekly AND daily timeframes deserve higher ranking.
    """
    total = 0.0

    # Stage-2 quality (max raw s2 score is 10)
    s2_norm = min(result.stage_score.get("s2", 0) / 10.0, 1.0)
    total += s2_norm * 0.40

    # Mansfield RS — benchmark outperformance (18%)
    # Using magnitude: strongly outperforming (MRS > 10) gets more than barely (MRS 0–5).
    # This replaces the flat 20% binary flag with a graduated reward.
    mrs = result.mansfield_rs  # Positive = outperforming, 0 = in-line, negative = lagging
    if mrs > 20:    total += 0.18   # strongly outperforming (+20% vs index RS-MA)
    elif mrs > 10:  total += 0.15   # solidly outperforming
    elif mrs > 0:   total += 0.10   # marginally outperforming — still credit
    # mrs ≤ 0: lagging the index — no RS credit

    # Momentum (22%)
    if "↑↑" in result.mom_label:   total += 0.22
    elif "↑"  in result.mom_label:  total += 0.11

    # ── Weekly stage confirmation (8%) — Weinstein primary ────────────────────
    # W-S2 ✓:      price above rising 30-week SMA → full weekly confirmation
    # W-S1 Accum:  transitioning into S2 from below — valid early entry, partial credit
    # W-Unknown:   insufficient weekly history — no penalty, small uncertainty discount
    # W-S3/S4:     already gated out — never reaches here
    if result.weekly_stage == 2:
        total += 0.08    # fully confirmed on weekly — highest conviction
    elif result.weekly_stage == 1:
        total += 0.03    # transitioning — potential early entry, lower confidence
    elif result.weekly_stage == 0:
        total += 0.02    # unknown history — small credit, no hard penalty

    # Volume dry-up in base (3%) — Weinstein's modern emphasis
    # Right-side of base showing declining volume = clean accumulation,
    # no distribution. Absent this, the base may be churning (dangerous).
    if result.weekly_vol_dry:
        total += 0.03

    # Entry signal (6%) — tiebreaker, not primary qualifier
    total += entry_score * 0.06

    # Volume conviction (3%)
    if result.vol_conviction == "Very High": total += 0.03
    elif result.vol_conviction == "High":    total += 0.015

    # Beta penalty: very high beta (>2.0) is riskier
    if result.beta > 2.0:
        total -= 0.05

    return round(min(max(total, 0.0), 1.0), 4)


# =============================================================================
# ROW BUILDER
# =============================================================================

def _result_to_row(
    result:        StageAnalysisResult,
    meta:          dict,
    raw_ticker:    str,
    entry_score:   float,
    entry_label:   str,
    score:         float,
    breakout_vol:  bool = False,
) -> dict:
    ticker_display = raw_ticker.replace(".NS", "").replace(".BO", "")

    if ".NS" in raw_ticker:
        tv_symbol = f"NSE:{ticker_display}"
    elif ".BO" in raw_ticker:
        tv_symbol = f"BSE:{ticker_display}"
    else:
        tv_symbol = ticker_display

    # Mansfield RS label — clearer than a raw number
    mrs = result.mansfield_rs
    if mrs > 20:    mrs_label = f"+{mrs:.1f} (Strong ↑)"
    elif mrs > 10:  mrs_label = f"+{mrs:.1f} (Solid ↑)"
    elif mrs > 0:   mrs_label = f"+{mrs:.1f} (Mild ↑)"
    elif mrs > -10: mrs_label = f"{mrs:.1f} (Mild ↓)"
    else:           mrs_label = f"{mrs:.1f} (Weak ↓)"

    # Weekly distance from 30w SMA — useful for sizing conviction
    if result.weekly_sma30 > 0:
        last_close = float(meta.get("_last_close", 0)) or result.ema_dist_slow  # fallback
        # We don't have raw price here, but we can use the label directly
        w_dist_label = ""   # populated by caller if needed — omit for now
    else:
        w_dist_label = "—"

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "Ticker":           ticker_display,
        "Company":          meta.get("name", raw_ticker),

        # ── Score + verdict ───────────────────────────────────────────────────
        "Score":            score,
        "Entry Signal":     entry_label,

        # ── Weekly stage — Weinstein primary (NEW) ────────────────────────────
        # This is now the primary classification. Read this column first.
        # W-S2 ✓     = price above RISING 30-week SMA → confirmed Stage 2
        # W-S1 Accum = transitioning (price below 30w SMA but SMA still rising)
        # W-Unknown  = insufficient weekly history (< 32 weekly bars)
        "Weekly Stage":     result.weekly_stage_label or "W-Unknown",
        "W-Vol Dry":        "✓" if result.weekly_vol_dry else "·",
        "W-Breakout Vol":   "✓" if breakout_vol else "·",

        # ── Mansfield RS — Weinstein's RS method ─────────────────────────────
        # Positive = outperforming the benchmark. Weinstein requires RS > 0
        # before entry. > +10 = clearly outperforming.
        "Mansfield RS":     mrs_label,

        # ── Daily stage — entry timing detail ────────────────────────────────
        "Stage":            result.stage_full,
        "Stage Score S2":   round(result.stage_score.get("s2", 0), 1),
        "Duration (bars)":  result.stage_duration,
        "Cheat Entry":      "✓ Yes" if result.is_cheat_entry else "·",
        "RS Status":        result.rs_status,
        "Momentum":         result.mom_label,
        "ROC Fast %":       result.roc_fast,
        "Mom Accel":        result.mom_accel,

        # ── Volume ───────────────────────────────────────────────────────────
        "Avg $ Vol":        _fmt_dollar_vol(result.avg_dollar_vol),
        "Vol Trend":        result.vol_trend_str,
        "Vol Conviction":   result.vol_conviction,
        "Vol Ratio":        f"{result.vol_ratio:.2f}x",

        # ── EMA distances ─────────────────────────────────────────────────────
        "EMA Dist Fast":    f"{result.ema_dist_fast:+.1f}%",
        "EMA Dist Mid":     f"{result.ema_dist_medium:+.1f}%",
        "EMA Dist Slow":    f"{result.ema_dist_slow:+.1f}%",

        # ── Risk / context ────────────────────────────────────────────────────
        "Beta":             result.beta,
        "Beta Label":       result.beta_label,
        "PEAD %":           result.pead_pct,
        "PEAD Label":       result.pead_label,
        "Market Cap":       _fmt_mcap(result.market_cap),
        "Sector":           meta.get("sector", "Unknown"),
        "TradingView":      f"https://www.tradingview.com/chart/?symbol={tv_symbol}",
        "Last Updated":     datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def _fmt_dollar_vol(adv: float) -> str:
    if adv >= 1e9:  return f"${adv / 1e9:.2f}B"
    if adv >= 1e6:  return f"${adv / 1e6:.2f}M"
    if adv >= 1e7:  return f"₹{adv / 1e7:.2f}Cr"
    return f"${adv:,.0f}"


def _fmt_mcap(cap: float) -> str:
    if not cap:      return "N/A"
    if cap >= 1e12:  return f"${cap / 1e12:.1f}T"
    if cap >= 1e9:   return f"${cap / 1e9:.1f}B"
    if cap >= 1e7:   return f"₹{cap / 1e7:.0f}Cr"
    return f"${cap:,.0f}"
