# =============================================================================
# RANKER — RS Leaders edition
# =============================================================================
#
# PURPOSE:
#   Identify stocks showing Relative Strength leadership during market
#   corrections — i.e. stocks whose RS line is at or near 52-week highs even
#   while the broader benchmark is declining. These are the stocks most likely
#   to lead the next bull market rally (Minervini RS Leader concept).
#
# WHEN TO USE:
#   Run this screener when the market is in a correction (benchmark down
#   5–20%+ from its peak) to build your watchlist of stocks to buy when a
#   Follow-Through Day (FTD) signal is confirmed.
#   It complements the Stage and SEPA screeners:
#     Stage  → find best Stage-2 uptrends (bull market watchlist)
#     SEPA   → rank by entry quality / pivot proximity
#     RS     → find leaders during corrections (bear market watchlist)
#
# SCORE (0–100):
#   RS Line at/near 52-week high    35 pts  (primary signal — showing strength)
#   Relative resilience vs bench    25 pts  (holding up vs the market decline)
#   Volume accumulation             20 pts  (institutional buying/holding)
#   Structural integrity            15 pts  (EMA200, EMA stack, slope)
#   Base formation quality           5 pts  (consolidating tightly)
#
# OUTPUT:
#   Top-N stocks sorted by RS Leader score descending.
#   Written to Google Sheets as a "{date}-rs" dated tab + fixed "RS Leaders" tab.
# =============================================================================

import logging
import pandas as pd
from datetime import datetime

from screeners.rs_leaders import RSLeaderResult, run_rs_leaders_analysis
from config import (
    TOP_N_RS_INDIA, TOP_N_RS_US,
    MIN_AVG_DOLLAR_VOL_NSE, MIN_AVG_DOLLAR_VOL_BSE, MIN_AVG_DOLLAR_VOL_US,
)

logger = logging.getLogger(__name__)

# Minimum score threshold — only genuine RS leaders reach the output.
# Set low so we don't miss stocks in very deep corrections (whole market weak).
MIN_RS_SCORE = 20.0


def run_screens_rs(
    ohlcv:     dict,
    metadata:  dict,
    benchmark: pd.DataFrame,
    market:    str = "india",
) -> pd.DataFrame:
    """
    Run RS Leaders scan: find stocks with RS line near 52-week highs.

    Args:
        ohlcv:     dict of {ticker: OHLCV DataFrame}
        metadata:  dict of {ticker: {name, sector, market_cap, ...}}
        benchmark: benchmark OHLCV DataFrame (aligned with stock data)
        market:    "india" or "us"

    Returns:
        DataFrame of top-N RS Leaders sorted by score descending.
    """
    top_n    = TOP_N_RS_US if market == "us" else TOP_N_RS_INDIA
    rows     = []
    total    = len(ohlcv)
    leaders  = 0

    # Benchmark stats for context (shown on every row)
    bench_close   = benchmark["close"].dropna() if benchmark is not None else pd.Series()
    bench_off_52w = 0.0
    if len(bench_close) >= 2:
        b_hi = float(bench_close.iloc[-252:].max()) if len(bench_close) >= 252 else float(bench_close.max())
        bench_off_52w = (b_hi - float(bench_close.iloc[-1])) / b_hi * 100

    bench_label = _bench_regime_label(bench_off_52w)

    logger.info(
        f"RS Leaders screener: {total} {market.upper()} tickers | "
        f"Benchmark {bench_off_52w:.1f}% off 52w high ({bench_label})"
    )

    for i, (ticker, df) in enumerate(ohlcv.items(), 1):
        if i % 100 == 0:
            logger.info(f"  {i}/{total} | RS Leaders found: {leaders}")

        if len(df) < 60:
            continue

        meta = metadata.get(ticker, {})

        try:
            result = run_rs_leaders_analysis(
                df           = df,
                benchmark_df = benchmark,
                ticker       = ticker,
                market       = market,
            )
            result.market_cap = meta.get("market_cap", 0)

            # ── Liquidity filter (exchange-aware) ────────────────────────────
            # BSE-only stocks (.BO) have thinner order books — require higher bar.
            min_adv = (
                MIN_AVG_DOLLAR_VOL_US  if market == "us"
                else MIN_AVG_DOLLAR_VOL_BSE if ticker.endswith(".BO")
                else MIN_AVG_DOLLAR_VOL_NSE
            )
            if result.avg_dollar_vol < min_adv:
                continue

            # ── Minimum score filter ──────────────────────────────────────
            if result.rs_score < MIN_RS_SCORE:
                continue

            leaders += 1

            row = _result_to_row(
                result      = result,
                meta        = meta,
                raw_ticker  = ticker,
                bench_off   = bench_off_52w,
                bench_label = bench_label,
            )
            rows.append(row)

        except Exception as e:
            logger.debug(f"RS Leaders analysis failed for {ticker}: {e}")

    logger.info(f"RS Leaders screener: {leaders} candidates → top {top_n} returned")

    if not rows:
        logger.warning("RS Leaders screener: no results above minimum score")
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)

    # ── Three-tier sort — no stocks filtered out, just reordered ─────────────
    #
    # Tier 1: 🌟 Leads   — RS line at new high, price still >5% below its own 52w high
    #           These are future leaders caught BEFORE the price breakout.
    #           Minervini: "The RS line leading the price is the most reliable
    #           signal in the whole system."
    #
    # Tier 2: ✓ Confirms — RS at new high AND price near / at its high
    #           Strong setup — RS confirming a simultaneous price breakout.
    #
    # Tier 3: · (no RS leadership) — sorted by score only
    #
    # Within each tier: descending RS Score.
    # NOTE: The 8-pt score bonus for 🌟 already lifts these stocks in raw score;
    #       the explicit tier sort is a belt-and-suspenders guarantee that a
    #       🌟 stock with score 62 never ranks below a · stock with score 65.
    _sort_key = {"🌟 Leads": 0, "✓ Confirms": 1, "·": 2}
    df_out["_tier_sort"] = df_out["RS Leads Price"].map(_sort_key).fillna(2)
    df_out = (
        df_out
        .sort_values(["_tier_sort", "RS Score"], ascending=[True, False])
        .drop(columns=["_tier_sort"])
        .reset_index(drop=True)
    )
    df_out.insert(0, "Rank", range(1, len(df_out) + 1))
    return df_out.head(top_n)


# =============================================================================
# ROW BUILDER
# =============================================================================

def _result_to_row(
    result:     RSLeaderResult,
    meta:       dict,
    raw_ticker: str,
    bench_off:  float,
    bench_label: str,
) -> dict:
    ticker_display = raw_ticker.replace(".NS", "").replace(".BO", "")

    if ".NS" in raw_ticker:
        tv_symbol = f"NSE:{ticker_display}"
    elif ".BO" in raw_ticker:
        tv_symbol = f"BSE:{ticker_display}"
    else:
        tv_symbol = ticker_display

    tv_url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"

    # Score breakdown as readable string
    bd = result.score_breakdown
    lead_bonus = bd.get("rs_leads_price_bonus", 0)
    breakdown_str = (
        f"RS:{bd.get('rs_at_high', 0):.0f} "
        f"Res:{bd.get('resilience', 0):.0f} "
        f"Vol:{bd.get('volume', 0):.0f} "
        f"Str:{bd.get('structure', 0):.0f} "
        f"Base:{bd.get('base', 0):.0f}"
        + (f" Lead:+{lead_bonus:.0f}" if lead_bonus else "")
    )

    # ── RS Leads Price label (three tiers) ───────────────────────────────────
    # 🌟 Leads   — RS line at new high while price is still >5% below its own 52w high
    #              = institutions accumulating before the price breakout (highest conviction)
    # ✓ Confirms — RS line at new high and price also near / at its own 52w high
    #              = RS confirming a simultaneous price breakout (also strong)
    # ·          — RS line not at new high
    if result.rs_leads_price:
        rs_leads_label = "🌟 Leads"     # pre-breakout RS leadership
    elif result.rs_at_52w_high:
        rs_leads_label = "✓ Confirms"   # RS confirming a price breakout
    else:
        rs_leads_label = "·"

    return {
        "Ticker":           ticker_display,
        "Company":          meta.get("name", raw_ticker),
        "RS Score":         result.rs_score,
        "Resilience":       result.resilience_label,
        "RS % from High":   f"{result.rs_52w_high_pct:.1f}%",
        "RS at 52w High":   "✓" if result.rs_at_52w_high else "·",
        "RS New High":      "✓" if result.rs_new_high    else "·",
        # ── Leads Price — the primary pre-breakout signal ────────────────────────
        # 🌟 Leads   = RS line at new 52w high, stock price still >5% below its own high
        #              → institutions buying before the price breakout (most actionable)
        # ✓ Confirms = RS at new high and price also near its 52w high (breakout confirming)
        # ·           = RS not at new high
        "RS Leads Price":   rs_leads_label,
        "Stock Off 52w %":  f"-{result.stock_pct_off_52w:.1f}%",
        "Bench Off 52w %":  f"-{result.bench_pct_off_52w:.1f}%",
        "Resilience Δ":     f"{result.resilience:+.1f}%",
        "RS vs Bench 1M":   f"{result.rs_vs_bench_1m:+.1f}%",
        "RS vs Bench 3M":   f"{result.rs_vs_bench_3m:+.1f}%",
        "Accum Ratio":      f"{result.accum_ratio:.2f}",
        "Vol Dry":          "✓" if result.vol_dry else "·",
        "Vol Dry Ratio":    f"{result.vol_dry_ratio:.2f}x",
        "Stage":            result.stage_label,
        "Above EMA200":     "✓" if result.above_ema200 else "·",
        "EMA Stack":        "✓" if result.bullish_ema_stack else "·",
        "EMA200 Slope":     f"{result.ema200_slope:+.2f}%",
        "Price vs EMA200":  f"{result.price_vs_ema200:+.1f}%",
        "Base Forming":     "✓" if result.base_forming else "·",
        "Base Depth %":     f"{result.base_depth_pct:.1f}%",
        "Consol Bars":      result.consolidation_bars,
        "Score Breakdown":  breakdown_str,
        "Market Regime":    bench_label,
        "Bench Off 52w":    f"-{bench_off:.1f}%",
        "FTD Signal":       "✓ FTD" if result.ftd_detected else "·",
        "Avg $ Vol":        _fmt_dollar_vol(result.avg_dollar_vol),
        "Market Cap":       _fmt_mcap(result.market_cap),
        "Sector":           meta.get("sector", "Unknown"),
        "TradingView":      tv_url,
        "Last Updated":     datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _bench_regime_label(pct_off: float) -> str:
    """Human-readable label for how far the benchmark is off its 52w high."""
    if pct_off <= 2.0:
        return "✅ At High"
    elif pct_off <= 5.0:
        return f"🟡 -{pct_off:.0f}% (Pullback)"
    elif pct_off <= 10.0:
        return f"🟠 -{pct_off:.0f}% (Correction)"
    elif pct_off <= 20.0:
        return f"🔴 -{pct_off:.0f}% (Deep Correction)"
    else:
        return f"🚨 -{pct_off:.0f}% (Bear Market)"


def _fmt_dollar_vol(adv: float) -> str:
    if adv >= 1e9:   return f"${adv / 1e9:.2f}B"
    if adv >= 1e6:   return f"${adv / 1e6:.2f}M"
    if adv >= 1e7:   return f"₹{adv / 1e7:.2f}Cr"
    return f"${adv:,.0f}"


def _fmt_mcap(cap: float) -> str:
    if not cap:      return "N/A"
    if cap >= 1e12:  return f"${cap / 1e12:.1f}T"
    if cap >= 1e9:   return f"${cap / 1e9:.1f}B"
    if cap >= 1e7:   return f"₹{cap / 1e7:.0f}Cr"
    return f"${cap:,.0f}"
