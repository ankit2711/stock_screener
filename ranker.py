# =============================================================================
# RANKER — Run all screens on each ticker, compute composite score, return top N
# =============================================================================

import logging
import pandas as pd
from datetime import datetime
from config import SCREEN_WEIGHTS, TOP_N_US, TOP_N_INDIA
from screeners import (
    screen_vcp, screen_darvas, screen_volume_breakout,
    screen_52w_high, screen_rs_rating, screen_ma_alignment,
)

logger = logging.getLogger(__name__)

SCREENS = {
    "vcp":             screen_vcp,
    "darvas":          screen_darvas,
    "volume_breakout": screen_volume_breakout,
    "high_52w":        screen_52w_high,
    "ma_alignment":    screen_ma_alignment,
    # rs_rating handled separately (needs benchmark df)
}


def run_screens(
    ohlcv:      dict[str, pd.DataFrame],
    metadata:   dict[str, dict],
    benchmark:  pd.DataFrame,
    market:     str = "us",
) -> pd.DataFrame:
    """
    Run all screens on every ticker and return a ranked DataFrame.

    Args:
        ohlcv:     {ticker: OHLCV DataFrame}
        metadata:  {ticker: {name, sector, market_cap}}
        benchmark: OHLCV DataFrame for the index (S&P 500 or Nifty 50)
        market:    'us' or 'india'

    Returns:
        DataFrame sorted by composite score, top N rows.
    """
    top_n    = TOP_N_US if market == "us" else TOP_N_INDIA
    rows     = []
    total    = len(ohlcv)

    for i, (ticker, df) in enumerate(ohlcv.items(), 1):
        if i % 50 == 0:
            logger.info(f"  Screening {i}/{total}...")

        if len(df) < 60:
            continue

        row = _screen_ticker(ticker, df, benchmark, metadata.get(ticker, {}))
        if row:
            rows.append(row)

    if not rows:
        logger.warning("No results from screener")
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("Score", ascending=False).reset_index(drop=True)
    result_df.insert(0, "Rank", range(1, len(result_df) + 1))

    logger.info(f"Screened {total} tickers → {len(result_df)} scored → returning top {top_n}")
    return result_df.head(top_n)


def _screen_ticker(
    ticker:    str,
    df:        pd.DataFrame,
    benchmark: pd.DataFrame,
    meta:      dict,
) -> dict | None:
    """Run all 6 screens on a single ticker and compute composite score."""

    screen_results = {}

    # Standard screens
    for name, fn in SCREENS.items():
        try:
            screen_results[name] = fn(df)
        except Exception as e:
            screen_results[name] = {"passed": False, "score": 0.0, "detail": str(e)}

    # RS Rating (needs benchmark)
    try:
        screen_results["rs_rating"] = screen_rs_rating(df, benchmark)
    except Exception as e:
        screen_results["rs_rating"] = {"passed": False, "score": 0.0, "detail": str(e)}

    # Composite weighted score
    composite = sum(
        SCREEN_WEIGHTS.get(name, 0) * res["score"]
        for name, res in screen_results.items()
    )

    # Screens passed (for display)
    passed_screens = [
        _screen_label(name)
        for name, res in screen_results.items()
        if res["passed"]
    ]

    # Price and change
    latest_close = df["close"].iloc[-1]
    prev_close   = df["close"].iloc[-2] if len(df) > 1 else latest_close
    pct_change   = (latest_close - prev_close) / prev_close * 100

    rs_score = screen_results["rs_rating"].get("rs_score", 0)

    return {
        "Ticker":       ticker.replace(".NS", "").replace(".BO", ""),
        "Company":      meta.get("name", ticker),
        "Price":        round(latest_close, 2),
        "Change %":     round(pct_change, 2),
        "Score":        round(composite, 4),
        "Screens":      ", ".join(passed_screens) if passed_screens else "—",
        "VCP":          "✓" if screen_results["vcp"]["passed"]             else "·",
        "Darvas":       "✓" if screen_results["darvas"]["passed"]          else "·",
        "Vol Breakout": "✓" if screen_results["volume_breakout"]["passed"] else "·",
        "52W High":     "✓" if screen_results["high_52w"]["passed"]        else "·",
        "RS Rating":    rs_score,
        "MA Align":     "✓" if screen_results["ma_alignment"]["passed"]    else "·",
        "Market Cap":   _fmt_mcap(meta.get("market_cap", 0)),
        "Sector":       meta.get("sector", "Unknown"),
        "TradingView":  f"https://www.tradingview.com/chart/?symbol={ticker.replace('.NS', 'NSE:').replace('.BO', 'BSE:')}",
        "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def _screen_label(name: str) -> str:
    labels = {
        "vcp":             "VCP",
        "darvas":          "Darvas",
        "volume_breakout": "Vol Break",
        "high_52w":        "52W High",
        "rs_rating":       "RS",
        "ma_alignment":    "MA Stack",
    }
    return labels.get(name, name)


def _fmt_mcap(cap: float) -> str:
    if not cap:
        return "N/A"
    if cap >= 1e12:
        return f"${cap/1e12:.1f}T"
    if cap >= 1e9:
        return f"${cap/1e9:.1f}B"
    if cap >= 1e7:
        return f"₹{cap/1e7:.0f}Cr"
    return f"${cap:,.0f}"
