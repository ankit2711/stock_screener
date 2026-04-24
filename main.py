#!/usr/bin/env python3
# =============================================================================
# Stock Screener — Stage Analysis + SEPA (Minervini) Edition
#
# USAGE:
#   python main.py                                   # India, Stage screener
#   python main.py --market india                    # India, Stage screener
#   python main.py --market india --screener sepa    # India, SEPA entry-quality
#   python main.py --market india --screener trade   # India, top 5-10 trade candidates
#   python main.py --market india --screener rs      # India, RS Leaders scan
#   python main.py --market us                       # US, Stage screener
#   python main.py --market us    --screener sepa    # US, SEPA entry-quality
#   python main.py --market us    --screener trade   # US, trade candidates
#   python main.py --market us    --screener rs      # US, RS Leaders scan
#   python main.py --market both  --screener sepa    # both markets, SEPA
#   python main.py --market both  --screener trade   # both markets, trade candidates
#   python main.py --market both  --screener rs      # both markets, RS Leaders
#   python main.py --market india --schedule         # daily schedule at config IST time
#
# SCREENER MODES:
#   stage  (default) — Weinstein Stage Analysis: finds best Stage-2 uptrends
#   sepa             — Minervini SEPA: ranks Stage-2 stocks by entry quality
#                      (pivot proximity, RS line, VCP tightness, volume dry-up …)
#                      Best for finding low-risk entries, not just strong trends.
#   trade            — Trade Candidates: top 5-10 actionable stocks with ₹ entry,
#                      stop-loss, risk %, and position size. Combines SEPA + Stage
#                      into a single ranked list ready to execute. Use this daily.
#   rs               — RS Leaders: stocks whose RS line is at/near 52-week highs
#                      while the market corrects. Build your watchlist during
#                      corrections; buy when a Follow-Through Day is confirmed.
# =============================================================================

import argparse
import logging
import sys
import time
import schedule
from datetime import datetime

import pandas as pd

import fetcher
import universe
import universe_ai
import sheets_writer
import cache
from drive_exporter import export_results
from ranker_stage import run_screens_stage
from ranker_sepa  import run_screens_sepa
from ranker_trade import run_trade_scan
from ranker_rs    import run_screens_rs
from screeners.stage_analysis import StageAnalysisConfig
from config import (
    RS_RATING, SCHEDULE_ENABLED, SCHEDULE_TIME_IST, FETCH_MARKETS_ORDER,
)

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("screener.log"),
    ],
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# STAGE ANALYSIS CONFIG
# ma_length=200  → 200-day EMA = 40-week EMA (Weinstein structural MA).
#                  This matches ranker_stage.DEFAULT_CFG and ranker_sepa._STAGE_CFG
#                  so ALL three sheets (Stage, SEPA, Trade) classify S2/S4 identically.
# slope_lookback=10 → 10-day (2-week) slope on EMA200 for direction signal.
# -----------------------------------------------------------------------------
STAGE_CFG = StageAnalysisConfig(
    sensitivity    = "Aggressive",
    ma_length      = 200,   # 200-day EMA = 40-week EMA  (was 30 — wrong)
    slope_lookback = 10,    # 2-week EMA slope             (was 5)
    ema_fast       = 10,
    ema_medium     = 21,
    ema_slow       = 50,
    rs_ma_length   = 52,
    vol_avg_len    = 50,
    mom_fast       = 10,
    mom_slow       = 20,
    beta_length    = 52,
    pead_threshold = 10.0,
    pead_window    = 5,
)


# -----------------------------------------------------------------------------
# RESULT LOGGER — top-N table to console/log before writing to Sheets
# -----------------------------------------------------------------------------

def _log_top_results(df, market_label: str, screener: str):
    """Log a compact top-N table so results are visible in the console/log."""
    if df is None or (hasattr(df, "empty") and df.empty):
        logger.info(f"  [{market_label}] No results to display.")
        return

    if screener == "trade":
        cols = ["Rank", "Ticker", "Company", "Action", "Entry ₹", "Stop ₹",
                "Risk %", "Pos Size %", "Trade Score", "Regime ⚠", "Signal Summary"]
    elif screener == "sepa":
        cols = ["Rank", "Ticker", "Company", "SEPA Score", "Setup Stage",
                "Pivot Dist %", "Stop Dist %", "RS Status", "Avg $ Vol", "Sector"]
    elif screener == "rs":
        cols = ["Rank", "Ticker", "Company", "RS Score", "Resilience",
                "RS % from High", "RS at 52w High", "Resilience Δ",
                "Stage", "Market Regime", "Sector"]
    else:
        cols = ["Rank", "Ticker", "Company", "Score", "Stage",
                "RS Status", "Momentum", "Avg $ Vol", "Sector"]

    available = [c for c in cols if c in df.columns]
    display   = df[available].copy()

    if "Company" in display.columns:
        display["Company"] = display["Company"].str.slice(0, 22)
    if "Sector" in display.columns:
        display["Sector"]  = display["Sector"].str.slice(0, 14)

    mode_tag = f"{screener.upper()}"
    sep = "─" * 110
    logger.info(f"\n  ┌─ TOP {len(display)} {market_label} [{mode_tag}] RESULTS {'─' * max(0, 70 - len(market_label))}┐")
    logger.info(f"  {display.to_string(index=False)}")
    logger.info(f"  └{sep}┘")


# -----------------------------------------------------------------------------
# MARKET RUNNERS
# -----------------------------------------------------------------------------

def _fetch_market(market: str):
    """
    Shared data-fetch step. Returns (ohlcv, meta, bench) for a market.
    Separated so Stage and SEPA runners share the same data fetch.
    """
    if market == "india":
        logger.info("STEP 1: Building India ticker universe...")
        tickers  = universe.get_india_tickers()
        logger.info(f"  India: {len(tickers)} tickers")

        logger.info("STEP 2: Fetching OHLCV data (India)...")
        ohlcv    = fetcher.fetch_ohlcv(tickers, market="india")

        logger.info(f"  Fetching benchmark ({RS_RATING['benchmark_india']})...")
        bench_raw = fetcher.fetch_benchmarks([RS_RATING["benchmark_india"]])
        bench     = bench_raw.get(RS_RATING["benchmark_india"])

        if bench is None:
            logger.error(
                f"India benchmark fetch failed ({RS_RATING['benchmark_india']}) — aborting. "
                "Run check_benchmark.py to find the correct symbol."
            )
            return None, None, None

        logger.info("STEP 3: Fetching metadata (India)...")
        meta = fetcher.fetch_metadata(list(ohlcv.keys()))

    elif market == "ai":
        logger.info("STEP 1: Building AI Theme universe...")
        tickers = universe_ai.get_ai_tickers()
        logger.info(f"  AI Theme: {len(tickers)} tickers across {len(universe_ai.get_ai_groups())} groups")

        logger.info("STEP 2: Fetching OHLCV data (AI Theme)...")
        ohlcv = fetcher.fetch_ohlcv(tickers, market="ai")

        bench_sym = RS_RATING["benchmark_ai"]
        logger.info(f"  Fetching benchmark ({bench_sym} — Nasdaq 100)...")
        bench_raw = fetcher.fetch_benchmarks([bench_sym])
        bench     = bench_raw.get(bench_sym)

        if bench is None:
            logger.error(f"AI benchmark fetch failed ({bench_sym}) — aborting.")
            return None, None, None

        logger.info("STEP 3: Fetching metadata (AI Theme)...")
        meta = fetcher.fetch_metadata(list(ohlcv.keys()))

        # Override sector with AI supply-chain group names from the watchlist
        ai_group_meta = universe_ai.get_ai_metadata()
        for ticker, ai_info in ai_group_meta.items():
            if ticker in meta:
                meta[ticker]["sector"] = ai_info["sector"]
            else:
                meta[ticker] = {
                    "name":       ticker,
                    "sector":     ai_info["sector"],
                    "market_cap": 0,
                }

    else:  # us
        logger.info("STEP 1: Building US ticker universe...")
        tickers  = universe.get_us_tickers()
        logger.info(f"  US: {len(tickers)} tickers")

        logger.info("STEP 2: Fetching OHLCV data (US)...")
        ohlcv    = fetcher.fetch_ohlcv(tickers, market="us")

        logger.info(f"  Fetching benchmark ({RS_RATING['benchmark_us']})...")
        bench_raw = fetcher.fetch_benchmarks([RS_RATING["benchmark_us"]])
        bench     = bench_raw.get(RS_RATING["benchmark_us"])

        if bench is None:
            logger.error(f"US benchmark fetch failed ({RS_RATING['benchmark_us']}) — aborting.")
            return None, None, None

        logger.info("STEP 3: Fetching metadata (US)...")
        meta = fetcher.fetch_metadata(list(ohlcv.keys()))

    return ohlcv, meta, bench


def run_india(screener: str = "stage") -> bool:
    """Fetch, screen, log and write India results. Returns True on success."""
    logger.info("─" * 55)
    logger.info(f"▶  INDIA  [{screener.upper()}]")
    logger.info("─" * 55)

    ohlcv, meta, bench = _fetch_market("india")
    if ohlcv is None:
        return False

    logger.info(f"STEP 4: Running {screener.upper()} analysis (India)...")

    if screener == "trade":
        results = run_trade_scan(ohlcv, meta, bench, market="india")
    elif screener == "sepa":
        results = run_screens_sepa(ohlcv, meta, bench, market="india")
    elif screener == "rs":
        results = run_screens_rs(ohlcv, meta, bench, market="india")
    else:
        results = run_screens_stage(ohlcv, meta, bench, market="india", cfg=STAGE_CFG)

    if isinstance(results, dict):
        count = sum(len(v) for v in results.values() if hasattr(v, "__len__"))
        logger.info(f"  India [{screener}]: Stage={len(results.get('stage', []))}, "
                    f"SEPA={len(results.get('sepa', []))}, "
                    f"RS={len(results.get('rs', []))}, "
                    f"Trade={len(results.get('trade', []))}")
        _log_top_results(results.get("trade", pd.DataFrame()), "INDIA", "trade")
    else:
        count = len(results) if results is not None and hasattr(results, "__len__") else 0
        logger.info(f"  India [{screener}]: {count} stocks returned")
        _log_top_results(results, "INDIA", screener)

    logger.info("STEP 5: Writing India results to Google Sheets...")
    sheets_writer.write_results(results, market="india", screener=screener)

    logger.info("STEP 6: Exporting India results to JSON → Google Drive...")
    export_results(results, market="india", screener=screener)

    return True


def run_ai(screener: str = "stage") -> bool:
    """Fetch AI Theme universe, screen it, and write results. Returns True on success."""
    logger.info("─" * 55)
    logger.info(f"▶  AI THEME  [{screener.upper()}]")
    logger.info("─" * 55)

    ohlcv, meta, bench = _fetch_market("ai")
    if ohlcv is None:
        return False

    logger.info(f"STEP 4: Running {screener.upper()} analysis (AI Theme)...")

    if screener == "trade":
        results = run_trade_scan(ohlcv, meta, bench, market="ai")
    elif screener == "sepa":
        results = run_screens_sepa(ohlcv, meta, bench, market="ai")
    elif screener == "rs":
        results = run_screens_rs(ohlcv, meta, bench, market="ai")
    else:
        results = run_screens_stage(ohlcv, meta, bench, market="ai", cfg=STAGE_CFG)

    if isinstance(results, dict):
        logger.info(
            f"  AI [{screener}]: Stage={len(results.get('stage', []))}, "
            f"SEPA={len(results.get('sepa', []))}, "
            f"RS={len(results.get('rs', []))}, "
            f"Trade={len(results.get('trade', []))}"
        )
        _log_top_results(results.get("trade", pd.DataFrame()), "AI THEME", "trade")
    else:
        count = len(results) if results is not None and hasattr(results, "__len__") else 0
        logger.info(f"  AI [{screener}]: {count} stocks returned")
        _log_top_results(results, "AI THEME", screener)

    logger.info("STEP 5: Writing AI Theme results to Google Sheets...")
    sheets_writer.write_results(results, market="ai", screener=screener)

    logger.info("STEP 6: Exporting AI Theme results to JSON → Google Drive...")
    export_results(results, market="ai", screener=screener)

    return True


def run_us(screener: str = "stage") -> bool:
    """Fetch, screen, log and write US results. Returns True on success."""
    logger.info("─" * 55)
    logger.info(f"▶  US  [{screener.upper()}]")
    logger.info("─" * 55)

    ohlcv, meta, bench = _fetch_market("us")
    if ohlcv is None:
        return False

    logger.info(f"STEP 4: Running {screener.upper()} analysis (US)...")

    if screener == "trade":
        results = run_trade_scan(ohlcv, meta, bench, market="us")
    elif screener == "sepa":
        results = run_screens_sepa(ohlcv, meta, bench, market="us")
    elif screener == "rs":
        results = run_screens_rs(ohlcv, meta, bench, market="us")
    else:
        results = run_screens_stage(ohlcv, meta, bench, market="us", cfg=STAGE_CFG)

    if isinstance(results, dict):
        logger.info(f"  US [{screener}]: Stage={len(results.get('stage', []))}, "
                    f"SEPA={len(results.get('sepa', []))}, "
                    f"RS={len(results.get('rs', []))}, "
                    f"Trade={len(results.get('trade', []))}")
        _log_top_results(results.get("trade", pd.DataFrame()), "US", "trade")
    else:
        count = len(results) if results is not None and hasattr(results, "__len__") else 0
        logger.info(f"  US [{screener}]: {count} stocks returned")
        _log_top_results(results, "US", screener)

    logger.info("STEP 5: Writing US results to Google Sheets...")
    sheets_writer.write_results(results, market="us", screener=screener)

    logger.info("STEP 6: Exporting US results to JSON → Google Drive...")
    export_results(results, market="us", screener=screener)

    return True


# -----------------------------------------------------------------------------
# MAIN RUN
# -----------------------------------------------------------------------------

def run(market: str = "india", screener: str = "stage"):
    """
    Run the screener for the given market and mode.

    Args:
        market:   "india", "us", or "both"
        screener: "stage" (original) or "sepa" (Minervini entry quality)
    """
    start = datetime.now()
    logger.info("=" * 60)
    logger.info(
        f"SCREENER [{screener.upper()}] [{market.upper()}] — "
        f"{start.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info("=" * 60)

    try:
        # --- 0. Init cache ---
        cache.init_cache()
        stats = cache.get_cache_stats()
        logger.info(
            f"Cache: {stats['ohlcv_tickers']} tickers | "
            f"OHLCV: {stats['ohlcv_db_mb']}MB | "
            f"US universe: {stats['us_universe_size']} | "
            f"India universe: {stats['india_universe_size']}"
        )

        success = True

        if market in ("india", "both"):
            ok = run_india(screener)
            success = success and ok

        if market in ("us", "both"):
            ok = run_us(screener)
            success = success and ok

        if market == "ai":
            ok = run_ai(screener)
            success = success and ok

        elapsed = (datetime.now() - start).seconds
        status  = "✓ Completed" if success else "⚠ Completed with errors"
        logger.info(f"{status} in {elapsed}s")

    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.exception(f"Run failed: {e}")
        raise


# -----------------------------------------------------------------------------
# SCHEDULER
# -----------------------------------------------------------------------------

def run_with_scheduler(market: str, screener: str):
    logger.info(
        f"Scheduler active — daily at {SCHEDULE_TIME_IST} IST  "
        f"[market={market}] [screener={screener}]"
    )
    schedule.every().day.at(SCHEDULE_TIME_IST).do(run, market=market, screener=screener)
    run(market, screener)   # run immediately on start
    while True:
        schedule.run_pending()
        time.sleep(60)


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stock Screener — Stage Analysis + SEPA (Minervini) Edition",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--market",
        choices=["india", "us", "both", "ai"],
        default="india",
        help=(
            "Market to screen (default: india)\n"
            "  india  — NSE/BSE stocks only\n"
            "  us     — US stocks only\n"
            "  both   — India first, then US\n"
            "  ai     — AI Theme: ~150 curated global AI supply-chain stocks\n"
            "           (benchmark: QQQ; sector = AI supply-chain group)"
        ),
    )
    parser.add_argument(
        "--screener",
        choices=["stage", "sepa", "trade", "rs"],
        default="stage",
        help=(
            "Screener mode (default: stage)\n"
            "  stage  — Weinstein Stage Analysis: best Stage-2 uptrends\n"
            "  sepa   — Minervini SEPA: Stage-2 stocks ranked by entry quality\n"
            "           (pivot proximity, VCP tightness, RS line, volume dry-up)\n"
            "  trade  — Top 5-10 actionable candidates with ₹ entry, stop-loss,\n"
            "           risk % and position size. Use this for daily trading decisions.\n"
            "  rs     — RS Leaders: stocks with RS line at/near 52w highs during\n"
            "           market corrections. Build correction watchlist; buy on FTD."
        ),
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help=f"Run on a daily schedule at {SCHEDULE_TIME_IST} IST",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    use_scheduler = args.schedule or SCHEDULE_ENABLED
    if use_scheduler:
        run_with_scheduler(args.market, args.screener)
    else:
        run(args.market, args.screener)
