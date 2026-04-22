# =============================================================================
# FETCHER — OHLCV data with local SQLite cache
# Based on: "Stage Analysis Screener (TheWrap Clone)"
# Original Pine Script author: Atlas (OpenClaw) for Ankit
# Python port: Claude (Anthropic)
#
# Data source priority:
#   PRIMARY  → yfinance  (free, handles US + India .NS natively)
#   FALLBACK → Twelve Data (API key required, rate-limited 8 req/min on free tier)
#
# Flow per ticker:
#   1. Check local SQLite cache → get last stored date
#   2. Only fetch dates AFTER that from yfinance (primary) / Twelve Data (fallback)
#   3. Append new rows to cache
#   4. Return full merged DataFrame (cache + new)
# =============================================================================

import os
import time
import logging
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta

import cache
from config import (
    TWELVE_DATA_API_KEY, HISTORY_DAYS,
    YF_CHUNK_SIZE, YF_RETRY_LIMIT, YF_RETRY_DELAY,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Redirect yfinance's internal SQLite timezone-cache to our project cache dir.
#
# WHY: yfinance 0.2.x writes a SQLite file to ~/.cache/py-yfinance/ by default.
# If that directory doesn't exist (Docker, CI, new machine) SQLite throws:
#   OperationalError: unable to open database file
# Setting it to None causes: TypeError: stat: path should be string … not NoneType
#
# SOLUTION: point it at our cache/ subdir which cache.init_cache() always creates.
# -----------------------------------------------------------------------------
_YF_TZ_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "yf_tz")
os.makedirs(_YF_TZ_CACHE, exist_ok=True)

try:
    yf.set_tz_cache_location(_YF_TZ_CACHE)   # yfinance < 0.2.28
except AttributeError:
    pass

try:
    from yfinance import cache as _yf_cache_mod
    _yf_cache_mod.set_cache_dir(_YF_TZ_CACHE)  # yfinance >= 0.2.28
except (ImportError, AttributeError):
    pass

# Twelve Data constants (fallback only)
TWELVE_DATA_BASE    = "https://api.twelvedata.com"
TD_RATE_LIMIT_DELAY = 8   # seconds between batches (free tier: 8 req/min)
TD_BATCH_SIZE       = 8   # tickers per request

_LOG_EVERY = 5  # log after every N chunks


# -----------------------------------------------------------------------------
# PUBLIC: OHLCV
# -----------------------------------------------------------------------------

def fetch_ohlcv(
    tickers: list[str],
    market:  str = "us",
    db_path: str = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for tickers, using local cache. Only fetches new days.
    Returns {ticker: DataFrame[open, high, low, close, volume]}.

    Primary:  yfinance  (chunked, sequential within chunk, free)
    Fallback: Twelve Data (for any tickers yfinance cannot serve)
    """
    if db_path is None:
        db_path = cache.OHLCV_DB

    results       = {}
    need_fetch    = {}
    already_fresh = 0

    for ticker in tickers:
        start, end = cache.get_missing_date_range(ticker, HISTORY_DAYS, db_path)
        if start is None:
            df = cache.load_ohlcv(ticker, db_path)
            if df is not None and len(df) >= 50:
                results[ticker] = df
                already_fresh += 1
        else:
            need_fetch[ticker] = (start, end)

    logger.info(f"{market.upper()}: {already_fresh} cached / {len(need_fetch)} to fetch")

    if need_fetch:
        # ── PRIMARY: yfinance ─────────────────────────────────────────────────
        yf_results, yf_failed = _fetch_yfinance_chunked(need_fetch, market)

        # ── FALLBACK: Twelve Data ─────────────────────────────────────────────
        td_results = {}
        if yf_failed and TWELVE_DATA_API_KEY:
            logger.info(
                f"  Twelve Data fallback for {len(yf_failed)} ticker(s) "
                f"that yfinance could not serve..."
            )
            td_results, td_still_failed = _fetch_twelve_data_ranged(
                {t: need_fetch[t] for t in yf_failed}
            )
            if td_still_failed:
                logger.warning(
                    f"  Both APIs failed for {len(td_still_failed)} tickers: "
                    f"{td_still_failed[:10]}"
                )
        elif yf_failed:
            logger.warning(
                f"  {len(yf_failed)} tickers failed yfinance and no Twelve Data "
                f"key is configured — skipping."
            )

        fetched = {**yf_results, **td_results}
        logger.info(
            f"  Fetched {len(fetched)}/{len(need_fetch)} tickers — saving to cache..."
        )

        saved = 0
        for ticker, new_df in fetched.items():
            try:
                merged = cache.merge_with_cache(ticker, new_df, db_path)
                if merged is not None and len(merged) >= 50:
                    results[ticker] = merged
                    saved += 1
            except Exception as e:
                # Cache write failed — use the in-memory data so the run continues
                logger.warning(f"  Cache save failed for {ticker}: {e} — using in-memory data")
                if new_df is not None and len(new_df) >= 50:
                    results[ticker] = new_df
                    saved += 1

        logger.info(f"  Saved {saved} tickers to cache")

        # Partial-cache fallback for still-missing tickers
        for ticker in need_fetch:
            if ticker not in results:
                cached_df = cache.load_ohlcv(ticker, db_path)
                if cached_df is not None and len(cached_df) >= 50:
                    results[ticker] = cached_df

        still_failed = [t for t in need_fetch if t not in results]
        if still_failed:
            logger.warning(
                f"  No usable data for {len(still_failed)} tickers: {still_failed[:10]}"
            )

    logger.info(f"{market.upper()}: {len(results)}/{len(tickers)} tickers ready")
    return results


# -----------------------------------------------------------------------------
# PUBLIC: BENCHMARKS
# -----------------------------------------------------------------------------

def fetch_benchmarks(benchmark_tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch benchmark index OHLCV using a separate cache DB."""
    return fetch_ohlcv(benchmark_tickers, market="index", db_path=cache.BENCHMARK_DB)


# -----------------------------------------------------------------------------
# YFINANCE — PRIMARY FETCHER (chunked, per-ticker fallback)
# -----------------------------------------------------------------------------

def _fetch_yfinance_chunked(
    need_fetch: dict[str, tuple],
    market: str = "us",
) -> tuple[dict, list]:
    """
    Download OHLCV via yfinance in chunks of YF_CHUNK_SIZE.
    For each chunk: tries batch download first, then falls back to
    per-ticker individual downloads for any that fail in the batch.
    Returns (results_dict, failed_tickers_list).
    """
    results = {}
    failed  = []

    range_groups: dict[tuple, list] = {}
    for ticker, (start, end) in need_fetch.items():
        range_groups.setdefault((str(start), str(end)), []).append(ticker)

    total     = len(need_fetch)
    processed = 0
    chunk_num = 0

    for (start_str, end_str), group in range_groups.items():
        # yfinance end is exclusive — add 1 day to include today's bar
        end_yf = (date.fromisoformat(end_str) + timedelta(days=1)).isoformat()

        chunks = [
            group[i: i + YF_CHUNK_SIZE]
            for i in range(0, len(group), YF_CHUNK_SIZE)
        ]

        for chunk in chunks:
            chunk_num += 1
            ok, fail = _yf_chunk_with_retry(chunk, start_str, end_yf)
            results.update(ok)
            failed.extend(fail)
            processed += len(chunk)

            if chunk_num % _LOG_EVERY == 0 or processed == total:
                pct = 100 * processed // max(total, 1)
                logger.info(
                    f"  [{market.upper()}] {processed}/{total} ({pct}%) — "
                    f"{len(results)} ok | {len(failed)} failed so far"
                )

    return results, failed


def _yf_chunk_with_retry(
    tickers: list[str],
    start_str: str,
    end_str: str,
) -> tuple[dict, list]:
    """
    Download a chunk via yfinance.

    Strategy:
      1. Try a batch download with retries (fast path).
      2. For any ticker that fails in the batch, retry it individually
         (slow path — isolates bad tickers from good ones).

    NOTE: threads=False is intentional.
      yfinance's threaded download causes race conditions when certain
      Indian tickers return None metadata, producing:
        TypeError: 'NoneType' object is not subscriptable
      Sequential mode avoids this entirely.
    """
    results = {}
    failed  = list(tickers)  # start assuming all failed

    # ── Step 1: batch download (fast) ────────────────────────────────────────
    for attempt in range(YF_RETRY_LIMIT):
        try:
            raw = yf.download(
                tickers,
                start=start_str,
                end=end_str,
                auto_adjust=True,
                progress=False,
                threads=False,   # sequential: avoids NoneType race conditions
                # group_by="ticker" intentionally omitted:
                # yfinance 1.x always returns (Price, Ticker) MultiIndex regardless
            )
            ok, fail = _parse_yf_download(raw, tickers)
            results.update(ok)
            failed = fail
            break

        except Exception as e:
            if attempt < YF_RETRY_LIMIT - 1:
                wait = YF_RETRY_DELAY * (attempt + 1)
                logger.debug(
                    f"yfinance batch attempt {attempt + 1}/{YF_RETRY_LIMIT} failed "
                    f"({e}) — retrying in {wait}s"
                )
                time.sleep(wait)

    # ── Step 2: per-ticker fallback for anything still failing ───────────────
    still_failed = []
    for ticker in failed:
        try:
            raw_s = yf.download(
                [ticker],
                start=start_str,
                end=end_str,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            ok_s, _ = _parse_yf_download(raw_s, [ticker])
            if ok_s:
                results.update(ok_s)
            else:
                still_failed.append(ticker)
        except Exception as e:
            logger.debug(f"yfinance single-ticker {ticker} failed: {e}")
            still_failed.append(ticker)

    return results, still_failed


def _parse_yf_download(raw: pd.DataFrame, tickers: list[str]) -> tuple[dict, list]:
    """
    Parse a yfinance download result into per-ticker DataFrames.

    yfinance column formats across versions (detected at runtime):

      Format B — yfinance 1.x (ALL downloads, single or multi-ticker):
          MultiIndex level-0 = Price field, level-1 = Ticker
          e.g. ('Close','RELIANCE.NS'), ('High','RELIANCE.NS'), …
          Extract via: raw.xs(ticker, axis=1, level=1)

      Format A — yfinance 0.2.x with group_by='ticker':
          MultiIndex level-0 = Ticker, level-1 = Price field
          e.g. ('RELIANCE.NS','Close'), ('RELIANCE.NS','High'), …
          Extract via: raw[ticker]

      Flat — very old yfinance, single-ticker without group_by:
          Plain columns: Close, High, Low, Open, Volume
          Extract via: raw itself

    The format is detected once from the actual returned DataFrame,
    so this function is version-agnostic.
    """
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return {}, list(tickers)

    results = {}
    failed  = []

    is_multi = isinstance(raw.columns, pd.MultiIndex)
    if is_multi:
        lvl0 = set(raw.columns.get_level_values(0))
        lvl1 = set(raw.columns.get_level_values(1))
        # Format B: tickers sit at level-1 (yfinance 1.x — all downloads)
        fmt_b = any(t in lvl1 for t in tickers)
        # Format A: tickers sit at level-0 (old yfinance with group_by='ticker')
        fmt_a = (not fmt_b) and any(t in lvl0 for t in tickers)
    else:
        fmt_a = fmt_b = False   # flat columns

    for ticker in tickers:
        try:
            if is_multi and fmt_b and ticker in lvl1:
                # ── yfinance 1.x: (Price, Ticker) ───────────────────────────
                df = raw.xs(ticker, axis=1, level=1).copy()

            elif is_multi and fmt_a and ticker in lvl0:
                # ── old yfinance: (Ticker, Price) ───────────────────────────
                df = raw[ticker].copy()

            elif not is_multi:
                # ── flat columns (very old yfinance, single ticker) ──────────
                df = raw.copy()

            else:
                failed.append(ticker)
                continue

            if df is None or df.empty:
                failed.append(ticker)
                continue

            # Normalise column names to lowercase
            df.columns = [c.lower() for c in df.columns]
            needed = ["open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in needed):
                failed.append(ticker)
                continue

            df = df[needed].dropna()
            df.index = pd.to_datetime(df.index)

            if len(df) > 0:
                results[ticker] = df
            else:
                failed.append(ticker)

        except Exception as e:
            logger.debug(f"yfinance parse error [{ticker}]: {e}")
            failed.append(ticker)

    return results, failed


# -----------------------------------------------------------------------------
# TWELVE DATA — FALLBACK (date-ranged batch fetch)
# -----------------------------------------------------------------------------

def _fetch_twelve_data_ranged(
    need_fetch: dict[str, tuple],
) -> tuple[dict, list]:
    results = {}
    failed  = []

    range_groups: dict[tuple, list] = {}
    for ticker, (start, end) in need_fetch.items():
        range_groups.setdefault((str(start), str(end)), []).append(ticker)

    for (start_str, end_str), group in range_groups.items():
        batches = [
            group[i: i + TD_BATCH_SIZE]
            for i in range(0, len(group), TD_BATCH_SIZE)
        ]
        for b_idx, batch in enumerate(batches):
            ok, fail = _td_batch(batch, start_str, end_str)
            results.update(ok)
            failed.extend(fail)
            if b_idx < len(batches) - 1:
                time.sleep(TD_RATE_LIMIT_DELAY)

    return results, failed


def _td_batch(tickers: list[str], start: str, end: str) -> tuple[dict, list]:
    results = {}
    failed  = []

    def td_sym(t):
        if t.endswith(".NS"):  return t.replace(".NS", "") + ":NSE"
        if t.endswith(".BO"):  return t.replace(".BO", "") + ":BSE"
        if t.startswith("^"): return t.lstrip("^")
        return t

    symbols = ",".join(td_sym(t) for t in tickers)
    params  = {
        "symbol":     symbols,
        "interval":   "1day",
        "start_date": start,
        "end_date":   end,
        "outputsize": HISTORY_DAYS + 60,
        "format":     "JSON",
        "apikey":     TWELVE_DATA_API_KEY,
    }

    try:
        resp = requests.get(
            f"{TWELVE_DATA_BASE}/time_series", params=params, timeout=20
        )
        resp.raise_for_status()
        data = resp.json()

        for ticker in tickers:
            raw = data if len(tickers) == 1 else data.get(td_sym(ticker), {})
            if not raw or raw.get("status") == "error":
                failed.append(ticker)
                continue
            df = _parse_td(raw)
            if df is not None:
                results[ticker] = df
            else:
                failed.append(ticker)

    except Exception as e:
        logger.warning(f"Twelve Data batch error: {e}")
        failed.extend(tickers)

    return results, failed


def _parse_td(data: dict) -> pd.DataFrame | None:
    try:
        values = data.get("values", [])
        if not values:
            return None
        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df if len(df) > 0 else None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# METADATA — weekly-cached via cache.py
# -----------------------------------------------------------------------------

def fetch_metadata(tickers: list[str]) -> dict[str, dict]:
    """
    Return metadata for tickers. Uses weekly cache — only calls
    yfinance for tickers not already stored.
    """
    cached  = cache.load_metadata() or {}
    missing = [t for t in tickers if t not in cached]

    if missing:
        logger.info(f"Fetching metadata for {len(missing)} new tickers...")
        new_meta = {}
        for ticker in missing:
            try:
                info = yf.Ticker(ticker).info
                new_meta[ticker] = {
                    "name":       info.get("longName") or info.get("shortName", ticker),
                    "sector":     info.get("sector", "Unknown"),
                    "market_cap": info.get("marketCap", 0),
                }
            except Exception:
                new_meta[ticker] = {"name": ticker, "sector": "Unknown", "market_cap": 0}
        cache.update_metadata(new_meta)
        cached.update(new_meta)
    else:
        logger.info(f"Metadata: all {len(tickers)} tickers cached")

    return {
        t: cached.get(t, {"name": t, "sector": "Unknown", "market_cap": 0})
        for t in tickers
    }
