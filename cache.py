# =============================================================================
# CACHE — Local storage layer
# Based on: "Stage Analysis Screener (TheWrap Clone)"
# Original Pine Script author: Atlas (OpenClaw) for Ankit
# Python port: Claude (Anthropic)
#
# Storage layout (all inside stock_screener/cache/):
#   ohlcv.db          — SQLite: OHLCV price history, one table per ticker
#   universe_us.json  — US ticker list (refreshed weekly)
#   universe_in.json  — India ticker list (refreshed weekly)
#   metadata.json     — name/sector/market cap (refreshed weekly)
#   benchmarks.db     — SQLite: benchmark index OHLCV (same append logic)
#
# Strategy:
#   - OHLCV:     on each run, only fetch dates AFTER the last cached date
#   - Universe:  fetched fresh if cache is older than UNIVERSE_TTL_DAYS
#   - Metadata:  fetched fresh if cache is older than METADATA_TTL_DAYS
#   - Benchmark: same append logic as OHLCV
# =============================================================================

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, date

import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# SQLite connection helper
# -----------------------------------------------------------------------------

def _connect(db_path: str) -> sqlite3.Connection:
    """
    Open a SQLite connection with:
      • timeout=30  — waits up to 30 s for a lock instead of failing instantly
      • WAL mode    — allows concurrent readers + one writer without blocking
    Use this everywhere instead of sqlite3.connect() directly.
    """
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

CACHE_DIR         = os.path.join(os.path.dirname(__file__), "cache")
OHLCV_DB          = os.path.join(CACHE_DIR, "ohlcv.db")
BENCHMARK_DB      = os.path.join(CACHE_DIR, "benchmarks.db")
UNIVERSE_US_FILE  = os.path.join(CACHE_DIR, "universe_us.json")
UNIVERSE_IN_FILE  = os.path.join(CACHE_DIR, "universe_in.json")
METADATA_FILE     = os.path.join(CACHE_DIR, "metadata.json")

# Cache TTLs
UNIVERSE_TTL_DAYS = 7    # re-fetch ticker list every 7 days
METADATA_TTL_DAYS = 7    # re-fetch name/sector/mcap every 7 days

# -----------------------------------------------------------------------------
# INIT
# -----------------------------------------------------------------------------

def init_cache():
    """Create cache directory and databases if they don't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    _init_db(OHLCV_DB)
    _init_db(BENCHMARK_DB)
    logger.info(f"Cache initialised at: {CACHE_DIR}")


def _init_db(db_path: str):
    """Create SQLite DB file if missing."""
    conn = _connect(db_path)
    conn.close()


# -----------------------------------------------------------------------------
# OHLCV CACHE — append-only, one table per ticker
# -----------------------------------------------------------------------------

def get_last_cached_date(ticker: str, db_path: str = OHLCV_DB) -> date | None:
    """Return the most recent date stored for a ticker, or None if not cached."""
    table = _safe_table_name(ticker)
    try:
        conn = _connect(db_path)
        cur  = conn.cursor()
        cur.execute(f"SELECT MAX(date) FROM \"{table}\"")
        row = cur.fetchone()
        conn.close()
        if row and row[0]:
            return datetime.strptime(row[0], "%Y-%m-%d").date()
    except sqlite3.OperationalError:
        pass  # table doesn't exist yet
    return None


def load_ohlcv(ticker: str, db_path: str = OHLCV_DB) -> pd.DataFrame | None:
    """
    Load all cached OHLCV rows for a ticker.
    Returns DataFrame with DatetimeIndex or None if no cache exists.
    """
    table = _safe_table_name(ticker)
    try:
        conn = _connect(db_path)
        df = pd.read_sql(
            f"SELECT * FROM \"{table}\" ORDER BY date ASC",
            conn, parse_dates=["date"], index_col="date"
        )
        conn.close()
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


def save_ohlcv(ticker: str, df: pd.DataFrame, db_path: str = OHLCV_DB):
    """
    Append new OHLCV rows to the ticker's cache table.
    Skips rows that already exist (upsert by date).
    df must have DatetimeIndex and columns [open, high, low, close, volume].
    """
    if df is None or df.empty:
        return

    table = _safe_table_name(ticker)
    df_out = df.copy()
    df_out.index = df_out.index.strftime("%Y-%m-%d")
    df_out.index.name = "date"

    conn = _connect(db_path)
    try:
        # Create table if it doesn't exist
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS "{table}" (
                date    TEXT PRIMARY KEY,
                open    REAL,
                high    REAL,
                low     REAL,
                close   REAL,
                volume  REAL
            )
        """)
        # Insert or ignore (skip duplicates)
        df_out.to_sql(table, conn, if_exists="append", index=True,
                      method=_insert_or_ignore)
        conn.commit()
    finally:
        conn.close()


def _insert_or_ignore(table, conn, keys, data_iter):
    """Custom to_sql method that uses INSERT OR IGNORE.
    NOTE: pandas passes a SQLiteTable *object* as `table`, not a plain string.
    We must use table.name to get the actual table-name string.
    """
    table_name = table.name  # ← fix: was using `table` directly (object, not string)
    cols = ", ".join([f'"{k}"' for k in keys])
    placeholders = ", ".join(["?" for _ in keys])
    sql = f'INSERT OR IGNORE INTO "{table_name}" ({cols}) VALUES ({placeholders})'
    conn.executemany(sql, data_iter)


def get_missing_date_range(ticker: str, history_days: int,
                           db_path: str = OHLCV_DB) -> tuple[date, date]:
    """
    Given a ticker and how many days of history we want,
    return (fetch_start, fetch_end) — only the dates we don't already have.
    """
    today      = datetime.today().date()
    full_start = today - timedelta(days=history_days + 60)  # buffer for weekends
    last_cached = get_last_cached_date(ticker, db_path)

    if last_cached is None:
        # No cache — fetch full history
        return full_start, today
    elif last_cached >= today - timedelta(days=1):
        # Already up to date (yesterday or today)
        return None, None
    else:
        # Fetch from day after last cached to today
        fetch_start = last_cached + timedelta(days=1)
        return fetch_start, today


def merge_with_cache(ticker: str, new_df: pd.DataFrame | None,
                     db_path: str = OHLCV_DB) -> pd.DataFrame | None:
    """
    Save new_df rows to cache, then return the full merged DataFrame
    (cache + new rows combined and sorted).
    """
    if new_df is not None and not new_df.empty:
        save_ohlcv(ticker, new_df, db_path)

    return load_ohlcv(ticker, db_path)


# -----------------------------------------------------------------------------
# UNIVERSE CACHE — JSON, refreshed weekly
# -----------------------------------------------------------------------------

def load_universe(market: str) -> list[str] | None:
    """
    Load cached ticker list for 'us' or 'india'.
    Returns None if cache is missing or older than UNIVERSE_TTL_DAYS.
    """
    path = UNIVERSE_US_FILE if market == "us" else UNIVERSE_IN_FILE
    return _load_json_if_fresh(path, UNIVERSE_TTL_DAYS)


def save_universe(market: str, tickers: list[str]):
    """Save ticker list to cache with current timestamp."""
    path = UNIVERSE_US_FILE if market == "us" else UNIVERSE_IN_FILE
    _save_json(path, {"tickers": tickers, "saved_at": _now_str()})
    logger.info(f"Universe cached: {market.upper()} — {len(tickers)} tickers")


def _load_json_if_fresh(path: str, ttl_days: int):
    """Load JSON file if it exists and is within TTL. Returns None if stale/missing."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        saved_at = datetime.strptime(data["saved_at"], "%Y-%m-%d %H:%M:%S")
        if datetime.now() - saved_at > timedelta(days=ttl_days):
            logger.info(f"Cache stale ({path}) — will refresh")
            return None
        return data.get("tickers") or data.get("metadata")
    except Exception as e:
        logger.warning(f"Cache read error ({path}): {e}")
        return None


# -----------------------------------------------------------------------------
# METADATA CACHE — JSON, refreshed weekly
# -----------------------------------------------------------------------------

def load_metadata() -> dict | None:
    """
    Load cached metadata dict {ticker: {name, sector, market_cap}}.
    Returns None if missing or stale.
    """
    if not os.path.exists(METADATA_FILE):
        return None
    try:
        with open(METADATA_FILE) as f:
            data = json.load(f)
        saved_at = datetime.strptime(data["saved_at"], "%Y-%m-%d %H:%M:%S")
        if datetime.now() - saved_at > timedelta(days=METADATA_TTL_DAYS):
            logger.info("Metadata cache stale — will refresh")
            return None
        return data.get("metadata", {})
    except Exception as e:
        logger.warning(f"Metadata cache read error: {e}")
        return None


def save_metadata(meta: dict):
    """Save metadata dict to cache."""
    _save_json(METADATA_FILE, {"metadata": meta, "saved_at": _now_str()})
    logger.info(f"Metadata cached: {len(meta)} tickers")


def update_metadata(new_meta: dict):
    """
    Merge new_meta into existing metadata cache (additive update).
    Useful for adding metadata for newly discovered tickers.
    """
    existing = load_metadata() or {}
    existing.update(new_meta)
    save_metadata(existing)


# -----------------------------------------------------------------------------
# CACHE STATS — useful for debugging / monitoring
# -----------------------------------------------------------------------------

def get_cache_stats() -> dict:
    """Return a summary of what's currently in the cache."""
    stats = {
        "cache_dir":         CACHE_DIR,
        "ohlcv_tickers":     0,
        "benchmark_tickers": 0,
        "us_universe_size":  0,
        "india_universe_size": 0,
        "metadata_size":     0,
        "ohlcv_db_mb":       0,
        "benchmark_db_mb":   0,
    }

    # OHLCV DB stats
    if os.path.exists(OHLCV_DB):
        stats["ohlcv_db_mb"] = round(os.path.getsize(OHLCV_DB) / 1e6, 1)
        conn = sqlite3.connect(OHLCV_DB)
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        stats["ohlcv_tickers"] = cur.fetchone()[0]
        conn.close()

    if os.path.exists(BENCHMARK_DB):
        stats["benchmark_db_mb"] = round(os.path.getsize(BENCHMARK_DB) / 1e6, 1)
        conn = sqlite3.connect(BENCHMARK_DB)
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        stats["benchmark_tickers"] = cur.fetchone()[0]
        conn.close()

    # JSON caches
    us_uni = load_universe("us")
    in_uni = load_universe("india")
    meta   = load_metadata()

    stats["us_universe_size"]    = len(us_uni)  if us_uni  else 0
    stats["india_universe_size"] = len(in_uni)  if in_uni  else 0
    stats["metadata_size"]       = len(meta)    if meta    else 0

    return stats


def clear_cache(confirm: bool = False):
    """
    Wipe all cached data. Requires confirm=True to prevent accidents.
    """
    if not confirm:
        logger.warning("clear_cache() called without confirm=True — skipping")
        return
    import shutil
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    _init_db(OHLCV_DB)
    _init_db(BENCHMARK_DB)
    logger.info("Cache cleared")


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def _safe_table_name(ticker: str) -> str:
    """Convert ticker to a safe SQLite table name."""
    return ticker.replace(".", "_").replace("-", "_").replace("^", "IDX_").upper()


def _save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
