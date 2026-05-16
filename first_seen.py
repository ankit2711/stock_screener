"""
first_seen.py — Persistent "First Reported" tracker for every screener.

Each time a screener returns results, this module:
  1. Loads the registry from cache/first_seen.json
  2. For every ticker in the results:
       • If new → stamps today's date (never overwritten on future runs)
       • If seen before → keeps the original date
  3. Saves the updated registry
  4. Adds two columns to the DataFrame:
       "First Entry"  — YYYY-MM-DD date string (first time this stock appeared)
       "Days Listed"  — integer: today − first_entry (how long on the list)

Usage (called at the end of each screener):
    from first_seen import annotate_df
    return annotate_df(df_out, screener="stage")

Screener keys: "stage" | "sepa" | "rs" | "trade"

Storage: cache/first_seen.json
    {
      "stage": {"TCS.NS": "2026-04-10", "HDFCBANK.NS": "2026-04-15", ...},
      "sepa":  {"TCS.NS": "2026-04-08", ...},
      "rs":    {"RELIANCE.NS": "2026-04-01", ...},
      "trade": {"TCS.NS": "2026-04-10", ...}
    }
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# BUG FIX: was Path("cache/first_seen.json") — relative path resolves from CWD.
# Running from any directory other than the project root silently loses all first-seen
# history. Anchor to module directory so the path is always correct regardless of CWD.
_CACHE_FILE = Path(__file__).parent / "cache" / "first_seen.json"


# =============================================================================
# INTERNAL REGISTRY HELPERS
# =============================================================================

def _load() -> dict:
    """Load registry from disk. Returns {} if file is missing or corrupt."""
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"first_seen: could not load registry ({e}) — starting fresh")
    return {}


def _save(registry: dict) -> None:
    """Atomic write: write to .tmp then rename so no corrupt state on crash."""
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(
            json.dumps(registry, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(_CACHE_FILE)
    except Exception as e:
        logger.warning(f"first_seen: could not save registry ({e})")


def _clean_ticker(raw: str) -> str:
    """
    Strip any Google Sheets HYPERLINK formula that sheets_writer may have
    embedded in the Ticker cell value.
    =HYPERLINK("https://...","HDFCBANK.NS") → "HDFCBANK.NS"
    """
    if raw.startswith("=HYPERLINK"):
        m = re.search(r'",\s*"([^"]+)"\s*\)', raw)
        if m:
            return m.group(1)
    return raw.strip()


# =============================================================================
# PUBLIC API
# =============================================================================

def annotate_df(
    df:          pd.DataFrame,
    screener:    str,
    data_as_of:  "str | None" = None,
) -> pd.DataFrame:
    """
    Stamp every ticker in *df* with its first-seen date for *screener*,
    then add "First Entry" and "Days Listed" columns.

    Rules:
    - First appearance → data_as_of date saved to registry (never overwritten).
    - Subsequent appearances → original date kept.
    - If Ticker column is absent or df is empty → returned unchanged.

    Args:
        df:          screener result DataFrame; must have a "Ticker" column.
        screener:    registry key — "stage", "sepa", "rs", or "trade".
        data_as_of:  last OHLCV bar date string (YYYY-MM-DD).
                     Uses date.today() if not provided.
                     Pass this for idempotency — first_seen and Days Listed
                     are anchored to the data date, not the run date, so
                     re-running with the same data produces identical columns.

    Returns:
        A copy of df with two new columns appended.
    """
    if df is None or (hasattr(df, "empty") and df.empty):
        return df
    if "Ticker" not in df.columns:
        return df

    # Use OHLCV data date, not run date — ensures identical output on re-runs.
    today    = data_as_of or date.today().isoformat()
    today_dt = date.fromisoformat(today)
    registry = _load()
    bucket   = registry.setdefault(screener, {})

    updated        = False
    first_reported = []
    days_listed    = []

    for raw_ticker in df["Ticker"]:
        ticker = _clean_ticker(str(raw_ticker))

        if ticker not in bucket:
            bucket[ticker] = today
            updated = True

        first_date_str = bucket[ticker]
        first_reported.append(first_date_str)

        try:
            first_dt = date.fromisoformat(first_date_str)
            days     = (today_dt - first_dt).days
        except Exception:
            days = 0
        days_listed.append(days)

    if updated:
        registry[screener] = bucket
        try:
            _save(registry)
            logger.debug(
                f"first_seen [{screener}]: {sum(d == today for d in first_reported)} "
                f"new tickers stamped today"
            )
        except Exception as e:
            logger.warning(f"first_seen: save failed ({e})")

    df = df.copy()
    df["First Entry"] = first_reported   # renamed from "First Reported" — consistent with Streak tab
    df["Days Listed"] = days_listed
    return df


def get_first_seen(ticker: str, screener: str) -> Optional[str]:
    """
    Return the first-seen date string for a single ticker, or None if unknown.
    Useful for point lookups without modifying any DataFrame.
    """
    registry = _load()
    return registry.get(screener, {}).get(_clean_ticker(ticker))


def get_registry_summary() -> dict:
    """Return {screener: count} for quick health-check logging."""
    registry = _load()
    return {k: len(v) for k, v in registry.items()}
