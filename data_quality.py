"""
data_quality.py — OHLCV data quality scanner.

Runs on the full ohlcv dict every trade scan and flags tickers with bad data
so you know exactly which stocks the screeners are silently skipping.

ISSUES DETECTED:
  NaN Price     — last close is NaN or 0 (yfinance returned no price for latest bar)
  NaN Volume    — last volume is NaN or 0 (no trading data)
  Stale Data    — last bar is > 5 calendar days old (feed not updating)
  Too Few Bars  — fewer than 100 bars total (insufficient for EMA200 / RS 52w)
  Price Spike   — last bar moved > 50% from prior close (split, bad tick, error)
  All NaN       — entire close series is NaN (complete feed failure)

OUTPUT:
  DataFrame with one row per bad ticker, sorted by Issue severity then Ticker.
  Written to the fixed "Data Issues" tab in Google Sheets on every run.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
STALE_DAYS      = 5     # calendar days — last bar older than this = stale
MIN_BARS        = 100   # below this = too few bars for reliable signals
SPIKE_THRESHOLD = 0.50  # 50% single-bar move = likely bad tick or unadjusted split

# Severity order for sorting (lower = shown first)
_SEVERITY = {
    "All NaN":     0,
    "NaN Price":   1,
    "NaN Volume":  2,
    "Stale Data":  3,
    "Price Spike": 4,
    "Too Few Bars":5,
}


def run_data_quality_scan(
    ohlcv:    dict,
    metadata: dict = None,
    market:   str  = "india",
) -> pd.DataFrame:
    """
    Scan the full ohlcv dict and return a DataFrame of tickers with data issues.

    Args:
        ohlcv:    {ticker: OHLCV DataFrame} — the full fetched universe
        metadata: optional {ticker: {name, sector, ...}} for display names
        market:   "india" | "us" | "ai"

    Returns:
        DataFrame sorted by severity then ticker.
        Empty DataFrame if no issues found (rare but possible).
    """
    metadata  = metadata or {}
    today     = date.today()
    rows: list[dict] = []

    for ticker, df in ohlcv.items():
        issues = _check_ticker(ticker, df, today)
        for issue_type, details in issues:
            ticker_display = ticker.replace(".NS", "").replace(".BO", "")
            meta           = metadata.get(ticker, {})
            company        = meta.get("name", ticker_display)
            sector         = meta.get("sector", "Unknown")

            rows.append({
                "Ticker":     ticker_display,
                "Company":    company,
                "Issue":      issue_type,
                "Details":    details,
                "Last Date":  _last_date(df),
                "Last Price": _last_price(df),
                "Last Vol":   _last_vol(df),
                "Bars":       len(df),
                "Sector":     sector,
                "Checked At": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

    if not rows:
        logger.info(f"Data Quality [{market.upper()}]: no issues found in {len(ohlcv)} tickers ✓")
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out["_sev"] = df_out["Issue"].map(_SEVERITY).fillna(99)
    df_out = (
        df_out
        .sort_values(["_sev", "Ticker"])
        .drop(columns=["_sev"])
        .reset_index(drop=True)
    )
    df_out.insert(0, "Rank", range(1, len(df_out) + 1))

    counts = df_out["Issue"].value_counts().to_dict()
    summary = " | ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: _SEVERITY.get(x[0], 99)))
    logger.warning(
        f"Data Quality [{market.upper()}]: {len(df_out)} issues in {len(ohlcv)} tickers — {summary}"
    )
    return df_out


# =============================================================================
# PER-TICKER CHECKS
# =============================================================================

def _check_ticker(
    ticker: str,
    df:     pd.DataFrame,
    today:  date,
) -> list[tuple[str, str]]:
    """
    Run all checks for one ticker.
    Returns list of (issue_type, detail_string) — may be empty or multiple issues.
    """
    issues = []

    if df is None or df.empty:
        issues.append(("All NaN", "DataFrame is empty — fetch returned nothing"))
        return issues

    if "close" not in df.columns:
        issues.append(("All NaN", "No 'close' column in DataFrame"))
        return issues

    close  = df["close"]
    volume = df.get("volume", pd.Series(dtype=float))

    # ── All NaN ───────────────────────────────────────────────────────────────
    if close.isna().all():
        issues.append(("All NaN", f"All {len(close)} close values are NaN — complete feed failure"))
        return issues   # no point checking further

    # ── NaN Price (last bar) ──────────────────────────────────────────────────
    last_close = close.iloc[-1]
    if pd.isna(last_close) or last_close == 0:
        # How many trailing NaN bars?
        trailing = int(close.iloc[::-1].isna().cumprod().sum())
        last_valid_price = close.dropna().iloc[-1] if close.notna().any() else float("nan")
        last_valid_date  = close.dropna().index[-1].date() if close.notna().any() else "—"
        issues.append((
            "NaN Price",
            f"Last {trailing} bar(s) have NaN close. "
            f"Last valid price ₹{last_valid_price:,.2f} on {last_valid_date}"
        ))

    # ── NaN Volume (last bar) ─────────────────────────────────────────────────
    if not volume.empty:
        last_vol = volume.iloc[-1]
        if pd.isna(last_vol) or last_vol == 0:
            issues.append((
                "NaN Volume",
                f"Last bar volume is {'NaN' if pd.isna(last_vol) else '0'} — "
                f"no trade data for {df.index[-1].date() if hasattr(df.index[-1], 'date') else '?'}"
            ))

    # ── Stale Data ────────────────────────────────────────────────────────────
    try:
        last_idx = df.index[-1]
        last_bar_date = last_idx.date() if hasattr(last_idx, "date") else date.fromisoformat(str(last_idx)[:10])
        gap = (today - last_bar_date).days
        if gap > STALE_DAYS:
            issues.append((
                "Stale Data",
                f"Last bar is {gap} calendar days old ({last_bar_date}) — feed may have stopped"
            ))
    except Exception:
        pass

    # ── Too Few Bars ──────────────────────────────────────────────────────────
    if len(close) < MIN_BARS:
        issues.append((
            "Too Few Bars",
            f"Only {len(close)} bars — need {MIN_BARS}+ for EMA200 and RS 52-week signals"
        ))

    # ── Price Spike ───────────────────────────────────────────────────────────
    valid_close = close.dropna()
    if len(valid_close) >= 2:
        prev  = float(valid_close.iloc[-2])
        last  = float(valid_close.iloc[-1])
        if prev > 0 and last > 0:
            move = abs(last - prev) / prev
            if move >= SPIKE_THRESHOLD:
                direction = "UP" if last > prev else "DOWN"
                issues.append((
                    "Price Spike",
                    f"Last bar moved {move*100:.0f}% {direction} "
                    f"(₹{prev:,.2f} → ₹{last:,.2f}) — possible unadjusted split or bad tick"
                ))

    return issues


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _last_date(df: pd.DataFrame) -> str:
    try:
        idx = df.index[-1]
        return str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
    except Exception:
        return "—"


def _last_price(df: pd.DataFrame) -> str:
    try:
        v = df["close"].iloc[-1]
        return f"₹{float(v):,.2f}" if not pd.isna(v) else "NaN"
    except Exception:
        return "—"


def _last_vol(df: pd.DataFrame) -> str:
    try:
        v = df["volume"].iloc[-1]
        if pd.isna(v):    return "NaN"
        v = float(v)
        if v >= 1e7:      return f"₹{v/1e7:.1f}Cr"
        if v >= 1e5:      return f"{v/1e5:.1f}L"
        return f"{v:,.0f}"
    except Exception:
        return "—"
