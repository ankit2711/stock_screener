"""
persistence.py — Streak tracker for the Daily BUY Conviction screener.
=======================================================================

Tracks consecutive days a stock appears in the conviction top-20.

STORAGE: cache/persistence.json
    {
      "conviction": {
        "TICKER": {
          "streak":      5,           # consecutive days in conviction list (active run)
          "max_streak":  8,           # all-time longest streak
          "first_seen":  "2026-01-15",
          "last_seen":   "2026-05-06",
          "exit_date":   null,         # date streak broke (null = still active)
          "company":     "...",        # stored so exited rows can still show names
          "sector":      "...",
          "last_price":  "₹1,234.56"  # last price seen (for exited row display)
        }
      }
    }

STREAK RULES:
  Gap ≤ 3 calendar days = consecutive (covers Fri → Mon weekends).
  Gap > 3 days = streak resets to 1 (genuine miss).
  When stock drops out: exit_date = last_seen (stamped on next run).
  When stock re-enters: exit_date cleared, streak restarts.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_FILE = Path("cache/persistence.json")


def _load() -> dict:
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"persistence: could not load ({e}) — starting fresh")
    return {}


def _save(data: dict) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(
            json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(_CACHE_FILE)
    except Exception as e:
        logger.warning(f"persistence: could not save ({e})")


# =============================================================================
# PUBLIC API
# =============================================================================

def annotate_conviction_df(
    df:     pd.DataFrame,
    bucket: str = "conviction",
) -> pd.DataFrame:
    """
    Update streak state for every active ticker (those in df), mark exit_date
    for tickers that just dropped out, and annotate df with tracking columns.

    Columns added to df:
      Streak      — consecutive days in conviction list
      First Seen  — first date ever in conviction list
      Days Here   — calendar days since first_seen (active) or first→exit (exited)
      Left On     — blank while active; last-seen date when streak breaks

    Args:
        df:     active conviction DataFrame; must have "Ticker" column.
        bucket: registry key (default "conviction").

    Returns:
        Annotated copy of df.
    """
    if df is None or (hasattr(df, "empty") and df.empty):
        return df
    if "Ticker" not in df.columns:
        return df

    today    = date.today().isoformat()
    today_dt = date.today()
    data     = _load()
    state    = data.setdefault(bucket, {})
    active   = {str(t) for t in df["Ticker"]}

    # ── Step 1: update active tickers ─────────────────────────────────────────
    for _, row in df.iterrows():
        ticker  = str(row["Ticker"])
        company = str(row.get("Company", ticker))
        sector  = str(row.get("Sector",  "Unknown"))
        price   = str(row.get("Price ₹", "—"))

        rec = state.setdefault(ticker, {
            "streak": 0, "max_streak": 0,
            "first_seen": today, "last_seen": None, "exit_date": None,
            "company": company, "sector": sector, "last_price": price,
        })

        last = rec.get("last_seen")
        if last is None:
            rec["streak"]     = 1
            rec["first_seen"] = today
        else:
            try:
                gap = (today_dt - date.fromisoformat(last)).days
                rec["streak"] = rec["streak"] + 1 if gap <= 3 else 1
            except Exception:
                rec["streak"] = 1

        rec["max_streak"] = max(rec.get("max_streak", 0), rec["streak"])
        rec["last_seen"]  = today
        rec["exit_date"]  = None       # still active — clear any prior exit stamp
        rec["company"]    = company    # refresh in case metadata changed
        rec["sector"]     = sector
        rec["last_price"] = price

    # ── Step 2: stamp exit_date for tickers that just dropped out ─────────────
    for ticker, rec in state.items():
        if ticker not in active:
            if rec.get("last_seen") and rec.get("exit_date") is None:
                rec["exit_date"] = rec["last_seen"]

    data[bucket] = state
    _save(data)

    # ── Step 3: annotate DataFrame ────────────────────────────────────────────
    df = df.copy()
    streaks     = []
    first_seens = []
    days_here   = []
    left_ons    = []

    for raw in df["Ticker"]:
        t   = str(raw)
        rec = state.get(t, {})
        streak  = rec.get("streak", 1)
        fs      = rec.get("first_seen", today)
        exit_d  = rec.get("exit_date") or ""

        streaks.append(streak)
        first_seens.append(fs)
        left_ons.append(exit_d)

        try:
            dh = (today_dt - date.fromisoformat(fs)).days
        except Exception:
            dh = 0
        days_here.append(dh)

    df["Streak"]     = streaks
    df["First Seen"] = first_seens
    df["Days Here"]  = days_here
    df["Left On"]    = left_ons   # blank while active, date when dropped out

    return df


def get_recent_exits(
    bucket: str = "conviction",
    days:   int = 30,
) -> list[dict]:
    """
    Return tickers that exited the conviction list within the last `days` calendar days.
    Sorted by exit_date descending (most recent first).
    """
    data  = _load()
    state = data.get(bucket, {})
    today = date.today()
    exits = []

    for ticker, rec in state.items():
        ed = rec.get("exit_date")
        if not ed:
            continue
        try:
            ed_dt = date.fromisoformat(ed)
            if (today - ed_dt).days <= days:
                fs = rec.get("first_seen", ed)
                try:
                    dh = (ed_dt - date.fromisoformat(fs)).days
                except Exception:
                    dh = 0
                exits.append({
                    "ticker":      ticker,
                    "streak":      rec.get("streak", 0),
                    "first_seen":  fs,
                    "days_here":   dh,
                    "last_seen":   rec.get("last_seen", ed),
                    "exit_date":   ed,
                    "company":     rec.get("company", ticker),
                    "sector":      rec.get("sector",  "Unknown"),
                    "last_price":  rec.get("last_price", "—"),
                })
        except Exception:
            continue

    exits.sort(key=lambda x: x["exit_date"], reverse=True)
    return exits


def get_all_active(bucket: str = "conviction") -> dict:
    """Return {ticker: rec} for all tickers currently active (exit_date is None)."""
    data  = _load()
    state = data.get(bucket, {})
    return {t: r for t, r in state.items() if r.get("exit_date") is None}


def append_screener_exits(
    df:              pd.DataFrame,
    bucket:          str,
    key_col:         str = "Ticker",
    days:            int = 14,
    exit_col:        str = "Exit Date",
    exit_reason:     str = "Left scan",
    exit_reason_col: str = "Exit Reason",
) -> pd.DataFrame:
    """
    Generic exit tracker for any screener tab.

    Tracks which tickers are in `df` each run, stamps an exit_date when they
    drop out, and appends the exited rows (last `days` calendar days) at the
    bottom of the returned DataFrame.

    STATE STORED (in cache/persistence.json under `bucket`):
      {ticker: {last_seen, exit_date, exit_reason, row: {col: val}}}

    The full row from the last active appearance is saved so exited rows
    still display their scores/signals from when they last qualified.

    ACTIVE rows:  unchanged. exit_col and exit_reason_col added as blank strings.
    EXITED rows:  appended at bottom; exit_col shows exit_date; Rank = "—";
                  exit_reason_col shows the reason stamped at time of exit.

    Args:
        df:              current screener output DataFrame (must have key_col column)
        bucket:          unique key per screener+market  e.g. "stage_india", "trade_us"
        key_col:         column holding ticker identifiers (default "Ticker")
        days:            calendar days to keep exited rows visible (default 14)
        exit_col:        column name for exit date (default "Exit Date")
        exit_reason:     short reason stamped when a ticker drops out (default "Left scan")
        exit_reason_col: column name for the reason (default "Exit Reason")

    Returns:
        DataFrame — active rows (top) + exited rows (bottom), sorted by exit_date DESC.
    """
    if df is None or (hasattr(df, "empty") and df.empty):
        return df
    if key_col not in df.columns:
        return df

    today    = date.today().isoformat()
    today_dt = date.today()
    data     = _load()
    state    = data.setdefault(bucket, {})
    active   = {str(t) for t in df[key_col]}

    # ── Step 1: update active tickers — save full row ─────────────────────────
    for _, row in df.iterrows():
        ticker   = str(row[key_col])
        row_dict = {k: v for k, v in row.to_dict().items()
                    if k not in (exit_col, exit_reason_col)}
        rec = state.setdefault(ticker, {})
        rec["last_seen"]   = today
        rec["exit_date"]   = None   # still active — clear any prior exit stamp
        rec["exit_reason"] = None   # clear on re-entry
        rec["row"]         = row_dict

    # ── Step 2: stamp exit_date + reason for tickers that just dropped out ────
    # exit_reason can be a plain string OR callable(ticker, saved_row) -> str
    # so callers can pass OHLCV-based diagnostic logic per screener.
    #
    # Backfill: if a ticker was already exited (exit_date set) but has no
    # reason stored yet (exited before this feature was deployed), compute and
    # persist the reason now so it shows on this run rather than staying blank.
    def _compute_reason(t: str, row: dict) -> str:
        if callable(exit_reason):
            try:
                return exit_reason(t, row)
            except Exception as _re:
                logger.debug(f"exit_reason_fn failed for {t}: {_re}")
                return "—"
        return str(exit_reason)

    for ticker, rec in state.items():
        if ticker not in active and rec.get("last_seen"):
            if rec.get("exit_date") is None:
                # First exit this run — stamp date and reason
                rec["exit_date"]   = rec["last_seen"]
                rec["exit_reason"] = _compute_reason(ticker, rec.get("row", {}))
            elif not rec.get("exit_reason"):
                # Already exited on a prior run but reason was never stored
                # (feature was added after the exit) — backfill now
                rec["exit_reason"] = _compute_reason(ticker, rec.get("row", {}))

    data[bucket] = state
    _save(data)

    # ── Step 3: add exit_col + exit_reason_col to active rows (blank = active) ─
    df = df.copy()
    df[exit_col]        = ""
    df[exit_reason_col] = ""

    # ── Step 4: build exited rows from saved state ────────────────────────────
    exited_rows = []
    for ticker, rec in state.items():
        ed = rec.get("exit_date")
        if not ed:
            continue                 # still active
        if ticker in active:
            continue                 # re-entered this run — already in active rows
        try:
            ed_dt = date.fromisoformat(ed)
            if (today_dt - ed_dt).days > days:
                continue             # too old — past the window
        except Exception:
            continue

        saved_row = dict(rec.get("row", {}))
        saved_row[exit_col]         = ed
        saved_row[exit_reason_col]  = rec.get("exit_reason") or "—"
        saved_row[key_col]          = ticker      # ensure ticker is present

        # Stale rank is misleading — blank it out for exited rows
        if "Rank" in saved_row:
            saved_row["Rank"] = "—"

        # Fill any column gaps so concat doesn't fail
        for col in df.columns:
            if col not in saved_row:
                saved_row[col] = "—"

        exited_rows.append(saved_row)

    if not exited_rows:
        return df

    # Sort exited rows: most recent exits first
    exited_rows.sort(key=lambda r: r.get(exit_col, ""), reverse=True)

    # ── Visual separator between live section and exited section ──────────────
    separator = {col: "" for col in df.columns}
    separator[key_col] = "─── Exited — last 14 days ───"

    df_sep    = pd.DataFrame([separator], columns=df.columns)
    df_exited = pd.DataFrame(exited_rows, columns=df.columns)
    return pd.concat([df, df_sep, df_exited], ignore_index=True)
