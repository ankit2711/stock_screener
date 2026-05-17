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
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# BUG FIX: was Path("cache/persistence.json") — a relative path that resolves to
# whatever the shell's CWD is at launch. If the process is started from any directory
# other than the project root, a new empty cache is silently created, losing all
# streak/exit history. Fix: anchor to the module's own directory via __file__.
_CACHE_FILE = Path(__file__).parent / "cache" / "persistence.json"


def get_data_as_of(benchmark) -> str:
    """
    Return the date of the last available trading bar from a benchmark DataFrame.

    WHY THIS EXISTS:
      All streak/first_seen annotations must use the OHLCV data date, NOT
      date.today().  If the script runs on Saturday with Friday's data and again
      on Monday with the same Friday data, using date.today() would see a 2-day
      gap and incorrectly increment the streak.  Using the last bar date means
      both runs produce the same "today" → gap = 0 → streak unchanged.

    Falls back to date.today().isoformat() if the benchmark is unavailable.

    Usage (call once at the top of each run function, then pass data_as_of= to
    annotate_streak_df / annotate_conviction_df / annotate_df / append_screener_exits):

        data_as_of = get_data_as_of(benchmark)
    """
    try:
        if benchmark is not None and hasattr(benchmark, "empty") and not benchmark.empty:
            close = benchmark["close"].dropna()
            if not close.empty:
                last_idx = close.index[-1]
                if hasattr(last_idx, "date"):
                    return last_idx.date().isoformat()
                return str(last_idx)[:10]
    except Exception:
        pass
    return date.today().isoformat()


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
    df:          pd.DataFrame,
    bucket:      str = "conviction",
    data_as_of:  Optional[str] = None,
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
        df:          active conviction DataFrame; must have "Ticker" column.
        bucket:      registry key (default "conviction").
        data_as_of:  last OHLCV bar date (from get_data_as_of(benchmark)).
                     Uses date.today() if not provided.
                     IMPORTANT: pass this so that re-running with the same data
                     on a different calendar day doesn't increment the streak.

    Returns:
        Annotated copy of df.
    """
    if df is None or (hasattr(df, "empty") and df.empty):
        return df
    if "Ticker" not in df.columns:
        return df

    # Use OHLCV data date, not run date — ensures idempotency when the same
    # data is processed more than once (e.g. weekend re-runs, debug reruns).
    today    = data_as_of or date.today().isoformat()
    today_dt = date.fromisoformat(today)
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
                if gap == 0:
                    pass           # same data re-run — streak unchanged
                elif gap <= 3:
                    rec["streak"] = rec["streak"] + 1
                else:
                    rec["streak"] = 1
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
    bucket:      str           = "conviction",
    days:        int           = 30,
    data_as_of:  Optional[str] = None,
) -> list[dict]:
    """
    Return tickers that exited the conviction list within the last `days` calendar days.
    Sorted by exit_date descending (most recent first).

    BUG FIX: added data_as_of parameter so the 30-day exit window is evaluated
    relative to the OHLCV data date rather than wall-clock date.  Using date.today()
    violates idempotency — re-running on the same data from a different calendar day
    could drop exits from the window even though the underlying data hasn't changed.
    Falls back to date.today() when data_as_of is not provided.
    """
    data  = _load()
    state = data.get(bucket, {})
    today = date.fromisoformat(data_as_of) if data_as_of else date.today()
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


def annotate_streak_df(
    df:          pd.DataFrame,
    bucket:      str,
    ticker_col:  str = "Ticker",
    data_as_of:  Optional[str] = None,
) -> pd.DataFrame:
    """
    Annotate df with "Streak" and "First Entry" columns.

    Streak     — consecutive calendar days this ticker has appeared in this
                 screener's output.  Gap ≤ 3 days = consecutive (covers weekends).
                 A stock in Stage Leaders for 15 days straight has been
                 institutionally-confirmed in a Stage-2 uptrend for 3 weeks.

    First Entry — date this ticker FIRST appeared in this screener bucket.
                  Proxy for: first day in Stage 2 / first day in portfolio.

    State stored in cache/persistence.json under the given bucket key.
    Separator / blank rows (ticker starts with "─") are skipped.

    Args:
        df:          screener output DataFrame; must have ticker_col column.
        bucket:      unique key per screener+market, e.g. "streak_stage_india".
                     Use distinct buckets per market so India/US streaks don't mix.
        ticker_col:  column holding ticker identifiers (default "Ticker").
        data_as_of:  last OHLCV bar date (from get_data_as_of(benchmark)).
                     Uses date.today() if not provided.
                     IMPORTANT: pass this so that re-running with the same data
                     on a different calendar day doesn't increment the streak.

    Returns:
        df copy with "Streak" and "First Entry" columns inserted right after
        the ticker_col position (or appended if ticker_col not found).
    """
    if df is None or (hasattr(df, "empty") and df.empty):
        return df
    if ticker_col not in df.columns:
        return df

    # Use OHLCV data date, not run date — ensures idempotency when the same
    # data is processed more than once (e.g. weekend re-runs, debug reruns).
    today    = data_as_of or date.today().isoformat()
    today_dt = date.fromisoformat(today)
    data     = _load()
    state    = data.setdefault(bucket, {})

    # Build the set of active tickers, skipping separator rows
    active = {
        str(t) for t in df[ticker_col]
        if str(t) and not str(t).startswith("─")
    }

    # ── Update active tickers ──────────────────────────────────────────────────
    for ticker in active:
        rec  = state.setdefault(ticker, {"streak": 0, "last_seen": None, "first_seen": today})
        last = rec.get("last_seen")

        # first_seen: stamp once, never overwrite
        if not rec.get("first_seen"):
            rec["first_seen"] = today

        if last is None:
            rec["streak"] = 1
        else:
            try:
                gap = (today_dt - date.fromisoformat(last)).days
                if gap == 0:
                    pass           # same data re-run — streak unchanged
                elif gap <= 3:
                    rec["streak"] = rec["streak"] + 1
                else:
                    rec["streak"] = 1
            except Exception:
                rec["streak"] = 1

        rec["last_seen"] = today

    data[bucket] = state
    _save(data)

    # ── Annotate DataFrame ─────────────────────────────────────────────────────
    df = df.copy()
    streaks      = []
    first_entries = []

    for raw in df[ticker_col]:
        t = str(raw)
        if not t or t.startswith("─"):    # separator / blank row
            streaks.append("")
            first_entries.append("")
        else:
            rec = state.get(t, {})
            streaks.append(rec.get("streak", 1))
            first_entries.append(rec.get("first_seen", today))

    df["Streak"]      = streaks
    df["First Entry"] = first_entries
    return df


def append_screener_exits(
    df:              pd.DataFrame,
    bucket:          str,
    key_col:         str = "Ticker",
    days:            int = 7,
    exit_col:        str = "Exit Date",
    exit_reason:     str = "Left scan",
    exit_reason_col: str = "Exit Reason",
    reentry_pool:    "pd.DataFrame | None" = None,
    data_as_of:      Optional[str] = None,
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
        days:            calendar days to keep exited rows visible (default 7).
                         New exits accumulate within the window; entries older
                         than `days` calendar days are automatically dropped on
                         the next run — no manual cleanup needed.
        exit_col:        column name for exit date (default "Exit Date")
        exit_reason:     short reason stamped when a ticker drops out (default "Left scan")
        exit_reason_col: column name for the reason (default "Exit Reason")
        reentry_pool:    optional full pool DataFrame (e.g. the 80-stock pool when only
                         top-30 are in df). Tickers present in the pool but not in df are
                         still qualifying — their exit_date is cleared and their state is
                         kept fresh, but they are NOT added to the displayed active rows.
                         This prevents the "ranked out of display but still in pool" false-exit
                         where a stock at rank 32 would otherwise be permanently stuck in the
                         exit section after one run outside the top-30 display window.
        data_as_of:      last OHLCV bar date (from get_data_as_of(benchmark)).
                         Uses date.today() if not provided. Pass this for full idempotency:
                         exit_date and last_seen are stamped with the data date, not the
                         run date, so re-running with the same data produces the same state.

    Returns:
        DataFrame — active rows (top) + exited rows (bottom), sorted by exit_date DESC.
    """
    if df is None or (hasattr(df, "empty") and df.empty):
        return df
    if key_col not in df.columns:
        return df

    # Use OHLCV data date, not run date — see get_data_as_of() for rationale.
    today    = data_as_of or date.today().isoformat()
    today_dt = date.fromisoformat(today)
    data     = _load()
    state    = data.setdefault(bucket, {})
    active   = {str(t) for t in df[key_col]}

    # ── Step 1a: update displayed active tickers — save full row ─────────────
    for _, row in df.iterrows():
        ticker   = str(row[key_col])
        row_dict = {k: v for k, v in row.to_dict().items()
                    if k not in (exit_col, exit_reason_col)}
        rec = state.setdefault(ticker, {})
        rec["last_seen"]   = today
        rec["exit_date"]   = None   # still active — clear any prior exit stamp
        rec["exit_reason"] = None   # clear on re-entry
        rec["row"]         = row_dict

    # ── Step 1b: pool-only tickers — still qualifying, just not in display ───
    # When screeners use a pool (e.g. top-80) but only display the top-30,
    # stocks ranked 31–80 are genuinely still Stage-2 / RS-leading / etc.
    # We must clear their exit_date and update last_seen so they aren't
    # wrongly shown in the exit section on subsequent runs.
    # They are NOT appended to the displayed rows — only the display df is shown.
    pool_tickers: set[str] = set()
    if reentry_pool is not None and not reentry_pool.empty and key_col in reentry_pool.columns:
        for _, prow in reentry_pool.iterrows():
            pticker = str(prow[key_col])
            if pticker in active:
                continue   # already handled above
            pool_tickers.add(pticker)
            prow_dict = {k: v for k, v in prow.to_dict().items()
                         if k not in (exit_col, exit_reason_col)}
            rec = state.setdefault(pticker, {})
            was_exited = rec.get("exit_date") is not None
            rec["last_seen"]   = today
            rec["exit_date"]   = None   # re-entered pool — clear exit stamp
            rec["exit_reason"] = None
            rec["row"]         = prow_dict
            if was_exited:
                logger.debug(
                    f"{bucket}: {pticker} re-entered pool (rank > display cutoff) — exit cleared"
                )

    # All tickers considered "active" for exit-detection: display + pool
    all_active = active | pool_tickers

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
        if ticker not in all_active and rec.get("last_seen"):
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

        # Fill any column gaps so concat doesn't fail.
        # For text-label columns (Company, Sector, etc.) that are missing from
        # old saved rows (schema changed), fall back to the ticker rather than
        # "—" so the row is still identifiable without a company lookup.
        for col in df.columns:
            if col not in saved_row:
                if col == "Company":
                    # Ticker is always meaningful; "—" is not
                    saved_row[col] = ticker
                else:
                    saved_row[col] = "—"

        exited_rows.append(saved_row)

    if not exited_rows:
        return df

    # Sort exited rows: most recent exits first.
    # The 7-day window is the natural cap — entries older than `days` calendar
    # days are dropped above, so this list only contains genuinely recent exits.
    exited_rows.sort(key=lambda r: r.get(exit_col, ""), reverse=True)

    # ── Visual separator between live section and exited section ──────────────
    separator = {col: "" for col in df.columns}
    separator[key_col] = f"─── Exited — last {days} days ───"

    df_sep    = pd.DataFrame([separator], columns=df.columns)
    df_exited = pd.DataFrame(exited_rows, columns=df.columns)
    return pd.concat([df, df_sep, df_exited], ignore_index=True)
