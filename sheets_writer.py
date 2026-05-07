# =============================================================================
# SHEETS WRITER — Push screener results to Google Sheets
# Uses gspread with a service account JSON key.
#
# SETUP (one-time):
#   1. Go to https://console.cloud.google.com/
#   2. Create a project → Enable "Google Sheets API" + "Google Drive API"
#   3. Create a Service Account → Download JSON key → save as credentials.json
#   4. Create TWO Google Sheets:
#        • One for India results → share with service account (Editor)
#        • One for US results    → share with service account (Editor)
#   5. Paste both Sheet IDs into config.py:
#        GOOGLE_SHEET_ID_INDIA = "..."
#        GOOGLE_SHEET_ID_US    = "..."
#
# SHEET STRUCTURE (per market):
#   • One tab per run date, e.g. "2026-03-29", accumulating history.
#   • "Run Log" tab appended each run (never cleared).
#   • Oldest date tabs are pruned once MAX_HISTORY_TABS is exceeded.
# =============================================================================

import logging
import gspread
import pandas as pd
from datetime import datetime
from google.oauth2.service_account import Credentials
from config import (
    GOOGLE_SHEET_ID_INDIA,
    GOOGLE_SHEET_ID_US,
    GOOGLE_SHEET_ID_AI,
    GOOGLE_SHEETS_CREDENTIALS_FILE,
    SHEET_TABS,
    OUTPUT_COLUMNS,
    OUTPUT_COLUMNS_SEPA,
    OUTPUT_COLUMNS_TRADE,
    OUTPUT_COLUMNS_TRADE_STAGE,
    OUTPUT_COLUMNS_TRADE_SEPA,
    OUTPUT_COLUMNS_RS,
    OUTPUT_COLUMNS_HOLDINGS_ALERT,
    OUTPUT_COLUMNS_CONVICTION,
    OUTPUT_COLUMNS_DATA_QUALITY,
    MAX_HISTORY_TABS,
)

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


# -----------------------------------------------------------------------------
# TRADINGVIEW URL HELPERS
# -----------------------------------------------------------------------------

def _tv_url(ticker: str, market: str) -> str:
    """Return a TradingView chart URL for the given ticker."""
    if market == "india":
        if ticker.endswith(".NS"):
            sym = "NSE:" + ticker[:-3]
        elif ticker.endswith(".BO"):
            sym = "BSE:" + ticker[:-3]
        else:
            sym = "NSE:" + ticker
    else:
        # US: plain symbol; TradingView resolves exchange automatically
        sym = ticker
    return f"https://www.tradingview.com/chart/?symbol={sym}"


def _ticker_hyperlink(ticker: str, market: str) -> str:
    """Return a Google Sheets HYPERLINK formula: ticker name as text, TV chart as URL."""
    url = _tv_url(ticker, market)
    return f'=HYPERLINK("{url}", "{ticker}")'


# -----------------------------------------------------------------------------
# CLIENT
# -----------------------------------------------------------------------------

def get_client() -> gspread.Client:
    creds = Credentials.from_service_account_file(
        GOOGLE_SHEETS_CREDENTIALS_FILE, scopes=SCOPES
    )
    return gspread.authorize(creds)


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------

def write_results(df: pd.DataFrame, market: str, screener: str = "stage") -> None:
    """
    Write screener results to the appropriate Google Sheet.

    Args:
        df:       Screener results DataFrame (top-N rows).
        market:   "india" or "us"
        screener: "stage" | "sepa" | "trade"
    """
    if market == "india":
        sheet_id = GOOGLE_SHEET_ID_INDIA
        sheet_key = "GOOGLE_SHEET_ID_INDIA"
    elif market == "ai":
        sheet_id  = GOOGLE_SHEET_ID_AI
        sheet_key = "GOOGLE_SHEET_ID_AI"
    else:
        sheet_id  = GOOGLE_SHEET_ID_US
        sheet_key = "GOOGLE_SHEET_ID_US"

    if not sheet_id:
        logger.error(
            f"No Google Sheet ID configured for market='{market}'. "
            f"Set {sheet_key} in config.py."
        )
        return

    try:
        client = get_client()
        sheet  = client.open_by_key(sheet_id)
        logger.info(f"Connected to Google Sheets ({market.upper()})")
    except Exception as e:
        logger.error(f"Google Sheets connection failed ({market}): {e}")
        raise

    today = datetime.now().strftime("%Y-%m-%d")

    if screener == "trade":
        # Trade mode returns a dict with DataFrames — write each to its own tab.
        # df is either the dict from run_trade_scan, or a plain DataFrame (legacy).
        if isinstance(df, dict):
            stage_df          = df.get("stage",          pd.DataFrame())
            sepa_df           = df.get("sepa",           pd.DataFrame())
            rs_df             = df.get("rs",             pd.DataFrame())
            trade_df          = df.get("trade",          pd.DataFrame())
            holdings_alert_df = df.get("holdings_alert", pd.DataFrame())
            conviction_df     = df.get("conviction",     pd.DataFrame())
            data_quality_df   = df.get("data_quality",   pd.DataFrame())
            sector_res        = df.get("sectors",        {})
        else:
            # Legacy: single DataFrame passed — write to trade tab only
            stage_df = sepa_df = rs_df = holdings_alert_df = pd.DataFrame()
            trade_df  = df
            sector_res = {}

        # Stage / SEPA / RS: fixed tabs only — overwritten each run, always current.
        _write_tab(sheet, SHEET_TABS["stage_trade"], stage_df, market, OUTPUT_COLUMNS_TRADE_STAGE)
        _write_tab(sheet, SHEET_TABS["sepa_trade"],  sepa_df,  market, OUTPUT_COLUMNS_TRADE_SEPA)
        _write_tab(sheet, SHEET_TABS["rs_trade"],    rs_df,    market, OUTPUT_COLUMNS_RS)

        # Trade Candidates: fixed tab only — exited stocks stay at the bottom for 14 days,
        # so there is no need for a separate dated archive tab.
        _write_tab(sheet, SHEET_TABS["trade"], trade_df, market, OUTPUT_COLUMNS_TRADE)

        # Daily BUY Conviction: stocks confirmed by 2+ screeners simultaneously.
        # Sorted by Streak DESC → consecutive days = institutional persistence signal.
        # This is the primary daily BUY watchlist tab — open this first every morning.
        _write_tab(sheet, SHEET_TABS["conviction"], conviction_df, market, OUTPUT_COLUMNS_CONVICTION)

        # Data Issues: tickers with NaN price/volume, stale feeds, or price spikes.
        # Shows exactly which stocks the screeners are silently skipping and why.
        _write_tab(sheet, SHEET_TABS["data_quality"], data_quality_df, market, OUTPUT_COLUMNS_DATA_QUALITY)

        # Holdings Alert: ONLY held positions sorted by TheWrap urgency.
        # Open this tab first every morning before checking your broker.
        _write_tab(sheet, SHEET_TABS["holdings_alert"], holdings_alert_df, market,
                   OUTPUT_COLUMNS_HOLDINGS_ALERT)

        # Sector Rotation: fixed tab — flat ranked list.
        # Always written (even if sector scan returned nothing) so the tab exists.
        try:
            from ranker_sector import sector_results_to_df
            sector_df = sector_results_to_df(sector_res) if sector_res else pd.DataFrame()
            _write_tab(sheet, SHEET_TABS["sectors"], sector_df, market, col_list=None)
        except Exception as _se:
            logger.warning(f"Sector Rotation tab write failed: {_se}")

        # Sector Overview: fixed tab — sectors grouped into Leading → Lagging buckets.
        # Always written so the tab exists even when sector data is unavailable.
        _write_sector_buckets_tab(sheet, SHEET_TABS["sector_overview"], sector_res, market)

    elif screener == "rs":
        # Standalone RS scan — fixed tab only (consistent with above)
        _write_tab(sheet, SHEET_TABS["rs_trade"], df, market, OUTPUT_COLUMNS_RS)

    else:
        tab_label = f"{today}-{screener}"
        col_list  = OUTPUT_COLUMNS_SEPA if screener == "sepa" else OUTPUT_COLUMNS
        _write_tab(sheet, tab_label, df, market, col_list)

    _prune_old_tabs(sheet)
    _write_log(sheet, df, market, screener)
    logger.info(f"Google Sheets ({market.upper()}/{screener.upper()}) updated")


# -----------------------------------------------------------------------------
# INTERNAL HELPERS
# -----------------------------------------------------------------------------

def _write_sector_buckets_tab(
    sheet,
    tab_name:      str,
    sector_results: dict,
    market:        str,
) -> None:
    """
    Write the bucketed sector overview tab.

    Layout: LEADING → IMPROVING → NEUTRAL → WEAKENING → LAGGING.
    Each bucket has a bold shaded header, a column-header row, and data rows.
    Bucket header rows and column header rows are formatted bold + light grey.
    """
    try:
        from ranker_sector import sector_results_to_bucketed_rows

        ws = _get_or_create_worksheet(sheet, tab_name)
        ws.clear()

        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        if not sector_results:
            ws.update("A1", [[
                f"Sector scan returned no data — {now}",
                "Check screener.log for details. "
                "Ensure sector indices (^CNXIT, ^NSEBANK …) are reachable via yfinance."
            ]])
            logger.warning(f"Sector Overview tab '{tab_name}': no sector data to display")
            return

        rows, bucket_hdr_rows, col_hdr_rows = sector_results_to_bucketed_rows(
            sector_results, market=market, as_of=now,
        )

        if not rows:
            ws.update("A1", [[f"No sector data — {now}"]])
            return

        # Write all rows in one call
        ws.update("A1", rows, value_input_option="USER_ENTERED")

        # ── Optional formatting (best-effort — gspread ≥ 5.x) ────────────────
        # Row 1 (summary) bold + dark background
        # Bucket label rows: bold + light grey fill
        # Column header rows: bold
        try:
            num_cols = 14  # matches len(_BUCKET_COLS)
            last_col_letter = chr(ord("A") + num_cols - 1)  # "N"

            def _fmt(row_idx: int, bold: bool, bg: dict | None = None):
                fmt: dict = {"textFormat": {"bold": bold}}
                if bg:
                    fmt["backgroundColor"] = bg
                ws.format(f"A{row_idx}:{last_col_letter}{row_idx}", fmt)

            # Summary row
            _fmt(1, bold=True, bg={"red": 0.20, "green": 0.20, "blue": 0.20})
            ws.format("A1:A1", {
                "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
                "backgroundColor": {"red": 0.20, "green": 0.20, "blue": 0.20},
            })

            # Bucket header rows — bold + light silver background
            for r_idx in bucket_hdr_rows:
                ws.format(f"A{r_idx}:{last_col_letter}{r_idx}", {
                    "textFormat": {"bold": True, "fontSize": 10},
                    "backgroundColor": {"red": 0.88, "green": 0.88, "blue": 0.88},
                })

            # Column header rows — bold
            for r_idx in col_hdr_rows:
                ws.format(f"A{r_idx}:{last_col_letter}{r_idx}", {
                    "textFormat": {"bold": True},
                    "backgroundColor": {"red": 0.95, "green": 0.95, "blue": 0.95},
                })

        except Exception:
            pass  # formatting is optional — data is already written

        logger.info(
            f"Sector Overview tab '{tab_name}' written "
            f"({len(sector_results)} sectors, {len(rows)} rows)"
        )

    except Exception as e:
        logger.warning(f"Sector Overview tab write failed: {e}")


def _write_tab(
    sheet,
    tab_name:  str,
    df:        pd.DataFrame,
    market:    str,
    col_list:  list = None,
) -> None:
    """Create (or overwrite) a date tab with the screener results."""
    if col_list is None:
        col_list = OUTPUT_COLUMNS

    try:
        ws = _get_or_create_worksheet(sheet, tab_name)
        ws.clear()

        label = market.upper()

        if df.empty:
            ws.update(
                "A1",
                [[f"No results for {label} — {datetime.now().strftime('%Y-%m-%d %H:%M')}"]],
            )
            logger.warning(f"{label}: empty results — sheet tab cleared")
            return

        # Build output in column order, only keep columns that exist in df
        cols_to_write = [c for c in col_list if c in df.columns]
        out = df[cols_to_write].copy()

        # ── Ticker column → clickable hyperlink pointing to TradingView ─────
        # The ranker already built the correct TV URL in the TradingView column,
        # so we reuse it rather than re-derive it from the (already-stripped) ticker.
        if "Ticker" in out.columns and "TradingView" in df.columns:
            tv_urls = df["TradingView"].values  # original (before slicing to cols_to_write)
            out["Ticker"] = [
                f'=HYPERLINK("{url}", "{ticker}")' if pd.notna(url) and url else ticker
                for ticker, url in zip(out["Ticker"], tv_urls)
            ]
        elif "Ticker" in out.columns:
            # Fallback: rebuild URL from ticker + market if TradingView col absent
            out["Ticker"] = out["Ticker"].apply(
                lambda t: _ticker_hyperlink(str(t), market)
            )

        # ── TradingView column → "Chart" link text ───────────────────────────
        if "TradingView" in out.columns:
            out["TradingView"] = out["TradingView"].apply(
                lambda url: f'=HYPERLINK("{url}", "Chart")' if pd.notna(url) and url else ""
            )

        # Header + data rows
        header   = [cols_to_write]
        data     = out.values.tolist()
        all_rows = header + data

        ws.update(all_rows, value_input_option="USER_ENTERED")

        # Format header row bold
        ws.format("1:1", {"textFormat": {"bold": True}})

        # Freeze header row
        sheet.batch_update({
            "requests": [{
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": ws.id,
                        "gridProperties": {"frozenRowCount": 1},
                    },
                    "fields": "gridProperties.frozenRowCount",
                }
            }]
        })

        logger.info(f"{label}: wrote {len(df)} rows to tab '{tab_name}'")

    except Exception as e:
        logger.error(f"Failed to write {market} tab '{tab_name}': {e}")
        raise


def _prune_old_tabs(sheet) -> None:
    """
    Remove the oldest date-named tabs if there are more than MAX_HISTORY_TABS.
    Only date-format tabs ("YYYY-MM-DD") are considered; Run Log is never pruned.
    """
    if MAX_HISTORY_TABS <= 0:
        return

    all_ws = sheet.worksheets()
    date_tabs = []
    for ws in all_ws:
        try:
            datetime.strptime(ws.title, "%Y-%m-%d")
            date_tabs.append(ws)
        except ValueError:
            pass  # not a date tab — skip

    # Sort oldest first
    date_tabs.sort(key=lambda ws: ws.title)

    excess = len(date_tabs) - MAX_HISTORY_TABS
    if excess > 0:
        for ws in date_tabs[:excess]:
            try:
                sheet.del_worksheet(ws)
                logger.info(f"Pruned old tab: '{ws.title}'")
            except Exception as e:
                logger.warning(f"Could not prune tab '{ws.title}': {e}")


def _write_log(sheet, df: pd.DataFrame, market: str, screener: str = "stage") -> None:
    """Append one row to the Run Log tab (never cleared).

    Uses explicit row targeting (column A scan) rather than gspread's
    append_row(), which can misplace entries if the sheet has content
    in non-log columns (e.g. a reference table pasted alongside the log).
    """
    try:
        ws  = _get_or_create_worksheet(sheet, SHEET_TABS["log"])
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Read only column A to find the true last log row — ignores any
        # content pasted in columns B+ or J+ that confuses append_row.
        col_a = ws.col_values(1)  # list of strings, one per row
        # Strip trailing empty strings
        last_used = len(col_a)
        while last_used > 0 and not col_a[last_used - 1].strip():
            last_used -= 1

        # Write header on row 1 if missing or sheet was freshly created
        if last_used == 0:
            ws.update("A1", [["Timestamp", "Market", "Screener", "Results", "Tab", "Status"]])
            last_used = 1

        # Append the new log row immediately after the last used row
        next_row = last_used + 1
        today = datetime.now().strftime("%Y-%m-%d")
        ws.update(
            f"A{next_row}",
            [[
                now,
                market.upper(),
                screener.upper(),
                len(df) if not df.empty else 0,
                f"{today}-{screener}",
                "✓ Success",
            ]],
        )
    except Exception as e:
        logger.warning(f"Log write failed (non-critical): {e}")


def _get_or_create_worksheet(sheet, name: str) -> gspread.Worksheet:
    """Return existing worksheet by name, or create it."""
    try:
        return sheet.worksheet(name)
    except gspread.WorksheetNotFound:
        return sheet.add_worksheet(title=name, rows=500, cols=30)
