# =============================================================================
# HOLDINGS READER — Load current portfolio positions from Om-Holdings sheet
# =============================================================================
#
# Reads the user's personal portfolio tracker (Om-Holdings Google Sheet) and
# returns a normalized ticker → portfolio mapping used by the Exit Monitor to
# flag stocks the user actually holds.
#
# India holdings come from three tabs:
#   • Niveshaay — externally managed positions
#   • Self       — self-managed India positions
#   • Trading    — shorter-term trading book
#
# US holdings come from:
#   • International — US / international positions
#
# Ticker normalization:
#   Om-Holdings uses "NSE:SANSERA", "BOM:522195", "NSE-SME:VILAS" format.
#   The screener stores display tickers as "SANSERA" (suffix stripped).
#   This module strips the exchange prefix so the two formats match.
# =============================================================================

import logging
import re

logger = logging.getLogger(__name__)

# Om-Holdings sheet ID (portfolio tracker — separate from screener output sheet)
OM_HOLDINGS_SHEET_ID = "1yBmRYPiUmMT9z4bBjQ4aG1up3jIMbxeioyhEdz6PrF0"

# Tab names in Om-Holdings and which market they belong to
INDIA_HOLDING_TABS = ["Niveshaay", "Self", "Trading"]
US_HOLDING_TABS    = ["International"]


def _normalize_ticker(raw: str) -> str:
    """
    Strip exchange prefix from Om-Holdings ticker format.

    Examples:
        "NSE:SANSERA"    → "SANSERA"
        "BOM:522195"     → "522195"
        "NSE-SME:VILAS"  → "VILAS"
        "TXG"            → "TXG"          (US — already plain)
    """
    raw = raw.strip()
    if ":" in raw:
        return raw.split(":")[-1].strip()
    return raw


def load_holdings(client, market: str) -> dict:
    """
    Read Om-Holdings Google Sheet and return holdings for the given market.

    Args:
        client:  authenticated gspread.Client
        market:  "india" or "us"

    Returns:
        dict mapping normalized_ticker (str) → portfolio_info (dict) with keys:
            "portfolio"  : tab name  (e.g. "Self", "Trading", "International")
            "qty"        : quantity held (int)
            "name"       : stock name / display label from the sheet
    """
    try:
        sheet = client.open_by_key(OM_HOLDINGS_SHEET_ID)
    except Exception as e:
        logger.warning(f"Could not open Om-Holdings sheet: {e}")
        return {}

    tabs = INDIA_HOLDING_TABS if market == "india" else US_HOLDING_TABS
    holdings = {}

    for tab_name in tabs:
        try:
            ws   = sheet.worksheet(tab_name)
            rows = ws.get_all_values()
        except Exception as e:
            logger.warning(f"Could not read Om-Holdings tab '{tab_name}': {e}")
            continue

        if len(rows) < 2:
            continue

        headers = [h.strip().lower() for h in rows[0]]

        # Locate required columns flexibly
        tk_col  = _find_col(headers, ["ticker", "symbol"])
        qty_col = _find_col(headers, ["quantity", "qty"])
        nm_col  = _find_col(headers, ["stock", "name"])

        if tk_col is None:
            logger.warning(f"Om-Holdings tab '{tab_name}': no ticker column found")
            continue

        # Locate buy price column — varies by tab
        bp_col = _find_col(headers, ["buy price", "breakeven price", "buyprice", "avg price", "avg cost"])

        # Locate Change% column — the sheet already computes gain/loss for us.
        # Try every common header variant (case-insensitive, already lowercased).
        chg_col = _find_col(headers, [
            "change%", "change %", "chg%", "chg %",
            "gain%", "gain %", "gain/loss%", "gain / loss%",
            "p&l%", "p&l %", "pl%", "pl %",
            "return%", "return %",
            "unrealised%", "unrealised %", "unrealized%", "unrealized %",
        ])

        for row in rows[1:]:
            if len(row) <= tk_col or not row[tk_col].strip():
                continue

            raw_ticker = row[tk_col].strip()
            ticker     = _normalize_ticker(raw_ticker)
            if not ticker:
                continue

            # Parse quantity — treat blank / 0 as not held
            qty = 0
            if qty_col is not None and qty_col < len(row):
                try:
                    qty = int(float(row[qty_col].replace(",", "").strip() or "0"))
                except ValueError:
                    qty = 0
            if qty == 0:
                continue

            name = ""
            if nm_col is not None and nm_col < len(row):
                name = row[nm_col].strip()

            # Parse Change% — primary gain/loss source.
            # Sheet values may be "12.34%", "+12.34", "-5.67 %", etc.
            change_pct = None
            if chg_col is not None and chg_col < len(row):
                raw_val = row[chg_col].replace(",", "").replace("%", "").strip()
                if raw_val:
                    try:
                        change_pct = round(float(raw_val), 2)
                    except ValueError:
                        change_pct = None

            # Parse buy price — fallback when Change% column is absent
            buy_price = 0.0
            if bp_col is not None and bp_col < len(row):
                try:
                    buy_price = float(row[bp_col].replace(",", "").strip() or "0")
                except ValueError:
                    buy_price = 0.0

            holdings[ticker] = {
                "portfolio":  tab_name,
                "qty":        qty,
                "name":       name or ticker,
                "change_pct": change_pct,   # direct from sheet — preferred
                "buy_price":  buy_price,    # fallback if change_pct absent
            }

    logger.info(
        f"Holdings loaded ({market.upper()}): {len(holdings)} positions "
        f"from tabs: {', '.join(tabs)}"
    )
    return holdings


def _find_col(headers: list, candidates: list) -> int | None:
    """Return index of first matching header, case-insensitive."""
    for name in candidates:
        if name in headers:
            return headers.index(name)
    return None
