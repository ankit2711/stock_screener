# =============================================================================
# UNIVERSE — Fetch all eligible tickers for US and India
# Applies market cap filters from config
#
# INDIA UNIVERSE STRATEGY:
#   1. Fetch NSE equity list  → symbols with .NS suffix (primary)
#   2. Fetch BSE equity list  → symbols with .BO suffix
#   3. Deduplicate by symbol: if a stock is on both NSE and BSE, keep .NS only
#      Reason: NSE has better liquidity, tighter spreads, and more complete
#      yfinance data. .BO is only used for genuinely BSE-only stocks.
#   4. Apply ₹500Cr+ market cap filter to the combined list
#
#   Deduplication method: ISIN matching (primary) + symbol matching (fallback)
#   ISIN is the definitive company identifier — same company on both exchanges
#   will have the same 12-character ISIN. Symbol matching catches cases where
#   the NSE CSV does not include ISIN data.
# =============================================================================

import logging
import requests
import pandas as pd
import yfinance as yf
from config import US_MIN_MARKET_CAP_USD, INDIA_MIN_MARKET_CAP_INR

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# US UNIVERSE — All NYSE + NASDAQ, filtered by market cap
# -----------------------------------------------------------------------------

def get_us_tickers() -> list[str]:
    """
    Fetch all NYSE + NASDAQ tickers and filter by $2B+ market cap.
    Uses the free SEC/NASDAQ listings as source.
    Returns a list of ticker strings e.g. ['AAPL', 'MSFT', ...]
    """
    import cache as cache_module
    cached = cache_module.load_universe("us")
    if cached:
        logger.info(f"US universe from cache: {len(cached)} tickers")
        return cached
    logger.info("Fetching US ticker universe (cache miss)...")
    tickers = set()

    # NASDAQ listed
    try:
        url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange=nasdaq"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        data = resp.json()
        rows = data["data"]["table"]["rows"]
        tickers.update(r["symbol"].strip() for r in rows if r.get("symbol"))
    except Exception as e:
        logger.warning(f"NASDAQ fetch failed: {e}")

    # NYSE listed
    try:
        url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange=nyse"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        data = resp.json()
        rows = data["data"]["table"]["rows"]
        tickers.update(r["symbol"].strip() for r in rows if r.get("symbol"))
    except Exception as e:
        logger.warning(f"NYSE fetch failed: {e}")

    # Remove known bad patterns (warrants, units, preferred shares)
    clean = [t for t in tickers if t.isalpha() and 1 <= len(t) <= 5]

    logger.info(f"US raw universe: {len(clean)} tickers — filtering by market cap...")
    filtered = _filter_by_marketcap_us(clean)
    logger.info(f"US filtered universe: {len(filtered)} tickers above ${US_MIN_MARKET_CAP_USD/1e9:.0f}B market cap")
    cache_module.save_universe("us", filtered)
    return filtered


def _filter_by_marketcap_us(tickers: list[str]) -> list[str]:
    """
    Filter US tickers by market cap using yfinance in batches.
    Batching avoids rate limits.
    """
    passed = []
    batch_size = 100

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(
                batch, period="1d", group_by="ticker",
                auto_adjust=True, progress=False, threads=True
            )
            for ticker in batch:
                try:
                    info = yf.Ticker(ticker).fast_info
                    mcap = getattr(info, "market_cap", None)
                    if mcap and mcap >= US_MIN_MARKET_CAP_USD:
                        passed.append(ticker)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Batch market cap filter error: {e}")

    return passed


# -----------------------------------------------------------------------------
# INDIA UNIVERSE — NSE + BSE-only stocks filtered by ₹500Cr+ market cap
# -----------------------------------------------------------------------------

def get_india_tickers() -> list[str]:
    """
    Fetch NSE + BSE-only equity tickers, filtered by ₹500 Crore+ market cap.

    Strategy:
      - NSE stocks  → .NS suffix (preferred: better liquidity + yfinance data)
      - BSE-only    → .BO suffix (stocks not listed on NSE)
      - Overlap     → NSE wins: one ticker per company, no duplicates
      - Dedup       → ISIN-based (primary) + symbol-based (fallback)

    Returns list like ['RELIANCE.NS', 'TCS.NS', 'XYZCO.BO', ...]
    """
    import cache as cache_module
    cached = cache_module.load_universe("india")
    if cached:
        logger.info(f"India universe from cache: {len(cached)} tickers")
        return cached
    logger.info("Fetching India ticker universe (cache miss)...")

    # ── Step 1: NSE equity list ───────────────────────────────────────────────
    nse_tickers, nse_isin_map = _fetch_nse_equity_list()
    nse_symbols = {t.replace(".NS", "") for t in nse_tickers}
    logger.info(f"NSE: {len(nse_tickers)} tickers ({len(nse_isin_map)} with ISIN)")

    # ── Step 2: BSE equity list ───────────────────────────────────────────────
    bse_tickers, bse_isin_map = _fetch_bse_equity_list()
    logger.info(f"BSE total: {len(bse_tickers)} tickers ({len(bse_isin_map)} with ISIN)")

    # ── Step 3: Deduplicate — keep NSE for any stock listed on both ───────────
    # Primary: ISIN match (same ISIN = same company regardless of symbol)
    nse_isins = set(nse_isin_map.values())
    bse_only_tickers = []
    for ticker in bse_tickers:
        symbol = ticker.replace(".BO", "")
        isin   = bse_isin_map.get(symbol)

        # Skip if same ISIN exists on NSE (company is already covered)
        if isin and isin in nse_isins:
            continue
        # Fallback: skip if same symbol exists on NSE (catches missing ISIN cases)
        if symbol in nse_symbols:
            continue
        bse_only_tickers.append(ticker)

    logger.info(
        f"BSE-only (not on NSE): {len(bse_only_tickers)} tickers — "
        f"({len(bse_tickers) - len(bse_only_tickers)} deduplicated as NSE-listed)"
    )

    # ── Step 4: Combine and filter by market cap ──────────────────────────────
    all_tickers = nse_tickers + bse_only_tickers
    logger.info(
        f"India combined universe: {len(all_tickers)} tickers "
        f"({len(nse_tickers)} NSE + {len(bse_only_tickers)} BSE-only) — "
        f"filtering by ₹{INDIA_MIN_MARKET_CAP_INR/1e7:.0f}Cr+ market cap..."
    )

    filtered = _filter_by_marketcap_india(all_tickers)
    nse_kept = sum(1 for t in filtered if t.endswith(".NS"))
    bse_kept = sum(1 for t in filtered if t.endswith(".BO"))
    logger.info(
        f"India filtered universe: {len(filtered)} tickers "
        f"({nse_kept} NSE + {bse_kept} BSE-only) above "
        f"₹{INDIA_MIN_MARKET_CAP_INR/1e7:.0f}Cr market cap"
    )

    cache_module.save_universe("india", filtered)
    return filtered


def _fetch_nse_equity_list() -> tuple[list[str], dict[str, str]]:
    """
    Download NSE's official equity list CSV.
    Returns:
        (tickers_with_ns_suffix, {symbol: isin} map)
    """
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer":    "https://www.nseindia.com/",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        df.columns = df.columns.str.strip()

        symbols = df["SYMBOL"].dropna().str.strip().tolist()
        tickers = [f"{s}.NS" for s in symbols if _valid_india_symbol(s)]

        # Build ISIN map where available (column name varies: ISIN, ISIN No, etc.)
        isin_col = next((c for c in df.columns if "ISIN" in c.upper()), None)
        isin_map = {}
        if isin_col:
            for _, row in df.iterrows():
                sym  = str(row.get("SYMBOL", "")).strip()
                isin = str(row.get(isin_col, "")).strip()
                if sym and isin and isin.startswith("IN") and len(isin) == 12:
                    isin_map[sym] = isin

        return tickers, isin_map

    except Exception as e:
        logger.error(f"NSE equity list fetch failed: {e}")
        fallback = _nifty500_fallback()
        return fallback, {}


def _fetch_bse_equity_list() -> tuple[list[str], dict[str, str]]:
    """
    Download BSE's official active equity list.
    Returns:
        (tickers_with_bo_suffix, {symbol: isin} map)

    BSE API returns JSON array with fields:
        SCRIP_CD:   BSE scrip code (numeric)
        SCRIP_ID:   Ticker symbol (e.g. "RELIANCE")
        SCRIP_NAME: Company name
        ISIN_NO:    ISIN code (INE...)
        STATUS:     "Active" | "Suspended"
        GROUP:      Exchange group (A, B, T, etc.)
    """
    url = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
    params = {
        "Group":     "",
        "Scripcode": "",
        "industry":  "",
        "segment":   "Equity",
        "status":    "Active",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer":    "https://www.bseindia.com/",
        "Accept":     "application/json",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=25)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list):
            # Some response formats wrap in a key
            data = data.get("Table", data.get("data", []))

        tickers  = []
        isin_map = {}

        for row in data:
            # BSE API field names (verified against live response):
            #   scrip_id    → ticker symbol  (lowercase key)
            #   ISIN_NUMBER → ISIN code
            #   Status      → "Active" | "Suspended"
            symbol = str(row.get("scrip_id", row.get("SCRIP_ID", ""))).strip()
            isin   = str(row.get("ISIN_NUMBER", row.get("ISIN_NO", ""))).strip()
            status = str(row.get("Status", row.get("STATUS", ""))).strip().lower()

            if not symbol or status == "suspended":
                continue
            if not _valid_india_symbol(symbol):
                continue

            tickers.append(f"{symbol}.BO")
            if isin and isin.startswith("IN") and len(isin) == 12:
                isin_map[symbol] = isin

        logger.info(f"BSE API: {len(tickers)} active equity tickers fetched")
        return tickers, isin_map

    except Exception as e:
        logger.warning(f"BSE equity list fetch failed: {e} — BSE-only stocks will be skipped this run")
        return [], {}


def _valid_india_symbol(s: str) -> bool:
    """
    Accept symbols that are valid NSE/BSE equity ticker names.
    Allows letters, digits, hyphens, ampersands — rejects pure numbers (scrip codes).
    """
    s = s.strip()
    if not s or len(s) > 20:
        return False
    if s.isdigit():          # pure numeric = BSE scrip code, not a symbol
        return False
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-&")
    return all(c in allowed for c in s.upper())


def _filter_by_marketcap_india(tickers: list[str]) -> list[str]:
    """
    Filter India tickers by market cap using yfinance fast_info.
    Processes in batches to stay within rate limits.
    NSE (.NS) and BSE (.BO) tickers are handled identically — yfinance
    supports both suffixes natively.
    """
    passed     = []
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        for ticker in batch:
            try:
                info = yf.Ticker(ticker).fast_info
                mcap = getattr(info, "market_cap", None)
                if mcap and mcap >= INDIA_MIN_MARKET_CAP_INR:
                    passed.append(ticker)
            except Exception:
                pass

    return passed


def _nifty500_fallback() -> list[str]:
    """Hardcoded fallback — Nifty 500 top tickers — used if NSE CSV fetch fails."""
    symbols = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BAJFINANCE", "KOTAKBANK",
        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
        "SUNPHARMA", "ULTRACEMCO", "WIPRO", "ONGC", "NTPC",
        "POWERGRID", "TECHM", "HCLTECH", "INDUSINDBK", "NESTLEIND",
        "BAJAJFINSV", "COALINDIA", "JSWSTEEL", "GRASIM", "ADANIENT",
    ]
    return [f"{s}.NS" for s in symbols]
