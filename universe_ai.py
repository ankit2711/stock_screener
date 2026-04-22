# =============================================================================
# AI THEME UNIVERSE — ~150 curated global stocks across the AI supply chain
# =============================================================================
#
# Parsed from the TradingView watchlist (exchange:symbol → yfinance format).
# Groups are used as the "Sector" column in screener output so you can see
# exactly which part of the AI supply chain each stock belongs to.
#
# Exchange mapping:
#   US (NASDAQ/NYSE/AMEX)  → plain symbol (NVDA)
#   HKEX                   → zero-padded 4-digit + .HK (700 → 0700.HK)
#   KRX                    → .KS  (005930 → 005930.KS)
#   TSE (Tokyo)            → .T   (8035 → 8035.T)
#   TWSE (Taiwan main)     → .TW  (2454 → 2454.TW)
#   TPEX (Taiwan OTC)      → .TWO (4971 → 4971.TWO)
#   SSE (Shanghai)         → .SS  (600584 → 600584.SS)
#   SZSE (Shenzhen)        → .SZ  (002281 → 002281.SZ)
#   SGX (Singapore)        → .SI  (AJBU → AJBU.SI)
#   ASX (Australia)        → .AX  (NXT → NXT.AX)
#   LSE (London)           → .L   (OXIG → OXIG.L)
#   XETR (Frankfurt)       → .DE  (IFX → IFX.DE)
#   MIL (Milan)            → .MI  (PRY → PRY.MI)
#   OMXSTO (Stockholm)     → .ST  (ABB → ABB.ST)
#   MYX (Malaysia)         → .KL  (UNISEM → UNISEM.KL)
#   VIE (Vienna)           → .VI  (ATS → ATS.VI)
#   TSX (Canada)           → .TO  (HPS.A → HPS-A.TO)
#   EURONEXT               → manual country mapping (see _EURONEXT_MAP)
#   BCBA/LatAm             → manual mapping (WEGE3 → WEGE3.SA)
# =============================================================================

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# RAW WATCHLIST — paste-exported from TradingView
# =============================================================================

_AI_THEME_RAW = (
    "### HYPERSCALERS/CLOUD,NASDAQ:META,NYSE:ORCL,NASDAQ:MSFT,NASDAQ:AMZN,NYSE:SAP,"
    "NYSE:IBM,NASDAQ:GOOGL,HKEX:700,HKEX:9988,NASDAQ:BIDU,"
    "### AI INFRA/NEOCLOUD/GPU CLOUD,NASDAQ:VNET,NASDAQ:EQIX,NYSE:DLR,NASDAQ:GDS,"
    "NYSE:DBRG,NASDAQ:APLD,NASDAQ:CRWV,NASDAQ:NBIS,NASDAQ:GLXY,"
    "### DATA CENTER REITS/COLOCATION/TOWERS,SGX:AJBU,NYSE:IRM,SGX:ME8U,NYSE:CCI,"
    "NASDAQ:SBAC,NYSE:AMT,ASX:NXT,"
    "### AI APPLICATIONS/ENTERPRISE,NASDAQ:ADBE,NYSE:CRM,NASDAQ:DDOG,NASDAQ:PLTR,"
    "NYSE:NOW,NYSE:SNOW,"
    "### AI COMPUTE,TSE:6723,NASDAQ:AVGO,NASDAQ:AMD,NYSE:STM,NASDAQ:NVDA,NASDAQ:MCHP,"
    "NASDAQ:ADI,NASDAQ:MRVL,NASDAQ:QCOM,NASDAQ:NXPI,NASDAQ:ARM,NASDAQ:ON,NASDAQ:TXN,"
    "NASDAQ:INTC,NASDAQ:MPWR,"
    "### NETWORKING SILICON & EQUIPMENT,NYSE:ANET,NASDAQ:ALAB,NYSE:HPE,NYSE:CIEN,"
    "NASDAQ:VIAV,NASDAQ:CSCO,NASDAQ:MTSI,NYSE:NOK,NASDAQ:MXL,NASDAQ:CRDO,"
    "### OPTICS/TRANSCEIVERS/PHOTONIC,SZSE:002281,NASDAQ:AAOI,LSE:OXIG,NYSE:FN,"
    "HKEX:763,NYSE:GLW,NASDAQ:SMTC,NASDAQ:IPGP,NASDAQ:LITE,NYSE:COHR,TSE:6702,"
    "TSE:6701,SZSE:300308,TPEX:4971,TPEX:3081,SZSE:300502,LSE:IQE,NASDAQ:AXTI,"
    "### HBM/MEMORY/STORAGE,KRX:005930,KRX:000660,NASDAQ:SNDK,TWSE:2408,NYSE:PSTG,"
    "TPEX:8299,NASDAQ:SIMO,NASDAQ:NTAP,NASDAQ:STX,NASDAQ:WDC,NASDAQ:MU,"
    "### ADVANCED PACKAGING/OSAT/SUBSTRATES,MYX:UNISEM,TWSE:6239,NASDAQ:IMOS,"
    "SSE:600584,TSE:2802,VIE:ATS,TWSE:3711,TSE:4062,NYSE:ASX,NASDAQ:AMKR,TWSE:3037,"
    "TWSE:3189,TWSE:8046,"
    "### FOUNDRY/LOGIC & MEMORY FAB,HKEX:1347,HKEX:981,EURONEXT:XFAB,NYSE:UMC,"
    "NASDAQ:GFS,NYSE:TSM,KRX:000990,NASDAQ:TSEM,"
    "### EDA/DESIGN/TEST,NASDAQ:CDNS,NASDAQ:SNPS,NYSE:KEYS,"
    "### WAFERS/MATERIALS/CHEMICALS,TSE:3436,NASDAQ:ENTG,NYSE:DD,EURONEXT:AI,"
    "NYSE:APD,EURONEXT:SOI,TPEX:6488,NYSE:MMM,NASDAQ:LIN,TSE:4063,XETR:WAF,"
    "### SEMICAP EQUIPMENT,TSE:8035,EURONEXT:ASM,TSE:7735,TSE:6146,TSE:7751,"
    "HKEX:522,NASDAQ:VECO,TSE:7731,NASDAQ:ASML,NASDAQ:MKSI,XETR:AIXA,NASDAQ:KLAC,"
    "NASDAQ:KLIC,NASDAQ:AMAT,NASDAQ:LRCX,EURONEXT:BESI,"
    "### TEST/MEASUREMENT/BURN-IN,NASDAQ:AEHR,AMEX:INTT,TSE:6857,NASDAQ:COHU,"
    "NASDAQ:TER,NASDAQ:FORM,"
    "### SERVER OEM/ODM/EMS,TWSE:2345,NASDAQ:SMCI,TWSE:4938,NYSE:DELL,TWSE:2356,"
    "NYSE:CLS,NYSE:JBL,TWSE:6669,NASDAQ:FLEX,TWSE:2317,TWSE:2382,NASDAQ:SANM,"
    "TWSE:3231,SSE:600756,"
    "### POWER & ELECTRICAL,XETR:ENR,MIL:PRY,EURONEXT:SU,XETR:SIE,OMXSTO:ABB,"
    "TSE:6501,EURONEXT:LR,TSE:6503,NYSE:HUBB,BCBA:WEGE3,NYSE:BDC,EURONEXT:NEX,"
    "NYSE:ETN,NYSE:TEL,TSX:HPS.A,TWSE:2308,NYSE:NVT,NYSE:APH,"
    "### THERMAL/HVAC/LIQUID COOLING,NYSE:VRT,NASDAQ:AAON,TSE:6367,NYSE:LII,"
    "NYSE:CARR,NYSE:MOD,NYSE:JCI,NYSE:XYL,NYSE:CE,NYSE:TT,NYSE:CC,NYSE:ROG,"
    "### ON-SITE POWER GENERATION,NASDAQ:FCEL,NASDAQ:PLUG,NYSE:GEV,NYSE:BE,"
    "NYSE:CAT,NYSE:GNRC,NYSE:CMI,"
    "### DATACENTER CONSTRUCTION/ENGINEERING,NASDAQ:TTEK,NYSE:MTZ,NYSE:FLR,"
    "NYSE:J,NYSE:PWR,NYSE:ACM,NYSE:EME,"
    "### GAN CAMP (POWER),XETR:IFX,NASDAQ:VICR,NASDAQ:NVTS,NASDAQ:POWI,"
    "### SIC CAMP (POWER),TSE:6963,NYSE:WOLF,"
    "### POWER MGMT/ANALOG,NASDAQ:ALGM,NYSE:VSH,"
    "### METALS (COPPER/ALUMINUM),NYSE:SCCO,NYSE:FCX,NYSE:AA,NYSE:NUE,"
    "### AI OPS/SECURITY/DATA PIPELINES,NASDAQ:FTNT,NASDAQ:PANW,NYSE:NET,"
    "NASDAQ:CRWD,NASDAQ:ZS,"
    "### EDGE AI/SMARTPHONES,TWSE:2454,NASDAQ:AAPL,TSE:6758,HKEX:1810,"
    "### COMPUTE MINING / HPC,NASDAQ:RIOT,NASDAQ:HUT,NASDAQ:IREN,"
    "NASDAQ:WULF,NASDAQ:CIFR,NASDAQ:MARA"
)

# =============================================================================
# EXCHANGE → yfinance SUFFIX
# =============================================================================

_TV_SUFFIX: dict[str, str] = {
    "NASDAQ":   "",
    "NYSE":     "",
    "AMEX":     "",
    "NYSEARCA": "",
    "HKEX":     ".HK",   # numeric tickers zero-padded to 4 digits
    "KRX":      ".KS",
    "TSE":      ".T",    # Tokyo
    "TWSE":     ".TW",   # Taiwan main board
    "TPEX":     ".TWO",  # Taiwan OTC
    "SSE":      ".SS",   # Shanghai
    "SZSE":     ".SZ",   # Shenzhen
    "SGX":      ".SI",   # Singapore
    "ASX":      ".AX",   # Australia
    "LSE":      ".L",    # London
    "XETR":     ".DE",   # Frankfurt XETRA
    "MIL":      ".MI",   # Milan
    "OMXSTO":   ".ST",   # Stockholm
    "MYX":      ".KL",   # Malaysia
    "VIE":      ".VI",   # Vienna
    "TSX":      ".TO",   # Toronto (hyphens replace dots — see _tv_to_yf)
}

# EURONEXT spans national exchanges — manual per-symbol country routing
_EURONEXT_MAP: dict[str, str] = {
    "XFAB":  "XFAB.PA",   # X-Fab Silicon Foundries    — Paris
    "ASM":   "ASM.AS",    # ASM International          — Amsterdam
    "AI":    "AI.PA",     # Air Liquide                — Paris
    "BESI":  "BESI.AS",   # BE Semiconductor           — Amsterdam
    "SU":    "SU.PA",     # Schneider Electric         — Paris
    "NEX":   "NEX.PA",    # Nexans                     — Paris
    "LR":    "LR.PA",     # Legrand                    — Paris
    "SOI":   "SOI.PA",    # Soitec                     — Paris
}

# BCBA / LatAm overrides
_BCBA_MAP: dict[str, str] = {
    "WEGE3": "WEGE3.SA",  # WEG S.A. — B3 Brazil (TradingView lists as BCBA)
}


# =============================================================================
# CONVERSION
# =============================================================================

def _tv_to_yf(exchange: str, symbol: str) -> str | None:
    """
    Convert a TradingView exchange:symbol pair to a yfinance ticker.
    Returns None if the exchange is unknown or has no mapping.
    """
    exchange = exchange.upper().strip()
    symbol   = symbol.strip()

    # US exchanges — plain symbol, no suffix
    if exchange in ("NASDAQ", "NYSE", "AMEX", "NYSEARCA"):
        return symbol

    # HKEX — numeric tickers must be zero-padded to 4 digits
    if exchange == "HKEX":
        padded = symbol.zfill(4) if symbol.isdigit() else symbol
        return padded + ".HK"

    # EURONEXT — country-specific suffix from manual map
    if exchange == "EURONEXT":
        yf_sym = _EURONEXT_MAP.get(symbol)
        if yf_sym is None:
            logger.warning(f"  AI universe: EURONEXT:{symbol} — no yfinance mapping, skipped")
        return yf_sym

    # BCBA / LatAm manual overrides
    if exchange == "BCBA":
        yf_sym = _BCBA_MAP.get(symbol)
        if yf_sym is None:
            logger.warning(f"  AI universe: BCBA:{symbol} — no yfinance mapping, skipped")
        return yf_sym

    # TSX Canada — dots → hyphens (HPS.A → HPS-A.TO)
    if exchange == "TSX":
        return symbol.replace(".", "-") + ".TO"

    # Standard suffix table
    suffix = _TV_SUFFIX.get(exchange)
    if suffix is not None:
        return symbol + suffix

    logger.warning(f"  AI universe: unknown exchange '{exchange}' for {symbol}, skipped")
    return None


# =============================================================================
# PARSER
# =============================================================================

def _parse(raw: str) -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Parse raw TradingView watchlist text.

    Format: ### GROUP NAME,EXCHANGE:SYMBOL,...,### NEXT GROUP,...

    Returns:
        groups          : {group_name: [yf_ticker, ...]}
        ticker_to_group : {yf_ticker: group_name}
    """
    groups:          dict[str, list[str]] = {}
    ticker_to_group: dict[str, str]       = {}
    current_group = "AI Theme"

    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue

        if token.startswith("###"):
            current_group = token.lstrip("#").strip()
            groups.setdefault(current_group, [])
            continue

        if ":" not in token:
            logger.debug(f"  AI universe: skipping malformed token {token!r}")
            continue

        exchange, symbol = token.split(":", 1)
        yf_ticker = _tv_to_yf(exchange, symbol)

        if yf_ticker:
            groups.setdefault(current_group, []).append(yf_ticker)
            # First-seen group wins for duplicate tickers across groups
            if yf_ticker not in ticker_to_group:
                ticker_to_group[yf_ticker] = current_group

    return groups, ticker_to_group


# Build at import time — pure string parsing, no I/O
_AI_GROUPS, _AI_TICKER_TO_GROUP = _parse(_AI_THEME_RAW)


# =============================================================================
# PUBLIC API
# =============================================================================

def get_ai_tickers() -> list[str]:
    """Return all AI theme yfinance tickers (deduplicated, group order preserved)."""
    seen: set[str] = set()
    result: list[str] = []
    for tickers in _AI_GROUPS.values():
        for t in tickers:
            if t not in seen:
                seen.add(t)
                result.append(t)
    logger.info(
        f"AI Theme universe: {len(result)} tickers across {len(_AI_GROUPS)} groups"
    )
    return result


def get_ai_groups() -> dict[str, list[str]]:
    """Return {group_name: [yf_ticker, ...]} mapping."""
    return dict(_AI_GROUPS)


def get_ai_metadata() -> dict[str, dict]:
    """
    Return per-ticker metadata with the watchlist group as 'sector'.
    Merged over yfinance metadata in main.py so group names show in output.
    """
    return {
        ticker: {
            "group":  group,
            "sector": group,         # used as the Sector column in screener output
        }
        for ticker, group in _AI_TICKER_TO_GROUP.items()
    }


def tv_url_for(yf_ticker: str) -> str:
    """
    Best-effort TradingView chart URL for an AI theme ticker.
    Reverses the yfinance suffix back to the TV exchange prefix.
    """
    suffix_to_exchange = {
        ".HK":  "HKEX",
        ".KS":  "KRX",
        ".T":   "TSE",
        ".TW":  "TWSE",
        ".TWO": "TPEX",
        ".SS":  "SSE",
        ".SZ":  "SZSE",
        ".SI":  "SGX",
        ".AX":  "ASX",
        ".L":   "LSE",
        ".DE":  "XETR",
        ".MI":  "MIL",
        ".ST":  "OMXSTO",
        ".KL":  "MYX",
        ".VI":  "VIE",
        ".TO":  "TSX",
        ".PA":  "EURONEXT",
        ".AS":  "EURONEXT",
        ".SA":  "BMFBOVESPA",
    }
    for suffix, exchange in suffix_to_exchange.items():
        if yf_ticker.endswith(suffix):
            sym = yf_ticker[: -len(suffix)]
            return f"https://www.tradingview.com/chart/?symbol={exchange}:{sym}"
    # US — plain symbol
    return f"https://www.tradingview.com/chart/?symbol={yf_ticker}"
