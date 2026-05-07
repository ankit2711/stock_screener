# =============================================================================
# SECTOR ROTATION SCREENER
# =============================================================================
#
# Scores each market sector across 5 dimensions:
#   1. Stage Analysis  — is the sector index in Stage 1/2/3/4?
#   2. RS vs Benchmark — 13W + 26W outperformance vs Nifty500 / SPY
#   3. Breadth         — % of sector stocks above their SMA200
#   4. New Highs       — % of sector stocks near 52W high
#   5. RS Momentum     — is sector RS improving or deteriorating?
#
# Output: dict of SectorResult per sector  (display_name → SectorResult)
#
# Sector multiplier for trade_executions:
#   LEADING   (80-100): ×1.25  — strong tailwind
#   IMPROVING (65-80):  ×1.12  — early rotation, favour
#   NEUTRAL   (45-65):  ×1.00  — no adjustment
#   WEAKENING (30-45):  ×0.88  — headwind, reduce conviction
#   LAGGING   (0-30):   ×0.75  — strong headwind, demote to Tier B
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# =============================================================================
# SECTOR INDEX MAPS
# =============================================================================

# Maps yfinance sector string → (index_ticker, display_name)
#
# Ground truth: yfinance returns exactly 11 sector strings for Indian stocks
# (verified from metadata cache — these are ALL the strings that appear):
#   Technology, Financial Services, Healthcare, Consumer Defensive,
#   Consumer Cyclical, Industrials, Basic Materials, Energy,
#   Real Estate, Communication Services, Utilities
#
# Additional alias strings are included for robustness (industry sub-types
# or alternative yfinance labels that may appear for specific tickers).
#
# NSE index used for Stage + RS scoring per sector:
#   ^CNXIT       — Nifty IT
#   ^CNXFIN      — Nifty Financial Services  (banks + NBFCs + insurance)
#   ^CNXPHARMA   — Nifty Pharma
#   ^CNXFMCG     — Nifty FMCG
#   ^CNXAUTO     — Nifty Auto  (largest mkt-cap companies in Consumer Cyclical)
#   ^CNXINFRA    — Nifty Infrastructure
#   ^CNXMETAL    — Nifty Metal  (metals dominate Basic Materials by mkt cap)
#   ^CNXENERGY   — Nifty Energy  (oil/gas + power: ONGC, NTPC, Power Grid all here)
#   ^CNXREALTY   — Nifty Realty
#   ^CNXMEDIA    — Nifty Media
#
# "Energy" (oil/gas) and "Utilities" (power) are both measured against ^CNXENERGY
# because Nifty Energy includes NTPC and Power Grid — it is the closest available
# NSE index for the power sector.  They appear as separate buckets with different
# breadth pools, so the output shows "Energy & Oil" and "Power & Utilities" as
# distinct sectors while sharing the same Stage/RS baseline.

_NSE_SECTOR_MAP: dict[str, tuple[str, str]] = {

    # ── Technology ───────────────────────────────────────────────────────────
    # 510 stocks — TCS, Infosys, Wipro, HCL Tech, Tech Mahindra, LTIMindtree …
    "Technology":             ("^CNXIT",    "IT & Technology"),
    "Information Technology": ("^CNXIT",    "IT & Technology"),
    "Software":               ("^CNXIT",    "IT & Technology"),
    "IT Services":            ("^CNXIT",    "IT & Technology"),

    # ── Financial Services ────────────────────────────────────────────────────
    # 525 stocks — HDFC Bank, ICICI Bank, SBI, Bajaj Finance, Kotak, Axis …
    # ^CNXFIN (Nifty Financial Services) is broader than ^NSEBANK:
    # it includes private banks, NBFCs, insurance, and AMCs — matching the
    # full "Financial Services" bucket from yfinance.
    "Financial Services":     ("^CNXFIN",  "Banking & Fin. Services"),
    "Financial":              ("^CNXFIN",  "Banking & Fin. Services"),
    "Banks":                  ("^CNXFIN",  "Banking & Fin. Services"),
    "Insurance":              ("^CNXFIN",  "Banking & Fin. Services"),
    "Asset Management":       ("^CNXFIN",  "Banking & Fin. Services"),

    # ── Healthcare ───────────────────────────────────────────────────────────
    # 387 stocks — Sun Pharma, Dr Reddy, Cipla, Divi's, Apollo, Fortis …
    "Healthcare":             ("^CNXPHARMA","Pharma & Healthcare"),
    "Health Care":            ("^CNXPHARMA","Pharma & Healthcare"),
    "Pharmaceuticals":        ("^CNXPHARMA","Pharma & Healthcare"),
    "Drug Manufacturers":     ("^CNXPHARMA","Pharma & Healthcare"),
    "Biotechnology":          ("^CNXPHARMA","Pharma & Healthcare"),
    "Medical Devices":        ("^CNXPHARMA","Pharma & Healthcare"),

    # ── Consumer Defensive ───────────────────────────────────────────────────
    # 212 stocks — HUL, ITC, Nestle, Britannia, Dabur, Marico, Colgate …
    "Consumer Defensive":     ("^CNXFMCG", "FMCG & Consumer Staples"),
    "Consumer Staples":       ("^CNXFMCG", "FMCG & Consumer Staples"),
    "Household & Personal":   ("^CNXFMCG", "FMCG & Consumer Staples"),
    "Food Distribution":      ("^CNXFMCG", "FMCG & Consumer Staples"),
    "Beverages":              ("^CNXFMCG", "FMCG & Consumer Staples"),

    # ── Consumer Cyclical ────────────────────────────────────────────────────
    # 546 stocks — Maruti, M&M, Tata Motors, Hero, Bajaj Auto, Eicher,
    #              Titan, Zomato, Avenue Supermarts, Jubilant Foods …
    # ^CNXAUTO captures the largest mkt-cap names in this bucket
    "Consumer Cyclical":      ("^CNXAUTO", "Auto & Consumer Discretionary"),
    "Consumer Discretionary": ("^CNXAUTO", "Auto & Consumer Discretionary"),
    "Automobiles":            ("^CNXAUTO", "Auto & Consumer Discretionary"),
    "Auto Parts":             ("^CNXAUTO", "Auto & Consumer Discretionary"),
    "Restaurants":            ("^CNXAUTO", "Auto & Consumer Discretionary"),
    "Retail":                 ("^CNXAUTO", "Auto & Consumer Discretionary"),

    # ── Industrials ──────────────────────────────────────────────────────────
    # 672 stocks (largest bucket) — L&T, Siemens, ABB, Thermax, Cummins,
    #   Adani Ports, Interglobe Aviation, HAL, BEL, GRSE, COCHIN Shipyard …
    "Industrials":            ("^CNXINFRA","Capital Goods & Infra"),
    "Capital Goods":          ("^CNXINFRA","Capital Goods & Infra"),
    "Engineering":            ("^CNXINFRA","Capital Goods & Infra"),
    "Infrastructure":         ("^CNXINFRA","Capital Goods & Infra"),
    "Aerospace & Defense":    ("^CNXINFRA","Capital Goods & Infra"),
    "Defense":                ("^CNXINFRA","Capital Goods & Infra"),
    "Transportation":         ("^CNXINFRA","Capital Goods & Infra"),
    "Logistics":              ("^CNXINFRA","Capital Goods & Infra"),

    # ── Basic Materials ──────────────────────────────────────────────────────
    # 425 stocks — Tata Steel, JSW Steel, Hindalco, Vedanta, SAIL, Nalco,
    #   UltraTech (cement), ACC, SRF, PI Industries (chemicals), Aarti …
    # Metals dominate by market cap; ^CNXMETAL is the best single-index proxy.
    # Chemicals and cement are also here (no separate NSE index reliably
    # available on yfinance for these sub-sectors).
    "Basic Materials":        ("^CNXMETAL","Metals & Materials"),
    "Materials":              ("^CNXMETAL","Metals & Materials"),
    "Metals & Mining":        ("^CNXMETAL","Metals & Materials"),
    "Steel":                  ("^CNXMETAL","Metals & Materials"),
    "Chemicals":              ("^CNXMETAL","Metals & Materials"),
    "Specialty Chemicals":    ("^CNXMETAL","Metals & Materials"),
    "Construction Materials": ("^CNXMETAL","Metals & Materials"),
    "Aluminum":               ("^CNXMETAL","Metals & Materials"),
    "Copper":                 ("^CNXMETAL","Metals & Materials"),

    # ── Energy & Oil ─────────────────────────────────────────────────────────
    # 162 stocks — ONGC, IOC, BPCL, Reliance (energy), Coal India,
    #              GAIL, Petronet, Oil India, MRPL …
    "Energy":                 ("^CNXENERGY","Energy & Oil"),
    "Oil & Gas":              ("^CNXENERGY","Energy & Oil"),
    "Coal":                   ("^CNXENERGY","Energy & Oil"),
    "Oil & Gas Refining":     ("^CNXENERGY","Energy & Oil"),

    # ── Power & Utilities ────────────────────────────────────────────────────
    # 115 stocks — NTPC, Power Grid, Tata Power, Adani Power, Adani Green,
    #              CESC, Torrent Power, NHPC, SJVN, JSW Energy,
    #              Power Finance Corp, REC Ltd …
    # yfinance tags all of these as "Utilities". ^CNXENERGY is the best
    # available NSE proxy: NTPC and Power Grid are its top constituents.
    "Utilities":              ("^CNXENERGY","Power & Utilities"),
    "Electric Utilities":     ("^CNXENERGY","Power & Utilities"),
    "Renewable Energy":       ("^CNXENERGY","Power & Utilities"),
    "Independent Power":      ("^CNXENERGY","Power & Utilities"),

    # ── Real Estate ──────────────────────────────────────────────────────────
    # 168 stocks — DLF, Godrej Properties, Prestige, Oberoi, Sobha, Macrotech …
    "Real Estate":            ("^CNXREALTY","Real Estate"),
    "Realty":                 ("^CNXREALTY","Real Estate"),
    "REIT":                   ("^CNXREALTY","Real Estate"),

    # ── Communication Services ───────────────────────────────────────────────
    # 136 stocks — Reliance Jio (Reliance), Bharti Airtel, Indus Towers,
    #              Zee, Sun TV, PVR-INOX, Info Edge, Zomato* …
    "Communication Services": ("^CNXMEDIA", "Media & Telecom"),
    "Media":                  ("^CNXMEDIA", "Media & Telecom"),
    "Telecom":                ("^CNXMEDIA", "Media & Telecom"),
    "Telecommunication":      ("^CNXMEDIA", "Media & Telecom"),
    "Entertainment":          ("^CNXMEDIA", "Media & Telecom"),
    "Internet Content":       ("^CNXMEDIA", "Media & Telecom"),
}

_US_SECTOR_MAP: dict[str, tuple[str, str]] = {
    "Technology":             ("XLK",  "Technology"),
    "Information Technology": ("XLK",  "Technology"),
    "Financial Services":     ("XLF",  "Financials"),
    "Financial":              ("XLF",  "Financials"),
    "Healthcare":             ("XLV",  "Healthcare"),
    "Health Care":            ("XLV",  "Healthcare"),
    "Pharmaceuticals":        ("XLV",  "Healthcare"),
    "Energy":                 ("XLE",  "Energy"),
    "Industrials":            ("XLI",  "Industrials"),
    "Capital Goods":          ("XLI",  "Industrials"),
    "Communication Services": ("XLC",  "Communication"),
    "Media":                  ("XLC",  "Communication"),
    "Consumer Cyclical":      ("XLY",  "Consumer Discretionary"),
    "Consumer Discretionary": ("XLY",  "Consumer Discretionary"),
    "Consumer Defensive":     ("XLP",  "Consumer Staples"),
    "Consumer Staples":       ("XLP",  "Consumer Staples"),
    "Basic Materials":        ("XLB",  "Materials"),
    "Materials":              ("XLB",  "Materials"),
    "Real Estate":            ("XLRE", "Real Estate"),
    "Utilities":              ("XLU",  "Utilities"),
}


# =============================================================================
# DATA CLASS
# =============================================================================

@dataclass
class SectorResult:
    sector_name:      str             # display name e.g. "IT & Technology"
    index_ticker:     str             # e.g. "^CNXIT"
    sector_label:     str             # LEADING / IMPROVING / NEUTRAL / WEAKENING / LAGGING
    sector_score:     float           # composite 0-100
    sector_mult:      float           # score multiplier for trade ranking (0.75-1.25)
    stage:            int             # 1/2/3/4  (Weinstein stage of sector index)
    stage_label:      str             # "Stage 2 ↑" etc.
    rs_13w:           float           # sector RS vs benchmark 13W (%)
    rs_26w:           float           # sector RS vs benchmark 26W (%)
    rs_momentum:      float           # rs_13w - rs_13w_4wks_ago (positive = improving)
    rs_at_52w_high:   bool            # sector RS ratio at new 52W high
    breadth_pct:      float           # % sector stocks above SMA200
    new_highs_pct:    float           # % sector stocks within 5% of 52W high
    stock_count:      int             # number of universe stocks in this sector
    top_stocks:       list[str]       = field(default_factory=list)  # top 5 by RS
    rotation_signal:  str             = ""    # e.g. "Early rotation starting"
    caution:          str             = ""    # e.g. "RS momentum flattening"


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_sector_rotation(
    ohlcv:     dict,
    metadata:  dict,
    benchmark: pd.DataFrame,
    market:    str = "india",
) -> dict[str, SectorResult]:
    """
    Score all sectors found in the current universe.

    Args:
        ohlcv:     ticker → OHLCV DataFrame (already fetched by main pipeline)
        metadata:  ticker → {name, sector, market_cap} dict
        benchmark: benchmark OHLCV DataFrame (Nifty500 / SPY)
        market:    "india" | "us" | "ai"

    Returns:
        dict: display_name → SectorResult, sorted by sector_score descending
    """
    sector_map = _NSE_SECTOR_MAP if market in ("india", "ai") else _US_SECTOR_MAP

    # ── Step 1: Group universe tickers by sector display name ─────────────────
    sector_tickers: dict[str, list[str]] = {}  # display_name → [tickers]
    sector_index:   dict[str, str]       = {}  # display_name → index_ticker

    for ticker, meta in metadata.items():
        raw_sector = str(meta.get("sector", "Unknown")).strip()
        if raw_sector in ("Unknown", "None", "", "nan"):
            continue
        mapping = sector_map.get(raw_sector)
        if mapping is None:
            # Fuzzy fallback: check if any key is a substring
            mapping = _fuzzy_sector_match(raw_sector, sector_map)
        if mapping is None:
            continue

        idx_ticker, display_name = mapping
        sector_tickers.setdefault(display_name, []).append(ticker)
        sector_index[display_name] = idx_ticker

    if not sector_tickers:
        logger.warning("SectorRotation: No sectors found in metadata. Returning empty.")
        return {}

    logger.info(f"SectorRotation: Found {len(sector_tickers)} sectors across "
                f"{sum(len(v) for v in sector_tickers.values())} tickers")

    # ── Step 2: Fetch sector index OHLCV (one call per unique index ticker) ───
    unique_indices = list(set(sector_index.values()))
    index_ohlcv    = _fetch_index_ohlcv(unique_indices)

    # ── Step 3: Score each sector ─────────────────────────────────────────────
    bench_close = _resolve_close(benchmark) if benchmark is not None and not benchmark.empty else None

    results: dict[str, SectorResult] = {}

    for display_name, tickers in sector_tickers.items():
        idx_ticker = sector_index[display_name]
        idx_df     = index_ohlcv.get(idx_ticker)

        try:
            result = _score_sector(
                display_name  = display_name,
                index_ticker  = idx_ticker,
                idx_df        = idx_df,
                sector_tickers= tickers,
                ohlcv         = ohlcv,
                bench_close   = bench_close,
            )
            if result is not None:
                results[display_name] = result
        except Exception as e:
            logger.debug(f"SectorRotation: Failed to score {display_name}: {e}")

    # ── Step 4: Sort by score descending ─────────────────────────────────────
    results = dict(sorted(results.items(), key=lambda x: x[1].sector_score, reverse=True))
    logger.info(f"SectorRotation: Scored {len(results)} sectors successfully")

    return results


# =============================================================================
# SECTOR SCORING
# =============================================================================

def _score_sector(
    display_name:   str,
    index_ticker:   str,
    idx_df:         Optional[pd.DataFrame],
    sector_tickers: list[str],
    ohlcv:          dict,
    bench_close:    Optional[pd.Series],
) -> Optional[SectorResult]:
    """
    Score one sector across 5 dimensions and return a SectorResult.
    Returns None if insufficient data.
    """
    # ── Dimension 1: Stage Analysis on sector index ───────────────────────────
    stage, stage_label, ema200_val = _compute_stage(idx_df)

    # ── Dimension 2: RS vs Benchmark ─────────────────────────────────────────
    rs_13w, rs_26w, rs_momentum, rs_at_high = _compute_rs(idx_df, bench_close)

    # ── Dimension 3 + 4: Breadth and New Highs (from universe stocks) ─────────
    breadth_pct, new_highs_pct, top_stocks = _compute_breadth(
        sector_tickers, ohlcv
    )

    # ── Need at least some data to score ─────────────────────────────────────
    stock_count = len(sector_tickers)
    if stock_count < 2 and idx_df is None:
        return None

    # ── Composite Score (weighted) ────────────────────────────────────────────
    # Stage score: Stage2=100, Stage1=60, Stage3=25, Stage4=0, Unknown=50
    stage_score_map = {1: 60.0, 2: 100.0, 3: 25.0, 4: 0.0, 0: 50.0}
    stage_s = stage_score_map.get(stage, 50.0)

    # RS score: clamp rs_13w to [-25, +25] → scale to [0, 100]
    rs_s = min(max((rs_13w + 25) / 50 * 100, 0), 100)

    # Breadth score: 0%=0, 50%=50, 100%=100
    breadth_s = min(max(breadth_pct, 0), 100)

    # New highs score: 0%=0, 30%+=100
    new_high_s = min(new_highs_pct / 30 * 100, 100)

    # RS momentum score: positive improving=100, negative=0
    rs_mom_s = min(max((rs_momentum + 10) / 20 * 100, 0), 100)

    composite = (
        stage_s    * 0.25 +
        rs_s       * 0.30 +
        breadth_s  * 0.20 +
        new_high_s * 0.15 +
        rs_mom_s   * 0.10
    )
    composite = round(min(max(composite, 0), 100), 1)

    # ── Label ─────────────────────────────────────────────────────────────────
    label = _score_to_label(composite)
    mult  = _label_to_mult(label)

    # ── Rotation Signal ───────────────────────────────────────────────────────
    rotation_signal = _detect_rotation_signal(
        label, rs_momentum, breadth_pct, stage, rs_13w
    )
    caution = _detect_caution(label, rs_momentum, rs_at_high, stage)

    return SectorResult(
        sector_name    = display_name,
        index_ticker   = index_ticker,
        sector_label   = label,
        sector_score   = composite,
        sector_mult    = mult,
        stage          = stage,
        stage_label    = stage_label,
        rs_13w         = round(rs_13w, 1),
        rs_26w         = round(rs_26w, 1),
        rs_momentum    = round(rs_momentum, 1),
        rs_at_52w_high = rs_at_high,
        breadth_pct    = round(breadth_pct, 1),
        new_highs_pct  = round(new_highs_pct, 1),
        stock_count    = stock_count,
        top_stocks     = top_stocks[:5],
        rotation_signal= rotation_signal,
        caution        = caution,
    )


# =============================================================================
# DIMENSION CALCULATORS
# =============================================================================

def _compute_stage(idx_df: Optional[pd.DataFrame]) -> tuple[int, str, float]:
    """
    Weinstein Stage from sector index OHLCV.
    Returns (stage_int, stage_label, ema200_value)
    """
    if idx_df is None or len(idx_df) < 40:
        return 0, "Unknown", 0.0

    close = _resolve_close(idx_df)
    if close is None or len(close) < 40:
        return 0, "Unknown", 0.0
    ema200 = close.ewm(span=200, adjust=False).mean()
    slope  = float(ema200.iloc[-1]) - float(ema200.iloc[-20])
    price  = float(close.iloc[-1])
    ema_v  = float(ema200.iloc[-1])

    if price > ema_v and slope > 0:
        return 2, "Stage 2 ↑", ema_v
    elif price > ema_v and slope <= 0:
        return 3, "Stage 3 ↔", ema_v
    elif price <= ema_v and slope > 0:
        return 1, "Stage 1 →", ema_v
    else:
        return 4, "Stage 4 ↓", ema_v


def _compute_rs(
    idx_df:      Optional[pd.DataFrame],
    bench_close: Optional[pd.Series],
) -> tuple[float, float, float, bool]:
    """
    Compute sector RS vs benchmark.
    Returns (rs_13w, rs_26w, rs_momentum, rs_at_52w_high)
    """
    if idx_df is None or bench_close is None:
        return 0.0, 0.0, 0.0, False

    close = _resolve_close(idx_df)
    if close is None:
        return 0.0, 0.0, 0.0, False
    n     = min(len(close), len(bench_close))
    if n < 30:
        return 0.0, 0.0, 0.0, False

    c = close.iloc[-n:].values
    b = bench_close.iloc[-n:].values

    # RS ratio
    rs_ratio = pd.Series(c / b)

    # 13W RS
    lb13 = min(63, n - 1)
    rs_13w = float(c[-1] / c[-lb13] - b[-1] / b[-lb13]) * 100 if lb13 > 0 else 0.0

    # 26W RS
    lb26 = min(126, n - 1)
    rs_26w = float(c[-1] / c[-lb26] - b[-1] / b[-lb26]) * 100 if lb26 > 0 else 0.0

    # RS momentum: compare RS 13W now vs RS 13W 4 weeks ago
    lb13_4w = min(63, n - 20)
    rs_13w_4wago = float(c[-20] / c[-(lb13_4w + 20)] - b[-20] / b[-(lb13_4w + 20)]) * 100 \
        if (lb13_4w > 0 and n > 20) else rs_13w
    rs_momentum = rs_13w - rs_13w_4wago

    # RS at 52W high
    rs_52w_high = float(rs_ratio.iloc[-252:].max()) if len(rs_ratio) >= 20 else float(rs_ratio.max())
    rs_current  = float(rs_ratio.iloc[-1])
    rs_at_high  = rs_current >= rs_52w_high * 0.97  # within 3% of 52W RS high

    return rs_13w, rs_26w, rs_momentum, rs_at_high


def _compute_breadth(
    tickers: list[str],
    ohlcv:   dict,
) -> tuple[float, float, list[str]]:
    """
    Compute breadth metrics from universe stocks in this sector.
    Returns (breadth_pct, new_highs_pct, top_stocks_by_rs)
    """
    above_sma200 = 0
    near_52w_high = 0
    valid         = 0
    stock_rs: list[tuple[str, float]] = []

    for ticker in tickers:
        df = ohlcv.get(ticker)
        if df is None or len(df) < 50:
            continue

        # fetcher.fetch_ohlcv() returns lowercase columns ("close"); yfinance raw
        # returns title-case ("Close").  Accept either so this works in both paths.
        col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None:
            continue
        close = df[col]
        if len(close) < 50:
            continue

        try:
            sma200    = float(close.rolling(200, min_periods=50).mean().iloc[-1])
            high52w   = float(close.rolling(252, min_periods=50).max().iloc[-1])
            price     = float(close.iloc[-1])
            price_4w  = float(close.iloc[-21]) if len(close) > 21 else price

            # Breadth: above SMA200
            if sma200 > 0 and price > sma200:
                above_sma200 += 1

            # New highs: within 5% of 52W high
            if high52w > 0 and price >= high52w * 0.95:
                near_52w_high += 1

            # RS proxy: 21-day ROC
            rs_proxy = (price / price_4w - 1) * 100 if price_4w > 0 else 0.0
            stock_rs.append((ticker.replace(".NS", "").replace(".BO", ""), rs_proxy))
            valid += 1
        except Exception:
            continue

    if valid == 0:
        return 0.0, 0.0, []

    breadth_pct   = above_sma200 / valid * 100
    new_highs_pct = near_52w_high / valid * 100

    # Top 5 stocks by recent RS
    top_stocks = [t for t, _ in sorted(stock_rs, key=lambda x: x[1], reverse=True)[:5]]

    return breadth_pct, new_highs_pct, top_stocks


# =============================================================================
# CLASSIFICATION HELPERS
# =============================================================================

def _score_to_label(score: float) -> str:
    if score >= 80: return "LEADING"
    if score >= 65: return "IMPROVING"
    if score >= 45: return "NEUTRAL"
    if score >= 30: return "WEAKENING"
    return "LAGGING"


def _label_to_mult(label: str) -> float:
    return {
        "LEADING":   1.25,
        "IMPROVING": 1.12,
        "NEUTRAL":   1.00,
        "WEAKENING": 0.88,
        "LAGGING":   0.75,
    }.get(label, 1.00)


def _detect_rotation_signal(
    label:       str,
    rs_momentum: float,
    breadth_pct: float,
    stage:       int,
    rs_13w:      float,
) -> str:
    """Detect early rotation signals before they are obvious."""
    if label in ("WEAKENING", "LAGGING") and rs_momentum > 3.0:
        return "⚡ Early rotation starting — RS momentum turning positive"
    if label == "LAGGING" and breadth_pct > 40:
        return "⚡ Breadth recovering — watch for Stage2 confirmation"
    if label == "NEUTRAL" and rs_momentum > 5.0 and stage == 2:
        return "↑ RS improving — potential IMPROVING next week"
    if label == "IMPROVING" and rs_13w > 10 and rs_momentum > 2.0:
        return "↑↑ Accelerating — approaching LEADING territory"
    if label == "LEADING" and rs_momentum < -3.0:
        return "⚠ Momentum fading — watch for leadership change"
    return ""


def _detect_caution(
    label:       str,
    rs_momentum: float,
    rs_at_high:  bool,
    stage:       int,
) -> str:
    if label == "LEADING" and rs_momentum < -2.0 and not rs_at_high:
        return "RS momentum fading — do not add new positions"
    if label == "IMPROVING" and stage != 2:
        return "Stage not confirmed — wait for Stage2 on index"
    if label == "NEUTRAL" and rs_momentum < -3.0:
        return "RS deteriorating — avoid new entries"
    return ""


# =============================================================================
# LOOKUP HELPERS
# =============================================================================

def get_sector_for_ticker(
    ticker:       str,
    metadata:     dict,
    sector_results: dict[str, "SectorResult"],
    market:       str = "india",
) -> Optional["SectorResult"]:
    """
    Return the SectorResult for a given ticker, or None if not found.
    Used by ranker_trade to apply sector multiplier per stock.
    """
    meta = metadata.get(ticker, {})
    raw_sector = str(meta.get("sector", "Unknown")).strip()
    if raw_sector in ("Unknown", "None", "", "nan"):
        return None

    sector_map = _NSE_SECTOR_MAP if market in ("india", "ai") else _US_SECTOR_MAP
    mapping = sector_map.get(raw_sector) or _fuzzy_sector_match(raw_sector, sector_map)
    if mapping is None:
        return None

    _, display_name = mapping
    return sector_results.get(display_name)


def get_regime_from_sectors(sector_results: dict[str, "SectorResult"]) -> tuple[str, float]:
    """
    Compute market regime from sector breadth count.
    Returns (regime_label_suffix, confidence_0_to_1)
    More reliable than price-only regime detection.
    """
    if not sector_results:
        return "", 0.0

    total    = len(sector_results)
    leading  = sum(1 for s in sector_results.values() if s.sector_label in ("LEADING", "IMPROVING"))
    lagging  = sum(1 for s in sector_results.values() if s.sector_label in ("LAGGING", "WEAKENING"))

    lead_pct = leading / total
    lag_pct  = lagging / total

    if lead_pct >= 0.65:
        return "Broad Bull (sectors confirm)", lead_pct
    elif lead_pct >= 0.45:
        return "Selective Bull (mixed sectors)", lead_pct
    elif lag_pct >= 0.65:
        return "Broad Bear (sectors confirm)", 1 - lead_pct
    else:
        return "Mixed (sector rotation active)", 0.5


# =============================================================================
# PRIVATE UTILITIES
# =============================================================================

def _resolve_close(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Return the close-price Series from a DataFrame that may use either
    lowercase ("close") or title-case ("Close") column names.
    fetcher.fetch_ohlcv / fetch_benchmarks normalise to lowercase;
    yf.download() returns title-case.  This handles both.
    """
    for col in ("close", "Close", "CLOSE"):
        if col in df.columns:
            s = df[col]
            return s.squeeze() if hasattr(s, "squeeze") else s
    return None


def _fetch_index_ohlcv(index_tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Fetch 2 years of daily OHLCV for sector indices (one per call).

    Downloading one ticker at a time avoids the yfinance MultiIndex ambiguity
    that changed across versions (0.1.x vs 0.2.x group_by format).
    We only have 10-14 unique index tickers so the extra round-trips are trivial.
    `show_errors` was removed in yfinance ≥ 0.2.x — not used here.
    """
    result: dict[str, pd.DataFrame] = {}
    unique = list(set(index_tickers))

    for ticker in unique:
        try:
            df = yf.download(
                tickers     = ticker,
                period      = "2y",
                interval    = "1d",
                auto_adjust = True,
                progress    = False,
            )
            # yfinance may wrap a single ticker in a MultiIndex — unwrap if so
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(how="all")
            if len(df) >= 20:
                result[ticker] = df
                logger.debug(f"SectorRotation: fetched {ticker} ({len(df)} bars)")
            else:
                logger.debug(f"SectorRotation: {ticker} — too few bars ({len(df)}), skipped")
        except Exception as e:
            logger.warning(f"SectorRotation: failed to fetch {ticker}: {e}")

    logger.info(f"SectorRotation: fetched index data for {len(result)}/{len(unique)} indices")
    return result


def _fuzzy_sector_match(
    raw_sector: str,
    sector_map: dict[str, tuple[str, str]],
) -> Optional[tuple[str, str]]:
    """
    Case-insensitive substring match for sector names not in the map.
    Returns mapping tuple or None.
    """
    raw_lower = raw_sector.lower()
    for key, val in sector_map.items():
        if key.lower() in raw_lower or raw_lower in key.lower():
            return val
    return None
