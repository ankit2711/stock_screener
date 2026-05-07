#!/usr/bin/env python3
# =============================================================================
# SECTOR ROTATION RANKER — Standalone runner
# =============================================================================
#
# USAGE (standalone):
#   python ranker_sector.py                     # India sectors
#   python ranker_sector.py --market us         # US sectors
#   python ranker_sector.py --market india --json  # JSON output only
#
# Called automatically by run_trade_scan() — results flow into
# trade_executions scoring as a sector multiplier.
# =============================================================================

import argparse
import json
import logging
import sys
from datetime import datetime

import pandas as pd

import fetcher
import universe
from config import RS_RATING
from screeners.sector_rotation import (
    SectorResult,
    run_sector_rotation,
    get_regime_from_sectors,
)

logger = logging.getLogger(__name__)

# Label display config
_LABEL_META = {
    "LEADING":   {"icon": "▲", "color_hint": "strong uptrend",  "action": "deploy capital here"},
    "IMPROVING": {"icon": "↑", "color_hint": "early rotation",  "action": "watch for setups"},
    "NEUTRAL":   {"icon": "→", "color_hint": "market perform",  "action": "selective only"},
    "WEAKENING": {"icon": "↓", "color_hint": "losing momentum", "action": "reduce exposure"},
    "LAGGING":   {"icon": "✕", "color_hint": "underperforming", "action": "no new entries"},
}


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def print_sector_dashboard(
    sector_results: dict[str, SectorResult],
    market:         str = "india",
    show_detail:    bool = True,
) -> None:
    """
    Print the full sector rotation dashboard to stdout.
    Structured as: ranking table → detail cards → rotation signals → regime.
    """
    if not sector_results:
        print("\n  [SectorRotation] No sector data available.\n")
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    mkt = market.upper()

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print("╔" + "═" * 78 + "╗")
    print(f"║  SECTOR ROTATION SNAPSHOT — {mkt} — {now:<35}  ║")
    print("╚" + "═" * 78 + "╝")

    # ── Ranking Table ─────────────────────────────────────────────────────────
    print()
    hdr = (f"  {'Sector':<26} {'Score':>5}  {'Stage':<10} "
           f"{'RS 13W':>7}  {'RS 26W':>7}  {'Breadth':>7}  {'Label'}")
    print(hdr)
    print("  " + "─" * 76)

    for name, r in sector_results.items():
        meta = _LABEL_META.get(r.sector_label, {})
        icon = meta.get("icon", "→")
        print(
            f"  {name:<26} {r.sector_score:>5.0f}  {r.stage_label:<10} "
            f"{r.rs_13w:>+6.1f}%  {r.rs_26w:>+6.1f}%  "
            f"{r.breadth_pct:>6.1f}%  {icon} {r.sector_label}"
        )

    # ── Rotation Signals ──────────────────────────────────────────────────────
    signals = [(n, r) for n, r in sector_results.items() if r.rotation_signal]
    if signals:
        print()
        print("  ── ROTATION SIGNALS " + "─" * 57)
        for name, r in signals:
            print(f"  {r.rotation_signal}")
            print(f"    └─ {name}  (score {r.sector_score:.0f}, RS 13W {r.rs_13w:+.1f}%)")

    # ── Detail Cards (grouped by label) ───────────────────────────────────────
    if show_detail:
        _print_detail_cards(sector_results)

    # ── Market Regime from Sectors ─────────────────────────────────────────────
    regime_label, confidence = get_regime_from_sectors(sector_results)
    leading   = [n for n, r in sector_results.items() if r.sector_label in ("LEADING", "IMPROVING")]
    weakening = [n for n, r in sector_results.items() if r.sector_label in ("WEAKENING", "LAGGING")]

    print()
    print("  ── MARKET REGIME FROM SECTOR BREADTH " + "─" * 39)
    total = len(sector_results)
    lead_n = sum(1 for r in sector_results.values() if r.sector_label in ("LEADING", "IMPROVING"))
    lag_n  = sum(1 for r in sector_results.values() if r.sector_label in ("WEAKENING", "LAGGING"))
    print(f"  Leading + Improving : {lead_n}/{total} sectors")
    print(f"  Weakening + Lagging : {lag_n}/{total} sectors")
    print(f"  Regime signal       : {regime_label}")
    if leading:
        print(f"  Focus on            : {', '.join(leading[:4])}")
    if weakening:
        print(f"  Avoid               : {', '.join(weakening[:4])}")
    print()


def _print_detail_cards(sector_results: dict[str, SectorResult]) -> None:
    """Print grouped detail cards for each label bucket."""
    groups = {
        "LEADING":   "▲ LEADING SECTORS  (deploy capital here)",
        "IMPROVING": "↑ IMPROVING SECTORS  (early rotation — watch for setups)",
        "NEUTRAL":   "→ NEUTRAL SECTORS  (market perform — selective only)",
        "WEAKENING": "↓ WEAKENING SECTORS  (reduce exposure)",
        "LAGGING":   "✕ LAGGING SECTORS  (no new entries)",
    }

    for label, heading in groups.items():
        bucket = [(n, r) for n, r in sector_results.items() if r.sector_label == label]
        if not bucket:
            continue

        print()
        print(f"  {'─' * 76}")
        print(f"  {heading}")
        print(f"  {'─' * 76}")

        for name, r in bucket:
            print()
            high_flag = " ★ RS at 52W high" if r.rs_at_52w_high else ""
            print(f"  {label} · {name:<28} Score: {r.sector_score:.0f}/100{high_flag}")
            print(f"    {r.stage_label:<12}  "
                  f"RS 13W: {r.rs_13w:+.1f}%   RS 26W: {r.rs_26w:+.1f}%   "
                  f"Breadth: {r.breadth_pct:.0f}%   New Highs: {r.new_highs_pct:.0f}%")
            print(f"    RS momentum: {r.rs_momentum:+.1f}%/month  "
                  f"Universe stocks: {r.stock_count}")
            if r.top_stocks:
                print(f"    Top setups  : {', '.join(r.top_stocks)}")
            if r.rotation_signal:
                print(f"    ⚡ {r.rotation_signal}")
            if r.caution:
                print(f"    ⚠  {r.caution}")


def sector_results_to_df(sector_results: dict[str, SectorResult]) -> pd.DataFrame:
    """
    Convert sector results to a DataFrame for export to Google Sheets.
    """
    if not sector_results:
        return pd.DataFrame()

    rows = []
    for rank, (name, r) in enumerate(sector_results.items(), 1):
        icon = _LABEL_META.get(r.sector_label, {}).get("icon", "→")
        rows.append({
            "Rank":           rank,
            "Sector":         name,
            "Label":          f"{icon} {r.sector_label}",
            "Score":          r.sector_score,
            "Mult ×":         r.sector_mult,
            "Stage":          r.stage_label,
            "RS 13W %":       f"{r.rs_13w:+.1f}%",
            "RS 26W %":       f"{r.rs_26w:+.1f}%",
            "RS Momentum":    f"{r.rs_momentum:+.1f}%",
            "RS 52W High":    "✓" if r.rs_at_52w_high else "·",
            "Breadth %":      f"{r.breadth_pct:.0f}%",
            "New Highs %":    f"{r.new_highs_pct:.0f}%",
            "Stocks":         r.stock_count,
            "Top Setups":     ", ".join(r.top_stocks),
            "Rotation Signal":r.rotation_signal,
            "Caution":        r.caution,
            "Last Updated":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

    return pd.DataFrame(rows)


# =============================================================================
# BUCKETED SHEET BUILDER — for the "Sector Overview" Google Sheets tab
# =============================================================================

# Columns written in bucket rows
_BUCKET_COLS = [
    "Sector", "Score", "Mult ×", "Stage",
    "RS 13W %", "RS 26W %", "RS Momentum", "RS 52W High",
    "Breadth %", "New Highs %", "Stocks", "Top Setups",
    "Signal", "Caution",
]

# Bucket display labels (in priority order)
_BUCKET_DISPLAY = {
    "LEADING":   "▲  LEADING  —  Deploy capital here",
    "IMPROVING": "↑  IMPROVING  —  Early rotation, watch for setups",
    "NEUTRAL":   "→  NEUTRAL  —  Market perform, selective only",
    "WEAKENING": "↓  WEAKENING  —  Reduce exposure",
    "LAGGING":   "✕  LAGGING  —  No new entries",
}

_BUCKET_ORDER = ["LEADING", "IMPROVING", "NEUTRAL", "WEAKENING", "LAGGING"]


def sector_results_to_bucketed_rows(
    sector_results: dict[str, SectorResult],
    market:         str = "india",
    as_of:          str = "",
) -> tuple[list[list], list[int], list[int]]:
    """
    Build raw rows for a Google Sheets bucketed-layout tab.

    Returns:
        rows               — list of lists, one per sheet row
        bucket_hdr_rows    — 1-based row numbers that are bucket label rows
                             (caller bolds + grey-backgrounds these)
        col_hdr_rows       — 1-based row numbers that are column header rows
                             (caller bolds these)

    Layout per bucket:
        ▲  LEADING  —  Deploy capital here    ← bucket header (bold + shaded)
        Sector | Score | Mult × | …           ← column header  (bold)
        IT & Technology | 85 | ×1.25 | …      ← data
        Banking & Finance | 72 | ×1.25 | …
        [blank spacer row]
    """
    if not sector_results:
        return [[f"No sector data available — {as_of or datetime.now().strftime('%Y-%m-%d %H:%M')}"]], [], []

    regime_label, _ = get_regime_from_sectors(sector_results)
    now = as_of or datetime.now().strftime("%Y-%m-%d %H:%M")

    # Count leading/lagging for the top summary line
    total    = len(sector_results)
    lead_n   = sum(1 for r in sector_results.values() if r.sector_label in ("LEADING", "IMPROVING"))
    lag_n    = sum(1 for r in sector_results.values() if r.sector_label in ("WEAKENING", "LAGGING"))

    rows           = []
    bucket_hdr_rows = []
    col_hdr_rows   = []

    # ── Top summary row ───────────────────────────────────────────────────────
    rows.append([
        f"SECTOR ROTATION — {market.upper()} — {now}  |  Regime: {regime_label}  "
        f"|  Leading+Improving: {lead_n}/{total}  |  Weakening+Lagging: {lag_n}/{total}"
    ])
    rows.append([])  # spacer

    # ── One block per bucket ──────────────────────────────────────────────────
    for label in _BUCKET_ORDER:
        bucket = [
            (name, r) for name, r in sector_results.items()
            if r.sector_label == label
        ]
        if not bucket:
            continue

        # Sort highest score first within the bucket
        bucket.sort(key=lambda x: x[1].sector_score, reverse=True)

        # Bucket label row
        rows.append([_BUCKET_DISPLAY[label]] + [""] * (len(_BUCKET_COLS) - 1))
        bucket_hdr_rows.append(len(rows))   # 1-based

        # Column header row
        rows.append(_BUCKET_COLS)
        col_hdr_rows.append(len(rows))      # 1-based

        # Data rows
        for name, r in bucket:
            rows.append([
                name,
                round(r.sector_score, 0),
                f"×{r.sector_mult:.2f}",
                r.stage_label,
                f"{r.rs_13w:+.1f}%",
                f"{r.rs_26w:+.1f}%",
                f"{r.rs_momentum:+.1f}%/mo",
                "✓" if r.rs_at_52w_high else "·",
                f"{r.breadth_pct:.0f}%",
                f"{r.new_highs_pct:.0f}%",
                r.stock_count,
                ", ".join(r.top_stocks) if r.top_stocks else "—",
                r.rotation_signal or "—",
                r.caution or "—",
            ])

        rows.append([])  # spacer between buckets

    return rows, bucket_hdr_rows, col_hdr_rows


def sector_results_to_json(sector_results: dict[str, SectorResult]) -> dict:
    """
    Serialise sector results to a JSON-compatible dict for Drive export.
    """
    regime_label, _ = get_regime_from_sectors(sector_results)

    sectors_out = {}
    for name, r in sector_results.items():
        sectors_out[name] = {
            "sector_label":    r.sector_label,
            "sector_score":    r.sector_score,
            "sector_mult":     r.sector_mult,
            "stage":           r.stage_label,
            "rs_13w":          r.rs_13w,
            "rs_26w":          r.rs_26w,
            "rs_momentum":     r.rs_momentum,
            "rs_at_52w_high":  r.rs_at_52w_high,
            "breadth_pct":     r.breadth_pct,
            "new_highs_pct":   r.new_highs_pct,
            "stock_count":     r.stock_count,
            "top_stocks":      r.top_stocks,
            "rotation_signal": r.rotation_signal,
            "caution":         r.caution,
        }

    return {
        "as_of":            datetime.now().isoformat(),
        "regime_from_sectors": regime_label,
        "sectors":          sectors_out,
    }


# =============================================================================
# MAIN — standalone run
# =============================================================================

def run_sector_screener(market: str = "india") -> dict[str, SectorResult]:
    """
    Fetch data and run sector rotation. Returns sector_results dict.
    Can be called directly from main.py or run standalone.
    """
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(levelname)-8s | %(message)s",
        stream = sys.stdout,
    )

    logger.info(f"SectorRotation ▶ Fetching {market.upper()} universe...")

    if market == "india":
        import universe
        tickers  = universe.get_india_tickers()
        bench_sym = RS_RATING["benchmark_india"]
    else:
        import universe
        tickers  = universe.get_us_tickers()
        bench_sym = RS_RATING["benchmark_us"]

    logger.info(f"SectorRotation: {len(tickers)} tickers  |  Benchmark: {bench_sym}")

    ohlcv = fetcher.fetch_ohlcv(tickers, market=market)
    meta  = fetcher.fetch_metadata(list(ohlcv.keys()))

    bench_raw = fetcher.fetch_benchmarks([bench_sym])
    bench     = bench_raw.get(bench_sym)

    if bench is None:
        logger.error(f"SectorRotation: Benchmark fetch failed ({bench_sym})")
        return {}

    results = run_sector_rotation(ohlcv, meta, bench, market=market)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sector Rotation Screener")
    parser.add_argument("--market", choices=["india", "us"], default="india")
    parser.add_argument("--json",   action="store_true", help="Print JSON output only")
    parser.add_argument("--no-detail", action="store_true", help="Skip detail cards")
    args = parser.parse_args()

    results = run_sector_screener(market=args.market)

    if args.json:
        print(json.dumps(sector_results_to_json(results), indent=2))
    else:
        print_sector_dashboard(results, market=args.market, show_detail=not args.no_detail)
