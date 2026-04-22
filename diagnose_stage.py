#!/usr/bin/env python3
"""
Diagnostic: run Stage Analysis on a target list of manually verified Stage-2
stocks and print every intermediate signal so we can see exactly why any stock
is failing or scoring lower than expected.

Usage:
    python diagnose_stage.py
"""

import sys
import logging
import numpy as np
import pandas as pd
import yfinance as yf

# ── silence yfinance noise ────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
for noisy in ("yfinance", "peewee", "urllib3", "requests"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

sys.path.insert(0, ".")

from screeners.stage_analysis import (
    StageAnalysisConfig, _sensitivity_mult, _calc_rs, _calc_beta,
    _calc_atr, _calc_pead_approx,
)
from ranker_stage import DEFAULT_CFG

# ─────────────────────────────────────────────────────────────────────────────
# TARGET TICKERS — manually verified Stage-2 by Mark Minervini 5-Apr-2025
# ─────────────────────────────────────────────────────────────────────────────
TARGET_BASE = [
    "LLOYDSME", "TRAVELFOOD", "INOXINDIA", "SHILPAMED", "RPSGVENT",
    "KRISHANA", "INDOTHAI", "GANECOS", "CONFIPET", "LOKESHMACH",
    "YAAP", "PUSHPA", "MEGASTAR", "MADHUSUDAN",
]
TICKERS = [f"{t}.NS" for t in TARGET_BASE]
BENCHMARK = "^CRSLDX"   # CNX500 — same as India run


# ─────────────────────────────────────────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch(tickers, period="2y"):
    print(f"\nFetching {len(tickers)} tickers …")
    raw = yf.download(tickers, period=period, auto_adjust=True,
                      group_by="ticker", progress=False, threads=True)
    result = {}
    for t in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[t].copy()
            df.columns = [c.lower() for c in df.columns]
            df = df.dropna(subset=["close"])
            if len(df) >= 100:
                result[t] = df
            else:
                print(f"  ⚠  {t}: only {len(df)} bars — skipped")
        except Exception as e:
            print(f"  ✗  {t}: {e}")
    print(f"  Got data for {len(result)}/{len(tickers)} tickers")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL EXTRACTOR — mirrors run_stage_analysis internals exactly
# ─────────────────────────────────────────────────────────────────────────────

def extract_signals(ticker, df, bench_df, cfg):
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # 1. Stage MA + slope
    stage_ma = close.ewm(span=cfg.ma_length, adjust=False).mean()
    ma_slope = (stage_ma - stage_ma.shift(cfg.slope_lookback)) / stage_ma.shift(cfg.slope_lookback) * 100

    sens_mult   = _sensitivity_mult(cfg.sensitivity)
    slope_thresh = 0.5 * sens_mult

    ma_rising  = ma_slope.iloc[-1] >  slope_thresh
    ma_falling = ma_slope.iloc[-1] < -slope_thresh
    ma_flat    = not ma_rising and not ma_falling

    # 2. EMAs
    ema_fast   = close.ewm(span=cfg.ema_fast,   adjust=False).mean()
    ema_medium = close.ewm(span=cfg.ema_medium,  adjust=False).mean()
    ema_slow   = close.ewm(span=cfg.ema_slow,    adjust=False).mean()

    ef = ema_fast.iloc[-1]
    em = ema_medium.iloc[-1]
    es = ema_slow.iloc[-1]
    px = close.iloc[-1]
    ma_val = stage_ma.iloc[-1]

    bullish_ema = ef > em and em > es
    bearish_ema = ef < em and em < es

    ema_dist_slow   = (px - es) / es * 100
    ema_dist_medium = (px - em) / em * 100

    # 3. RS
    _, _, rs_vs_bm, rs_is_strong, rs_rising, rs_falling = _calc_rs(
        close, bench_df["close"], cfg.rs_ma_length
    )

    # 4. Momentum
    roc_f = float(close.pct_change(cfg.mom_fast).iloc[-1] * 100)
    roc_s = float(close.pct_change(cfg.mom_slow).iloc[-1] * 100)
    mom_accel = float(close.pct_change(cfg.mom_fast).iloc[-1] * 100) - \
                float(close.pct_change(cfg.mom_fast).iloc[-4] * 100)

    # 5. Volume
    avg_vol = volume.rolling(cfg.vol_avg_len).mean()
    vol_ratio = float(volume.iloc[-1] / avg_vol.iloc[-1]) if avg_vol.iloc[-1] > 0 else 1.0
    vol_expansion = volume.iloc[-1] > avg_vol.iloc[-1] * cfg.vol_expansion_ratio

    # 6. HH / LL (40-bar)
    hh = high.rolling(40).max().iloc[-1] > high.rolling(40).max().iloc[-41]
    ll = low.rolling(40).min().iloc[-1]  < low.rolling(40).min().iloc[-41]

    # 7. Price vs MA
    price_above_ma = px > ma_val
    price_below_ma = px < ma_val

    # 8. S2 score components — mirrors updated _calc_stage (graduated core)
    if price_above_ma and ma_rising:
        s2_core = 4.0
    elif price_above_ma:
        s2_core = 2.0
    else:
        s2_core = 0.0
    s2_ema      = 2.0 if bullish_ema    else 0.0
    s2_rs       = 2.0 if rs_is_strong   else 0.0
    s2_hh       = 1.0 if hh             else 0.0
    s2_roc      = 1.0 if roc_f > 0      else 0.0
    s2_total    = s2_core + s2_ema + s2_rs + s2_hh + s2_roc

    # 9. Full stage scores (for reference)
    slope_improving = ma_slope.iloc[-1] > ma_slope.iloc[-cfg.slope_lookback - 1]
    s1 = 0.0
    s1 += 3.0 if ma_flat else 0.0
    s1 += 2.0 if abs(ema_dist_slow) < 5 else 0.0
    s1 += 1.0 if (not hh and not ll) else 0.0
    s1 += 2.0 if (ma_falling is False and slope_improving and ma_flat) else 0.0
    s1 += 1.0 if rs_rising else 0.0

    s3 = 0.0
    s3 += 3.0 if (ma_flat and (s2_total > 0 or price_above_ma)) else 0.0
    s3 += 1.0 if (price_above_ma and not ma_rising) else 0.0
    s3 += 2.0 if (not rs_is_strong) else 0.0
    s3 += 1.5 if (roc_f < 0 and roc_s > 0) else 0.0
    s3 += 1.0 if (not bullish_ema and not bearish_ema) else 0.0
    slope_deteriorating = ma_slope.iloc[-1] < ma_slope.iloc[-cfg.slope_lookback - 1]
    s3 += 2.0 if (slope_deteriorating and not ma_falling) else 0.0

    s4 = 0.0
    s4 += 4.0 if (price_below_ma and ma_falling) else 0.0
    s4 += 2.0 if bearish_ema else 0.0
    s4 += 1.0 if (not rs_is_strong) else 0.0
    s4 += 1.0 if ll else 0.0
    s4 += 1.0 if roc_f < 0 else 0.0
    s4 += 2.0 if (price_below_ma and vol_expansion and ma_falling) else 0.0

    S2_MIN = 5.0
    S4_MIN = 4.0
    max_sc = max(s1, s2_total, s3, s4)
    if   max_sc == s2_total and s2_total >= S2_MIN: stage = 2
    elif max_sc == s4       and s4       >= S4_MIN: stage = 4
    elif max_sc == s1:                              stage = 1
    else:                                           stage = 3

    # 10. Cheat entry
    atr14  = _calc_atr(high, low, close, 14)
    atr_avg = atr14.rolling(20).mean()
    atr_contracting = atr14.iloc[-1] < atr_avg.iloc[-1]
    vol_dry = volume.iloc[-1] < avg_vol.iloc[-1] * cfg.cheat_vol_ratio
    cheat_pullback = abs(ema_dist_medium) < cfg.cheat_pullback_pct
    cheat_above_ma = px > ma_val and ma_val > 0
    is_cheat = (stage == 2 and cheat_pullback and vol_dry and atr_contracting and cheat_above_ma)

    # 11. Entry signal (from ranker_stage)
    high_4w = float(high.iloc[-21:-1].max()) if len(high) > 21 else float(high.max())
    dist_4wh = (px - high_4w) / high_4w * 100
    vol_10 = float(volume.iloc[-10:].mean())
    avg_vol_50 = float(volume.rolling(50).mean().iloc[-1])
    vol_dry_entry = vol_10 < avg_vol_50 * 0.75
    dist_ema21 = abs(px - em) / em * 100

    if is_cheat:
        entry_label = "🟢 Cheat Entry"
    elif dist_ema21 <= 5.0 and vol_dry_entry:
        entry_label = "🟡 EMA Pullback"
    elif -3.0 <= dist_4wh <= 1.0:
        entry_label = "🔵 Near Pivot"
    else:
        entry_label = "⚪ Extended"

    return {
        "ticker":          ticker.replace(".NS",""),
        "price":           round(px, 2),
        "ma200":           round(ma_val, 2),
        "slope_%":         round(float(ma_slope.iloc[-1]), 3),
        "slope_thresh":    round(slope_thresh, 3),
        "ma_rising":       ma_rising,
        "price>ma200":     price_above_ma,
        "bullish_ema":     bullish_ema,
        "rs_strong":       rs_is_strong,
        "rs_vs_bm%":       round(rs_vs_bm, 2),
        "hh":              hh,
        "roc_f>0":         roc_f > 0,
        "roc_f%":          round(roc_f, 2),
        "s2_core":         s2_core,
        "s2_ema":          s2_ema,
        "s2_rs":           s2_rs,
        "s2_hh":           s2_hh,
        "s2_roc":          s2_roc,
        "S2_TOTAL":        round(s2_total, 1),
        "S1":              round(s1, 1),
        "S3":              round(s3, 1),
        "S4":              round(s4, 1),
        "STAGE":           stage,
        "is_cheat":        is_cheat,
        "entry":           entry_label,
        "vol_ratio":       round(vol_ratio, 2),
        "ema_dist_med%":   round(ema_dist_medium, 2),
        "ema_dist_slow%":  round(ema_dist_slow, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = DEFAULT_CFG

    print("\n" + "=" * 70)
    print("STAGE DIAGNOSTIC — Minervini verified Stage-2 list")
    print(f"Config: ma_length={cfg.ma_length}, slope_thresh=0.5×{_sensitivity_mult(cfg.sensitivity)}={0.5*_sensitivity_mult(cfg.sensitivity)}, S2_MIN=5")
    print("=" * 70)

    # Fetch benchmark
    print(f"\nFetching benchmark {BENCHMARK} …")
    bench_raw = yf.download(BENCHMARK, period="2y", auto_adjust=True,
                            progress=False)
    # yfinance >= 0.2 returns MultiIndex columns — flatten them
    if isinstance(bench_raw.columns, pd.MultiIndex):
        bench_raw.columns = [c[0].lower() for c in bench_raw.columns]
    else:
        bench_raw.columns = [c.lower() for c in bench_raw.columns]
    bench_df = bench_raw.dropna(subset=["close"])
    print(f"  Benchmark: {len(bench_df)} bars")

    # Fetch stock data
    ohlcv = fetch(TICKERS, period="2y")

    # Run diagnostics
    rows = []
    missing = []
    for t in TICKERS:
        base = t.replace(".NS", "")
        if t not in ohlcv:
            missing.append(base)
            continue
        sig = extract_signals(t, ohlcv[t], bench_df, cfg)
        rows.append(sig)

    if missing:
        print(f"\n⚠  No data for: {', '.join(missing)}")

    if not rows:
        print("No data at all — check internet / ticker symbols")
        return

    diag = pd.DataFrame(rows)

    # ── Print full signal table ───────────────────────────────────────────────
    print("\n── FULL SIGNAL TABLE ──────────────────────────────────────────────────")
    key_cols = ["ticker", "price", "ma200", "slope_%", "ma_rising", "price>ma200",
                "bullish_ema", "rs_strong", "hh", "roc_f%",
                "s2_core", "s2_ema", "s2_rs", "s2_hh", "s2_roc", "S2_TOTAL",
                "S1", "S3", "S4", "STAGE", "entry"]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(diag[key_cols].to_string(index=False))

    # ── Summary: what's failing ───────────────────────────────────────────────
    print("\n── FAILURE BREAKDOWN ──────────────────────────────────────────────────")
    print(f"{'Ticker':<14} {'Stage':>5} {'S2':>5} {'PASS?':>6}  Failing conditions")
    print("-" * 75)
    for _, r in diag.iterrows():
        s2   = r["S2_TOTAL"]
        stg  = int(r["STAGE"])
        ok   = stg == 2 or r["is_cheat"]
        mark = "✅" if ok else "❌"
        fails = []
        if not r["ma_rising"]:    fails.append(f"ma_rising=False(slope={r['slope_%']:.3f}<{r['slope_thresh']:.3f})")
        if not r["price>ma200"]:  fails.append(f"price({r['price']:.1f})<MA200({r['ma200']:.1f})")
        if not r["bullish_ema"]:  fails.append("ema_not_bullish")
        if not r["rs_strong"]:    fails.append(f"rs_weak(vs_bm={r['rs_vs_bm%']:.1f}%)")
        if not r["hh"]:           fails.append("no_HH(40bar)")
        if not r["roc_f>0"]:      fails.append(f"roc_f<0({r['roc_f%']:.1f}%)")
        if s2 < 5.0:              fails.append(f"S2={s2:.1f}<S2_MIN=5")
        if stg != 2 and r["S1"] > s2: fails.append(f"S1({r['S1']:.1f})>S2({s2:.1f})")
        if stg != 2 and r["S3"] > s2: fails.append(f"S3({r['S3']:.1f})>S2({s2:.1f})")
        status = " | ".join(fails) if fails else "All conditions met"
        print(f"{r['ticker']:<14} {stg:>5} {s2:>5.1f} {mark:>6}  {status}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_s2    = (diag["STAGE"] == 2).sum()
    n_cheat = diag["is_cheat"].sum()
    n_pass  = n_s2  # cheat entries are always S2
    print(f"\nSummary: {n_pass}/{len(diag)} correctly classified as Stage 2")
    print(f"Cheat entries: {n_cheat}")

    n_no_ma_rising    = (~diag["ma_rising"]).sum()
    n_below_ma        = (~diag["price>ma200"]).sum()
    n_no_bullish_ema  = (~diag["bullish_ema"]).sum()
    n_rs_weak         = (~diag["rs_strong"]).sum()
    n_no_hh           = (~diag["hh"]).sum()
    n_neg_roc         = (~diag["roc_f>0"]).sum()
    print(f"\nGate failures across all {len(diag)} stocks:")
    print(f"  ma_rising=False   : {n_no_ma_rising}")
    print(f"  price<MA200       : {n_below_ma}")
    print(f"  EMA not bullish   : {n_no_bullish_ema}")
    print(f"  RS weak           : {n_rs_weak}")
    print(f"  No 40-bar HH      : {n_no_hh}")
    print(f"  ROC_fast < 0      : {n_neg_roc}")

    # ── Which gate is most commonly blocking stage 2 ──────────────────────────
    failing = diag[diag["STAGE"] != 2]
    if not failing.empty:
        print(f"\nFor the {len(failing)} non-S2 stocks, component scores:")
        for _, r in failing.iterrows():
            print(f"  {r['ticker']:<14} s2_core={r['s2_core']:.0f} "
                  f"s2_ema={r['s2_ema']:.0f} s2_rs={r['s2_rs']:.0f} "
                  f"s2_hh={r['s2_hh']:.0f} s2_roc={r['s2_roc']:.0f} "
                  f"→ S2={r['S2_TOTAL']:.1f}  "
                  f"(S1={r['S1']:.1f} S3={r['S3']:.1f} S4={r['S4']:.1f})")


if __name__ == "__main__":
    main()
