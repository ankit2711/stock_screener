# =============================================================================
# SCREEN: Relative Strength (RS) Rating
# IBD-style RS — compares stock performance vs benchmark over multiple periods.
# Score 0–100. Stocks scoring 70+ pass.
#
# *** PASTE YOUR PINE SCRIPT LOGIC HERE ***
# =============================================================================

import numpy as np
import pandas as pd
from config import RS_RATING


# Module-level cache: benchmark data keyed by ticker symbol
_benchmark_cache: dict[str, pd.Series] = {}


def screen_rs_rating(df: pd.DataFrame, benchmark_df: pd.DataFrame) -> dict:
    """
    Args:
        df:           Stock OHLCV DataFrame
        benchmark_df: Benchmark (S&P 500 or Nifty 50) OHLCV DataFrame
    """
    try:
        return _run_rs_rating(df, benchmark_df)
    except Exception as e:
        return {"passed": False, "score": 0.0, "detail": f"Error: {e}"}


def _run_rs_rating(df: pd.DataFrame, benchmark_df: pd.DataFrame) -> dict:
    cfg = RS_RATING
    close     = df["close"]
    bench_cls = benchmark_df["close"]

    # Align dates
    common_idx = close.index.intersection(bench_cls.index)
    if len(common_idx) < max(cfg["period_weights"].keys()):
        return {"passed": False, "score": 0.0, "detail": "Not enough common dates with benchmark"}

    close     = close.loc[common_idx]
    bench_cls = bench_cls.loc[common_idx]

    # Weighted performance score (IBD-style)
    weighted_stock = 0.0
    weighted_bench = 0.0
    total_weight   = 0.0

    for period, weight in cfg["period_weights"].items():
        if len(close) < period:
            continue
        stock_perf = (close.iloc[-1] / close.iloc[-period]) - 1
        bench_perf = (bench_cls.iloc[-1] / bench_cls.iloc[-period]) - 1
        weighted_stock += stock_perf * weight
        weighted_bench += bench_perf * weight
        total_weight   += weight

    if total_weight == 0:
        return {"passed": False, "score": 0.0, "detail": "No valid periods"}

    # RS line: stock relative to benchmark
    rs_value = weighted_stock / total_weight - weighted_bench / total_weight

    # Convert to 0–100 score
    # rs_value of +0.30 = top performer (score ~95), -0.30 = bottom (score ~5)
    rs_score = int(np.clip(50 + rs_value * 150, 0, 99))

    passed = rs_score >= cfg["min_rs_score"]
    detail = f"RS Rating: {rs_score}/100" + (" ✓" if passed else f" (need {cfg['min_rs_score']}+)")

    return {
        "passed":   passed,
        "score":    round(rs_score / 100, 2),
        "detail":   detail,
        "rs_score": rs_score,
    }
