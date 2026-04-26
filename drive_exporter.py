# =============================================================================
# JSON EXPORTER — Write screener results to a JSON file after each run
# =============================================================================
#
# Writes to LOCAL_JSON_OUTPUT_DIR (set in config.py).
# Point it at your iCloud Drive, Dropbox, or OneDrive folder and the file
# will sync automatically — no Drive API, no quota issues.
#
# If LOCAL_JSON_OUTPUT_DIR is empty, it auto-detects common sync folders:
#   1. iCloud Drive   ~/Library/Mobile Documents/com~apple~CloudDocs/Screener
#   2. Dropbox        ~/Dropbox/Screener
#   3. OneDrive       ~/OneDrive/Screener
#   4. Fallback       <project dir>/exports/
#
# File naming: screener_results_YYYY-MM-DD_<market>_<screener>.json
# Re-running the same day overwrites — no duplicates.
#
# JSON structure (trade mode):
#   {
#     "run_date":  "2026-04-26",
#     "run_time":  "14:30:15",
#     "market":    "india",
#     "screener":  "trade",
#     "summary":   { "stage": 30, "sepa": 25, "rs": 30, "trade": 12, "holdings_alert": 8 },
#     "stage":          [ {...}, ... ],
#     "sepa":           [ {...}, ... ],
#     "rs":             [ {...}, ... ],
#     "trade":          [ {...}, ... ],
#     "holdings_alert": [ {...}, ... ]
#   }
#
# JSON structure (single-screener mode — stage / sepa / rs):
#   {
#     "run_date":  "2026-04-26",
#     "run_time":  "14:30:15",
#     "market":    "india",
#     "screener":  "sepa",
#     "summary":   { "total": 25 },
#     "results":   [ {...}, ... ]
#   }
# =============================================================================

from __future__ import annotations
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import LOCAL_JSON_OUTPUT_DIR

logger = logging.getLogger(__name__)


# =============================================================================
# OUTPUT DIRECTORY RESOLUTION
# =============================================================================

def _resolve_output_dir() -> Path:
    """
    Resolve the directory to write JSON files into.

    Priority:
      1. LOCAL_JSON_OUTPUT_DIR from config.py  (explicit user setting)
      2. iCloud Drive                           (Mac personal account)
      3. Dropbox                                (cross-platform sync)
      4. OneDrive                               (Windows / Microsoft)
      5. <project dir>/exports/                 (local-only fallback)
    """
    home = Path.home()

    # 1. Explicit config
    if LOCAL_JSON_OUTPUT_DIR:
        p = Path(LOCAL_JSON_OUTPUT_DIR).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # 2. iCloud Drive (macOS)
    icloud = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Screener Results"
    if (home / "Library" / "Mobile Documents" / "com~apple~CloudDocs").exists():
        icloud.mkdir(parents=True, exist_ok=True)
        return icloud

    # 3. Dropbox
    dropbox = home / "Dropbox" / "Screener Results"
    if (home / "Dropbox").exists():
        dropbox.mkdir(parents=True, exist_ok=True)
        return dropbox

    # 4. OneDrive (common locations)
    for candidate in [home / "OneDrive", home / "OneDrive - Personal"]:
        if candidate.exists():
            out = candidate / "Screener Results"
            out.mkdir(parents=True, exist_ok=True)
            return out

    # 5. Project-local fallback
    fallback = Path(os.path.dirname(os.path.abspath(__file__))) / "exports"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# =============================================================================
# PUBLIC API
# =============================================================================

def export_results(
    results,           # dict (trade mode) or pd.DataFrame (single screener)
    market:   str,
    screener: str,
) -> Path | None:
    """
    Serialise all screener results to a timestamped JSON file.

    Args:
        results:  dict with keys stage/sepa/rs/trade/holdings_alert  (trade mode)
                  OR a plain pd.DataFrame  (stage/sepa/rs modes)
        market:   "india" | "us" | "ai"
        screener: "stage" | "sepa" | "rs" | "trade"

    Returns:
        Path of the written file on success, None on failure.
    """
    now      = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    filename = f"screener_results_{date_str}_{market}_{screener}.json"

    # ── Build & serialise ─────────────────────────────────────────────────────
    payload   = _build_payload(results, market, screener, date_str, time_str)
    json_str  = json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
    json_bytes = json_str.encode("utf-8")

    # ── Write to sync folder ──────────────────────────────────────────────────
    try:
        out_dir   = _resolve_output_dir()
        out_path  = out_dir / filename
        out_path.write_bytes(json_bytes)
        logger.info(
            f"JSON exported → {out_path}  "
            f"({len(json_bytes) / 1024:.0f} KB, "
            f"{payload['summary']})"
        )
        return out_path
    except Exception as e:
        logger.error(f"JSON export failed for '{filename}': {e}")
        return None


# =============================================================================
# PAYLOAD BUILDER
# =============================================================================

def _build_payload(
    results,
    market:   str,
    screener: str,
    date_str: str,
    time_str: str,
) -> dict:
    """Convert results (dict or DataFrame) into a clean JSON-serialisable dict."""
    base = {
        "run_date": date_str,
        "run_time": time_str,
        "market":   market,
        "screener": screener,
    }

    if isinstance(results, dict):
        stage_df    = results.get("stage",          pd.DataFrame())
        sepa_df     = results.get("sepa",           pd.DataFrame())
        rs_df       = results.get("rs",             pd.DataFrame())
        trade_df    = results.get("trade",          pd.DataFrame())
        holdings_df = results.get("holdings_alert", pd.DataFrame())

        base["summary"] = {
            "stage":          len(stage_df),
            "sepa":           len(sepa_df),
            "rs":             len(rs_df),
            "trade":          len(trade_df),
            "holdings_alert": len(holdings_df),
        }
        base["stage"]          = _df_to_records(stage_df)
        base["sepa"]           = _df_to_records(sepa_df)
        base["rs"]             = _df_to_records(rs_df)
        base["trade"]          = _df_to_records(trade_df)
        base["holdings_alert"] = _df_to_records(holdings_df)

    elif isinstance(results, pd.DataFrame):
        base["summary"] = {"total": len(results)}
        base["results"] = _df_to_records(results)

    else:
        base["summary"] = {"total": 0}
        base["results"] = []

    return base


# =============================================================================
# SERIALISATION HELPERS
# =============================================================================

def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of clean dicts, handling NaN / Inf / Timestamps."""
    if df is None or df.empty:
        return []
    return [
        {k: _clean_val(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _clean_val(v):
    """Normalise a single value to something json.dumps can handle."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if hasattr(v, "item"):      # numpy scalar → native Python
        return v.item()
    return v


def _json_default(obj):
    """Fallback serialiser for types json.dumps can't handle natively."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)
