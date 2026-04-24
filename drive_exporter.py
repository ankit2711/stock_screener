# =============================================================================
# DRIVE EXPORTER — Write screener results to JSON and upload to Google Drive
# =============================================================================
#
# Produces: screener_results_YYYY-MM-DD_<market>_<screener>.json
# Uploads to: the configured GOOGLE_DRIVE_RESULTS_FOLDER_ID
#
# JSON structure (trade mode):
#   {
#     "run_date":  "2026-04-23",
#     "run_time":  "14:30:15",
#     "market":    "india",
#     "screener":  "trade",
#     "summary":   { "stage": 30, "sepa": 25, "rs": 30, "trade": 12, "holdings_alert": 8 },
#     "stage":     [ {...}, ... ],
#     "sepa":      [ {...}, ... ],
#     "rs":        [ {...}, ... ],
#     "trade":     [ {...}, ... ],
#     "holdings_alert": [ {...}, ... ]
#   }
#
# JSON structure (single-screener mode):
#   {
#     "run_date":  "2026-04-23",
#     "run_time":  "14:30:15",
#     "market":    "india",
#     "screener":  "sepa",
#     "summary":   { "total": 25 },
#     "results":   [ {...}, ... ]
#   }
# =============================================================================

from __future__ import annotations
import io
import json
import logging
import math
import os
from datetime import datetime

import pandas as pd
from google.oauth2.service_account import Credentials

from config import GOOGLE_SHEETS_CREDENTIALS_FILE, GOOGLE_DRIVE_RESULTS_FOLDER_ID

logger = logging.getLogger(__name__)

_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]


# =============================================================================
# PUBLIC API
# =============================================================================

def export_results(
    results,          # dict (trade mode) or pd.DataFrame (single screener)
    market:   str,
    screener: str,
) -> str | None:
    """
    Serialise screener results to JSON and upload to Google Drive.

    Args:
        results:  dict with keys stage/sepa/rs/trade/holdings_alert  (trade mode)
                  OR a plain pd.DataFrame  (stage/sepa/rs modes)
        market:   "india" | "us" | "ai"
        screener: "stage" | "sepa" | "rs" | "trade"

    Returns:
        Google Drive file ID on success, None on failure.
    """
    now       = datetime.now()
    date_str  = now.strftime("%Y-%m-%d")
    time_str  = now.strftime("%H:%M:%S")
    filename  = f"screener_results_{date_str}_{market}_{screener}.json"

    # ── Build payload ─────────────────────────────────────────────────────────
    payload = _build_payload(results, market, screener, date_str, time_str)

    # ── Serialise ─────────────────────────────────────────────────────────────
    json_str  = json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
    json_bytes = json_str.encode("utf-8")

    # ── Write local copy ──────────────────────────────────────────────────────
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    try:
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        logger.info(f"JSON written locally: {filename}  ({len(json_bytes)/1024:.0f} KB)")
    except Exception as e:
        logger.warning(f"Could not write local JSON: {e}")

    # ── Upload to Google Drive ────────────────────────────────────────────────
    if not GOOGLE_DRIVE_RESULTS_FOLDER_ID:
        logger.warning("GOOGLE_DRIVE_RESULTS_FOLDER_ID not set in config.py — skipping Drive upload")
        return None

    file_id = _upload_to_drive(json_bytes, filename)
    return file_id


# =============================================================================
# PAYLOAD BUILDER
# =============================================================================

def _build_payload(results, market: str, screener: str, date_str: str, time_str: str) -> dict:
    """Convert results (dict or DataFrame) into a clean JSON-serialisable dict."""

    base = {
        "run_date": date_str,
        "run_time": time_str,
        "market":   market,
        "screener": screener,
    }

    if isinstance(results, dict):
        # Trade mode — multiple DataFrames
        stage_df         = results.get("stage",          pd.DataFrame())
        sepa_df          = results.get("sepa",           pd.DataFrame())
        rs_df            = results.get("rs",             pd.DataFrame())
        trade_df         = results.get("trade",          pd.DataFrame())
        holdings_df      = results.get("holdings_alert", pd.DataFrame())

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


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of clean dicts, handling NaN / inf / Timestamps."""
    if df is None or df.empty:
        return []
    return [
        {k: _clean_val(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _clean_val(v):
    """Normalise a single cell value to something JSON-safe."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if hasattr(v, "item"):          # numpy scalar
        return v.item()
    return v


def _json_default(obj):
    """Fallback serialiser for types json.dumps can't handle."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


# =============================================================================
# GOOGLE DRIVE UPLOAD
# =============================================================================

def _upload_to_drive(data: bytes, filename: str) -> str | None:
    """
    Upload bytes to the configured Google Drive folder.
    Creates a new file or updates an existing file with the same name
    (so re-running the same day overwrites rather than duplicates).

    Returns the Drive file ID, or None on failure.
    """
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseUpload
    except ImportError:
        logger.error("google-api-python-client not installed — run: pip install google-api-python-client")
        return None

    try:
        creds   = Credentials.from_service_account_file(GOOGLE_SHEETS_CREDENTIALS_FILE, scopes=_SCOPES)
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        logger.error(f"Drive auth failed: {e}")
        return None

    folder_id = GOOGLE_DRIVE_RESULTS_FOLDER_ID
    mime      = "application/json"

    try:
        # Check if a file with this name already exists in the folder
        query = (
            f"name='{filename}' "
            f"and '{folder_id}' in parents "
            f"and mimeType='{mime}' "
            f"and trashed=false"
        )
        existing = service.files().list(q=query, fields="files(id,name)").execute()
        files    = existing.get("files", [])

        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=False)

        if files:
            # Update in place — same URL, no duplicate
            file_id = files[0]["id"]
            service.files().update(
                fileId=file_id,
                media_body=media,
            ).execute()
            logger.info(f"Drive: updated  '{filename}'  (id={file_id})")
        else:
            # Create new file in folder
            meta    = {"name": filename, "parents": [folder_id]}
            result  = service.files().create(
                body=meta,
                media_body=media,
                fields="id",
            ).execute()
            file_id = result["id"]
            logger.info(f"Drive: created  '{filename}'  (id={file_id})")

        return file_id

    except Exception as e:
        logger.error(f"Drive upload failed for '{filename}': {e}")
        return None
