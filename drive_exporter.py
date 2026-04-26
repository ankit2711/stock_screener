# =============================================================================
# DRIVE EXPORTER — Write screener results to Google Drive after each run
# =============================================================================
#
# Uses OAuth2 user credentials so files are uploaded under YOUR personal
# Google account (your own Drive quota — no service-account limits).
#
# ONE-TIME SETUP
# --------------
# 1. Google Cloud Console → your project → APIs & Services → Credentials
# 2. Create credential → OAuth 2.0 Client ID → Desktop application
# 3. Download JSON → save as the path in GOOGLE_OAUTH_CLIENT_FILE (config.py)
# 4. Enable "Google Drive API" in APIs & Services → Library
# 5. First screener run: a browser tab opens → sign in → grant access
#    A token.json is saved next to oauth_client.json — future runs are silent.
#
# Config keys (config.py)
# ------------------------
#   GOOGLE_OAUTH_CLIENT_FILE   = "oauth_client.json"   # OAuth2 Desktop credentials
#   GOOGLE_DRIVE_FOLDER_ID     = "<folder-id>"          # target Drive folder (shared link → id)
#
# File naming: screener_results_YYYY-MM-DD_<market>_<screener>.json
# Re-running the same day overwrites — no duplicates.
#
# LOCAL FALLBACK
# --------------
# If GOOGLE_OAUTH_CLIENT_FILE or GOOGLE_DRIVE_FOLDER_ID is not configured,
# the exporter falls back to writing JSON to a local sync folder:
#   1. iCloud Drive   ~/Library/Mobile Documents/com~apple~CloudDocs/Screener Results
#   2. Dropbox        ~/Dropbox/Screener Results
#   3. OneDrive       ~/OneDrive/Screener Results
#   4. Fallback       <project dir>/exports/
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

from config import GOOGLE_OAUTH_CLIENT_FILE, GOOGLE_DRIVE_FOLDER_ID, LOCAL_JSON_OUTPUT_DIR

logger = logging.getLogger(__name__)

# OAuth2 scopes — drive.file allows create/update of files this app created.
# Use drive if you also need to overwrite files created outside this app.
_SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# token.json lives next to the oauth client file
_TOKEN_FILE = Path(GOOGLE_OAUTH_CLIENT_FILE).parent / "token.json" if GOOGLE_OAUTH_CLIENT_FILE else None


# =============================================================================
# GOOGLE DRIVE UPLOAD (OAuth2)
# =============================================================================

def _get_drive_service():
    """
    Build an authenticated Drive API service using OAuth2 user credentials.
    On the first call a browser window opens for one-time consent.
    Subsequent calls load the saved token.json silently.
    Returns None if OAuth is not configured.
    """
    if not GOOGLE_OAUTH_CLIENT_FILE or not GOOGLE_DRIVE_FOLDER_ID:
        return None

    client_path = Path(GOOGLE_OAUTH_CLIENT_FILE).expanduser()
    if not client_path.exists():
        logger.warning(
            f"OAuth client file not found: {client_path}  "
            "→ falling back to local sync folder."
        )
        return None

    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        creds = None
        token_path = client_path.parent / "token.json"

        # Load saved token
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)

        # Refresh or re-auth if needed
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(client_path), _SCOPES)
                creds = flow.run_local_server(port=0)
            token_path.write_text(creds.to_json())
            logger.info(f"OAuth token saved → {token_path}")

        return build("drive", "v3", credentials=creds)

    except Exception as e:
        logger.warning(f"Drive OAuth setup failed ({e}) → falling back to local sync folder.")
        return None


def _upload_to_drive(service, json_bytes: bytes, filename: str) -> str | None:
    """
    Upload (or overwrite) a file in the configured Drive folder.
    Returns the Drive file ID on success, None on failure.
    """
    from googleapiclient.http import MediaInMemoryUpload

    folder_id = GOOGLE_DRIVE_FOLDER_ID
    mime = "application/json"

    try:
        # Check if file already exists in the folder (same-day overwrite)
        q = (
            f"name='{filename}' and "
            f"'{folder_id}' in parents and "
            "trashed=false"
        )
        existing = service.files().list(q=q, fields="files(id,name)").execute()
        files   = existing.get("files", [])

        media = MediaInMemoryUpload(json_bytes, mimetype=mime, resumable=False)

        if files:
            # Overwrite existing file
            file_id = files[0]["id"]
            service.files().update(
                fileId=file_id,
                media_body=media,
            ).execute()
            logger.info(f"Drive: updated existing file '{filename}' (id={file_id})")
        else:
            # Create new file
            meta = {"name": filename, "parents": [folder_id]}
            result = service.files().create(
                body=meta,
                media_body=media,
                fields="id",
            ).execute()
            file_id = result.get("id")
            logger.info(f"Drive: created new file '{filename}' (id={file_id})")

        return file_id

    except Exception as e:
        logger.error(f"Drive upload failed for '{filename}': {e}")
        return None


# =============================================================================
# LOCAL SYNC FOLDER (fallback)
# =============================================================================

def _resolve_local_dir() -> Path:
    """
    Resolve local sync directory.
    Priority: explicit config → iCloud Drive → Dropbox → OneDrive → project/exports/
    """
    home = Path.home()

    if LOCAL_JSON_OUTPUT_DIR:
        p = Path(LOCAL_JSON_OUTPUT_DIR).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    icloud = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Screener Results"
    if (home / "Library" / "Mobile Documents" / "com~apple~CloudDocs").exists():
        icloud.mkdir(parents=True, exist_ok=True)
        return icloud

    dropbox = home / "Dropbox" / "Screener Results"
    if (home / "Dropbox").exists():
        dropbox.mkdir(parents=True, exist_ok=True)
        return dropbox

    for candidate in [home / "OneDrive", home / "OneDrive - Personal"]:
        if candidate.exists():
            out = candidate / "Screener Results"
            out.mkdir(parents=True, exist_ok=True)
            return out

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
    Serialise all screener results to a timestamped JSON file, then:
      • Upload to Google Drive (if OAuth is configured), OR
      • Write to local iCloud/Dropbox/OneDrive sync folder (fallback).

    Args:
        results:  dict with keys stage/sepa/rs/trade/holdings_alert  (trade mode)
                  OR a plain pd.DataFrame  (stage/sepa/rs modes)
        market:   "india" | "us" | "ai"
        screener: "stage" | "sepa" | "rs" | "trade"

    Returns:
        Path of the written local file on success, None on failure.
    """
    now      = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    filename = f"screener_results_{date_str}_{market}_{screener}.json"

    # ── Build & serialise ─────────────────────────────────────────────────────
    payload   = _build_payload(results, market, screener, date_str, time_str)
    json_str  = json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
    json_bytes = json_str.encode("utf-8")

    # ── Try Google Drive first ────────────────────────────────────────────────
    service = _get_drive_service()
    if service:
        file_id = _upload_to_drive(service, json_bytes, filename)
        if file_id:
            logger.info(
                f"✓ Drive upload complete: '{filename}'  "
                f"({len(json_bytes) / 1024:.0f} KB, {payload['summary']})"
            )
            # Also write locally so you have a copy
            try:
                local_dir  = _resolve_local_dir()
                local_path = local_dir / filename
                local_path.write_bytes(json_bytes)
                logger.info(f"  Local copy → {local_path}")
                return local_path
            except Exception:
                return None  # Drive upload succeeded; local copy optional

    # ── Fallback: local sync folder ───────────────────────────────────────────
    try:
        out_dir  = _resolve_local_dir()
        out_path = out_dir / filename
        out_path.write_bytes(json_bytes)
        logger.info(
            f"JSON written → {out_path}  "
            f"({len(json_bytes) / 1024:.0f} KB, {payload['summary']})"
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
