#!/usr/bin/env python3
"""
check_benchmark.py — Deep diagnostic for India benchmark fetch issues.

Run from the project root:
    python check_benchmark.py

Tests each layer independently so you can see exactly where it breaks.
"""

import sys
import traceback
from datetime import datetime, timedelta

START = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
END   =  datetime.today().strftime("%Y-%m-%d")

SEP = "=" * 65


# ─── 0. Environment ───────────────────────────────────────────────────────────
print(SEP)
print("LAYER 0: Environment")
print(SEP)

import sys, platform
print(f"  Python : {sys.version}")
print(f"  OS     : {platform.system()} {platform.release()}")

try:
    import yfinance as yf
    print(f"  yfinance: {yf.__version__}")
except ImportError as e:
    print(f"  yfinance: NOT INSTALLED — {e}")
    sys.exit(1)

try:
    import pandas as pd
    print(f"  pandas  : {pd.__version__}")
except ImportError as e:
    print(f"  pandas  : NOT INSTALLED — {e}")
    sys.exit(1)


# ─── 1. Raw HTTP to Yahoo Finance (no yfinance) ───────────────────────────────
print()
print(SEP)
print("LAYER 1: Raw HTTP to Yahoo Finance API")
print(SEP)

import urllib.request, json

_test_url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?interval=1d&range=5d"
try:
    req = urllib.request.Request(_test_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        body   = json.loads(resp.read())
        result = body.get("chart", {}).get("result")
        if result:
            closes = result[0]["indicators"]["quote"][0]["close"]
            print(f"  ✓  Yahoo Finance reachable — ^NSEI last 5 closes: {closes}")
        else:
            err = body.get("chart", {}).get("error")
            print(f"  ✗  Yahoo Finance returned error: {err}")
except Exception as e:
    print(f"  ✗  Cannot reach Yahoo Finance: {type(e).__name__}: {e}")
    print("     → Check network / VPN / firewall. yfinance will also fail.")


# ─── 2. yfinance Ticker.history (single-ticker, simplest API) ─────────────────
print()
print(SEP)
print("LAYER 2: yfinance Ticker.history() — simplest call")
print(SEP)

for sym in ["^NSEI", "^CRSLDX", "RELIANCE.NS"]:
    try:
        t   = yf.Ticker(sym)
        df  = t.history(period="5d", auto_adjust=True)
        if df is not None and not df.empty:
            print(f"  ✓  {sym:<18} {len(df)} rows  last close={df['Close'].iloc[-1]:,.2f}")
        else:
            print(f"  ✗  {sym:<18} empty DataFrame")
    except Exception as e:
        print(f"  ✗  {sym:<18} {type(e).__name__}: {e}")


# ─── 3. yfinance download() — minimal parameters ──────────────────────────────
print()
print(SEP)
print("LAYER 3: yf.download() — minimal parameters (no threads/group_by)")
print(SEP)

for sym in ["^NSEI", "^CRSLDX", "RELIANCE.NS"]:
    try:
        df = yf.download(sym, start=START, end=END, progress=False)
        if df is not None and not df.empty:
            print(f"  ✓  {sym:<18} {len(df)} rows  cols={list(df.columns)}")
        else:
            print(f"  ✗  {sym:<18} empty DataFrame")
    except Exception as e:
        print(f"  ✗  {sym:<18} {type(e).__name__}: {e}")


# ─── 4. yfinance download() — with threads=False ──────────────────────────────
print()
print(SEP)
print("LAYER 4: yf.download() — threads=False (used by fetcher.py)")
print(SEP)

for sym in ["^NSEI", "^CRSLDX", "RELIANCE.NS"]:
    try:
        # In yfinance >= 0.2.38 the `threads` parameter was removed.
        # We detect this gracefully below.
        df = yf.download(sym, start=START, end=END, progress=False, threads=False)
        if df is not None and not df.empty:
            print(f"  ✓  {sym:<18} {len(df)} rows")
        else:
            print(f"  ✗  {sym:<18} empty DataFrame")
    except TypeError as e:
        if "threads" in str(e):
            print(f"  !  {sym:<18} 'threads' param not supported in this yfinance version — {e}")
            print(f"       → fetcher.py must remove threads=False for this version")
        else:
            print(f"  ✗  {sym:<18} TypeError: {e}")
    except Exception as e:
        print(f"  ✗  {sym:<18} {type(e).__name__}: {e}")


# ─── 5. yfinance download() — with auto_adjust=True ──────────────────────────
print()
print(SEP)
print("LAYER 5: yf.download() — auto_adjust=True (used by fetcher.py)")
print(SEP)

for sym in ["^NSEI", "^CRSLDX", "RELIANCE.NS"]:
    try:
        df = yf.download(sym, start=START, end=END, progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            print(f"  ✓  {sym:<18} {len(df)} rows  cols={list(df.columns)}")
        else:
            print(f"  ✗  {sym:<18} empty DataFrame")
    except Exception as e:
        print(f"  ✗  {sym:<18} {type(e).__name__}: {e}")


# ─── 6. yfinance cache location check ─────────────────────────────────────────
print()
print(SEP)
print("LAYER 6: yfinance internal cache")
print(SEP)

try:
    from yfinance import cache as yfc
    print(f"  yfinance.cache module found")
    if hasattr(yfc, "get_cache_dir"):
        print(f"  cache dir : {yfc.get_cache_dir()}")
    elif hasattr(yfc, "_TzCacheManager"):
        print(f"  TzCacheManager present")
    else:
        print(f"  cache attrs: {[a for a in dir(yfc) if not a.startswith('_')]}")
except ImportError:
    print("  yfinance.cache module not found (older version)")

try:
    # Try fetching after explicitly setting cache to a known good path
    import os, tempfile
    tmp = tempfile.mkdtemp()
    yf.set_tz_cache_location(tmp)
    df = yf.Ticker("RELIANCE.NS").history(period="5d")
    if df is not None and not df.empty:
        print(f"  ✓  After set_tz_cache_location({tmp!r}): RELIANCE.NS works")
    else:
        print(f"  ✗  After set_tz_cache_location: RELIANCE.NS still empty")
except AttributeError:
    print("  set_tz_cache_location not available (yfinance >= 0.2.28)")
except Exception as e:
    print(f"  ✗  Cache test failed: {e}")


# ─── Summary ──────────────────────────────────────────────────────────────────
print()
print(SEP)
print("DONE — share the full output above to diagnose the issue.")
print(SEP)
