# =============================================================================
# STOCK SCREENER — MASTER CONFIG
# Edit this file to tune all parameters, weights, and settings
# =============================================================================

# -----------------------------------------------------------------------------
# API KEYS
# -----------------------------------------------------------------------------
TWELVE_DATA_API_KEY = "e62b654349764ac98d51de0fe39c64e4"   # https://twelvedata.com/
GOOGLE_SHEETS_CREDENTIALS_FILE = "credentials.json" # path to your service account JSON
# Separate Google Sheet IDs for each market (each gets its own file with date-based tabs).
# Create two Google Sheets, share both with the service account email (Editor), then paste IDs below.
GOOGLE_SHEET_ID_INDIA = "1jXzBPKXXFWykibnK6KYHvPSDgCFJAScCDY0bI5EbV1c"  # India screener sheet
GOOGLE_SHEET_ID_US    = "1UlHcYMW56Pv0P8YAj33N9wCMI6obs8VrzFwIqUFCjsc"  # US screener sheet
GOOGLE_SHEET_ID_AI    = "1vqrDWPSFKtJhYqarrtfxNDCnnKSrc32J7ZN1mYPL-k8"  # AI Theme screener sheet

# Google Drive — JSON export (OAuth2 user credentials, NOT service account)
# -------------------------------------------------------------------------
# ONE-TIME SETUP:
#   1. Cloud Console → Credentials → Create → OAuth 2.0 Client ID → Desktop app
#   2. Download JSON → save as the path below
#   3. Enable "Google Drive API" in APIs & Services → Library
#   4. First run: browser opens for consent → token.json saved automatically
#
# Get folder ID from the Drive URL:
#   https://drive.google.com/drive/folders/<FOLDER_ID_HERE>
#
GOOGLE_OAUTH_CLIENT_FILE   = "oauth_client.json"   # OAuth2 Desktop credentials (not service account)
GOOGLE_DRIVE_FOLDER_ID     = "18eOnNUaZbOZp7ddZ8ZldZrLEditSDKBt"  # target Drive folder

# Local sync folder — used as fallback when Drive is not configured / offline.
# Leave blank ("") to auto-detect: iCloud Drive → Dropbox → OneDrive → <project>/exports/
LOCAL_JSON_OUTPUT_DIR = ""

# Backward-compat alias (used by any legacy code that still imports GOOGLE_SHEET_ID)
GOOGLE_SHEET_ID = GOOGLE_SHEET_ID_INDIA

# How many date-tabs to keep before the oldest one is pruned (0 = keep all)
MAX_HISTORY_TABS = 30

# -----------------------------------------------------------------------------
# MARKET UNIVERSE & FILTERS
# -----------------------------------------------------------------------------
US_MIN_MARKET_CAP_USD = 2_000_000_000        # $2B minimum — covers mid-cap growth (Minervini range)
INDIA_MIN_MARKET_CAP_INR = 3_000_000_000    # ₹300 Crore minimum

# Tickers that bypass the US_MIN_MARKET_CAP_USD filter.
# Use this for specific stocks you want tracked regardless of market cap
# (e.g. high-conviction setups in the $300M–$2B range).
US_WATCHLIST_TICKERS: list[str] = [
    "NNDM",    # Nano Dimension — $330M, below $2B floor
]

# Minimum average daily dollar volume — exchange-aware liquidity filter
# Applied in ranker_stage, ranker_sepa, ranker_rs to remove untradeable stocks.
# BSE-only stocks (.BO suffix) have wider spreads and thinner order books;
# they require a higher minimum to avoid slippage risk.
# These are in local currency (INR for India, USD for US).
MIN_AVG_DOLLAR_VOL_NSE = 10_000_000    # ₹1 crore/day  — NSE stocks
MIN_AVG_DOLLAR_VOL_BSE = 10_000_000    # ₹1 crore/day  — BSE-only stocks (wider spreads)
MIN_AVG_DOLLAR_VOL_US  = 10_000_000    # $1M/day        — US stocks
MIN_AVG_DOLLAR_VOL_AI  =  1_000_000    # $1M/day        — AI theme (curated global list, lower bar)

# Number of top stocks to write to each sheet tab (display cap)
TOP_N_US    = 30
TOP_N_INDIA = 30
TOP_N_AI    = 100    # AI theme is ~150 stocks — return all that pass Stage 2

# Computation pool size — used internally by run_trade_scan to build stage/sepa/rs
# maps BEFORE truncating to the display cap above.
#
# WHY THIS EXISTS:
#   In a bull market India can have 150-300 genuine Stage-2 stocks.  Capping the
#   individual screeners at TOP_N_INDIA=30 before building the Trade Candidates
#   pipeline means a stock ranked #35 in Stage + #8 in RS + #12 in SEPA (a
#   legitimately high-conviction setup) is invisible.  The pool is larger so Trade
#   Candidates can see more valid setups; the display cap keeps each tab readable.
#
POOL_N_INDIA = 80   # Stage/SEPA/RS internal pool fed into Trade Candidates
POOL_N_US    = 60
POOL_N_AI    = 100  # AI is a curated list — no extra pool needed

# How many calendar days of history to fetch per ticker (550 = ~2.2 years / ~390 trading days)
# WHY 550 NOT 365:
#   EMA200 needs 200 bars to initialise cleanly.  With 365 calendar days we only get
#   ~252 trading days — exactly enough, but any gap days, earnings suspensions, or
#   newly-listed stocks produce a partially-initialised EMA200 that creates false
#   Stage signals.  550 days gives ~390 trading bars: a 190-bar warm-up buffer for
#   EMA200 so the first displayed value is always well-converged.
HISTORY_DAYS = 550

# -----------------------------------------------------------------------------
# DATA FETCH SETTINGS
# -----------------------------------------------------------------------------
# yfinance (primary) — chunked parallel downloads, no API key needed
#   Larger chunks = faster but higher chance of a chunk-level timeout.
#   200 is a good balance; reduce to 100 if you see timeout errors.
YF_CHUNK_SIZE  = 200   # tickers per yfinance download call
YF_RETRY_LIMIT = 3     # max retry attempts per chunk
YF_RETRY_DELAY = 5     # seconds between retries (multiplied by attempt number)

# Market fetch order — first entry is processed first in Step 2.
# India (1 539 tickers) finishes faster for a quick smoke-test run.
# Change to ["us", "india"] to keep the original order.
FETCH_MARKETS_ORDER = ["india", "us"]

# -----------------------------------------------------------------------------
# SCREEN WEIGHTS (tune these — must sum to 1.0)
# Each stock gets: score = sum(weight * passed) for each screen
# -----------------------------------------------------------------------------
SCREEN_WEIGHTS = {
    "vcp":              0.25,   # Volatility Contraction Pattern
    "darvas":           0.20,   # Darvas Box breakout
    "volume_breakout":  0.20,   # Volume surge above average
    "high_52w":         0.15,   # Near / breaking 52-week high
    "rs_rating":        0.10,   # Relative Strength vs index
    "ma_alignment":     0.10,   # Moving average alignment (bullish stack)
}

# -----------------------------------------------------------------------------
# VCP PARAMETERS
# -----------------------------------------------------------------------------
VCP = {
    "trend_lookback_days":      50,     # Days to establish prior uptrend
    "min_trend_gain_pct":       20,     # Min % gain before contractions start
    "num_contractions":          3,     # Number of contractions to look for
    "max_contraction_range_pct": 15,    # Max high-low range during contraction (%)
    "volume_dry_up_ratio":       0.7,   # Volume during contraction vs avg (< this)
    "breakout_volume_ratio":     1.5,   # Volume on breakout day vs avg (> this)
    "pivot_lookback_days":       10,    # Days to look back for pivot high
}

# -----------------------------------------------------------------------------
# DARVAS BOX PARAMETERS
# -----------------------------------------------------------------------------
DARVAS = {
    "box_lookback_days":        20,     # Days to form a box
    "box_top_tolerance_pct":     1.0,   # % tolerance for box top confirmation
    "box_bottom_tolerance_pct":  1.0,   # % tolerance for box bottom confirmation
    "breakout_volume_ratio":     1.5,   # Volume on breakout vs avg
    "min_box_duration_days":     5,     # Minimum days price must consolidate in box
}

# -----------------------------------------------------------------------------
# VOLUME BREAKOUT PARAMETERS
# -----------------------------------------------------------------------------
VOLUME_BREAKOUT = {
    "avg_volume_days":          20,     # Rolling average window for volume
    "breakout_volume_ratio":     2.0,   # Today's volume must be > X * avg
    "min_price_change_pct":      2.0,   # Minimum price % move on breakout day
    "price_near_high_pct":       5.0,   # Price must be within X% of N-day high
    "price_high_lookback_days": 50,     # N-day high lookback
}

# -----------------------------------------------------------------------------
# 52-WEEK HIGH PARAMETERS
# -----------------------------------------------------------------------------
HIGH_52W = {
    "lookback_days":            252,    # Trading days in 52 weeks
    "within_pct":                 3.0,  # Price within X% of 52W high = passes
    "min_above_50dma_pct":        5.0,  # Price must be X% above 50-day MA
}

# -----------------------------------------------------------------------------
# RELATIVE STRENGTH (RS) RATING PARAMETERS
# -----------------------------------------------------------------------------
RS_RATING = {
    "benchmark_us":    "SPY",      # S&P 500 ETF — US benchmark
    "benchmark_india": "^CRSLDX",  # Nifty 500 — most reliable Indian index on yfinance
    "benchmark_ai":    "QQQ",      # Nasdaq 100 ETF — best proxy for AI/tech theme
    "period_weights": {                 # Performance weighting (IBid-style)
        63:  0.40,                      # 3-month performance weight
        126: 0.20,                      # 6-month
        189: 0.20,                      # 9-month
        252: 0.20,                      # 12-month
    },
    "min_rs_score": 70,                 # Minimum RS score to pass (0-100)
}

# -----------------------------------------------------------------------------
# MOVING AVERAGE ALIGNMENT PARAMETERS
# -----------------------------------------------------------------------------
MA_ALIGNMENT = {
    "short_ma":  20,    # Short-term MA (days)
    "mid_ma":    50,    # Mid-term MA
    "long_ma":  200,    # Long-term MA
    # Bullish stack: price > short > mid > long
    "require_price_above_short": True,
    "require_short_above_mid":   True,
    "require_mid_above_long":    True,
    "min_slope_pct":             0.5,   # 50-day MA must be sloping up X% over 10 days
}

# -----------------------------------------------------------------------------
# SEPA (Minervini) SCREENER PARAMETERS
# Tune these to adjust entry-quality scoring.
# Weights must sum to 1.0.
# -----------------------------------------------------------------------------
SEPA_WEIGHTS = {
    # Path B (VCP Base) weights — must sum to 1.0
    # Path A (Fresh Breakout) uses hardcoded weights in sepa.py
    #
    # pivot_proximity tripled to 0.15: the single biggest driver of trade output.
    # SEPA was sorting by base quality (VCP, RS) which put deep-in-base stocks at
    # the top and AT_PIVOT actionable stocks at the bottom — never reaching the trade
    # ranker's top-30 cap. Raising pivot weight surfaces near-pivot stocks naturally.
    # VCP reduced to 0.12: time_compressed bonus and tight-close bonus still reward
    # genuine VCPs; the base weight can be lower.
    # RS reduced to 0.20: still the primary leading indicator — just less dominant.
    # vol_character reduced to 0.18: slight trim to fund pivot increase.
    "vcp_contractions":  0.12,   # VCP contractions + time compression bonus
    "vol_character":     0.18,   # accumulation ratio (up-day vol / down-day vol) + churn
    "atr_contraction":   0.15,   # ATR first-half vs second-half of base (coiling signal)
    "rs_leading":        0.20,   # RS line at new high BEFORE price breakout (primary signal)
    "vol_dry_up":        0.12,   # recent 5d vol vs pre-base 20d vol + 3-bar spring completion
    "current_tightness": 0.08,   # CV of last-third of base closes + tight-close streak bonus
    "pivot_proximity":   0.15,   # distance from base high — raised 5%→15% to surface AT_PIVOT
}

# How many top stocks the SEPA screener returns (can differ from Stage screener)
TOP_N_SEPA_US    = 30
TOP_N_SEPA_INDIA = 30

# How many trade candidates the Trade ranker returns
TOP_N_TRADE = 10

# How many RS Leader stocks the RS Leaders screener returns
TOP_N_RS_INDIA = 30
TOP_N_RS_US    = 30

# SEPA output columns (match keys in ranker_sepa._result_to_row) — standalone --screener sepa mode
# Column order: identity → verdict → execution → primary signals → base detail → stage context → meta
OUTPUT_COLUMNS_SEPA = [
    "Rank",
    "Ticker",
    "Company",
    # ── Verdict + persistence ─────────────────────────────────────────────────
    "Setup",             # 🟢 Actionable VCP / 🟡 Ready / 🔵 Forming / 🔴 Extended / 🚫 No Base
    "First Entry",       # date this setup first appeared (base age proxy)
    "Days Listed",       # calendar days since first entry — base maturity indicator
    # ── Execution ─────────────────────────────────────────────────────────────
    "Breakout State",    # AT_PIVOT | IN_BASE | BREAKOUT | WEAK_BREAKOUT | FADING | EXTENDED
    "Pivot Dist %",      # how far from the pivot — the entry timing signal
    "Stop Dist %",       # (price − stop) / price — position size input
    # ── Primary conviction signals ────────────────────────────────────────────
    "RS Leading",        # ✓ RS line at new high before price — primary signal
    "VCP Count",         # ≥ 2 = proper VCP
    "Vol Dry %",         # recent vol vs pre-base vol — dry = constructive
    "Last Tightest",     # tightest recent contraction depth — coiling signal
    # ── Score & path ─────────────────────────────────────────────────────────
    "SEPA Score",        # 0–100 composite ranking metric
    "Path",              # A: Fresh Breakout | B: VCP Base
    # ── Path A columns ────────────────────────────────────────────────────────
    "Vol Surge",
    "S1 Base CV%",
    "Extension %",
    # ── Path B base detail ────────────────────────────────────────────────────
    "Base Bars",
    "Base Depth %",
    "ATR Contract",
    "Accum Ratio",
    "CV Tight %",
    "Churn Bars",
    "Base Count",
    "Base ×Mult",
    # ── Stage & RS context ────────────────────────────────────────────────────
    "Stage",
    "Stage S2",
    "Duration",
    "RS Status",
    "RS vs Bench %",
    "Momentum",
    "ROC Fast %",
    # ── Market regime ─────────────────────────────────────────────────────────
    "Market Regime",     # Bull (5/5) … Bear (≤1/5) — shown on every row for context
    # ── Liquidity & risk ──────────────────────────────────────────────────────
    "Avg $ Vol",
    "Beta",
    "EMA Dist Fast",
    "EMA Dist Mid",
    "EMA Dist Slow",
    "Vol Conv",
    "Market Cap",
    "Sector",
    # TradingView removed — Ticker is already a clickable hyperlink
    # Last Updated removed — operational noise, no trading value
]

# Streamlined Stage columns for the Stage tab written by trade mode
# (fewer columns than full Stage output — focused on watchlist quality)
# Column order: identity → action signal → RS quality → momentum → volume → context → tracking
OUTPUT_COLUMNS_TRADE_STAGE = [
    "Rank",
    "Ticker",           # clickable → TradingView
    "Company",
    # ── Action signal + persistence ───────────────────────────────────────────
    "Entry Signal",     # 🟢 Cheat Entry / 🟡 EMA Pullback / 🔵 Near Pivot / ⚪ Extended
    "Streak",           # consecutive days in Stage Leaders — higher = more institutionally confirmed
    "First Entry",      # date this stock first appeared in Stage Leaders (never resets)
    # ── Trend & RS quality ────────────────────────────────────────────────────
    "Stage",            # Stage 2 ↑ / Stage 1 / Stage 3 etc.
    "RS Status",        # RS Strong ↑↑ / RS Strong ↑ / Neutral / Weak
    "RS vs Bench %",    # outperformance vs benchmark — quantifies the RS signal
    # ── Momentum & volume ─────────────────────────────────────────────────────
    "Momentum",         # ↑↑ Strong / ↑ Rising / → Flat / ↓ Weak
    "Vol Conviction",   # Very High / High / Normal / Low
    "Avg $ Vol",
    "Sector",
    # ── Exit tracking (only relevant for exited rows at bottom) ───────────────
    "Exit Date",        # blank while active; date when stock dropped out (kept 14 days)
    "Exit Reason",      # why it left: e.g. "Stage 2 structure lost" / "EMA21 hold failed"
]

# Streamlined SEPA columns for the SEPA tab written by trade mode
# (prioritises entry execution over research detail)
# Column order: identity → verdict → execution prices → pattern quality → weekly context → tracking
OUTPUT_COLUMNS_TRADE_SEPA = [
    "Rank",
    "Ticker",
    "Company",
    # ── Verdict + persistence ─────────────────────────────────────────────────
    "Setup",            # 🟢/🟡/🔵/🔴 — action verdict at a glance
    "Streak",           # consecutive days in SEPA Setups — longer = base maturing, entry near
    "First Entry",      # date this setup first appeared (base age proxy)
    # ── Execution ─────────────────────────────────────────────────────────────
    "Breakout State",   # AT_PIVOT / BREAKOUT / WEAK_BREAKOUT / IN_BASE
    "Entry ₹",          # exact buy price
    "Stop ₹",           # exact stop-loss price
    "Pivot Dist %",     # how far from the pivot (+/-)
    "Stop Dist %",      # (price − stop) / price — your position size input
    # ── Pattern quality ───────────────────────────────────────────────────────
    "RS Leading",       # ✓ RS line at new high before price — primary conviction signal
    "VCP Count",        # ≥ 2 = proper VCP
    "Vol Signal",       # "dry 0.42×" (pre-breakout) or "surge 2.1×" (breakout)
    # ── Weekly structure ──────────────────────────────────────────────────────
    "Weekly Stage",     # W-S2 ✓ is the gold standard
    "TheWrap",          # weekly 10W/20W/40W EMA signal — TW_BULLISH best, TW_FADING reduce
    # ── Score & context ───────────────────────────────────────────────────────
    "SEPA Score",       # ranking metric — how good the base is
    "Sector",
    # ── Exit tracking ─────────────────────────────────────────────────────────
    "Exit Date",        # blank while active; date when stock dropped out (kept 14 days)
    "Exit Reason",      # why it left: e.g. "Setup invalidated" / "Breakout extended"
    # RSI(14) removed — entry timing detail; Breakout State + Vol Signal already cover timing
]

# RS Leaders output columns (matches ranker_rs._result_to_row)
# Column order: identity → primary RS signal → relative strength numbers → structure → context → tracking
OUTPUT_COLUMNS_RS = [
    "Rank",
    "Ticker",
    "Company",
    # ── Primary RS signal + persistence ──────────────────────────────────────
    "RS Leads Price",     # 🌟 Leads = RS at new high while price still >5% below its own 52w high
                          # ✓ Confirms = RS at new high with price also near its high — read this first
    "Streak",             # consecutive days in RS Leaders — key institutional accumulation signal
    "First Entry",        # date this stock first achieved RS Leader status
    # ── RS scoring ────────────────────────────────────────────────────────────
    "RS Score",           # 0–100 composite RS Leader score
    "RS at 52w High",     # ✓ if within 3% of 52w RS high (primary filter)
    "RS % from High",     # % below 52-week RS line high (0 = AT high)
    # ── Relative strength vs benchmark ───────────────────────────────────────
    "Resilience",         # Strong Leader / Leader / Neutral / Laggard
    "Resilience Δ",       # bench_off − stock_off (positive = stock holding up better; the key number)
    "Stock Off 52w %",    # how far stock price is below its own 52w high
    # ── Volume & structure ────────────────────────────────────────────────────
    "Vol Dry",            # ✓ if volume drying up — constructive pre-breakout behaviour
    "Stage",              # Stage 2 ↑ / Stage 1 Accum / Stage 3 / Stage 4
    # ── Market context ────────────────────────────────────────────────────────
    "FTD Signal",         # ✓ FTD if Follow-Through Day detected on benchmark — buy trigger
    "Market Regime",      # ✅ At High / 🟡 Pullback / 🟠 Correction / 🔴 Deep / 🚨 Bear
    # ── Liquidity ─────────────────────────────────────────────────────────────
    "Avg $ Vol",
    "Sector",
    # ── Exit tracking ─────────────────────────────────────────────────────────
    "Exit Date",          # blank while active; date when stock dropped out (kept 14 days)
    "Exit Reason",        # why it left: e.g. "RS leadership lost" / "Stage 2 broken"
    # Removed (too granular / redundant):
    #   RS New High — RS at 52w High already covers this
    #   Bench Off 52w % — market-wide, same for every row
    #   RS vs Bench 1M / 3M — RS % from High + Resilience Δ already tell the story
    #   Accum Ratio / Vol Dry Ratio — Vol Dry boolean is the actionable signal
    #   EMA200 Slope / Price vs EMA200 — Stage already encodes EMA200 relationship
    #   Base Forming / Base Depth % / Consol Bars — too granular for this tab
    #   Market Cap / First Reported / Days Listed — operational meta
]

# Trade Candidates output columns (matches ranker_trade._build_output)
# Column order: identity → tier/action → execution → conviction → scores → context → tracking
OUTPUT_COLUMNS_TRADE = [
    "Rank",
    "Ticker",           # clickable → TradingView
    "Company",
    # ── Action & entry ────────────────────────────────────────────────────────
    "Action",           # 🟢 BUY NOW / 🔔 NEAR PIVOT / 👁 WATCHLIST
    "Entry Quality",    # 🟢 Cheat Entry / 🟢 Fresh Breakout / 🔔 At Pivot / etc.
    "Streak",           # consecutive days as Trade Candidate
    "First Entry",      # date first appeared as trade candidate
    # ── Execution levels ──────────────────────────────────────────────────────
    "Entry ₹",          # exact price to enter
    "Stop ₹",           # hard stop-loss level
    "Risk %",           # (entry − stop) / entry × 100
    # ── Conviction ────────────────────────────────────────────────────────────
    "Signal Summary",   # RS Leading | Stage + SEPA + RS | W-Confirmed | VCP 3× ...
    "Trade Score",      # 0–110 unified composite score
    "RS Score",         # RS Leader score 0–100
    "SEPA Score",       # SEPA entry quality score
    "RSI(14)",
    # ── Context ───────────────────────────────────────────────────────────────
    "Regime ⚠",         # ✅ Bull / 🟡 Mild Bull / 🟠 Neutral / 🔴 Caution / 🚨 Bear
    "Sector",
    "Sector Label",     # LEADING ▲ / NEUTRAL → / LAGGING ✕
    # ── Exit tracking ─────────────────────────────────────────────────────────
    "Exit Date",
    "Exit Reason",
    "TradingView",
]

# Holdings Alert — TheWrap-only exit view: ONLY held positions, sorted by urgency.
# Open this tab first every morning before checking your broker.
# Column order: identity → urgency → action → signal → distances → levels → context
OUTPUT_COLUMNS_HOLDINGS_ALERT = [
    "Rank",
    "Ticker",
    "Company",
    # ── Priority (read these first, in order) ─────────────────────────────────
    "Urgency",         # 0-100 composite urgency (TheWrap base + slope + gain modifiers)
    "Action",          # gain-aware action: HOLD / REDUCE / EXIT + EMA reference levels
    "TheWrap",         # weekly 10W/20W/40W EMA signal — the driving reason for the action
    "Gain %",          # P&L context: "TW_EXIT with +80% gain" = take profit; "-15%" = stop-loss
    # ── EMA distance signals (how far are you from each EMA?) ─────────────────
    "vs 10W %",        # price distance from 10W EMA (positive = above)
    "vs 20W %",        # price distance from 20W EMA
    "vs 40W %",        # price distance from 40W EMA
    # ── EMA price levels (for limit/stop order placement) ─────────────────────
    "10W EMA",         # current 10-week EMA price level
    "20W EMA",         # current 20-week EMA price level
    "40W EMA",         # current 40-week EMA price level
    # ── Structure context ─────────────────────────────────────────────────────
    "40W Slope",       # 40W EMA velocity: Rising ↑↑ / Flat → / Falling ↓↓
    "Weekly Stage",    # Weinstein stage for context
    "Portfolio",       # Self / Trading / Niveshaay / International
    # First Entry / Streak removed — holdings appear every day until sold, so streak is
    # just a trivial counter and first_entry duplicates the buy date already in your broker.
    # TradingView removed — Ticker is already a clickable hyperlink
    # Last Updated removed — operational noise, no trading value
]

# Daily BUY Conviction output columns (matches ranker_conviction.run_conviction_scan)
# Active stocks shown first (sorted by streak DESC, then conviction DESC).
# Exited stocks (dropped out in last 30 days) appended at the bottom with "⚪ Exited" action.
# Column order: identity → action → signals → persistence → price/momentum → tracking
OUTPUT_COLUMNS_CONVICTION = [
    "Rank",
    "Ticker",        # clickable hyperlink → TradingView
    "Company",
    # ── Action + persistence ──────────────────────────────────────────────────
    "Action",        # 🟢 BUY NOW / 🔔 BUY STOP / 📋 SET ALERT / 👁 WATCHLIST / ⚪ Exited
    "Streak",        # consecutive days in conviction list — the key differentiator
    "First Seen",    # date this stock first entered the conviction list
    # ── Signals ───────────────────────────────────────────────────────────────
    "# Signals",     # 2 or 3 — how many of {Stage2, RS, SEPA} are firing
    "Signals",       # "Stage ✓ | RS ✓ | SEPA ✓" — compact signal breakdown
    "RS Signal",     # 🌟 RS Leads Price (pre-breakout) / ✓ RS at 52w High / ·
    # ── Price & momentum ──────────────────────────────────────────────────────
    "Price ₹",
    "ROC 5D %",      # 5-day rate of change — shows recent momentum, breaks ties
    "Sector",
    # ── Exit tracking (bottom of tab) ─────────────────────────────────────────
    "Left On",       # blank while active; exit date when stock dropped out of list
    "Exit Reason",   # why it left: "Ranked out of top 20" / "RS + SEPA lost (1/3 remain)"
    # Removed: Conviction score (top-20 by score + sorted by it — no display variance)
    # Removed: Pivot Dist % (always 0 in practice — near-pivot is already the entry condition)
    # Removed: Weekly Stage (empty for most stocks — sepa_map only covers SEPA pool, not full universe)
    # Removed: Days Here (never resets on re-entry — diverges from Streak and misleads)
    # Removed: TradingView (Ticker column is already a clickable hyperlink)
]

# Data Issues output columns (matches data_quality.run_data_quality_scan)
OUTPUT_COLUMNS_DATA_QUALITY = [
    "Rank",
    "Ticker",
    "Company",
    "Issue",       # NaN Price / NaN Volume / Stale Data / Price Spike / Too Few Bars / All NaN
    "Details",     # human-readable explanation of the issue
    "Last Date",   # last bar date in the OHLCV data
    "Last Price",  # last close price (or "NaN")
    "Last Vol",    # last volume (formatted)
    "Bars",        # total bars in history
    "Sector",
    "Checked At",  # timestamp of this scan
    # TradingView removed — Ticker is already a clickable hyperlink
]

# -----------------------------------------------------------------------------
# GOOGLE SHEETS OUTPUT CONFIG
# -----------------------------------------------------------------------------
# Tab names are now date-based ("YYYY-MM-DD") — these keys are used only for the Run Log tab.
SHEET_TABS = {
    "log":              "Run Log",
    "trade":            "Trade Candidates",   # fixed — overwritten each run
    "conviction":       "Daily BUY",          # fixed — stocks in 2+ screeners simultaneously (streak tracked)
    "data_quality":     "Data Issues",        # fixed — tickers with NaN price/volume, stale or bad data
    "stage_trade":      "Stage Leaders",      # fixed — updated with trade mode run
    "sepa_trade":       "SEPA Setups",        # fixed — updated with trade mode run
    "rs_trade":         "RS Leaders",         # fixed — shared with --screener rs
    "holdings_alert":   "Holdings Alert",     # fixed — ONLY held positions, TheWrap exit signals
    "sectors":          "Sector Rotation",    # fixed — sector rotation flat ranked list (trade mode)
    "sector_overview":  "Sector Overview",    # fixed — sectors grouped by Leading → Lagging buckets
}

# Columns written to each tab — Stage mode (matches ranker_stage._result_to_row) — standalone --screener stage
# Column order: identity → action signal → stage quality → RS quality → momentum → volume → risk → meta
OUTPUT_COLUMNS = [
    "Rank",
    "Ticker",
    "Company",
    # ── Action signal + persistence ───────────────────────────────────────────
    "Entry Signal",      # 🟢 Cheat Entry / 🟡 EMA Pullback / 🔵 Near Pivot / ⚪ Extended
    "First Entry",       # date this stock first appeared in Stage Leaders (never resets)
    "Days Listed",       # calendar days since first entry
    "Entry Score",       # numeric entry timing quality (underlies Entry Signal label)
    # ── Stage & trend quality ─────────────────────────────────────────────────
    "Stage",
    "Stage Score S2",    # composite Stage-2 quality score
    "Duration (bars)",   # bars in current EMA trend — longer = more established
    # ── Relative strength ─────────────────────────────────────────────────────
    "RS Status",         # RS Strong ↑↑ / RS Strong ↑ / Neutral / Weak
    "RS vs Bench %",     # outperformance vs benchmark
    # ── Momentum ──────────────────────────────────────────────────────────────
    "Momentum",          # ↑↑ Strong / ↑ Rising / → Flat / ↓ Weak
    "ROC Fast %",        # fast rate of change — recent price acceleration
    # ── Volume ────────────────────────────────────────────────────────────────
    "Vol Conviction",    # Very High / High / Normal / Low
    "Vol Trend",         # Rising / Flat / Declining — volume trend direction
    # ── Overall score ─────────────────────────────────────────────────────────
    "Score",             # composite ranking score
    # ── Risk & EMA distances ──────────────────────────────────────────────────
    "EMA Dist Fast",
    "EMA Dist Mid",
    "EMA Dist Slow",
    "Beta",
    "PEAD %",            # post-earnings drift — positive = earnings tailwind
    # ── Liquidity & meta ──────────────────────────────────────────────────────
    "Avg $ Vol",
    "Market Cap",
    "Sector",
    # Mom Accel removed — redundant with Momentum + ROC Fast % already present
    # Beta Label removed — derived from Beta number, adds no new information
    # Vol Ratio removed — Vol Conviction (High/Normal/Low) is the actionable form
    # PEAD Label removed — derived from PEAD %, sign is self-evident
    # TradingView removed — Ticker is already a clickable hyperlink
    # Last Updated removed — operational noise, no trading value
]

# -----------------------------------------------------------------------------
# SCHEDULER (used if running main.py as a daemon — otherwise set your own cron)
# -----------------------------------------------------------------------------
# Set these if you want main.py to self-schedule (optional)
SCHEDULE_ENABLED = False        # Set True to use built-in scheduler
SCHEDULE_TIME_IST = "07:00"     # HH:MM in IST — runs daily at this time
