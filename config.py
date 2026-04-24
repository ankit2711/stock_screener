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

# Google Drive folder — daily JSON exports are uploaded here after each run.
# Folder URL: https://drive.google.com/drive/folders/18eOnNUaZbOZp7ddZ8ZldZrLEditSDKBt
# Share this folder with your service account email (Editor) so it can write files.
GOOGLE_DRIVE_RESULTS_FOLDER_ID = "18eOnNUaZbOZp7ddZ8ZldZrLEditSDKBt"

# Backward-compat alias (used by any legacy code that still imports GOOGLE_SHEET_ID)
GOOGLE_SHEET_ID = GOOGLE_SHEET_ID_INDIA

# How many date-tabs to keep before the oldest one is pruned (0 = keep all)
MAX_HISTORY_TABS = 30

# -----------------------------------------------------------------------------
# MARKET UNIVERSE & FILTERS
# -----------------------------------------------------------------------------
US_MIN_MARKET_CAP_USD = 20_000_000_000       # $10B minimum
INDIA_MIN_MARKET_CAP_INR = 3_000_000_000    # ₹500 Crore minimum

# Minimum average daily dollar volume — exchange-aware liquidity filter
# Applied in ranker_stage, ranker_sepa, ranker_rs to remove untradeable stocks.
# BSE-only stocks (.BO suffix) have wider spreads and thinner order books;
# they require a higher minimum to avoid slippage risk.
# These are in local currency (INR for India, USD for US).
MIN_AVG_DOLLAR_VOL_NSE = 10_000_000    # ₹50 lakh/day  — NSE stocks
MIN_AVG_DOLLAR_VOL_BSE = 10_000_000    # ₹1 crore/day  — BSE-only stocks (wider spreads)
MIN_AVG_DOLLAR_VOL_US  = 10_000_000    # $1M/day        — US stocks
MIN_AVG_DOLLAR_VOL_AI  =  1_000_000    # $1M/day        — AI theme (curated global list, lower bar)

# Number of top stocks to write to each sheet tab
TOP_N_US    = 30
TOP_N_INDIA = 30
TOP_N_AI    = 100    # AI theme is ~150 stocks — return all that pass Stage 2

# How many calendar days of history to fetch per ticker (1 year = ~252 trading days)
HISTORY_DAYS = 365

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
    "vcp_contractions":  0.25,   # number + quality of shrinking price swings
    "vol_character":     0.20,   # accumulation ratio (up-day vol / down-day vol) + churn
    "atr_contraction":   0.15,   # ATR first-half vs second-half of base (coiling signal)
    "rs_leading":        0.15,   # RS line making new highs before price breakout
    "vol_dry_up":        0.10,   # recent 5d vol vs pre-base 20d vol (true dry-up)
    "current_tightness": 0.10,   # CV of last-third of base closes (final coil)
    "pivot_proximity":   0.05,   # distance from base high (buy-stop distance)
}

# How many top stocks the SEPA screener returns (can differ from Stage screener)
TOP_N_SEPA_US    = 30
TOP_N_SEPA_INDIA = 30

# How many trade candidates the Trade ranker returns
TOP_N_TRADE = 10

# How many RS Leader stocks the RS Leaders screener returns
TOP_N_RS_INDIA = 30
TOP_N_RS_US    = 30

# SEPA output columns (match keys in ranker_sepa._result_to_row)
OUTPUT_COLUMNS_SEPA = [
    "Rank",
    "Ticker",
    "Company",
    # Setup summary
    "Market Regime",     # Bull (5/5) … Bear (≤1/5) — shown on every row for context
    "Setup",             # 🟢 Actionable VCP / 🟡 Ready / 🔵 Forming / 🔴 Extended / 🚫 No Base
    "Path",              # A: Fresh Breakout | B: VCP Base
    "SEPA Score",
    "Breakout State",    # AT_PIVOT | IN_BASE | BREAKOUT | WEAK_BREAKOUT | FADING | EXTENDED
    "Pivot Dist %",
    "Stop Dist %",
    # Path A columns
    "Vol Surge",
    "S1 Base CV%",
    "Extension %",
    # Path B columns
    "Base Bars",
    "Base Depth %",
    "VCP Count",
    "Last Tightest",
    "ATR Contract",
    "Accum Ratio",
    "Churn Bars",
    "Vol Dry %",
    "CV Tight %",
    "RS Leading",
    "Base Count",
    "Base ×Mult",
    # Stage context
    "Stage",
    "Stage S2",
    "Duration",
    "Cheat Entry",
    "RS Status",
    "RS vs Bench %",
    "Momentum",
    "ROC Fast %",
    "Avg $ Vol",
    "EMA Dist Fast",
    "EMA Dist Mid",
    "EMA Dist Slow",
    "Beta",
    "Beta Label",
    "Vol Conv",
    "Market Cap",
    "Sector",
    "TradingView",
    "Last Updated",
]

# Streamlined Stage columns for the Stage tab written by trade mode
# (fewer columns than full Stage output — focused on watchlist quality)
OUTPUT_COLUMNS_TRADE_STAGE = [
    "Rank",
    "Ticker",           # clickable → TradingView
    "Company",
    "Score",            # composite Stage-2 score (0–1)
    "Stage",            # Stage 2 ↑ / Stage 1 / Stage 3 etc.
    "Stage Score S2",   # raw S2 score out of 10
    "RS Status",        # RS Strong ↑↑ / RS Strong ↑ / Neutral / Weak
    "Momentum",         # ↑↑ Strong / ↑ Rising / → Flat / ↓ Weak
    "Entry Signal",     # 🟢 Cheat Entry / 🟡 EMA Pullback / 🔵 Near Pivot / ⚪ Extended
    "Vol Conviction",   # Very High / High / Normal / Low
    "Avg $ Vol",
    "Sector",
    "TradingView",
]

# Streamlined SEPA columns for the SEPA tab written by trade mode
# (prioritises entry execution over research detail)
OUTPUT_COLUMNS_TRADE_SEPA = [
    "Rank",
    "Ticker",
    "Company",
    "SEPA Score",       # primary ranking metric
    "Setup",            # 🟢/🟡/🔵/🔴 — action verdict at a glance
    "Breakout State",   # AT_PIVOT / BREAKOUT / WEAK_BREAKOUT / IN_BASE
    "RSI(14)",          # entry timing: 50-65 ideal, >82 extended
    "Entry ₹",          # exact buy price
    "Stop ₹",           # exact stop-loss price
    "Pivot Dist %",     # how far from the pivot (+/-)
    "Stop Dist %",      # (price − stop) / price — your position size input
    "Vol Signal",       # "dry 0.42×" (pre-breakout) or "surge 2.1×" (breakout)
    "RS Leading",       # ✓ RS line at new high before price — primary signal
    "VCP Count",        # ≥ 2 = proper VCP
    "Weekly Stage",     # W-S2 ✓ is the gold standard
    "TheWrap",          # weekly 10W/20W/40W EMA signal — TW_BULLISH best, TW_FADING reduce
    "Raw Score",        # SEPA score before regime multiplier
    "Sector",
    "TradingView",
]

# RS Leaders output columns (matches ranker_rs._result_to_row)
OUTPUT_COLUMNS_RS = [
    "Rank",
    "Ticker",
    "Company",
    # ── Core RS signal ──────────────────────────────────────────────────────
    "RS Score",           # 0–100 composite RS Leader score
    "Resilience",         # Strong Leader / Leader / Neutral / Laggard
    "RS % from High",     # % below 52-week RS line high (0 = AT high)
    "RS at 52w High",     # ✓ if within 3% of 52w RS high (primary filter)
    "RS New High",        # ✓ if RS line at new all-time high in dataset
    "RS Leads Price",     # 🌟 Leads = RS at new high while price still >5% below its own 52w high
                          # ✓ Confirms = RS at new high with price also near its high
                          # · = RS not at new high
                          # Minervini's highest-conviction pre-breakout signal. Sorted to top.
    "Score Breakdown",    # RS:35 Res:25 Vol:20 Str:15 Base:5 Lead:+8 (debug)
    # ── Relative performance ─────────────────────────────────────────────────
    "Stock Off 52w %",    # how far stock is below its own 52w high
    "Bench Off 52w %",    # how far benchmark is below 52w high
    "Resilience Δ",       # bench_off - stock_off (positive = leader)
    "RS vs Bench 1M",     # 1-month RS vs benchmark
    "RS vs Bench 3M",     # 3-month RS vs benchmark
    # ── Volume ──────────────────────────────────────────────────────────────
    "Accum Ratio",        # up-day vol share (>0.55 = accumulating)
    "Vol Dry",            # ✓ if recent vol drying up (constructive)
    "Vol Dry Ratio",      # 5d avg / 20d avg (< 0.70 = dry)
    # ── Structure ───────────────────────────────────────────────────────────
    "Stage",              # Stage 2 ↑ / Stage 1 Accum / Stage 3 / Stage 4
    "Above EMA200",       # ✓ if price > EMA200
    "EMA Stack",          # ✓ if full bullish EMA alignment
    "EMA200 Slope",       # 10-day EMA200 slope (turning up = +ve)
    "Price vs EMA200",    # % above/below EMA200
    # ── Base ────────────────────────────────────────────────────────────────
    "Base Forming",       # ✓ if price range last 20 bars < 10%
    "Base Depth %",       # % range of last 20 bars (tighter = better)
    "Consol Bars",        # consecutive bars within 8% of current price
    # ── Market context ───────────────────────────────────────────────────────
    "Market Regime",      # ✅ At High / 🟡 Pullback / 🟠 Correction / 🔴 Deep / 🚨 Bear
    "Bench Off 52w",      # benchmark % off its 52w high
    "FTD Signal",         # ✓ FTD if Follow-Through Day detected on benchmark
    # ── Liquidity + meta ─────────────────────────────────────────────────────
    "Avg $ Vol",
    "Market Cap",
    "Sector",
    "TradingView",
    "Last Updated",
]

# Trade Candidates output columns (matches ranker_trade._build_output)
# Left-to-right: act → size → quality scores → conviction → context
OUTPUT_COLUMNS_TRADE = [
    "Rank",
    "Tier",            # 🟢 Trade Now (active entry) | 👁 Watchlist (wait for FTD)
    "Reason",          # which screeners confirmed it: Stage2+SEPA+RS / SEPA+RS / Stage2+SEPA / SEPA / RS Leader
    # ── Execute ────────────────────────────────────────────────────────────────
    "Ticker",          # clickable → TradingView
    "Company",
    "Action",          # 🟢 BUY NOW / 🔔 BUY STOP / 🟡 CONFIRM VOL / 📋 ALERT
    "Entry ₹",         # exact price to enter
    "Stop ₹",          # hard stop-loss level
    "Risk %",          # (entry − stop) / entry × 100
    "Pos Size %",      # regime-adjusted position size: (1%/stop%) × regime_factor, cap 8%
                       # Already accounts for market conditions — this is your actual allocation
    # ── Quality scores (3 lenses) ──────────────────────────────────────────────
    "Trade Score",     # 0–100 unified regime-aware composite
    "Stage S2",        # Stage-2 quality 0–10 (structural trend)
    "RS Score",        # RS Leader score 0–100 (institutional holding strength)
    "SEPA Score",      # SEPA entry quality (regime-adjusted)
    "RSI(14)",         # entry timing: 50-65 ideal, >82 extended
    # ── Conviction signals ─────────────────────────────────────────────────────
    "Signal Summary",  # RS Leading ✓ | VCP 3 | Weekly S2 ✓ | 1st base ...
    "Breakout State",  # BREAKOUT | AT_PIVOT | WEAK_BREAKOUT | WATCHLIST
    # ── Market context ─────────────────────────────────────────────────────────
    "Regime ⚠",        # ✅ Bull / 🟡 Mild Bull / 🟠 Neutral / 🔴 Caution / 🚨 Bear
    "Sector",
    "TradingView",
]

# Holdings Alert — TheWrap-only exit view: ONLY held positions, sorted by urgency.
# Open this tab first every morning before checking your broker.
OUTPUT_COLUMNS_HOLDINGS_ALERT = [
    "Rank",
    "Ticker",
    "Company",
    "Portfolio",        # Self / Trading / Niveshaay / International
    "Gain %",          # % gain vs buy price
    "Action",          # gain-aware action string with EMA reference levels
    "Urgency",         # 0-100 composite urgency (TheWrap base + slope + gain modifiers)
    "TheWrap",         # weekly 10W/20W/40W EMA signal label with emoji
    "Signal Code",     # machine-readable: TW_MAINTAIN / TW_BULLISH / … / TW_EXIT
    "10W EMA",         # current 10-week EMA price level
    "20W EMA",         # current 20-week EMA price level
    "40W EMA",         # current 40-week EMA price level
    "vs 10W %",        # price distance from 10W EMA (positive = above)
    "vs 20W %",        # price distance from 20W EMA
    "vs 40W %",        # price distance from 40W EMA
    "40W Slope",       # 40W EMA velocity label: Rising ↑↑ / Flat → / Falling ↓↓
    "Weekly Stage",    # Weinstein stage for context
    "TradingView",
    "Last Updated",
]

# -----------------------------------------------------------------------------
# GOOGLE SHEETS OUTPUT CONFIG
# -----------------------------------------------------------------------------
# Tab names are now date-based ("YYYY-MM-DD") — these keys are used only for the Run Log tab.
SHEET_TABS = {
    "log":              "Run Log",
    "trade":            "Trade Candidates",   # fixed — overwritten each run
    "stage_trade":      "Stage Leaders",      # fixed — updated with trade mode run
    "sepa_trade":       "SEPA Setups",        # fixed — updated with trade mode run
    "rs_trade":         "RS Leaders",         # fixed — shared with --screener rs
    "holdings_alert":   "Holdings Alert",     # fixed — ONLY held positions, TheWrap exit signals
}

# Columns written to each tab — Stage mode (matches ranker_stage._result_to_row)
OUTPUT_COLUMNS = [
    "Rank",
    "Ticker",
    "Company",
    "Score",
    "Entry Signal",      # 🟢 Cheat Entry / 🟡 EMA Pullback / 🔵 Near Pivot / ⚪ Extended
    "Entry Score",
    "Stage",
    "Stage Score S2",
    "Duration (bars)",
    "Cheat Entry",
    "RS Status",
    "RS vs Bench %",
    "Momentum",
    "ROC Fast %",
    "Mom Accel",
    "Avg $ Vol",
    "Vol Trend",
    "EMA Dist Fast",
    "EMA Dist Mid",
    "EMA Dist Slow",
    "Beta",
    "Beta Label",
    "Vol Conviction",
    "Vol Ratio",
    "PEAD %",
    "PEAD Label",
    "Market Cap",
    "Sector",
    "TradingView",
    "Last Updated",
]

# -----------------------------------------------------------------------------
# SCHEDULER (used if running main.py as a daemon — otherwise set your own cron)
# -----------------------------------------------------------------------------
# Set these if you want main.py to self-schedule (optional)
SCHEDULE_ENABLED = False        # Set True to use built-in scheduler
SCHEDULE_TIME_IST = "07:00"     # HH:MM in IST — runs daily at this time
