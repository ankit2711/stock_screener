# Stock Screener — Complete Implementation Specification

> Methodology: Stan Weinstein (Stage Analysis) + Mark Minervini (SEPA/VCP) + Jesse Livermore (RS Leadership)  
> This document is sufficient to reconstruct the entire system from scratch.

---

## 1. Architecture Overview

```
Universe → OHLCV Fetch → 3 Independent Screeners → Trade Scorer → Output
                              Stage (Weinstein)   ↘
                              SEPA  (Minervini)   → Union Pool → _score_candidate → Top 20 Trade Tab
                              RS    (Livermore)   ↗
```

**Markets**: `india` (NSE/BSE), `us` (NYSE/NASDAQ), `ai` (curated ~150 global AI supply chain)  
**Entry point**: `main.py --market india|us|ai --screener stage|sepa|rs|trade`  
**Scheduler**: built-in via `schedule` library; `SCHEDULE_TIME_IST = "07:00"` runs daily post-market  

---

## 2. File Map

```
main.py                      — orchestrator: fetch → screen → write → export
config.py                    — all tunable constants (single source of truth)
fetcher.py                   — yfinance (primary) + Twelve Data (fallback) OHLCV fetch
cache.py                     — SQLite OHLCV store, JSON universe/metadata cache
universe.py                  — NSE+BSE equity list, US NASDAQ+NYSE list, market cap filter
universe_ai.py               — hardcoded TradingView AI watchlist (~150 tickers, grouped)
persistence.py               — streak tracking, exit history (cache/persistence.json)
first_seen.py                — annotates "Days on List" for each screener tab
ranker_stage.py              — Weinstein Stage 2 screener
ranker_sepa.py               — Minervini SEPA/VCP entry screener
ranker_rs.py                 — Livermore RS Leaders screener
ranker_trade.py              — unified Trade Candidates scorer (combines all 3)
ranker_conviction.py         — daily conviction scan (2+ signals firing simultaneously)
ranker_sector.py             — sector rotation dashboard helper
sheets_writer.py             — Google Sheets output (gspread)
drive_exporter.py            — JSON export to Google Drive / iCloud / Dropbox fallback
data_quality.py              — NaN/stale data flag scan
screeners/
  stage_analysis.py          — core Stage 2 math (EMA stack, Mansfield RS, W-stage)
  sepa.py                    — base detection, VCP, ATR contraction, pocket pivot
  weekly_stage.py            — TheWrap TA signal (10W/20W/40W EMA decision tree)
  sector_rotation.py         — sector RS scoring (LEADING→LAGGING)
  rs_leaders.py              — per-stock RS line vs benchmark
  rs_rating.py               — Mansfield RS helper
  vcp.py                     — VCP contraction counter (used by sepa.py)
  holdings_reader.py         — TheWrap scan for currently held positions
```

---

## 3. Configuration (`config.py`)

```python
# Universe filters
US_MIN_MARKET_CAP_USD     = 2_000_000_000     # $2B — Minervini mid-cap floor
INDIA_MIN_MARKET_CAP_INR  = 3_000_000_000     # ₹300 Cr
US_WATCHLIST_TICKERS      = ["NNDM"]          # bypass market cap filter

# Liquidity gates (avg daily close × volume, 50-day rolling)
MIN_AVG_DOLLAR_VOL_NSE  = 10_000_000          # ₹1 Cr/day
MIN_AVG_DOLLAR_VOL_BSE  = 10_000_000          # ₹1 Cr/day (wider spreads)
MIN_AVG_DOLLAR_VOL_US   = 10_000_000          # $1M/day
MIN_AVG_DOLLAR_VOL_AI   =  1_000_000          # $100K/day (curated list, lower bar)

# Pool vs display (pool is scored; display is written to sheet)
POOL_N_INDIA = 80  ;  TOP_N_INDIA = 30
POOL_N_US    = 60  ;  TOP_N_US    = 30
POOL_N_AI    = 100 ;  TOP_N_AI    = 100

# History
HISTORY_DAYS   = 550    # ~2.2 years; buffer for EMA200 warm-up
YF_CHUNK_SIZE  = 200    # tickers per yfinance.download() call

# Benchmarks
benchmark_india = "^CRSLDX"   # Nifty 500
benchmark_us    = "SPY"        # S&P 500 ETF
benchmark_ai    = "QQQ"        # Nasdaq 100 ETF

# Cache TTLs
UNIVERSE_TTL_DAYS  = 7
METADATA_TTL_DAYS  = 7
MAX_HISTORY_TABS   = 30    # oldest Google Sheet date-tab pruned after this
```

---

## 4. Data Pipeline

### 4.1 Universe Building
- **India**: fetch NSE EQUITY_L.csv + BSE active equity API → deduplicate by ISIN (NSE wins) → filter by market cap via `yf.Ticker.fast_info` → cache weekly as `cache/universe_in.json`
- **US**: fetch NASDAQ + NYSE screener APIs → clean (alpha only, len 1–5) → market cap filter → merge `US_WATCHLIST_TICKERS` → cache weekly as `cache/universe_us.json`
- **AI**: parsed from hardcoded TradingView watchlist string in `universe_ai.py`; ~150 tickers across 20 groups (Hyperscalers, GPU Cloud, HBM Memory, etc.); no market cap filter

### 4.2 OHLCV Fetch (`fetcher.py`)
- Storage: SQLite `cache/ohlcv.db`, one table per ticker named `TICKER` (US) or `TICKER_NS`/`TICKER_BO` (India) via `_safe_table_name()` which replaces `.` and `-` with `_`
- Incremental: `cache.get_missing_date_range()` returns `(fetch_start, today)` where `fetch_start = last_cached_date + 1 day`; returns `(None, None)` if `last_cached >= today`
- **Always explicit `start/end` dates** — never `period='5d'`; end is always `today + 1 day` (yfinance end is exclusive). Reason: `period='5d'` returns NaN OHLC for current NSE trading day (volume arrives before prices are finalised in yfinance's index) causing today's bar to be silently dropped
- Batch: `YF_CHUNK_SIZE = 200` tickers per call; per-ticker retry if batch fails
- Fallback: Twelve Data API for any ticker yfinance cannot serve (rate limit: 8 req/min)
- Column normalisation: lowercase `open, high, low, close, volume`; drop rows where OHLC is NaN; fill NaN volume → 0 (volume lags a few minutes on NSE); drop rows where `close <= 0`
- Benchmarks stored in separate `cache/benchmarks.db`

### 4.3 Metadata
- `yf.Ticker(ticker).info` → `{name, sector, market_cap}` per ticker
- Cached weekly in `cache/metadata.json`; additive updates (new tickers only)
- AI universe overrides sector with watchlist group name (e.g. "HBM/MEMORY/STORAGE")

### 4.4 `data_as_of` Discipline
- Computed ONCE at the start of `run_trade_scan` from the last OHLCV bar date (not wall clock)
- All streak/first-seen stamps use this date
- Re-running the script on the same data never increments streaks or changes timestamps

---

## 5. Screener 1 — Stage Analysis (`ranker_stage.py` + `screeners/stage_analysis.py`)

### Purpose
Weinstein structural trend quality. A stock must be in Stage 2 (advancing) on both daily and weekly timeframes.

### Liquidity gate (applied first)
```
avg_dollar_vol = (close × volume).rolling(50).mean()
if avg_dollar_vol < MIN_AVG_DOLLAR_VOL_{EXCHANGE}: skip
```

### Daily Stage classification (EMA-based, daily bars)
```
EMA200 = ewm(span=200)   # Weinstein's 40-week equivalent
EMA50  = ewm(span=50)
EMA21  = ewm(span=21)    # Minervini's short-term trend line

Stage 2 criteria (ALL required):
  price > EMA200
  EMA200 is rising (slope over 10 bars > 0)
  EMA50 > EMA200
  Mansfield RS > 0 (stock outperforming benchmark)
```

### Weekly Stage classification (Weinstein's original)
Resample daily → weekly (W-FRI anchor, volume summed):
```
W30SMA = weekly_close.rolling(30).mean()   # Weinstein's 30-week SMA

W-S2 ✓    : weekly_close > W30SMA AND W30SMA slope > 0   ← confirmed Stage 2
W-S1 Accum: weekly_close < W30SMA AND W30SMA slope > 0   ← base forming
W-S3 Dist : weekly_close > W30SMA AND W30SMA slope < 0   ← distribution
W-S4 Decline: weekly_close < W30SMA AND W30SMA slope < 0  ← downtrend
```

### Mansfield RS
```
rs_line  = stock_close / benchmark_close
rs_ma    = rs_line.rolling(rs_ma_length).mean()   # default 200
mansfield_rs = (rs_line / rs_ma − 1) × 100
# Positive = outperforming benchmark. Weinstein requires > 0 for Stage 2.
```

### Cheat Entry detection (highest-conviction within-base signal)
```
EMA21 touch: price pulled back to within 3% of EMA21 while still in Stage 2
Confirms: price > EMA50, EMA50 > EMA200 still holds
Output: "🟢 Cheat Entry" in Entry Signal column → score_b = 25 in trade scorer
```

### Stage Score S2 (0–10) — feeds into trade scorer's Signal Breadth
Additive sub-scores based on EMA stack alignment, RS strength, weekly confirmation, volume dry-up in base, slope quality.

### Hard gates
- W-S1 Accum or W-S4 Decline → eliminated from Stage tab AND Trade tab

---

## 6. Screener 2 — SEPA / VCP (`ranker_sepa.py` + `screeners/sepa.py`)

### Purpose
Minervini entry quality. Finds stocks in valid VCP bases near a buyable pivot, or fresh breakouts with institutional volume.

### Base Detection (`detect_base()`)
Scans right-to-left through last 20–120 bars for the most recent significant local high:
```
1. Find absolute max in search window (excluding last 3 bars to avoid breakout spike)
2. Scan RIGHT-TO-LEFT for most recent local high ≥ 90% of absolute max
   (right-to-left avoids picking old V-shape peak as base_high for a recovered stock)
3. That peak = base_high (the pivot / resistance ceiling)
4. Validate:
   - base_length between base_min_bars(20) and base_max_bars(120)
   - max drawdown (high − low) / high < max_depth_pct (default 35%)
   - slope of closes within base < max_up_slope (not still trending up)
   - slope > −max_down_slope (not Stage 4 breakdown within base)
```

### Breakout State detection
```python
pivot_dist = (close − base_high) / base_high × 100

BREAKOUT      : pivot_dist > 0 AND volume ≥ 1.4× avg50d
WEAK_BREAKOUT : pivot_dist > 0 AND volume < 1.4× avg50d (suspect — possible distribution)
AT_PIVOT      : −3% < pivot_dist ≤ 0
IN_BASE       : pivot_dist < −3% OR extended > 8%
```

### Two Scoring Paths

**Path A — Fresh Breakout** (price already above base_high):
```
Breakout volume ratio (25%): volume / avg50d — surge confirms real demand
RS leading price      (20%): RS line at new high while price confirms
Extension above pivot (15%): penalises chasing (>10% above = too late)
Pivot proximity       (15%): how close price is to pivot
Volume character      (18%): up-day vs down-day vol ratio in base (accumulation)
Volume dry-up in base  (3%): Weinstein's low-vol consolidation signal
Weekly confirmation    (8%): W-S2 = full weight; W-S1 = partial; unknown = minimal
```

**Path B — VCP Base** (price still inside consolidation):
```
VCP contractions      (25%): count of valid swing contractions (each < 80% of prior)
                              Time compression bonus: last contraction shorter than prior
ATR contraction       (15%): ATR second-half / first-half of base < 1.0 = coiling
Volume dry-up         (12%): avg volume in final third of base vs first third
Volume character      (18%): same up/down day analysis as Path A
RS leading            (20%): RS line strength vs benchmark during base
Weekly confirm         (8%): same as Path A
```

### Pocket Pivot bonus (+15 pts, applied to both paths)
```
Today is an up day AND today's volume > every down-day volume in prior 10 sessions
→ buyers absorbed all recent sellers; institutional accumulation signal
→ precedes breakouts by 2–4 weeks
```

### Entry, Stop, Risk
```
Entry price = base_high × 1.005   (0.5% above pivot; buy-stop order)
Stop price  = base_high × 0.99    (1% below pivot — pivot becomes support after breakout)
            OR EMA21 × 0.97       (for cheat entries — tightest possible stop)
Stop dist % = (entry − stop) / entry × 100
```

### Regime multiplier
`sepa_score = raw_score × regime_mult`  where bull=1.0, mild_bull=0.95, neutral=0.85, caution=0.75, bear=0.70

### Output: `Raw Score` (0–100+), used in trade scorer's Signal Breadth

---

## 7. Screener 3 — RS Leaders (`ranker_rs.py` + `screeners/rs_leaders.py`)

### Purpose
Livermore leadership. Stocks whose relative strength vs benchmark is rising — the market's strongest names. Particularly valuable in corrections (they hold up while market falls).

### RS Score computation
```
rs_line  = stock_close / benchmark_close   (ratio series)
rs_score = percentile rank of current rs_line within its own 52-week history × 100
```

### RS Leads Price flag (🌟)
```
rs_new_high = rs_line made a new 52-week high in last 20 bars
price_new_high = price made a new 52-week high in last 20 bars

🌟 RS Leads Price = rs_new_high AND NOT price_new_high
# RS breaking out before price = institutional accumulation before breakout visible to all
```

### Tiered sorting
```
Tier 1: 🌟 RS Leads Price           ← most valuable
Tier 2: RS at new high (both RS and price)
Tier 3: RS rising (not at new high)
Tier 4: RS flat / lagging
```
Within each tier: sorted by RS Score descending.

### Output: `RS Score` (0–100), tier, `RS Leads Price` flag

---

## 8. TheWrap Signal (`screeners/weekly_stage.py`)

Weekly 10W / 20W / 40W EMA decision tree. Runs on weekly-resampled bars.

```
Priority order (checked top to bottom):

Below 40W EMA (structural breakdown):
  → TW_EXIT        🔴  — hard gate in trade scorer (eliminates candidate)

Below 20W, 40W rolling over:
  → TW_EXIT_40W    🔴  — 40W Break warning (demotes, doesn't eliminate)

EMA squeeze (10W ≈ 20W ≈ 40W, all converging):
  price > 10W AND 10W rising → TW_BULLISH   🟢  (explosive move setup)
  else                        → TW_WAIT      🟡

Full bull stack (price > 10W > 20W > 40W):
  40W rising   → TW_MAINTAIN   ✅  (trend intact)
  40W flat     → TW_FADING     🟡  (aging trend — × 0.82 penalty in current scorer)

Price > 10W but not full stack:
  → TW_BULLISH   🟢

Price between 10W and 40W:
  → TW_CAUTIOUS  ⚠️
```

**Lookup chain in trade scorer**: sepa_row → stage_row → recompute from OHLCV (last resort)

---

## 9. Trade Candidates Scorer (`ranker_trade.py`)

### 11-Step Pipeline

**Step 0** — `data_as_of`: last OHLCV bar date (not wall clock). Used for all streak/annotation timestamps.

**Step 1** — Market regime (display only, does NOT affect scoring or list size):
```
regime_mult from benchmark drawdown from 52w high:
  > −5%  → bull      ×1.0
  > −10% → mild_bull ×0.95
  > −15% → neutral   ×0.90
  > −25% → caution   ×0.80
  else   → bear      ×0.70
```

**Step 1b** — Sector rotation: run `screeners/sector_rotation.py` on sector index ETFs vs benchmark. Score each sector 0–100 → LEADING/IMPROVING/NEUTRAL/WEAKENING/LAGGING label.

**Step 2** — Run all 3 screeners with `POOL_N` size (80 India, 60 US) — larger than display cap so cross-lens high-scorers aren't prematurely cut.

**Step 3** — Build lookup dicts: `{ticker → stage_row}`, `{ticker → sepa_row}`, `{ticker → rs_row}` from full pools.

**Step 4** — Union all three pools → deduplicated candidate list.

**Step 5** — Score every candidate via `_score_candidate()` (see Section 10).

**Step 6** — Conviction streak boost (from prior run's streak tracking):
```
streak ≥ 3 days  → +2 pts
streak ≥ 7 days  → +4 pts
streak ≥ 15 days → +6 pts
```

**Step 7** — Holdings Alert: TheWrap scan on positions from `holdings_reader.py`. Flags TW_EXIT or TW_FADING for any currently held stock. Shown as a warning section in the trade tab.

**Step 8** — Sort descending by score → top 20. Assign action label:
```
score ≥ 72 AND state in (Cheat, BREAKOUT, AT_PIVOT) → 🟢 BUY NOW
score ≥ 55 AND state in (Cheat, BREAKOUT, AT_PIVOT) → 🔔 NEAR PIVOT
state in (WEAK_BREAKOUT, IN_BASE)                   → 📋 BASE BUILDING
else                                                → 👁  WATCHLIST
```

**Step 9** — Conviction scan (`ranker_conviction.py`): stocks firing 2+ of 3 independent signals (Stage2 + RS>0 + SEPA). Streak tracked. Triple-signal bonus. Top 20 by conviction score → "Daily BUY Conviction" tab.

**Step 10** — Data quality scan: flag tickers with NaN price, zero volume, stale data.

**Step 11** — Exit tracking appended to all 4 tabs (see Section 11).

---

## 10. Unified Scoring (`_score_candidate`)

### Hard Eliminators (return None immediately)
```
Weekly stage = W-S1 Accum or W-S4 Decline
TheWrap = TW_EXIT
OHLCV bars < 30
stop_dist_pct > 11%   (can't size the trade — do NOT fake a tighter stop)
Final score < 40
```

### Score Components

#### A. Signal Breadth (0–30)
How many of the three lenses see this stock:
```
stage_lens = min(Stage Score S2 / 10.0, 1.0) × 10     → 0–10
sepa_lens  = min(SEPA Raw Score / 100.0, 1.0) × 10    → 0–10
rs_lens    = min(RS Score / 100.0, 1.0) × 10          → 0–10
score_a    = stage_lens + sepa_lens + rs_lens          → 0–30
```
A stock in all three lenses scores 30. Stage-only scores proportionally lower.

#### B. Entry Signal Quality (0–25)
```
Cheat Entry (EMA21 pullback)         → 25
Fresh Breakout (pivot_dist ≤ 3%)     → 22
At Pivot — buy stop (0–3% below)     → 18
Breakout — slight extension (3–5%)   → 16
Back in Base — wait (3–8% below)     → 12
Building Base (>5% ext or >8% below) →  6
Watchlist (no clear state)           →  2
```

State resolution priority: sepa_row → OHLCV detection → cheat override if Stage confirms it.

#### C. Risk Quality (0–20)
```
stop_norm  = clamp((11.0 − stop_dist_pct) / 8.0, 0, 1)   # 3% stop → 1.0, 11% → 0.0
prox_norm  = max(0.0, 1.0 − abs(pivot_dist) / 8.0)       # 0% from pivot → 1.0, 8% → 0.0
score_c    = stop_norm × 10 + prox_norm × 10              → 0–20
```
`stop_dist_pct` clamped to [0.5, 15.0] — prevents data errors (0 or negative stop) from producing perfect risk score.

#### D. Volume Conviction (0–15)
Resolved via fallback chain: sepa_row `Vol Conv` → stage_row `Vol Conviction`:
```
"Very High" → 15
"High"      → 10
"Normal"    →  5
"Low"       →  1
```

#### E. Sector Strength (0–10)
From sector rotation result for this ticker's sector:
```
LEADING   → 10
NEUTRAL   →  6
LAGGING   →  2
```

#### Subtotal
```
score = score_a + score_b + score_c + score_d + score_e   → 0–100
```

### Bonuses (additive, applied after subtotal, cap at 110)
```
W-S2 ✓ confirmed         +5
TheWrap BULLISH           +4
TheWrap MAINTAIN          +2
🌟 RS Leads Price         +5
Conviction streak ≥ 3d    +2  (applied in Step 6, not here)
Conviction streak ≥ 7d    +4
Conviction streak ≥ 15d   +6
```

### Penalties (multiplicative, applied after bonuses)
```
RSI > 82               × 0.88   (overbought — no room to add)
RSI 75–82              × 0.95
Stage duration < 15    × 0.85   (base not seasoned)
TW_FADING              × 0.82   (demote, do not eliminate)
```

### Final
```
score = min(score × penalty_multipliers, 110.0)
if score < 40.0: return None
```

### Stop / Entry computation
```
Priority 1: use sepa_row values (SEPA already ran detect_base)
Priority 2: cheat entry from Stage — entry = close × 1.001, stop = EMA21 × 0.97
Priority 3: detect from OHLCV directly via detect_base()

For confirmed BREAKOUT/WEAK_BREAKOUT: stop = max(EMA21 × 0.97, base_high × 0.99)
(pivot becomes support; stop tightened to 1% below pivot, never widened)
```

---

## 11. Exit Tracking (`persistence.py — append_screener_exits`)

Called at Step 11 for all 4 output tabs (Stage, SEPA, RS, Trade).

### Logic
```
1. Load saved state for this bucket from cache/persistence.json
2. For each ticker in saved state that is NOT in current df:
   - If in reentry_pool (still in 80-stock pool, just not in top-30): NOT an exit
     (prevents false exits due to display cap, not genuine drop-off)
   - Else: compute exit reason → mark as exited with today's date
3. Keep exits visible for 7 calendar days (rolling window, no artificial cap)
4. Sort exits newest-first, append below active rows with Rank = "—"
5. Active rows get blank Exit Date / Exit Reason columns
```

### Exit Reason (5-tier diagnostic chain)
```
Priority 1 — Hard gate broken:
  "Price X% below EMA200 — structural trend broken"
  "Stop hit — price ₹X is Y% below stop ₹Z (entry was ₹W)"
  "TheWrap EXIT — EMA structure broken"
  "W-S4 Decline — weekly stage broken"

Priority 2 — Entry invalidated:
  "> 10% extended above pivot — risk/reward broken"
  "Breakout failed — price back below pivot after breakout attempt"
  "Base undercut >7% — consolidation structure failed"

Priority 3 — Lens loss (with specific EMA detail):
  "Stage 2 (EMA21 ₹X crossed below EMA50 ₹Y) lost + SEPA setup (Z% below pivot) lost | RS ✓ still holds"

Priority 4 — Score penalty:
  "Score penalties: RSI 83 → ×0.88 · 14% above EMA21 → score dropped from 74"

Priority 5 — Catch-all:
  "Scored below threshold — score 52, was BREAKOUT (🟡 slight extension)"
```

### Streak tracking (in `annotate_streak_df`)
```
State per ticker stored in persistence.json:
  {streak, max_streak, first_seen, last_seen, exit_date}

On each run:
  same data_as_of as last run → no change (idempotent re-run protection)
  gap ≤ 1 trading day         → streak + 1
  gap > 3 calendar days       → streak resets to 1 (genuine miss)
  stock absent                → exit_date stamped
  stock re-enters             → exit_date cleared, streak restarts from 1
```

---

## 12. Output Layer

### Google Sheets (`sheets_writer.py`)
- Separate sheet per market (India / US / AI) — IDs in `config.py`
- Tab per screener, overwritten each run (always current data):

| Tab | Content |
|---|---|
| Stage Analysis | Top 30 Stage 2 stocks + 7-day exits appended below |
| SEPA Setups | Top 30 SEPA/VCP entries + 7-day exits |
| RS Leaders | Top 30 RS leaders + 7-day exits |
| Trade Candidates | Top 20 scored candidates + 7-day exits |
| Daily BUY Conviction | Top 20 stocks firing 2+ signals simultaneously |
| Sector Overview | LEADING→LAGGING sector dashboard |
| Holdings Alert | TheWrap signals for held positions |
| Data Quality | Tickers with NaN/stale data flags |
| Log | Run timestamps and stats |

- Date-based tabs for historical record: `YYYY-MM-DD` suffix; oldest pruned after `MAX_HISTORY_TABS = 30`
- Auth: service account JSON (`credentials.json`) for Sheets; OAuth2 (`oauth_client.json`) for Drive

### JSON Export (`drive_exporter.py`)
- Filename: `screener_results_{date}_{market}_{screener}.json`
- Upload priority: Google Drive folder → iCloud Drive → Dropbox → OneDrive → `project/exports/`
- Each result is a dict of DataFrames serialised to JSON records

---

## 13. Simplified Scoring (Proposed — not yet implemented)

Replace the current 5-component + 4-bonus + 4-penalty (additive/multiplicative mix) with a clean **4-pillar linear system** summing to exactly 100. All signals baked into the pillar they belong to.

```
Score = Trend(0–25) + Entry(0–25) + Confirmation(0–25) + Risk(0–25)
```

**Pillar 1 — TREND (Weinstein)**
- EMA stack confirmed (price > EMA50 > EMA200, EMA200 rising): +10
- W-S2 ✓ (price above rising 30W SMA): +8
- TheWrap Bullish=+5, Maintain=+3, Fading=+1
- Stage duration ≥ 15 bars (seasoned base): +2
- RSI > 82: −3 | RSI 75–82: −1
- Hard gates unchanged: W-S4/W-S1 → None, TW_EXIT → None

**Pillar 2 — ENTRY (Minervini)**
- Cheat Entry=25, Fresh Breakout ≤3%=22, AT_PIVOT=18
- Breakout 3–5%=14, Back in Base=8, Building Base=3

**Pillar 3 — CONFIRMATION (Livermore)**
- Volume: Very High=10, High=7, Normal=3, Low=0
- RS Leads Price 🌟: +8
- Multi-lens: all 3 screeners=+7, any 2=+5, only 1=+0

**Pillar 4 — RISK (Minervini)**
- `risk_score = 25 × (11.0 − stop_dist_pct) / 8.0` clamped [0, 25]
- Hard gate: stop_dist_pct > 11% → None

**Post-score (separate, transparent)**
- Conviction streak: ≥3=+2, ≥7=+4, ≥15=+6 (capped at 100)
- Floor: score < 40 → eliminated

---

## 14. Key Design Rules

1. **Stop integrity**: if stop > 11%, eliminate — never fake a tighter stop (would give good risk score to an unsizeable trade)
2. **Pool > Display**: score on POOL_N (80/60), display TOP_N (30/30) — prevents a stock ranked #35 in Stage + #8 in RS from being invisible
3. **Regime is informational only**: bear market does not shrink the list; fewer stocks clear the 40-point floor organically
4. **`data_as_of` not wall clock**: all time-series annotations (streaks, first-seen) use last OHLCV bar date — running twice on the same data never double-counts
5. **Explicit fetch dates always**: never `period='5d'`; end = today+1 (exclusive)
6. **Right-to-left pivot scan**: finds most recent significant local high, not historical all-time high (avoids V-shape misclassification)
7. **NSE wins over BSE**: if stock listed on both, use `.NS` (better liquidity, better yfinance data)
8. **Cheat Entry overrides SEPA state**: if Stage confirms a Cheat Entry, state = AT_PIVOT and score_b = 25 regardless of SEPA row state
9. **`reentry_pool` guard on exits**: stock still in 80-pool but fell out of top-30 display is NOT marked as exited
10. **TheWrap lookup chain**: sepa_row → stage_row → recompute from OHLCV (never silently use "No Data")
