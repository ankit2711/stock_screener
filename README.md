# Stock Screener — Setup & Usage Guide (Mac)

> **Based on:** "Stage Analysis Screener (TheWrap Clone)"
> **Original Pine Script author:** Atlas (OpenClaw) for Ankit
> **Python port:** Claude (Anthropic)

Automated daily screener for US (NYSE + NASDAQ, $2B+ market cap) and Indian (NSE, ₹500Cr+) stocks across 6 strategies: VCP, Darvas Box, Volume Breakout, 52-Week High, RS Rating, and MA Alignment. Results pushed to Google Sheets daily.

---

## Project Structure

```
stock_screener/
├── main.py              # Orchestrator — run this
├── config.py            # ALL tunable parameters and weights (edit this)
├── universe.py          # Ticker universe fetcher (US + India)
├── fetcher.py           # OHLCV data: Twelve Data primary, yfinance fallback
├── ranker.py            # Runs all screens, scores, returns top 30
├── sheets_writer.py     # Writes results to Google Sheets
├── screeners/
│   ├── __init__.py
│   ├── vcp.py           # Volatility Contraction Pattern
│   ├── darvas.py        # Darvas Box breakout
│   ├── volume_breakout.py
│   ├── high_52w.py      # 52-week high breakout
│   ├── rs_rating.py     # IBD-style Relative Strength
│   └── ma_alignment.py  # MA stack (20 > 50 > 200)
├── requirements.txt
├── credentials.json     # ← you create this (Google service account)
└── screener.log         # auto-generated run log
```

---

## Step 1 — Install Python dependencies

```bash
cd stock_screener
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Step 2 — Get your Twelve Data API key (free)

1. Sign up at https://twelvedata.com/ (free tier: 800 requests/day)
2. Copy your API key
3. Open `config.py` and paste it:
   ```python
   TWELVE_DATA_API_KEY = "your_key_here"
   ```

---

## Step 3 — Set up Google Sheets access

### 3a. Create Google Cloud project + enable APIs
1. Go to https://console.cloud.google.com/
2. Create a new project (or use existing)
3. Search for and enable: **Google Sheets API**
4. Search for and enable: **Google Drive API**

### 3b. Create a service account
1. In Google Cloud Console → IAM & Admin → Service Accounts
2. Click **Create Service Account**
3. Give it any name (e.g. `stock-screener`)
4. Click **Create and Continue** → skip role → Done
5. Click the service account → **Keys** tab → **Add Key** → **JSON**
6. A JSON file downloads — rename it to `credentials.json`
7. Move it into the `stock_screener/` folder

### 3c. Share your Google Sheet
1. Create a new Google Sheet (or use existing)
2. Copy the Sheet ID from the URL:
   `https://docs.google.com/spreadsheets/d/`**THIS_PART**`/edit`
3. Open `config.py` and paste:
   ```python
   GOOGLE_SHEET_ID = "your_sheet_id_here"
   ```
4. In the Google Sheet → Share → paste the service account email
   (looks like `stock-screener@your-project.iam.gserviceaccount.com`)
5. Give it **Editor** access

---

## Step 4 — Paste your Pine Script strategies

Open each file in `screeners/` and replace the `_run_*()` function body with your translated logic. Each file has a clearly marked section:

```
# *** PASTE YOUR PINE SCRIPT LOGIC HERE ***
```

**Pine Script → Python translation cheatsheet:**

| Pine Script | Python (pandas) |
|---|---|
| `ta.highest(high, 52)` | `df['high'].rolling(252).max()` |
| `ta.lowest(low, 20)` | `df['low'].rolling(20).min()` |
| `ta.sma(close, 50)` | `df['close'].rolling(50).mean()` |
| `ta.ema(close, 20)` | `df['close'].ewm(span=20).mean()` |
| `ta.rsi(close, 14)` | `ta.momentum.RSIIndicator(df['close'], 14).rsi()` |
| `ta.atr(high, low, close, 14)` | `ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()` |
| `ta.bbands(close, 20)` | `ta.volatility.BollingerBands(df['close'], 20)` |
| `close[1]` (previous bar) | `df['close'].iloc[-2]` |
| `close > close[1]` | `df['close'].iloc[-1] > df['close'].iloc[-2]` |
| `volume > ta.sma(volume, 20) * 2` | `df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 2` |
| `crossover(fast, slow)` | `(fast.iloc[-1] > slow.iloc[-1]) and (fast.iloc[-2] <= slow.iloc[-2])` |

All parameters are in `config.py` — no magic numbers inside screener files.

---

## Step 5 — Test a single run

```bash
cd stock_screener
source venv/bin/activate
python main.py
```

Watch the log output. First run takes ~15–30 minutes (fetching full universe).
Subsequent runs are faster as Twelve Data caches responses.

---

## Step 6 — Schedule daily runs on Mac (launchd)

Mac's `launchd` is more reliable than cron for daily tasks. It survives sleep/wake.

### 6a. Create the plist file

Replace `/Users/YOUR_USERNAME/` with your actual home directory path.

```bash
cat > ~/Library/LaunchAgents/com.stockscreener.daily.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.stockscreener.daily</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/stock_screener/venv/bin/python</string>
        <string>/Users/YOUR_USERNAME/stock_screener/main.py</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>7</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>WorkingDirectory</key>
    <string>/Users/YOUR_USERNAME/stock_screener</string>

    <key>StandardOutPath</key>
    <string>/Users/YOUR_USERNAME/stock_screener/launchd.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/YOUR_USERNAME/stock_screener/launchd_error.log</string>

    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
EOF
```

### 6b. Load the schedule

```bash
launchctl load ~/Library/LaunchAgents/com.stockscreener.daily.plist
```

### 6c. Verify it's loaded

```bash
launchctl list | grep stockscreener
```

### 6d. To unload / disable

```bash
launchctl unload ~/Library/LaunchAgents/com.stockscreener.daily.plist
```

### 6e. Change run time
Edit the `Hour` and `Minute` values in the plist, then reload:
```bash
launchctl unload ~/Library/LaunchAgents/com.stockscreener.daily.plist
launchctl load   ~/Library/LaunchAgents/com.stockscreener.daily.plist
```

---

## Tuning the composite score

Open `config.py` and edit `SCREEN_WEIGHTS`:

```python
SCREEN_WEIGHTS = {
    "vcp":              0.25,
    "darvas":           0.20,
    "volume_breakout":  0.20,
    "high_52w":         0.15,
    "rs_rating":        0.10,
    "ma_alignment":     0.10,
}
```

Weights must sum to 1.0. Increase a screen's weight to rank it higher.
Individual screen parameters (lookbacks, thresholds) are in their named sections below `SCREEN_WEIGHTS`.

---

## Google Sheet output format

| Column | Description |
|---|---|
| Rank | 1 = highest composite score |
| Ticker | Exchange symbol |
| Company | Full company name |
| Price | Latest close |
| Change % | Day's price change |
| Score | Composite weighted score (0–1) |
| VCP / Darvas / Vol Breakout / 52W High / RS Rating / MA Align | ✓ = passed, · = not |
| Screens | List of screens passed |
| Market Cap | Formatted market cap |
| Sector | GICS sector |
| TradingView | Clickable chart link |
| Last Updated | Timestamp of last run |

Two tabs are created/updated: `🇺🇸 US Screener` and `🇮🇳 IND Screener`.
A `Run Log` tab tracks every run timestamp and result count.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside venv |
| `APIError: 429` from Twelve Data | Free tier rate limit hit — add `time.sleep()` or reduce universe size |
| Google Sheets auth error | Check `credentials.json` path in config, verify sheet is shared with service account |
| yfinance returns empty | Yahoo endpoint changed — update yfinance: `pip install --upgrade yfinance` |
| NSE CSV fetch fails | NSE changed headers — `universe.py` falls back to Nifty 500 hardcoded list |
| No results in sheet | Check `screener.log` — likely all stocks failed market cap filter |

---

## Adding your Pine Script (when ready)

Paste your Pine Script in a message and Claude will translate each strategy into the corresponding Python function, matching your exact parameters and conditions. The translation slots directly into the `_run_*()` function in each screener file — everything else stays unchanged.
