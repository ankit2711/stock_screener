#!/usr/bin/env python3
"""
Creates a Google Doc explaining the stock screener system logic and improvement ideas.
Uses the same service account credentials already configured for Google Sheets.
"""

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from config import GOOGLE_SHEETS_CREDENTIALS_FILE

SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]

# ── Document content ──────────────────────────────────────────────────────────
# Each entry: (text, style)
# style = "HEADING_1" | "HEADING_2" | "HEADING_3" | "NORMAL_TEXT"
# prefix "bold:" = bold normal text
# prefix "bullet:" = bullet list item
# prefix "table:" = special marker for table rows

CONTENT = [
    ("Stock Screener — System Logic & Improvement Roadmap", "TITLE"),
    ("Written for someone who knows the basics and wants to iterate toward better performance.", "SUBTITLE"),

    # ─────────────────────────────────────────────────────────────────────────
    ("Overview", "HEADING_1"),
    (
        "The system uses three separate screeners — Stage, SEPA, and RS Leaders — each looking "
        "at a stock from a different angle. The Trade sheet is where all three vote together to "
        "produce a final ranked list of actionable stocks.",
        "NORMAL_TEXT",
    ),
    (
        "Think of it this way:",
        "NORMAL_TEXT",
    ),
    ("Stage   →  Is this stock in a healthy uptrend? (structural watchlist)", "bullet:"),
    ("SEPA    →  Is there a low-risk entry point available right now? (entry shortlist)", "bullet:"),
    ("RS Leaders  →  Which stocks are institutions refusing to sell during this correction? (correction watchlist)", "bullet:"),
    ("Trade   →  What do I actually do today? (all three lenses combined)", "bullet:"),

    # ─────────────────────────────────────────────────────────────────────────
    ("1. Stage Screener", "HEADING_1"),
    ("Based on Stan Weinstein's Stage Analysis", "HEADING_3"),
    (
        "Every stock is always in one of four stages. Stage 1 = accumulation (sideways after a decline). "
        "Stage 2 = advancing (the only stage worth owning). Stage 3 = distribution (topping). "
        "Stage 4 = declining (avoid or short). The screener scores how deeply and convincingly a stock "
        "is in Stage 2.",
        "NORMAL_TEXT",
    ),
    ("What it measures (in order of weight):", "bold:"),
    ("45%  Stage-2 quality — price above EMA200, EMA200 rising, EMA50 above EMA200, positive ROC, higher highs", "bullet:"),
    ("20%  RS strength — stock has been outperforming the benchmark consistently", "bullet:"),
    ("25%  Momentum — ROC is positive and accelerating (↑↑) or just positive (↑)", "bullet:"),
    (" 6%  Entry proximity — tiebreaker: is there an entry signal nearby?", "bullet:"),
    (" 4%  Volume conviction — is buying volume above average?", "bullet:"),
    (
        "Output: top 30 stocks closest to a perfect score of 1.0. A stock with full Stage-2 "
        "structure, strong RS, and accelerating momentum scores 0.90–0.95. This is your structural "
        "watchlist — stocks you want to own, not necessarily today.",
        "NORMAL_TEXT",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    ("2. SEPA Screener", "HEADING_1"),
    ("Based on Mark Minervini's Specific Entry Point Analysis", "HEADING_3"),
    (
        "Takes Stage-2 stocks and asks: is the stock set up for a specific low-risk entry with "
        "a tight stop and clear risk/reward? It routes each stock through one of two scoring paths.",
        "NORMAL_TEXT",
    ),
    ("Path A — Fresh Breakout (stock just entered Stage 2, within 20 bars):", "bold:"),
    ("40%  Volume on breakout day — did institutions show up? (≥ 1.4× average = confirmed)", "bullet:"),
    ("25%  RS line behaviour — was the RS line already at new highs before price broke out?", "bullet:"),
    ("20%  Base tightness — was the pre-breakout Stage 1 base tight (low coefficient of variation)?", "bullet:"),
    ("15%  Extension — how far above the pivot is it? (< 5% = still in buy zone)", "bullet:"),
    ("Path B — VCP Base (in Stage 2 for weeks, now consolidating):", "bold:"),
    ("25%  VCP contractions — are price swings getting progressively smaller?", "bullet:"),
    ("20%  Volume character — is volume drying up before breakout, or surging on it?", "bullet:"),
    ("15%  ATR contraction — is daily volatility compressing (coiling like a spring)?", "bullet:"),
    ("15%  RS line leading — is RS at new highs while price consolidates?", "bullet:"),
    ("10%  Volume dry-up — last 5 days avg vol < 70% of 20-day avg?", "bullet:"),
    ("10%  Current tightness — is the last third of the base the tightest part?", "bullet:"),
    (" 5%  Pivot proximity — how close is price to the breakout level?", "bullet:"),
    ("RSI(14) timing modifier:", "bold:"),
    ("RSI 50–65 = +5% bonus (fresh momentum, room to run — ideal entry zone)", "bullet:"),
    ("RSI 65–75 = no change (strong but acceptable)", "bullet:"),
    ("RSI 75–82 = −7% penalty (somewhat extended)", "bullet:"),
    ("RSI > 82  = −18% penalty (very extended, pullback risk)", "bullet:"),
    ("RSI < 40  = −20% penalty (momentum absent — not a real breakout)", "bullet:"),
    (
        "The entire score is then multiplied by the market regime (Bear × 0.15, Bull × 1.0). "
        "Output: top 30 by SEPA score. This is your entry shortlist.",
        "NORMAL_TEXT",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    ("3. RS Leaders Screener", "HEADING_1"),
    ("Based on Minervini's RS Leader concept — built for corrections", "HEADING_3"),
    (
        "During a market correction, weak stocks fall with the market. Strong stocks hold up — "
        "because institutions are accumulating or simply won't sell. These leaders go up the most "
        "when the market recovers. The RS line = stock close ÷ benchmark close (daily ratio). "
        "If this ratio is near its 52-week high while the benchmark is down 10–20%, it means "
        "the stock is literally going up while everything else falls.",
        "NORMAL_TEXT",
    ),
    ("Scoring (0–100):", "bold:"),
    ("35%  RS line at/near 52-week high — RS within 3% of 52w high (the core signal)", "bullet:"),
    ("25%  Resilience — (benchmark % off 52w high) minus (stock % off 52w high). e.g. bench −15%, stock −3% → resilience = +12%", "bullet:"),
    ("20%  Volume accumulation — up-day volume / total volume in last 20 bars (> 0.55 = institutional buying)", "bullet:"),
    ("15%  Structural integrity — above EMA200, full EMA stack, EMA200 slope positive", "bullet:"),
    (" 5%  Base formation — price range last 20 bars < 10% (coiling quietly)", "bullet:"),
    (
        "Output: top 30 RS Leaders. This is your correction watchlist. "
        "Set price alerts at the pivot. Buy when the market gives a Follow-Through Day signal.",
        "NORMAL_TEXT",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    ("4. Trade Sheet — How It Is Computed", "HEADING_1"),
    ("Runs all three scans, combines them with regime-aware weights", "HEADING_3"),
    ("Step 1 — Regime Detection (from benchmark health):", "bold:"),
    ("Benchmark near 52w high (multiplier 1.0)  →  Bull market", "bullet:"),
    ("Down 5–15%  (multiplier 0.75)  →  Mild Bull", "bullet:"),
    ("Down 15–25% (multiplier 0.50)  →  Neutral", "bullet:"),
    ("Down 25–35% (multiplier 0.30)  →  Caution", "bullet:"),
    ("Down 35%+   (multiplier 0.15)  →  Bear", "bullet:"),
    ("Step 2 — Regime shifts the scoring weights:", "bold:"),
    ("Bull:     SEPA 35%  Stage 25%  RS 15%  State 15%  Stop 10%", "bullet:"),
    ("Mild:     SEPA 30%  Stage 25%  RS 20%  State 15%  Stop 10%", "bullet:"),
    ("Neutral:  SEPA 25%  Stage 22%  RS 28%  State 15%  Stop 10%", "bullet:"),
    ("Caution:  SEPA 18%  Stage 20%  RS 37%  State 15%  Stop 10%", "bullet:"),
    ("Bear:     SEPA 12%  Stage 15%  RS 48%  State 15%  Stop 10%  ← RS dominates", "bullet:"),
    (
        "In a bear or correction market (India today), RS Leadership carries nearly half the score. "
        "A stock confirmed by all three lenses (Reason = 'Stage2 + SEPA + RS') has a significant "
        "advantage over a SEPA-only stock.",
        "NORMAL_TEXT",
    ),
    ("Step 3 — Tier A (Trade Now):", "bold:"),
    ("Breakout State = BREAKOUT / AT_PIVOT / WEAK_BREAKOUT", "bullet:"),
    ("Pivot distance between −8% and +5%", "bullet:"),
    ("Stop distance ≤ 9%", "bullet:"),
    ("Setup not 🔴 (not fading or extended)", "bullet:"),
    ("Step 4 — Tier B (Watchlist):", "bold:"),
    ("RS Leaders score ≥ 45 in Stage 2 but no active SEPA entry yet", "bullet:"),
    ("Shown with a price alert level — these are your post-FTD buys", "bullet:"),
    ("Step 5 — Reason Column:", "bold:"),
    ("Stage2 + SEPA + RS  →  all three lenses agree — highest conviction", "bullet:"),
    ("Stage2 + SEPA       →  structural trend confirmed, entry ready, but not an RS Leader yet", "bullet:"),
    ("SEPA + RS           →  entry ready and showing correction leadership, but not in Stage top-30", "bullet:"),
    ("SEPA                →  entry signal only — weakest conviction in current bear regime", "bullet:"),
    ("Stage2 + RS         →  Tier B watchlist: great structure, leading RS, no entry yet", "bullet:"),
    ("RS Leader           →  Tier B watchlist: RS leader, structure uncertain", "bullet:"),

    # ─────────────────────────────────────────────────────────────────────────
    ("5. Improvement Ideas", "HEADING_1"),
    ("Reviewed through the lens of Weinstein, Minervini, and Livermore", "HEADING_3"),

    ("🔴 High Priority", "HEADING_2"),

    ("1. Weekly Stage as a hard gate for Tier A  [Weinstein]", "bold:"),
    (
        "Weinstein's first rule: the weekly chart is king. Currently weekly Stage is a bonus column. "
        "It should be a hard requirement: no stock enters Tier A unless the weekly chart also confirms "
        "Stage 2 (price above rising 30-week SMA). Right now a stock can reach Tier A with "
        "Weekly Stage = Unknown or W-S1 Accum.",
        "NORMAL_TEXT",
    ),

    ("2. Follow-Through Day as a Tier A action gate  [Livermore]", "bold:"),
    (
        "Livermore would never buy individual stocks until the market itself confirmed a new uptrend. "
        "The FTD detector already exists in the RS Leaders scan. The fix: if no FTD has occurred on "
        "the benchmark in the last 10 bars, all Tier A stocks show Action = '⏳ WAIT — no market "
        "confirmation' regardless of setup quality. This single rule would prevent most whipsaw losses "
        "in corrections.",
        "NORMAL_TEXT",
    ),

    ("3. Regime-adjusted position sizing  [Minervini]", "bold:"),
    (
        "Currently the 1% risk rule always calculates up to a 20% position size. In a Bear market, "
        "Minervini sizes at 0.25% risk per trade. The Pos Size % column should reflect the regime:",
        "NORMAL_TEXT",
    ),
    ("Bull market:    1.00% risk per trade → up to 20% position", "bullet:"),
    ("Neutral market: 0.50% risk per trade → up to 10% position", "bullet:"),
    ("Bear market:    0.25% risk per trade → up to 5% position", "bullet:"),

    ("🟡 Medium Priority", "HEADING_2"),

    ("4. Prior move requirement  [Minervini]", "bold:"),
    (
        "Before a VCP base qualifies, the stock must have already advanced at least 25–30% from its "
        "Stage 1 breakout point. This filters out weak stage changes and stocks attempting to bottom. "
        "Currently a stock up 8% from a low can qualify for Path B scoring.",
        "NORMAL_TEXT",
    ),

    ("5. Earnings date flag  [Minervini]", "bold:"),
    (
        "Entering a position 2–3 weeks before earnings is speculative even in a perfect setup — "
        "a gap can blow through any stop. Every Tier A candidate should show '⚠ Earnings in X days' "
        "so the trader decides consciously. This data is available via yfinance's .calendar property.",
        "NORMAL_TEXT",
    ),

    ("6. Sector leadership scoring  [Weinstein]", "bold:"),
    (
        "Weinstein's process: find leading sectors first, then find the leader within that sector. "
        "We track sector as a label but never score it. A stock in a top-decile sector should get a "
        "scoring boost; a stock in a bottom-decile sector should get a penalty regardless of its "
        "individual chart. This requires computing a rolling sector momentum rank.",
        "NORMAL_TEXT",
    ),

    ("7. Top 5 highest-conviction subset  [Livermore]", "bold:"),
    (
        "Livermore traded 4–5 leaders maximum. 15 candidates dilutes focus. The system should surface "
        "its top 3–5 'all green' stocks — Reason = Stage2+SEPA+RS, RSI 50–70, RS at 52w high, "
        "weekly S2 confirmed — as a HIGH CONVICTION block at the top of the trade sheet.",
        "NORMAL_TEXT",
    ),

    ("🟠 Lower Priority (Build Later)", "HEADING_2"),

    ("8. Weekly volume accumulation trend  [Weinstein]", "bold:"),
    (
        "Up-weeks on above-average volume versus down-weeks — this is Weinstein's institutional "
        "accumulation signal. We have daily accumulation ratio but not weekly. Weekly is more "
        "reliable: harder to fake, reduces noise significantly.",
        "NORMAL_TEXT",
    ),

    ("9. Group concentration detector  [Livermore]", "bold:"),
    (
        "If 3 or more stocks in the same sector are showing up in the RS Leaders scan simultaneously, "
        "institutional money is flowing into that sector. The system should flag this: "
        "'Healthcare: 4 RS Leaders detected this week.' Low implementation effort, high signal value.",
        "NORMAL_TEXT",
    ),

    ("10. Time stop flag  [Minervini]", "bold:"),
    (
        "If a stock doesn't advance 5%+ within 15 trading days of entry, Minervini exits regardless "
        "of whether the price stop was hit. A position that goes nowhere is dead capital. "
        "Implementing this requires tracking entry dates — possible with a small portfolio log file.",
        "NORMAL_TEXT",
    ),

    ("11. Pyramiding plan  [Livermore]", "bold:"),
    (
        "Livermore's greatest insight: the big money is made by adding to winning positions, not "
        "just buying once. For every Tier A stock, the system could output a pyramid plan: "
        "initial entry (50% of position), add-1 (30% at +5% above entry), add-2 (20% at +10%). "
        "This turns the trade sheet from a buy list into a full position management plan.",
        "NORMAL_TEXT",
    ),

    # ─────────────────────────────────────────────────────────────────────────
    ("6. Priority Summary", "HEADING_1"),
    ("Ranked by impact vs implementation effort", "HEADING_3"),
    ("🔴  Weekly Stage as hard gate for Tier A  [Weinstein]  — Low effort, high impact", "bullet:"),
    ("🔴  FTD as Tier A action gate  [Livermore]  — Low effort, prevents most whipsaw losses", "bullet:"),
    ("🔴  Regime-adjusted position sizing  [Minervini]  — Low effort, correct risk management", "bullet:"),
    ("🟡  Earnings date flag  [Minervini]  — Medium effort, yfinance .calendar", "bullet:"),
    ("🟡  Sector leadership scoring  [Weinstein]  — Medium effort, need sector momentum rank", "bullet:"),
    ("🟡  Prior move requirement (25%+)  [Minervini]  — Medium effort, add to SEPA Path B filter", "bullet:"),
    ("🟡  Top 5 highest-conviction subset  [Livermore]  — Low effort, filter existing output", "bullet:"),
    ("🟠  Weekly volume accumulation  [Weinstein]  — Medium effort, resample + compute", "bullet:"),
    ("🟠  Group concentration detector  [Livermore]  — Low effort, count sector hits", "bullet:"),
    ("🟠  Time stop flag  [Minervini]  — Medium effort, needs entry tracking log", "bullet:"),
    ("🟠  Pyramiding plan  [Livermore]  — Medium effort, extend trade output schema", "bullet:"),
]


def build_requests(content):
    """Convert content list into Google Docs API batchUpdate requests."""
    requests = []
    index = 1  # current insertion index in the document

    for item in content:
        text, style = item

        # Determine actual text to insert
        is_bullet = style.startswith("bullet:")
        is_bold   = style.startswith("bold:")

        if is_bullet:
            insert_text = text + "\n"
            para_style  = "NORMAL_TEXT"
        elif is_bold:
            insert_text = text + "\n"
            para_style  = "NORMAL_TEXT"
        else:
            insert_text = text + "\n"
            para_style  = style

        end_index = index + len(insert_text)

        # Insert the text
        requests.append({
            "insertText": {
                "location": {"index": index},
                "text": insert_text,
            }
        })

        # Apply paragraph style
        if para_style in ("TITLE", "SUBTITLE", "HEADING_1", "HEADING_2", "HEADING_3", "NORMAL_TEXT"):
            requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": index, "endIndex": end_index},
                    "paragraphStyle": {"namedStyleType": para_style},
                    "fields": "namedStyleType",
                }
            })

        # Apply bullet list formatting
        if is_bullet:
            requests.append({
                "createParagraphBullets": {
                    "range": {"startIndex": index, "endIndex": end_index - 1},
                    "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE",
                }
            })

        # Apply bold to entire paragraph
        if is_bold:
            requests.append({
                "updateTextStyle": {
                    "range": {"startIndex": index, "endIndex": end_index - 1},
                    "textStyle": {"bold": True},
                    "fields": "bold",
                }
            })

        index = end_index

    return requests


def main():
    creds = Credentials.from_service_account_file(
        GOOGLE_SHEETS_CREDENTIALS_FILE, scopes=SCOPES
    )

    docs_service  = build("docs",  "v1", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)

    # Create blank document
    print("Creating Google Doc...")
    doc = docs_service.documents().create(body={
        "title": "Stock Screener — System Logic & Improvement Roadmap"
    }).execute()

    doc_id  = doc["documentId"]
    doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"
    print(f"Doc created: {doc_url}")

    # Make it readable by anyone with the link
    drive_service.permissions().create(
        fileId=doc_id,
        body={"type": "anyone", "role": "reader"},
    ).execute()

    # Build and send content
    print("Writing content...")
    requests = build_requests(CONTENT)

    # Google Docs API has a limit per batchUpdate — split into chunks of 100
    chunk_size = 100
    for i in range(0, len(requests), chunk_size):
        chunk = requests[i : i + chunk_size]
        docs_service.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": chunk},
        ).execute()
        print(f"  Wrote batch {i // chunk_size + 1}/{(len(requests) - 1) // chunk_size + 1}")

    print(f"\n✅ Done! Open your doc here:\n{doc_url}")
    return doc_url


if __name__ == "__main__":
    main()
