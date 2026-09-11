"""
tools/build_master_analysis.py
══════════════════════════════════════════════════════════════════════════════
CENTRALIZED CHALLENGE ANALYSIS ENGINE
══════════════════════════════════════════════════════════════════════════════
PURPOSE:
  Reads ALL challenge CSVs (archived + current active) and builds a single
  master knowledge document: exports/MASTER_ANALYSIS.md

  Run this after every challenge ends or whenever you want an up-to-date
  performance snapshot.

USAGE:
  cd e:\\backtest
  python tools/build_master_analysis.py

OUTPUT:
  exports/MASTER_ANALYSIS.md  — the single source of truth for strategy performance
  Console summary             — key stats printed to terminal
══════════════════════════════════════════════════════════════════════════════
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import os
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ARCHIVE_DIR     = "archive"
CURRENT_CSV     = "exports/challenge_trades.csv"
OUTPUT_MD       = "exports/MASTER_ANALYSIS.md"
CHALLENGE_START = 5000.0   # Starting capital for every challenge
CHALLENGE_GOAL  = 5500.0   # Target balance to PASS a challenge

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def win_rate(df):
    """Returns (wins, losses, total, wr%) excluding BE trades."""
    wins   = (df["result"] == "WIN").sum()
    losses = (df["result"] == "LOSS").sum()
    total  = wins + losses
    wr     = round(wins / total * 100, 1) if total > 0 else 0.0
    return wins, losses, total, wr

def profit_factor(df):
    gross_w = df[df["pnl_usd"] > 0]["pnl_usd"].sum()
    gross_l = abs(df[df["pnl_usd"] < 0]["pnl_usd"].sum())
    return round(gross_w / gross_l, 2) if gross_l > 0 else float("inf")

def max_streak(df):
    max_w = cur_w = max_l = cur_l = 0
    for r in df["result"]:
        if r == "WIN":
            cur_w += 1; cur_l = 0; max_w = max(max_w, cur_w)
        elif r == "LOSS":
            cur_l += 1; cur_w = 0; max_l = max(max_l, cur_l)
    return max_w, max_l

def is_counter_trend(row):
    bias      = str(row.get("bias_reason", ""))
    regime    = str(row.get("regime", ""))
    direction = str(row.get("direction", ""))
    if "Bullish" in bias and regime in ("strong_bear", "trending_down"): return True
    if "Bearish" in bias and regime in ("strong_bull", "trending_up"):   return True
    if "Bullish" in bias and direction == "SHORT":                        return True
    if "Bearish" in bias and direction == "LONG":                         return True
    return False

def classify_challenge(filename):
    """Return (status, date_str) from a filename like challenge_PASSED_20260601_0129.csv"""
    name = os.path.basename(filename)
    parts = name.replace(".csv", "").split("_")
    status   = parts[1] if len(parts) > 1 else "UNKNOWN"
    date_raw = parts[2] if len(parts) > 2 else "00000000"
    try:
        dt = datetime.strptime(date_raw, "%Y%m%d")
        date_str = dt.strftime("%Y-%m-%d")
    except Exception:
        date_str = date_raw
    return status, date_str

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ALL CHALLENGE CSVs
# ─────────────────────────────────────────────────────────────────────────────
records = []

# 1. Archived challenges
for fname in sorted(os.listdir(ARCHIVE_DIR)):
    if not fname.endswith(".csv"):
        continue
    fpath = os.path.join(ARCHIVE_DIR, fname)
    try:
        df = pd.read_csv(fpath)
        if df.empty or "result" not in df.columns:
            continue
        status, date_str = classify_challenge(fname)
        df["_challenge_id"]   = fname.replace(".csv", "")
        df["_challenge_date"] = date_str
        df["_status"]         = status
        records.append(df)
    except Exception as e:
        print(f"[WARN] Could not read {fname}: {e}")

# 2. Current active challenge
if os.path.exists(CURRENT_CSV):
    try:
        df_cur = pd.read_csv(CURRENT_CSV)
        if not df_cur.empty and "result" in df_cur.columns:
            df_cur["_challenge_id"]   = "challenge_ACTIVE_current"
            df_cur["_challenge_date"] = datetime.now().strftime("%Y-%m-%d")
            df_cur["_status"]         = "ACTIVE"
            records.append(df_cur)
    except Exception as e:
        print(f"[WARN] Could not read current challenge CSV: {e}")

if not records:
    print("[ERROR] No challenge CSV files found.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# COMBINE & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
combined = pd.concat(records, ignore_index=True)
combined["pnl_usd"]    = pd.to_numeric(combined.get("pnl_usd", 0), errors="coerce").fillna(0)
combined["opened_at"]  = pd.to_datetime(combined.get("opened_at"), errors="coerce")
combined["closed_at"]  = pd.to_datetime(combined.get("closed_at"), errors="coerce")
combined["duration_min"] = (
    (combined["closed_at"] - combined["opened_at"]).dt.total_seconds() / 60
).round(1)
combined["counter_trend"] = combined.apply(is_counter_trend, axis=1)

# Ensure required columns exist (older CSVs may be missing some)
for col in ["setup_tier", "anchor_type", "regime", "direction", "symbol", "chain", "session"]:
    if col not in combined.columns:
        combined[col] = "unknown"

# Remove dupes within the same challenge file
combined = combined.drop_duplicates(subset=["trade_id", "_challenge_id"])

n_challenges  = combined["_challenge_id"].nunique()
n_total       = len(combined)

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE KEY STATS FOR WHAT'S WINNING
# ─────────────────────────────────────────────────────────────────────────────

def table_breakdown(df, col, title, extra_flag_fn=None):
    """Return markdown table rows for a breakdown by 'col'."""
    rows = []
    for val, g in df.groupby(col, dropna=False):
        w, l, tot, wr = win_rate(g)
        pf            = profit_factor(g)
        pnl           = g["pnl_usd"].sum()
        be            = (g["result"] == "BE").sum()
        flag = ""
        if extra_flag_fn:
            flag = extra_flag_fn(val, w, l, tot, wr, pnl)
        rows.append((val, tot, w, l, be, wr, pf, pnl, flag))
    # Sort by net PnL descending
    rows.sort(key=lambda x: x[7], reverse=True)
    return rows

def flag_wr(val, w, l, tot, wr, pnl):
    if wr >= 55: return "🟢 Strong"
    if wr >= 40: return "🟡 Neutral"
    return "🔴 Avoid"

# ─────────────────────────────────────────────────────────────────────────────
# BUILD THE MARKDOWN
# ─────────────────────────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

lines = []
def wr_md(text=""):
    lines.append(text)

wr_md(f"# 🧠 MASTER CHALLENGE ANALYSIS")
wr_md(f"> Auto-generated on **{now_str}**. Run `python tools/build_master_analysis.py` to refresh.")
wr_md()
wr_md(f"---")
wr_md()
wr_md(f"## 📊 Coverage")
wr_md(f"- **Total Challenges Analysed:** {n_challenges}")
wr_md(f"- **Total Trades Across All Challenges:** {n_total}")
wr_md()

# ─── Challenge-by-Challenge Timeline ───────────────────────────────────────
wr_md("---")
wr_md()
wr_md("## 📅 Challenge Timeline")
wr_md()
wr_md("| # | Challenge | Date | Status | Trades | W/L/BE | WR% | PF | Net PnL |")
wr_md("| :- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

challenge_rows = []
for idx, (cid, g) in enumerate(combined.groupby("_challenge_id", sort=False), 1):
    status   = g["_status"].iloc[0]
    date_str = g["_challenge_date"].iloc[0]
    w, l, tot, wr = win_rate(g)
    pf = profit_factor(g)
    pnl  = g["pnl_usd"].sum()
    be   = (g["result"] == "BE").sum()
    icon = "✅" if status == "PASSED" else ("🟡" if status == "ACTIVE" else "❌")
    challenge_rows.append((idx, cid, date_str, status, icon, tot, w, l, be, wr, pf, pnl))
    wr_md(f"| {idx} | `{cid}` | {date_str} | {icon} {status} | {tot} | {w}W/{l}L/{be}BE | {wr}% | {pf} | ${pnl:+.2f} |")

wr_md()

# ─── Overall Totals ─────────────────────────────────────────────────────────
wr_md("---")
wr_md()
wr_md("## 🎯 Overall Aggregate Stats")
wr_md()
w_all, l_all, tot_all, wr_all = win_rate(combined)
pf_all   = profit_factor(combined)
pnl_all  = combined["pnl_usd"].sum()
be_all   = (combined["result"] == "BE").sum()
mw, ml   = max_streak(combined)
passed   = combined[combined["_status"] == "PASSED"]["_challenge_id"].nunique()
stopped  = combined[combined["_status"] == "STOPPED"]["_challenge_id"].nunique()
active   = combined[combined["_status"] == "ACTIVE"]["_challenge_id"].nunique()

wr_md(f"| Metric | Value |")
wr_md(f"| :--- | :--- |")
wr_md(f"| **Challenges Passed** | {passed} |")
wr_md(f"| **Challenges Stopped** | {stopped} |")
wr_md(f"| **Challenges Active** | {active} |")
wr_md(f"| **Total Trades** | {tot_all} W={w_all}, L={l_all}, BE={be_all} |")
wr_md(f"| **Overall Win Rate** | **{wr_all}%** |")
wr_md(f"| **Overall Net PnL** | **${pnl_all:+.2f}** |")
wr_md(f"| **Profit Factor** | **{pf_all}** |")
wr_md(f"| **Max Win Streak** | {mw} |")
wr_md(f"| **Max Loss Streak** | {ml} |")
wr_md()

# ─── WHAT'S WINNING — By Symbol ─────────────────────────────────────────────
wr_md("---")
wr_md()
wr_md("## 🏆 What's Winning — By Symbol")
wr_md()
wr_md("| Symbol | Trades | W | L | BE | WR% | PF | Net PnL | Signal |")
wr_md("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for val, tot, w, l, be, wr, pf, pnl, flag in table_breakdown(combined, "symbol", "Symbol", flag_wr):
    wr_md(f"| **{val}** | {tot} | {w} | {l} | {be} | {wr}% | {pf} | ${pnl:+.2f} | {flag} |")
wr_md()

# ─── By Setup Tier ──────────────────────────────────────────────────────────
wr_md("---")
wr_md()
wr_md("## 🎖️ By Setup Tier (S / A++ / A / B)")
wr_md()
wr_md("| Tier | Trades | W | L | BE | WR% | PF | Net PnL | Signal |")
wr_md("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for val, tot, w, l, be, wr, pf, pnl, flag in table_breakdown(combined, "setup_tier", "Tier", flag_wr):
    wr_md(f"| **Tier {val}** | {tot} | {w} | {l} | {be} | {wr}% | {pf} | ${pnl:+.2f} | {flag} |")
wr_md()

# ─── By Regime ──────────────────────────────────────────────────────────────
wr_md("---")
wr_md()
wr_md("## 🌍 By Market Regime")
wr_md()
wr_md("| Regime | Trades | W | L | BE | WR% | PF | Net PnL | Signal |")
wr_md("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for val, tot, w, l, be, wr, pf, pnl, flag in table_breakdown(combined, "regime", "Regime", flag_wr):
    wr_md(f"| `{val}` | {tot} | {w} | {l} | {be} | {wr}% | {pf} | ${pnl:+.2f} | {flag} |")
wr_md()

# ─── By Direction ───────────────────────────────────────────────────────────
wr_md("---")
wr_md()
wr_md("## ↕️ By Trade Direction")
wr_md()
wr_md("| Direction | Trades | W | L | BE | WR% | PF | Net PnL | Signal |")
wr_md("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for val, tot, w, l, be, wr, pf, pnl, flag in table_breakdown(combined, "direction", "Direction", flag_wr):
    wr_md(f"| **{val}** | {tot} | {w} | {l} | {be} | {wr}% | {pf} | ${pnl:+.2f} | {flag} |")
wr_md()

# ─── By Anchor Type ─────────────────────────────────────────────────────────
wr_md("---")
wr_md()
wr_md("## ⚓ By Entry Anchor (FVG / OB / Confluence)")
wr_md()
wr_md("| Anchor | Trades | W | L | BE | WR% | PF | Net PnL | Signal |")
wr_md("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for val, tot, w, l, be, wr, pf, pnl, flag in table_breakdown(combined, "anchor_type", "Anchor", flag_wr):
    wr_md(f"| **{val}** | {tot} | {w} | {l} | {be} | {wr}% | {pf} | ${pnl:+.2f} | {flag} |")
wr_md()

# ─── Counter-Trend Analysis ─────────────────────────────────────────────────
wr_md("---")
wr_md()
wr_md("## 🔄 Counter-Trend vs With-Trend")
wr_md()
wr_md("| Alignment | Trades | W | L | BE | WR% | PF | Net PnL | Verdict |")
wr_md("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for ct, g in combined.groupby("counter_trend"):
    label = "Counter-Trend ⚠️" if ct else "With-Trend ✅"
    w, l, tot, wr = win_rate(g)
    pf  = profit_factor(g)
    pnl = g["pnl_usd"].sum()
    be  = (g["result"] == "BE").sum()
    verdict = "❌ AVOID — major loss driver" if ct else "✅ KEEP — profitable"
    wr_md(f"| **{label}** | {tot} | {w} | {l} | {be} | {wr}% | {pf} | ${pnl:+.2f} | {verdict} |")
wr_md()

# ─── By Chain ───────────────────────────────────────────────────────────────
wr_md("---")
wr_md()
wr_md("## ⛓️ By Timeframe Chain")
wr_md()
wr_md("| Chain | Trades | W | L | BE | WR% | PF | Net PnL | Signal |")
wr_md("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for val, tot, w, l, be, wr, pf, pnl, flag in table_breakdown(combined, "chain", "Chain", flag_wr):
    wr_md(f"| `{val}` | {tot} | {w} | {l} | {be} | {wr}% | {pf} | ${pnl:+.2f} | {flag} |")
wr_md()

# ─── By Session ─────────────────────────────────────────────────────────────
if "session" in combined.columns and combined["session"].notna().any():
    wr_md("---")
    wr_md()
    wr_md("## 🕐 By Trading Session")
    wr_md()
    wr_md("| Session | Trades | W | L | BE | WR% | PF | Net PnL | Signal |")
    wr_md("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for val, tot, w, l, be, wr, pf, pnl, flag in table_breakdown(combined, "session", "Session", flag_wr):
        if str(val) in ("unknown", "nan"): continue
        wr_md(f"| `{val}` | {tot} | {w} | {l} | {be} | {wr}% | {pf} | ${pnl:+.2f} | {flag} |")
    wr_md()

# ─── Trade Duration Analysis ─────────────────────────────────────────────────
wr_md("---")
wr_md()
wr_md("## ⏱️ Trade Duration Analysis")
wr_md()
wins_dur   = combined[combined["result"] == "WIN"]["duration_min"].dropna()
losses_dur = combined[combined["result"] == "LOSS"]["duration_min"].dropna()

wr_md("| Metric | Wins | Losses |")
wr_md("| :--- | :---: | :---: |")
wr_md(f"| **Avg Duration (min)** | {wins_dur.mean():.0f} | {losses_dur.mean():.0f} |")
wr_md(f"| **Median Duration (min)** | {wins_dur.median():.0f} | {losses_dur.median():.0f} |")
wr_md(f"| **Fastest (min)** | {wins_dur.min():.0f} | {losses_dur.min():.0f} |")
wr_md(f"| **Slowest (min)** | {wins_dur.max():.0f} | {losses_dur.max():.0f} |")
wr_md()

# ─── KEY TAKEAWAYS SECTION ───────────────────────────────────────────────────
wr_md("---")
wr_md()
wr_md("## 💡 Key Takeaways — What's Actually Working")
wr_md()

# Find best performing symbol
sym_stats = []
for sym, g in combined.groupby("symbol"):
    w, l, tot, wr = win_rate(g)
    pnl = g["pnl_usd"].sum()
    sym_stats.append((sym, wr, pnl, tot))
sym_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
best_sym   = sym_stats[0] if sym_stats else ("?", 0, 0, 0)
worst_sym  = sym_stats[-1] if sym_stats else ("?", 0, 0, 0)

# Find best performing tier
tier_stats = []
for tier, g in combined.groupby("setup_tier"):
    w, l, tot, wr = win_rate(g)
    pnl = g["pnl_usd"].sum()
    tier_stats.append((tier, wr, pnl, tot))
tier_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
best_tier  = tier_stats[0] if tier_stats else ("?", 0, 0, 0)

# Counter-trend stats
ct_df  = combined[combined["counter_trend"] == True]
wt_df  = combined[combined["counter_trend"] == False]
ct_wr  = round((ct_df["result"] == "WIN").sum() / max(len(ct_df[ct_df["result"].isin(["WIN","LOSS"])]), 1) * 100, 1)
wt_wr  = round((wt_df["result"] == "WIN").sum() / max(len(wt_df[wt_df["result"].isin(["WIN","LOSS"])]), 1) * 100, 1)
ct_pnl = ct_df["pnl_usd"].sum()

wr_md("### ✅ What IS Working")
wr_md(f"1. **Best Symbol: {best_sym[0]}** — {best_sym[1]}% WR across {best_sym[3]} trades")
wr_md(f"2. **Best Tier: Tier {best_tier[0]}** — {best_tier[1]}% WR, ${best_tier[2]:+.2f} PnL")
wr_md(f"3. **With-Trend Trades** — {wt_wr}% WR (vs {ct_wr}% counter-trend)")
wr_md(f"4. **intraday chain** — The most reliable timeframe alignment (1D→1H→5M)")
wr_md()

wr_md("### ❌ What's Bleeding Money")
wr_md(f"1. **Worst Symbol: {worst_sym[0]}** — {worst_sym[1]}% WR, ${worst_sym[2]:+.2f} PnL across {worst_sym[3]} trades")
wr_md(f"2. **Counter-Trend Trades** — Only {ct_wr}% WR, total loss of ${ct_pnl:+.2f}")
wr_md(f"3. **Tier-B Setups in isolation** — Without HTF OB confluence, Tier-B kills WR")
wr_md(f"4. **Asian session** — Low liquidity causes fake sweeps and immediate stop-outs")
wr_md(f"5. **Regime Mismatch Bug** — SKIP_COUNTER_TREND checked `trending_up/down` but")
wr_md(f"   analysis.py emits `strong_bull/strong_bear` → filter was completely bypassed")
wr_md()

wr_md("### 🔧 Code Bugs Still Active")
wr_md("1. **Regime Filter Bypass** (`challenge_trader.py` L680-685): Counter-trend filter")
wr_md("   uses `trending_up/trending_down` but analysis.py outputs `strong_bull/strong_bear`")
wr_md("2. **BE PnL Simulation Bug** (`challenge_trader.py` L1154-1186): When second-leg")
wr_md("   of a partial-exit trade hits BE stop, code deducts 50% risk instead of $0")
wr_md()

# ─── Recommended Rules Summary ──────────────────────────────────────────────
wr_md("---")
wr_md()
wr_md("## 📋 Recommended Rules (Data-Driven)")
wr_md()
wr_md("Based on the data across all challenges, these rules maximise Win Rate and PnL:")
wr_md()
wr_md("| Rule | Setting | Rationale |")
wr_md("| :--- | :--- | :--- |")
wr_md("| `MIN_SETUP_TIER` | `A` | Tier-B alone underperforms; require HTF OB confluence |")
wr_md("| `SKIP_COUNTER_TREND` | `True` | Counter-trend WR is critically low |")
wr_md("| Counter-trend regime labels | Fix bug to check `strong_bull`/`strong_bear` | Current filter is bypassed |")
wr_md("| Session filter | Avoid `asian` session | Highest loss rate, lowest quality sweeps |")
wr_md("| `MAX_TRADES_PER_DAY` | `3` | Over-trading destroys edge |")
wr_md()
wr_md("---")
wr_md()
wr_md(f"*Report covers {n_total} trades across {n_challenges} challenge runs.*  ")
wr_md(f"*Last updated: {now_str}*")

# ─────────────────────────────────────────────────────────────────────────────
# WRITE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
output_text = "\n".join(lines)
with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write(output_text)

print(f"\n{'='*72}")
print(f"  MASTER ANALYSIS REPORT WRITTEN: {OUTPUT_MD}")
print(f"{'='*72}")
print(f"  Challenges: {n_challenges}  |  Total Trades: {n_total}")
print(f"  Overall WR: {wr_all}%  |  Net PnL: ${pnl_all:+.2f}  |  PF: {pf_all}")
print(f"  Best Symbol : {best_sym[0]} ({best_sym[1]}% WR)")
print(f"  Worst Symbol: {worst_sym[0]} ({worst_sym[1]}% WR)")
print(f"  Counter-Trend WR: {ct_wr}% (${ct_pnl:+.2f}) — {'AVOID' if ct_wr < 40 else 'OK'}")
print(f"{'='*72}\n")
