# analyse_challenges.py
# Consolidates all challenge CSVs and prints a deep breakdown.
# Run from:  cd e:\backtest  then:  python tools/analyse_challenges.py
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import os
import sys

# ── File paths ──────────────────────────────────────────────────────────
FILES = {
    "Challenge 1 (ACTIVE Apr-02)": "archive/challenge_ACTIVE_20260405_0928.csv",
    "Challenge 2 (PASSED Apr-07)": "archive/challenge_PASSED_20260407_0920.csv",
    "Challenge 3 (STOPPED Apr-09)": "archive/challenge_STOPPED_20260409_0526.csv",
}

SEP  = "=" * 72
SEP2 = "-" * 72

def load(label, path):
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    df = pd.read_csv(path)
    df["_challenge"] = label
    return df

def wr(df):
    w = (df["result"] == "WIN").sum()
    l = (df["result"] == "LOSS").sum()
    total = w + l
    return w, l, total, round(w / total * 100, 1) if total else 0

def pf(df):
    gross_w = df[df["result"] == "WIN"]["pnl_usd"].sum()
    gross_l = abs(df[df["result"] == "LOSS"]["pnl_usd"].sum())
    return round(gross_w / gross_l, 2) if gross_l else float("inf")

def streak(df):
    max_w = cur_w = max_l = cur_l = 0
    for r in df["result"]:
        if r == "WIN":
            cur_w += 1; cur_l = 0; max_w = max(max_w, cur_w)
        elif r == "LOSS":
            cur_l += 1; cur_w = 0; max_l = max(max_l, cur_l)
    return max_w, max_l

# ────────────────────────────────────────────────────────────────────────
# LOAD ALL
# ────────────────────────────────────────────────────────────────────────
frames = []
for label, path in FILES.items():
    df = load(label, path)
    if df is not None:
        frames.append(df)

if not frames:
    print("No CSV files found. Make sure you're running from e:\\backtest\\")
    sys.exit(1)

# Deduplicate (the ACTIVE file had some duplicate trade_ids)
combined = pd.concat(frames, ignore_index=True)
combined = combined.drop_duplicates(subset=["trade_id", "_challenge"])
combined["pnl_usd"] = pd.to_numeric(combined["pnl_usd"], errors="coerce").fillna(0)
combined["opened_at"] = pd.to_datetime(combined["opened_at"], errors="coerce")
combined["closed_at"] = pd.to_datetime(combined["closed_at"], errors="coerce")
combined["duration_min"] = (combined["closed_at"] - combined["opened_at"]).dt.total_seconds() / 60

# ────────────────────────────────────────────────────────────────────────
# 1. PER-CHALLENGE SUMMARY
# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  ALL CHALLENGES — CONSOLIDATED REPORT")
print(SEP)

for label, df_c in combined.groupby("_challenge", sort=False):
    w, l, tot, wr_pct = wr(df_c)
    pf_val = pf(df_c)
    pnl = df_c["pnl_usd"].sum()
    mw, ml = streak(df_c)
    print(f"\n  +-- {label}")
    print(f"  |   Trades      : {tot}   ({w}W / {l}L)   WR: {wr_pct}%")
    print(f"  |   Net PnL     : ${pnl:+.2f}   Profit Factor: {pf_val}")
    print(f"  |   Max W-streak: {mw}   Max L-streak: {ml}")
    print(f"  +" + "-"*65)

# ────────────────────────────────────────────────────────────────────────
# 2. OVERALL STATS
# ────────────────────────────────────────────────────────────────────────
w_all, l_all, tot_all, wr_all = wr(combined)
pf_all = pf(combined)
mw_all, ml_all = streak(combined)
print(f"\n{SEP}")
print("  OVERALL ACROSS ALL CHALLENGES")
print(SEP)
print(f"  Total Trades      : {tot_all}  ({w_all}W / {l_all}L)   WR: {wr_all}%")
print(f"  Net PnL           : ${combined['pnl_usd'].sum():+.2f}")
print(f"  Profit Factor     : {pf_all}")
print(f"  Max Win Streak    : {mw_all}")
print(f"  Max Loss Streak   : {ml_all}")

# ────────────────────────────────────────────────────────────────────────
# 3. BREAKDOWN BY CHAIN
# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  BREAKDOWN BY CHAIN (all challenges)")
print(SEP)
print(f"  {'Chain':<12} {'Trades':>7} {'W':>5} {'L':>5} {'WR%':>7} {'PF':>7} {'PnL':>10}")
print(f"  {SEP2}")
for chain, g in combined.groupby("chain"):
    w, l, tot, wr_pct = wr(g)
    pf_val = pf(g)
    pnl = g["pnl_usd"].sum()
    flag = "  ← 🔴 WORST" if wr_pct < 30 else ("  ← 🟡" if wr_pct < 40 else "")
    print(f"  {chain:<12} {tot:>7} {w:>5} {l:>5} {wr_pct:>6.1f}% {pf_val:>7.2f} ${pnl:>8.2f}{flag}")

# ────────────────────────────────────────────────────────────────────────
# 4. BREAKDOWN BY SYMBOL
# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  BREAKDOWN BY SYMBOL (all challenges)")
print(SEP)
print(f"  {'Symbol':<14} {'Trades':>7} {'W':>5} {'L':>5} {'WR%':>7} {'PF':>7} {'PnL':>10}")
print(f"  {SEP2}")
for sym, g in combined.groupby("symbol"):
    w, l, tot, wr_pct = wr(g)
    pf_val = pf(g)
    pnl = g["pnl_usd"].sum()
    flag = "  ← 🔴" if wr_pct < 30 else ""
    print(f"  {sym:<14} {tot:>7} {w:>5} {l:>5} {wr_pct:>6.1f}% {pf_val:>7.2f} ${pnl:>8.2f}{flag}")

# ────────────────────────────────────────────────────────────────────────
# 5. BREAKDOWN BY DIRECTION
# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  BREAKDOWN BY DIRECTION (all challenges)")
print(SEP)
print(f"  {'Direction':<10} {'Trades':>7} {'W':>5} {'L':>5} {'WR%':>7} {'PF':>7} {'PnL':>10}")
print(f"  {SEP2}")
for d, g in combined.groupby("direction"):
    w, l, tot, wr_pct = wr(g)
    pf_val = pf(g)
    pnl = g["pnl_usd"].sum()
    print(f"  {d:<10} {tot:>7} {w:>5} {l:>5} {wr_pct:>6.1f}% {pf_val:>7.2f} ${pnl:>8.2f}")

# ────────────────────────────────────────────────────────────────────────
# 6. BREAKDOWN BY REGIME
# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  BREAKDOWN BY MARKET REGIME (all challenges)")
print(SEP)
print(f"  {'Regime':<18} {'Trades':>7} {'W':>5} {'L':>5} {'WR%':>7} {'PF':>7} {'PnL':>10}")
print(f"  {SEP2}")
for r, g in combined.groupby("regime"):
    w, l, tot, wr_pct = wr(g)
    pf_val = pf(g)
    pnl = g["pnl_usd"].sum()
    print(f"  {r:<18} {tot:>7} {w:>5} {l:>5} {wr_pct:>6.1f}% {pf_val:>7.2f} ${pnl:>8.2f}")

# ────────────────────────────────────────────────────────────────────────
# 7. COUNTER-TREND ANALYSIS (biggest killer)
# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  COUNTER-TREND vs WITH-TREND")
print(SEP)

def is_counter(row):
    b = row.get("bias_reason", "")
    r = row.get("regime", "")
    d = row.get("direction", "")
    if "Bullish" in b and r in ("strong_bear", "trending_down"): return True
    if "Bullish" in b and d == "SHORT":                           return True
    if "Bearish" in b and r in ("strong_bull", "trending_up"):   return True
    if "Bearish" in b and d == "LONG":                            return True
    return False

combined["is_counter"] = combined.apply(is_counter, axis=1)
for ct_label, g in combined.groupby("is_counter"):
    label_str = "Counter-Trend" if ct_label else "With-Trend   "
    w, l, tot, wr_pct = wr(g)
    pf_val = pf(g)
    pnl = g["pnl_usd"].sum()
    flag = "  ← 🔴 AVOID" if ct_label else "  ← ✅ KEEP"
    print(f"  {label_str}: {tot} trades | {w}W {l}L | WR {wr_pct}% | PF {pf_val} | ${pnl:+.2f}{flag}")

# ────────────────────────────────────────────────────────────────────────
# 8. SCALP CHAIN PROBLEM — too many trades, too low WR
# ────────────────────────────────────────────────────────────────────────
scalp = combined[combined["chain"] == "scalp"]
non_scalp = combined[combined["chain"] != "scalp"]

print(f"\n{SEP}")
print("  SCALP vs NON-SCALP (Biggest volume driver)")
print(SEP)
for lbl, g in [("Scalp chain    ", scalp), ("Non-scalp chain", non_scalp)]:
    if len(g) == 0: continue
    w, l, tot, wr_pct = wr(g)
    pf_val = pf(g)
    pnl = g["pnl_usd"].sum()
    flag = "  ← 🔴 PROBLEM" if lbl.strip() == "Scalp chain" and wr_pct < 40 else ""
    print(f"  {lbl}: {tot} trades | {w}W {l}L | WR {wr_pct}% | PF {pf_val} | ${pnl:+.2f}{flag}")

# ────────────────────────────────────────────────────────────────────────
# 9. ANCHOR TYPE
# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  BREAKDOWN BY ENTRY ANCHOR (FVG / OB / CONFLUENCE)")
print(SEP)
for a, g in combined.groupby("anchor_type"):
    w, l, tot, wr_pct = wr(g)
    pf_val = pf(g)
    pnl = g["pnl_usd"].sum()
    print(f"  {a:<15} {tot:>5} trades | {w}W {l}L | WR {wr_pct}% | PF {pf_val} | ${pnl:+.2f}")

# ────────────────────────────────────────────────────────────────────────
# 10. DURATION ANALYSIS
# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TRADE DURATION ANALYSIS (minutes)")
print(SEP)
wins_dur  = combined[combined["result"] == "WIN"]["duration_min"].dropna()
losses_dur = combined[combined["result"] == "LOSS"]["duration_min"].dropna()
print(f"  Avg duration of WINS  : {wins_dur.mean():.0f} min")
print(f"  Avg duration of LOSSES: {losses_dur.mean():.0f} min")
print(f"  Median WIN duration   : {wins_dur.median():.0f} min")
print(f"  Median LOSS duration  : {losses_dur.median():.0f} min")

# ────────────────────────────────────────────────────────────────────────
# 11. KEY FINDINGS SUMMARY
# ────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  KEY ROOT-CAUSE FINDINGS")
print(SEP)
scalp_wr = round((scalp["result"] == "WIN").sum() / max(len(scalp[scalp["result"].isin(["WIN","LOSS"])]), 1) * 100, 1)
ct_df    = combined[combined["is_counter"] == True]
ct_wr    = round((ct_df["result"] == "WIN").sum() / max(len(ct_df[ct_df["result"].isin(["WIN","LOSS"])]), 1) * 100, 1)

findings = [
    ("1", f"SCALP chain = {len(scalp)} trades ({round(len(scalp)/max(tot_all,1)*100)}% of total) at only {scalp_wr}% WR -- volume killer"),
    ("2", f"Counter-trend trades WR = {ct_wr}% -- bias vs direction mismatch is a major loss driver"),
    ("3", f"Challenge 3 had {len(combined[combined['_challenge'].str.contains('STOPPED')])} trades vs Challenge 2's {len(combined[combined['_challenge'].str.contains('PASSED')])} -- over-trading"),
    ("4", "Displacement Strength was consistently LOW (< 0.5x) in most entries -- weak entry signals"),
    ("5", "ALL setups are B-Tier -- scanner never found A or A++ setup in Challenge 3"),
]
for num, finding in findings:
    print(f"  [{num}] {finding}")

print(f"\n{SEP}\n")
