"""
Deep diagnostic: understand WHY trades are being filtered out at every stage.
Run with: python tools/diagnose.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from core.lux_fvg import detect_luxalgo_fvgs, detect_liquidity_sweeps
from core.config import *
from models.backtest_cisd import (
    calc_fvgs, get_liquidity_sweeps as get_sweeps,
    find_order_blocks, get_hourly_bias, fetch_ohlcv_full
)

SYMBOL = "BTC/USDT"

print("=" * 70)
print("  DEEP DIAGNOSTIC — Understanding why no trades are generated")
print("=" * 70)

# Load data
df_1d = fetch_ohlcv_full(SYMBOL, "1d", DAYS_BACK)
df_1h = fetch_ohlcv_full(SYMBOL, "1h", DAYS_BACK)
df_15m = fetch_ohlcv_full(SYMBOL, "15m", DAYS_BACK)
df_5m = fetch_ohlcv_full(SYMBOL, "5m", DAYS_BACK)

print("\n" + "=" * 70)
print("  1. DATA OVERVIEW")
print("=" * 70)
print(f"  1D candles : {len(df_1d)} ({df_1d.index[0].date()} to {df_1d.index[-1].date()})")
print(f"  1H candles : {len(df_1h)} ({df_1h.index[0].date()} to {df_1h.index[-1].date()})")
print(f"  15M candles: {len(df_15m)} ({df_15m.index[0].date()} to {df_15m.index[-1].date()})")
print(f"  5M candles : {len(df_5m)} ({df_5m.index[0].date()} to {df_5m.index[-1].date()})")

# ─── 2. BIAS ANALYSIS ────────────────────────────────
print("\n" + "=" * 70)
print("  2. HTF BIAS ANALYSIS (1H)")
print("=" * 70)

bias = get_hourly_bias(df_1h)
bull_bars = (bias == 1).sum()
bear_bars = (bias == -1).sum()
neutral_bars = (bias == 0).sum()
total = len(bias)

print(f"  Bullish bars : {bull_bars} ({bull_bars/total*100:.1f}%)")
print(f"  Bearish bars : {bear_bars} ({bear_bars/total*100:.1f}%)")
print(f"  Neutral bars : {neutral_bars} ({neutral_bars/total*100:.1f}%)")

# Check what HTF signals exist
fvgs_1h = calc_fvgs(df_1h)
sweeps_1h = detect_liquidity_sweeps(df_1h.copy(), lookback=10)
print(f"\n  1H Bull FVGs detected : {fvgs_1h['fvg_bull'].sum()}")
print(f"  1H Bear FVGs detected : {fvgs_1h['fvg_bear'].sum()}")
print(f"  1H Bull sweeps        : {sweeps_1h['sweep_bull'].sum()}")
print(f"  1H Bear sweeps        : {sweeps_1h['sweep_bear'].sum()}")

# ─── 3. LTF FVG DETECTION ────────────────────────────
for tf_name, df_ltf in [("15m", df_15m), ("5m", df_5m)]:
    print(f"\n{'=' * 70}")
    print(f"  3. LTF FVG DETECTION ({tf_name})")
    print(f"{'=' * 70}")

    fvgs = calc_fvgs(df_ltf)
    n_bull = fvgs['fvg_bull'].sum()
    n_bear = fvgs['fvg_bear'].sum()
    print(f"  Bull FVGs : {n_bull}")
    print(f"  Bear FVGs : {n_bear}")
    print(f"  Total FVGs: {n_bull + n_bear}")
    
    # Align bias to LTF
    bias_aligned = bias.reindex(df_ltf.index, method='ffill')
    
    # How many FVGs align with bias?
    bull_with_bias = ((fvgs['fvg_bull']) & (bias_aligned == 1)).sum()
    bear_with_bias = ((fvgs['fvg_bear']) & (bias_aligned == -1)).sum()
    print(f"\n  Bull FVGs with bullish bias: {bull_with_bias}")
    print(f"  Bear FVGs with bearish bias: {bear_with_bias}")
    print(f"  Aligned FVGs total         : {bull_with_bias + bear_with_bias}")

    # ─── 4. LTF ORDER BLOCKS ────────────────────────────
    ltf_obs = find_order_blocks(df_ltf)
    print(f"\n  LTF Order Blocks detected: {len(ltf_obs)}")
    bull_obs = [o for o in ltf_obs if o['type'] == 'bull']
    bear_obs = [o for o in ltf_obs if o['type'] == 'bear']
    print(f"    Bull OBs: {len(bull_obs)}")
    print(f"    Bear OBs: {len(bear_obs)}")

    # ─── 5. HTF OB CONFLUENCE ────────────────────────────
    htf_obs = find_order_blocks(df_1h)
    print(f"\n  HTF (1H) Order Blocks: {len(htf_obs)}")
    for o in htf_obs:
        print(f"    {o['type'].upper()} OB at {o['time']} — range: {o['low']:.2f} - {o['high']:.2f}")
    
    # ─── 6. SIMULATE WITH FILTERS DISABLED ONE BY ONE ─────
    print(f"\n{'=' * 70}")
    print(f"  4. FILTER IMPACT ANALYSIS ({tf_name})")
    print(f"{'=' * 70}")
    
    from models.backtest_cisd import simulate_trades
    
    configs = [
        ("ALL FILTERS OFF", dict(use_fvg_quality=False, use_htf_ob=False, use_pd_array=False, use_pullback=False, use_auto_be=False)),
        ("+ FVG Quality",    dict(use_fvg_quality=True, use_htf_ob=False, use_pd_array=False, use_pullback=False, use_auto_be=False)),
        ("+ Pullback",       dict(use_fvg_quality=True, use_htf_ob=False, use_pd_array=False, use_pullback=True, use_auto_be=False)),
        ("+ HTF OB",         dict(use_fvg_quality=True, use_htf_ob=True, use_pd_array=False, use_pullback=False, use_auto_be=False)),
        ("+ Premium/Disc",   dict(use_fvg_quality=True, use_htf_ob=False, use_pd_array=True, use_pullback=False, use_auto_be=False)),
        ("ALL FILTERS ON",   dict(use_fvg_quality=True, use_htf_ob=True, use_pd_array=True, use_pullback=True, use_auto_be=True)),
    ]
    
    print(f"  {'Config':<20} {'Trades':>7} {'Wins':>5} {'Loss':>5} {'WR%':>6} {'PnL':>10}")
    print(f"  {'-'*55}")
    
    for name, kwargs in configs:
        res = simulate_trades(df_ltf, df_1h, df_1d, symbol=SYMBOL, df_daily=df_1d, **kwargs)
        trades = res['trades']
        if trades:
            df_t = pd.DataFrame(trades)
            wins = (df_t['result'] == 'WIN').sum()
            losses = (df_t['result'] == 'LOSS').sum()
            wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            pnl = df_t['pnl'].sum()
        else:
            wins = losses = 0
            wr = pnl = 0
        
        final_count = res['funnel'].get('Final Trades', 0)
        print(f"  {name:<20} {final_count:>7} {wins:>5} {losses:>5} {wr:>5.1f}% ${pnl:>8.2f}")
        
        # Print funnel for first and last
        if name in ("ALL FILTERS OFF", "ALL FILTERS ON"):
            print(f"    Funnel: ", end="")
            for k, v in res['funnel'].items():
                if v > 0 or k in ('Final Trades',):
                    print(f"{k}={v} → ", end="")
            print()

# ─── 7. ROOT CAUSE: CISD TRIGGER ANALYSIS ────────────
print(f"\n{'=' * 70}")
print(f"  5. CISD TRIGGER ANALYSIS (why entries get blocked)")
print(f"{'=' * 70}")

# Run with no filters, manually trace the CISD trigger logic
df_ltf = df_15m
fvgs_ltf = calc_fvgs(df_ltf)
bias_aligned = bias.reindex(df_ltf.index, method='ffill')
avg_range_ltf = (df_ltf['high'] - df_ltf['low']).rolling(20).mean()

cisd_failures = {"no_breakout": 0, "not_impulsive": 0, "risk_too_big": 0, "risk_zero": 0}
cisd_details = []

for i in range(25, len(df_ltf)):
    row = df_ltf.iloc[i]
    b = bias_aligned.iloc[i]
    if b == 0: continue
    
    fvg_bull = fvgs_ltf['fvg_bull'].iloc[i]
    fvg_bear = fvgs_ltf['fvg_bear'].iloc[i]
    is_bull = (b == 1 and fvg_bull)
    is_bear = (b == -1 and fvg_bear)
    if not (is_bull or is_bear): continue
    
    entry = row['close']
    
    # CISD check
    if is_bull:
        prev_high = df_ltf['high'].iloc[i-3:i].max()
        if entry <= prev_high:
            cisd_failures["no_breakout"] += 1
            continue
        stop = df_ltf['low'].iloc[i-3:i+1].min()
        risk = entry - stop
    else:
        prev_low = df_ltf['low'].iloc[i-3:i].min()
        if entry >= prev_low:
            cisd_failures["no_breakout"] += 1
            continue
        stop = df_ltf['high'].iloc[i-3:i+1].max()
        risk = stop - entry
    
    # Impulsive check
    body = abs(row['close'] - row['open'])
    threshold = avg_range_ltf.iloc[i] * CISD_BODY_MULT
    if body < threshold:
        cisd_failures["not_impulsive"] += 1
        continue
    
    # Risk check
    max_risk = entry * 0.05
    if risk <= 0:
        cisd_failures["risk_zero"] += 1
        continue
    if risk > max_risk:
        cisd_failures["risk_too_big"] += 1
        continue
    
    cisd_details.append({
        "time": df_ltf.index[i],
        "direction": "BULL" if is_bull else "BEAR",
        "entry": entry,
        "stop": stop,
        "risk": risk,
        "risk_pct": risk/entry*100
    })

print(f"\n  CISD Failure Breakdown (15m, bias+FVG aligned candles):")
for k, v in cisd_failures.items():
    print(f"    {k:<20}: {v}")
print(f"    PASSED all checks : {len(cisd_details)}")

if cisd_details:
    print(f"\n  CISD-passed trades that need HTF OB / PD array:")
    for d in cisd_details[:10]:
        print(f"    {d['time']} | {d['direction']} | Entry: {d['entry']:.2f} | Risk: {d['risk']:.2f} ({d['risk_pct']:.3f}%)")

# ─── 8. FUNDAMENTAL PROBLEM ──────────────────────────
print(f"\n{'=' * 70}")
print(f"  6. FUNDAMENTAL PROBLEM SUMMARY")
print(f"{'=' * 70}")
print(f"""
  DAYS_BACK = {DAYS_BACK} days — this means only ~{len(df_15m)} candles on 15m
  and only ~{len(df_1h)} candles on 1H for bias.
  
  With only 7 days of data:
  - Very few FVGs form on 1H (need displacement + gaps)
  - Even fewer align with bias (need FVG/sweep → momentum confirmation)
  - Even fewer pass CISD (need breakout of 3-bar high/low)
  - Even fewer align with HTF OB (only {len(htf_obs)} in 7 days)
  
  RECOMMENDATION: Fetch AT LEAST 365 days (ideally 730) of data.
  The strategy needs hundreds of candles to find enough setups.
""")
