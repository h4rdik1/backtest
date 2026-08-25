#!/usr/bin/env python
import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backtest_cisd import (
    fetch_live_ohlcv, 
    find_latest_sfp, 
    find_order_blocks, 
    classify_regime,
    find_recent_swings
)
from core.config import format_price

def analyze_symbol(symbol: str):
    print("=" * 60)
    print(f"  SMC Trade Setup Analysis: {symbol}")
    print(f"  Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. Fetch data for all required timeframes
    # 4H = Context, 1H = Bias, 5M = Entry
    if ":" in symbol:
        sym_name, ex_id = symbol.split(":")
    else:
        sym_name, ex_id = symbol, "binance"

    try:
        df_4h = fetch_live_ohlcv(sym_name, "4h", 100, exchange_id=ex_id)
        df_1h = fetch_live_ohlcv(sym_name, "1h", 200, exchange_id=ex_id)
        df_5m = fetch_live_ohlcv(sym_name, "5m", 500, exchange_id=ex_id)
    except Exception as e:
        print(f"[ERROR] Could not fetch data for {symbol}: {e}")
        return

    if df_4h.empty or df_1h.empty or df_5m.empty:
        print(f"[ERROR] One or more dataframes are empty for {symbol}.")
        return

    # --- 4H CONTEXT ANALYSIS ---
    print("\n[HTF CONTEXT - 4H]")
    regime = classify_regime(df_4h, df_4h.index[-1])
    
    # Structural Range using Confirmed Fractal Swing High / Swing Low
    # (Excludes active unconfirmed expansion candles)
    range_high, range_low = find_recent_swings(df_4h, len(df_4h) - 1, lookback=60)
    current_price = df_5m['close'].iloc[-1]
    range_high = max(range_high, current_price)
    range_low = min(range_low, current_price)
    range_mid = (range_high + range_low) / 2
    pd_pct = min(100.0, max(0.0, ((current_price - range_low) / (range_high - range_low)) * 100))
    
    print(f"  Regime:       {regime.replace('_', ' ').upper()}")
    print(f"  4H Range:     {format_price(range_low)} - {format_price(range_high)}")
    print(f"  Current:      {format_price(current_price)} ({pd_pct:.1f}% of range)")
    
    if current_price > range_mid:
        print("  Status:       PREMIUM (Be careful with Longs)")
    else:
        print("  Status:       DISCOUNT (Optimal for Longs)")

    # --- 1H BIAS ANALYSIS ---
    print("\n[MTF BIAS - 1H]")
    sfp_1h_bull = find_latest_sfp(df_1h, is_long=True, lookback=48)
    sfp_1h_bear = find_latest_sfp(df_1h, is_long=False, lookback=48)
    
    if sfp_1h_bull:
        print(f"  [OK] BULLISH SFP found at {sfp_1h_bull[1].strftime('%H:%M')} | Price: {format_price(sfp_1h_bull[0])}")
    if sfp_1h_bear:
        print(f"  [!] BEARISH SFP found at {sfp_1h_bear[1].strftime('%H:%M')} | Price: {format_price(sfp_1h_bear[0])}")
        
    obs_1h = find_order_blocks(df_1h)
    active_obs_1h = [o for o in obs_1h if o['active']]
    if active_obs_1h:
        print(f"  Active 1H Order Blocks:")
        for ob in active_obs_1h[-2:]: # show last 2
            print(f"    - {ob['type'].upper()} OB: {format_price(ob['low'])} - {format_price(ob['high'])}")
    else:
        print("  No active 1H Order Blocks detected.")

    # --- 5M ENTRY ANALYSIS ---
    print("\n[LTF ENTRY - 5M]")
    sfp_5m_bull = find_latest_sfp(df_5m, is_long=True, lookback=100)
    sfp_5m_bear = find_latest_sfp(df_5m, is_long=False, lookback=100)
    
    if sfp_5m_bull:
        print(f"  [OK] 5M Sweep (Long) at {sfp_5m_bull[1].strftime('%H:%M')} | SL Anchor: < {format_price(sfp_5m_bull[0])}")
    if sfp_5m_bear:
        print(f"  [!] 5M Sweep (Short) at {sfp_5m_bear[1].strftime('%H:%M')} | SL Anchor: > {format_price(sfp_5m_bear[0])}")

    obs_5m = find_order_blocks(df_5m)
    active_obs_5m = [o for o in obs_5m if o['active']]
    if active_obs_5m:
        print(f"  Active 5M Order Blocks:")
        for ob in active_obs_5m[-3:]: # show last 3
             print(f"    - {ob['type'].upper()} OB: {format_price(ob['low'])} - {format_price(ob['high'])}")

    # --- FINAL VERDICT ---
    print("\n" + "=" * 60)
    print("  VERDICT & RISK NOTES")
    print("=" * 60)
    
    reasons = []
    if sfp_1h_bull and sfp_5m_bull:
        reasons.append("Nested SFP confirmed (1H + 5M). High conviction for Long.")
    elif sfp_1h_bear and sfp_5m_bear:
        reasons.append("Nested SFP confirmed (1H + 5M). High conviction for Short.")
    
    if regime == "strong_bull" and current_price < range_mid:
        reasons.append("Trend alignment: Bullish regime + Discount zone.")
    elif regime == "strong_bear" and current_price > range_mid:
        reasons.append("Trend alignment: Bearish regime + Premium zone.")
    
    if not reasons:
        print("  [!] Setup is secondary or counter-trend. Exercise caution.")
    else:
        for r in reasons:
            print(f"  - {r}")
            
    print("\n  [ADVICE]")
    if sfp_5m_bull:
        print(f"  For a Long: SL MUST be below {format_price(sfp_5m_bull[0])} (the 5M sweep low).")
    if sfp_5m_bear:
        print(f"  For a Short: SL MUST be above {format_price(sfp_5m_bear[0])} (the 5M sweep high).")
    
    print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a trade setup for a specific symbol.")
    parser.add_argument("symbol", help="The symbol to analyze (e.g., BTC/USDT)")
    args = parser.parse_args()
    
    analyze_symbol(args.symbol.upper())
