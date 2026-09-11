#!/usr/bin/env python
import os
import sys
import argparse
from datetime import datetime
import pandas as pd

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backtest_cisd import fetch_live_ohlcv
from core.config import format_price

# ANSI color codes for terminal formatting
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
YELLOW = "\033[93m"

def display_candles(symbol: str, timeframe: str, limit: int):
    # Parse exchange if present
    if ":" in symbol:
        sym_name, ex_id = symbol.split(":")
    else:
        sym_name, ex_id = symbol, "binance"
        
    ex_id_display = ex_id.upper()
    sym_name_display = sym_name.upper()

    print("=" * 70)
    print(f"  {BOLD}Raw Candle Viewer:{RESET} {CYAN}{sym_name_display}{RESET} on {YELLOW}{ex_id_display}{RESET} ({timeframe})")
    print(f"  Fetching the last {limit} candles...")
    print("=" * 70)

    try:
        df = fetch_live_ohlcv(sym_name_display, timeframe, limit=limit, exchange_id=ex_id.lower())
    except Exception as e:
        print(f"  {RED}[ERROR] Failed to fetch data: {e}{RESET}")
        return

    if df.empty:
        print(f"  {RED}[ERROR] No data returned for {sym_name_display} on {ex_id_display}. Check symbol or connection.{RESET}")
        return

    # Print Table Header
    print(f"{BOLD}{'Timestamp (UTC)':<20} | {'Open':<10} | {'High':<10} | {'Low':<10} | {'Close':<10} | {'Volume':<12} | {'Type'}{RESET}")
    print("-" * 75)

    bullish_count = 0
    bearish_count = 0
    
    for timestamp, row in df.iterrows():
        o, h, l, c, v = row['open'], row['high'], row['low'], row['close'], row['volume']
        
        # Determine if candle is bullish or bearish
        if c >= o:
            color = GREEN
            c_type = "BULL"
            bullish_count += 1
        else:
            color = RED
            c_type = "BEAR"
            bearish_count += 1
            
        time_str = timestamp.strftime('%Y-%m-%d %H:%M')
        
        # Display formatted row
        print(f"{color}{time_str:<20} | {format_price(o):<10} | {format_price(h):<10} | {format_price(l):<10} | {format_price(c):<10} | {v:<12,.2f} | {c_type}{RESET}")

    print("-" * 75)
    
    # Calculate and display stats
    period_high = df['high'].max()
    period_low = df['low'].min()
    avg_vol = df['volume'].mean()
    current_price = df['close'].iloc[-1]
    
    print(f"  {BOLD}Period Statistics (Last {len(df)} candles):{RESET}")
    print(f"    - Max High:     {GREEN}{format_price(period_high)}{RESET}")
    print(f"    - Min Low:      {RED}{format_price(period_low)}{RESET}")
    print(f"    - Current:      {CYAN}{format_price(current_price)}{RESET}")
    print(f"    - Avg Volume:   {format_price(avg_vol)}")
    print(f"    - Ratio:        {GREEN}{bullish_count} Bullish{RESET} / {RED}{bearish_count} Bearish{RESET}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal candle checking utility.")
    parser.add_argument("symbol", help="The symbol to check (e.g. BTC/USDT or HYPE/USDT:okx)")
    parser.add_argument("-t", "--timeframe", default="1h", help="Timeframe (e.g. 5m, 15m, 1h, 4h, 1d). Default: 1h")
    parser.add_argument("-l", "--limit", type=int, default=20, help="Number of candles to show. Default: 20")
    
    args = parser.parse_args()
    
    # Enable ANSI terminal processing on Windows if needed
    if os.name == 'nt':
        os.system('')
        
    display_candles(args.symbol, args.timeframe, args.limit)
