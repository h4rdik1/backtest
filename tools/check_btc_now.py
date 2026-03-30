import pandas as pd
import os
import sys

# Add root to sys.path to allow imports from core/ and models/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import lux_fvg as lux
from models import backtest_cisd as cisd

def check_immediate_btc():
    print("="*60)
    print("  BTC LIVE 'GO/NO-GO' CHECK")
    print("="*60)
    
    # 1. Load latest data
    df_15m = cisd.fetch_ohlcv_full("BTC/USDT", "15m", 3)
    df_5m = cisd.fetch_ohlcv_full("BTC/USDT", "5m", 3)
    
    # 2. Find the 15m FVG (The "Anchor")
    fvgs_15m = lux.detect_luxalgo_fvgs(df_15m)
    bear_fvgs = fvgs_15m[fvgs_15m["fvg_bear"]]
    
    if bear_fvgs.empty:
        print("[NO GO] No 15m Bearish FVG found in the last 3 days.")
        return
        
    last_fvg = bear_fvgs.iloc[-1]
    fvg_btm = last_fvg["fvg_bear_btm"]
    fvg_top = last_fvg["fvg_bear_top"]
    fvg_time = last_fvg.name
    
    print(f"[ANCHOR] 15m Bearish FVG found at {fvg_time}")
    print(f"         Zone: {fvg_btm:.2f} - {fvg_top:.2f}")

    # 3. Check for FVG Tap
    tapped = False
    tap_idx = -1
    for i in range(len(df_5m)):
        if df_5m.index[i] > fvg_time:
            if df_5m["high"].iloc[i] >= fvg_btm:
                tapped = True
                tap_idx = i
                break
                
    if not tapped:
        print("[NO GO] Price has not reached the 15m FVG yet. Limit sell at midpoint recommended.")
        return
        
    print(f"[STATUS] Price tapped FVG at {df_5m.index[tap_idx]}")

    # 4. Check for CISD (Market Structure Shift) on 5m AFTER the tap
    lookback = 10
    recent_swing_low = df_5m["low"].iloc[tap_idx-lookback:tap_idx].min()
    print(f"[FILTER] Waiting for 5m close below Swing Low: {recent_swing_low:.2f}")

    cisd_found = False
    for i in range(tap_idx + 1, len(df_5m)):
        if df_5m["close"].iloc[i] < recent_swing_low:
            cisd_found = True
            cisd_time = df_5m.index[i]
            cisd_price = df_5m["close"].iloc[i]
            break
            
    if cisd_found:
        print(f"\n[GO] SIGNAL TRIGGERED at {cisd_time}!")
        print(f"      Entry Price: {cisd_price:.2f}")
        print(f"      Stop Loss:   {fvg_top:.2f}")
        print(f"      Target(2R):  {cisd_price - (fvg_top - cisd_price)*2:.2f}")
    else:
        print("\n[NO GO] Waiting for CISD. Price is still inside or above the FVG.")
        print(f"      Alert: Watch for a 5m candle to close BELOW the recent low of {recent_swing_low:.2f}")

if __name__ == "__main__":
    check_immediate_btc()
