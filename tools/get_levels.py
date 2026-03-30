import pandas as pd
import lux_fvg as lux
import backtest_cisd as cisd
import os

def get_signal_levels(symbol, timeframe):
    try:
        df = cisd.fetch_ohlcv_full(symbol, timeframe, 7)
        fvgs = lux.detect_luxalgo_fvgs(df)
        
        # Get the latest bearish FVG
        bear_fvgs = fvgs[fvgs['fvg_bear']]
        if bear_fvgs.empty:
            print(f"{symbol}: No Bearish FVG found in last 7 days.")
            return
            
        last_fvg = bear_fvgs.iloc[-1]
        top = last_fvg['fvg_bear_top']
        btm = last_fvg['fvg_bear_btm']
        time = last_fvg.name
        
        # Calculate Risk/Reward (Standard 2.0R)
        # SL is top of FVG + small buffer
        sl = top * 1.001 
        entry = (top + btm) / 2 # Midpoint entry is safest
        target = entry - (sl - entry) * 2.0
        
        print(f"\n[SIGNAL]: {symbol} (15M Bearish Setup)")
        print(f"   FVG Time: {time}")
        print(f"   Entry Zone: {btm:.2f} - {top:.2f}")
        print(f"   Ideal Entry (50%): {entry:.2f}")
        print(f"   Stop Loss: {sl:.2f}")
        print(f"   Take Profit (2R): {target:.2f}")
        print(f"   Current Status: TAPPED (Wait for CISD Short)")
    except Exception as e:
        print(f"Error for {symbol}: {e}")

if __name__ == "__main__":
    get_signal_levels("BTC/USDT", "15m")
    get_signal_levels("ETH/USDT", "15m")
