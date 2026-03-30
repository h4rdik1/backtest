import os
import sys

# Add root to sys.path to allow imports from core/ and models/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import lux_fvg as lux
from models import backtest_cisd as cisd
from core.config import format_price
import argparse

def check_immediate(symbol):
    print("="*60)
    print(f"  {symbol} LIVE 'GO/NO-GO' CHECK")
    print("="*60)
    
    # 1. Load latest data (7 days for context)
    df_15m = cisd.fetch_ohlcv_full(symbol, "15m", 7)
    df_5m = cisd.fetch_ohlcv_full(symbol, "5m", 7)
    
    # 2. Find the 15m FVG (The "Anchor")
    fvgs_15m = lux.detect_luxalgo_fvgs(df_15m)
    bear_fvgs = fvgs_15m[fvgs_15m["fvg_bear"]]
    bull_fvgs = fvgs_15m[fvgs_15m["fvg_bull"]]
    
    # Determine Bias (Simple check of last state)
    df_bias = cisd.fetch_ohlcv_full(symbol, "1h", 7)
    df_bias['bias'] = cisd.get_hourly_bias(df_bias)
    current_bias = df_bias['bias'].iloc[-1]
    
    if current_bias == 0:
        print("[NO GO] Neutral Bias. Waiting for HTF FVG or Liquidity Sweep.")
        return

    bias_dir = "BULLISH" if current_bias == 1 else "BEARISH"
    print(f"[BIAS] Current HTF Bias is {bias_dir}")

    # Find the latest FVG in the bias direction
    if current_bias == -1:
        if bear_fvgs.empty:
            print("[NO GO] No 15m Bearish FVG found recently.")
            return
        last_fvg = bear_fvgs.iloc[-1]
        fvg_btm, fvg_top = last_fvg["fvg_bear_btm"], last_fvg["fvg_bear_top"]
    else:
        if bull_fvgs.empty:
            print("[NO GO] No 15m Bullish FVG found recently.")
            return
        last_fvg = bull_fvgs.iloc[-1]
        fvg_btm, fvg_top = last_fvg["fvg_bull_btm"], last_fvg["fvg_bull_top"]

    fvg_time = last_fvg.name
    print(f"[ANCHOR] 15m {bias_dir} FVG found at {fvg_time}")
    print(f"         Zone: {format_price(fvg_btm)} - {format_price(fvg_top)}")

    # 3. Check for FVG Tap
    tapped = False
    tap_idx = -1
    for i in range(len(df_5m)):
        if df_5m.index[i] > fvg_time:
            if current_bias == -1: # Bearish Look for HIGH >= btm
                if df_5m["high"].iloc[i] >= fvg_btm:
                    tapped = True
                    tap_idx = i
                    break
            else: # Bullish Look for LOW <= top
                if df_5m["low"].iloc[i] <= fvg_top:
                    tapped = True
                    tap_idx = i
                    break
                
    if not tapped:
        print(f"[NO GO] Price has not reached the {bias_dir} FVG yet.")
        mid = (fvg_top + fvg_btm) / 2
        print(f"         Pending Order Level: {format_price(mid)}")
        return
        
    print(f"[STATUS] Price tapped FVG at {df_5m.index[tap_idx]}")

    # 4. Check for CISD (Market Structure Shift) on 5m AFTER the tap
    lookback = 10
    if current_bias == -1: # Bearish: Break below recent low
        recent_target = df_5m["low"].iloc[tap_idx-lookback:tap_idx].min()
        print(f"[FILTER] Waiting for 5m close BELOW Swing Low: {format_price(recent_target)}")
    else: # Bullish: Break above recent high
        recent_target = df_5m["high"].iloc[tap_idx-lookback:tap_idx].max()
        print(f"[FILTER] Waiting for 5m close ABOVE Swing High: {format_price(recent_target)}")

    cisd_found = False
    for i in range(tap_idx + 1, len(df_5m)):
        close = df_5m["close"].iloc[i]
        if (current_bias == -1 and close < recent_target) or (current_bias == 1 and close > recent_target):
            cisd_found = True
            cisd_time = df_5m.index[i]
            cisd_price = close
            break
            
    if cisd_found:
        print(f"\n[GO] SIGNAL TRIGGERED at {cisd_time}!")
        print(f"      Entry Price: {format_price(cisd_price)}")
        sl = fvg_top if current_bias == -1 else fvg_btm
        dist = abs(sl - cisd_price)
        tp = cisd_price - (dist * 2) if current_bias == -1 else cisd_price + (dist * 2)
        print(f"      Stop Loss:   {format_price(sl)}")
        print(f"      Target(2R):  {format_price(tp)}")
    else:
        print("\n[NO GO] Waiting for Market Structure Shift (CISD).")
        print(f"      Watch for a 5m candle to close beyond {format_price(recent_target)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    args = parser.parse_args()
    check_immediate(args.symbol)
