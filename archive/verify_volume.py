import pandas as pd
import os
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import lux_fvg as lux
from core.config import FVG_MIN_VOLUME_MULT

def verify_volume_filter():
    print("="*60)
    print(f"  VERIFYING VOLUME FILTER (Mult: {FVG_MIN_VOLUME_MULT})")
    print("="*60)
    
    # Load 5m BTC data
    df = pd.read_csv('data/ohlcv/BTC_USDT/5m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Detect FVGs WITHOUT volume filter (mult=0)
    df_no_filter = lux.detect_luxalgo_fvgs(df.copy(), min_volume_mult=0)
    fvgs_no_filter = df_no_filter[df_no_filter['fvg_bull'] | df_no_filter['fvg_bear']]
    
    # Detect FVGs WITH volume filter (mult from config)
    df_with_filter = lux.detect_luxalgo_fvgs(df.copy(), min_volume_mult=FVG_MIN_VOLUME_MULT)
    fvgs_with_filter = df_with_filter[df_with_filter['fvg_bull'] | df_with_filter['fvg_bear']]
    
    print(f"Total FVGs (No Filter): {len(fvgs_no_filter)}")
    print(f"Total FVGs (With Filter): {len(fvgs_with_filter)}")
    print(f"Filtered out: {len(fvgs_no_filter) - len(fvgs_with_filter)}")
    
    # Specifically check the 20:20 UTC candle
    trap_time = '2026-03-29 20:20:00+00:00'
    trap_candle = df_with_filter[df_with_filter['timestamp'] == trap_time]
    
    if not trap_candle.empty:
        # Note: detect_luxalgo_fvgs evaluates FVG on Candle 3 (20:30 UTC for the 20:20 displacement)
        # So we look at the candle AFTER the displacement
        eval_time = '2026-03-29 20:30:00+00:00'
        res = df_with_filter[df_with_filter['timestamp'] == eval_time]
        if not res.empty and res['fvg_bull'].values[0]:
            print(f"\n[OK] 20:20 UTC Pump passed the filter (it reached 3.6x avg volume).")
        else:
            print(f"\n[INFO] 20:20 UTC Pump was filtered out (it did NOT reach the volume threshold).")

if __name__ == "__main__":
    verify_volume_filter()
