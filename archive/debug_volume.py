import pandas as pd
import os
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import lux_fvg as lux
from core.config import FVG_MIN_VOLUME_MULT

def debug_volume_filter():
    # Load 5m BTC data
    df = pd.read_csv('data/ohlcv/BTC_USDT/5m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Body & Wick Calculations
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Rolling Averages
    df['avg_body'] = df['body'].rolling(window=10, min_periods=1).mean()
    df['avg_range'] = (df['high'] - df['low']).rolling(window=10, min_periods=1).mean()
    df['avg_volume'] = df['volume'].shift(1).rolling(window=20, min_periods=1).mean() # SHIFT to exclude current
    
    # Target Candle (20:20 UTC)
    target_time = '2026-03-29 20:20:00+00:00'
    target = df[df['timestamp'] == target_time].iloc[0]
    
    print(f"Target Time: {target_time}")
    print(f"Candle Volume: {target['volume']:.2f}")
    print(f"Avg Vol (Prev 20): {target['avg_volume']:.2f}")
    print(f"Ratio: {target['volume'] / target['avg_volume']:.2f}")
    
    # Now check the lux_fvg detection
    df_res = lux.detect_luxalgo_fvgs(df.copy(), min_volume_mult=1.5)
    
    eval_time = '2026-03-29 20:30:00+00:00'
    res = df_res[df_res['timestamp'] == eval_time].iloc[0]
    print(f"\nFinal FVG Bull Result: {res['fvg_bull']}")

if __name__ == "__main__":
    debug_volume_filter()
