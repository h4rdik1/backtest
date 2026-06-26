import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════
# TASK 7 — FVG Expiry by Timeframe
# CHANGED: FVGs now track their creation bar index and expire
# after a configurable number of bars depending on the timeframe.
# WHY: A 5-minute FVG from 12 hours ago is no longer relevant
# to current price action. Holding expired FVGs wastes compute
# and can cause ghost entries from dead zones.
# ══════════════════════════════════════════════════════════════

from core.config import FVG_EXPIRY_BARS


def detect_luxalgo_fvgs(df: pd.DataFrame, wick_ratio: float = 0.36, avg_body_lookback: int = 10, min_size_mult: float = 0.5, min_volume_mult: float = 1.0, timeframe: str = None):
    """
    Implements exact LuxAlgo FVG logic using Pandas vectorization.
    Adds 'fvg_bull', 'fvg_bull_btm', 'fvg_bull_top', 'fvg_bear', 'fvg_bear_top', 'fvg_bear_btm' to df.

    TASK 7 ADDITION: Also adds 'fvg_bull_created_bar' and 'fvg_bear_created_bar'
    columns that store the bar index at which each FVG was created.
    These are used by is_fvg_expired() to check if an FVG is still valid.

    Args:
        df            : OHLCV DataFrame
        wick_ratio    : Max wick-to-body ratio for displacement candle
        avg_body_lookback: Bars used to compute average body size
        min_size_mult : Minimum FVG gap size as multiple of avg range
        min_volume_mult: Minimum volume multiplier for displacement
        timeframe     : Optional string like '5m', '15m', '1h', '4h', '1w'.
                        Used to populate expiry tracking columns.
    """
    # 1. Body & Wick Calculations
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # 2. Rolling Averages
    df['avg_body'] = df['body'].rolling(window=avg_body_lookback, min_periods=1).mean()
    df['avg_range'] = (df['high'] - df['low']).rolling(window=10, min_periods=1).mean()
    df['avg_volume'] = df['volume'].rolling(window=20, min_periods=1).mean().shift(1)
    
    # 3. Displacement Logic (Candle 2)
    # clean_body = wicks are small relative to body
    df['clean_body'] = (df['upper_wick'] < (df['body'] * wick_ratio)) & (df['lower_wick'] < (df['body'] * wick_ratio))
    df['large_body'] = df['body'] > df['avg_body']
    
    # Volume Filter: Displacement must have significant volume
    df['volume_valid'] = df['volume'] >= (df['avg_volume'] * min_volume_mult)
    
    df['is_displacement'] = df['clean_body'] & df['large_body'] & df['volume_valid']
    
    # Directional Displacement
    df['bull_disp'] = df['is_displacement'] & (df['close'] > df['open'])
    df['bear_disp'] = df['is_displacement'] & (df['close'] < df['open'])
    
    # 4. FVG Gap formation rules (Candle i is candle 3, so candle i-1 is the displacement candle)
    # We evaluate at candle 3 (i). Candle 2 (i-1) is the displacement. Candle 1 (i-2) is the origin.
    
    # Shift series to evaluate everything on the row of Candle 3
    disp_prev = df['bull_disp'].shift(1)
    c1_high = df['high'].shift(2)
    c3_low = df['low']
    
    # Bullish FVG
    bull_gap_exists = c3_low > c1_high
    is_bull_fvg = disp_prev & bull_gap_exists
    fvg_size_bull = c3_low - c1_high
    size_pass_bull = fvg_size_bull >= (min_size_mult * df['avg_range'].shift(1)) # evaluate against ATR at time of displacement
    
    df['fvg_bull'] = is_bull_fvg & size_pass_bull
    df['fvg_bull_btm'] = np.where(df['fvg_bull'], c1_high, np.nan)
    df['fvg_bull_top'] = np.where(df['fvg_bull'], c3_low, np.nan)
    df['fvg_bull_mid'] = np.where(df['fvg_bull'], c1_high + (fvg_size_bull / 2), np.nan)
    
    # Bearish FVG
    bear_disp_prev = df['bear_disp'].shift(1)
    c1_low = df['low'].shift(2)
    c3_high = df['high']
    
    bear_gap_exists = c1_low > c3_high
    is_bear_fvg = bear_disp_prev & bear_gap_exists
    fvg_size_bear = c1_low - c3_high
    size_pass_bear = fvg_size_bear >= (min_size_mult * df['avg_range'].shift(1))
    
    df['fvg_bear'] = is_bear_fvg & size_pass_bear
    df['fvg_bear_top'] = np.where(df['fvg_bear'], c1_low, np.nan)
    df['fvg_bear_btm'] = np.where(df['fvg_bear'], c3_high, np.nan)
    df['fvg_bear_mid'] = np.where(df['fvg_bear'], c3_high + (fvg_size_bear / 2), np.nan)

    # ── TASK 7: Store creation bar index for expiry tracking ──
    # We use a sequential integer index (position in DataFrame).
    # 'is_fvg_expired()' will compare current bar position vs creation bar.
    # Storing these allows other code to check FVG freshness without
    # re-scanning the entire history.
    bar_index_series = pd.Series(range(len(df)), index=df.index)
    df['fvg_bull_created_bar'] = np.where(df['fvg_bull'], bar_index_series, np.nan)
    df['fvg_bear_created_bar'] = np.where(df['fvg_bear'], bar_index_series, np.nan)
    df['fvg_timeframe'] = timeframe or ""  # store TF for expiry lookup

    # Cleanup intermediate columns
    df.drop(columns=['body', 'upper_wick', 'lower_wick', 'avg_body', 'avg_range', 'avg_volume', 'clean_body', 'large_body', 'volume_valid', 'is_displacement', 'bull_disp', 'bear_disp'], inplace=True)
    return df


def is_fvg_expired(
    fvg_created_bar: int,
    current_bar: int,
    timeframe: str,
) -> bool:
    """
    TASK 7 — Check if an FVG has expired based on the timeframe.

    Each timeframe has a different maximum number of bars an FVG
    is considered valid. Once an FVG has been open too long, it
    is stale and should not be used for new entries.

    Args:
        fvg_created_bar : Bar index (integer position) when the FVG formed.
        current_bar     : Current bar index to compare against.
        timeframe       : One of '1w', '4h', '1h', '15m', '5m'.

    Returns:
        True if the FVG has expired and should NOT be used.
        False if the FVG is still fresh and valid.
    """
    # Get the expiry limit for this timeframe (default 48 bars if unknown)
    expiry_limit = FVG_EXPIRY_BARS.get(timeframe, 48)

    bars_alive = current_bar - fvg_created_bar
    return bars_alive > expiry_limit


def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 10):
    """
    Detects if the current candle swept a recent 10-bar pivot high/low but closed back inside the range.
    Adds 'sweep_bull' and 'sweep_bear' to df.
    """
    df['swing_high'] = df['high'].rolling(window=lookback).max().shift(1)
    df['swing_low'] = df['low'].rolling(window=lookback).min().shift(1)
    
    # Bear sweep (Bullish Bias signal): price goes below recent low, but closes above it
    df['sweep_bull'] = (df['low'] < df['swing_low']) & (df['close'] > df['swing_low'])
    
    # Bull sweep (Bearish Bias signal): price goes above recent high, but closes below it
    df['sweep_bear'] = (df['high'] > df['swing_high']) & (df['close'] < df['swing_high'])
    
    df.drop(columns=['swing_high', 'swing_low'], inplace=True)
    return df
