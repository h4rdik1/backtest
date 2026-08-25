import pandas as pd
import os
import argparse
import ccxt
from typing import List, Dict

from core.lux_fvg import detect_luxalgo_fvgs, detect_liquidity_sweeps

from core.config import (
    RISK_REWARD, FIXED_SL_USDT, ACCOUNT_SIZE,
    DAYS_BACK, BIAS_EXPIRY_BARS, FVG_WICK_RATIO, FVG_AVG_BODY_LOOKBACK,
    FVG_MIN_SIZE_MULT, FVG_MIN_VOLUME_MULT, OB_MOVE_MULT, OB_SWING_LOOKBACK,
    AUTO_BREAKEVEN_R, BE_TRIGGER_R, PARTIAL_EXIT_R, PARTIAL_EXIT_PCT,
    USE_FVG_QUALITY, USE_HTF_OB_CONFLUENCE, USE_PREMIUM_DISCOUNT,
    USE_PULLBACK_FILTER, USE_AUTO_BREAKEVEN, USE_LTF_OB_ENTRY,
    SKIP_COUNTER_TREND, SKIP_RANGING_REGIME,
    MIN_SL_PCT_BTC, MIN_SL_PCT_ALTS,
    MAX_TRADE_DURATION,
    TEST_START, DATA_DIR,
    TF_CONTEXT, TF_BIAS, TF_ENTRY, ACTIVE_CHAIN, chain_label,
    PROTECTED_SWING_LOOKBACK, PROTECTED_SWING_BUFFER,
    MAX_STOP_DISTANCE, PREFER_HTF_STRUCTURAL_SL,
    # Task 1 — SFP constants
    MIN_SWEEP_BUFFER_PCT, MAX_SWEEP_WAIT_BARS, SFP_CLOSE_BUFFER_PCT,
)

from core.analysis import (
    print_enhanced_stats, classify_regime
)

# Module-level symbol (set by CLI or multi-asset runner)
SYMBOL = "BTC/USDT"


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def fetch_ohlcv_full(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Load cached OHLCV data from data/ohlcv/ directory."""
    safe_symbol = symbol.replace("/", "_")
    filename = os.path.join(DATA_DIR, safe_symbol, f"{timeframe}.csv")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing {filename}. Run: python fetch_candles.py --symbol {symbol} --days {days}")
    df = pd.read_csv(filename)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    print(f"   {timeframe}: {len(df):,} candles loaded from {filename}")
    return df


# -- Exchange caching for live scanner --
EXCHANGES = {}

def get_exchange(exchange_id: str = 'binance'):
    """Get or create a cached CCXT exchange instance."""
    global EXCHANGES
    exchange_id = exchange_id.lower().replace("usdt", "")
    if exchange_id not in EXCHANGES:
        try:
            exchange_class = getattr(ccxt, exchange_id)
            if exchange_id == "binance":
                EXCHANGES[exchange_id] = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 10000,
                    'options': {
                        'defaultType': 'spot',
                        'adjustForTimeDifference': True,
                        'urls': {
                            'api': {
                                'public': 'https://api3.binance.com/api/v3',
                                'private': 'https://api3.binance.com/api/v3',
                            }
                        }
                    }
                })
            else:
                EXCHANGES[exchange_id] = exchange_class({'enableRateLimit': True, 'timeout': 10000})
        except AttributeError:
            available = ", ".join(ccxt.exchanges[:10]) + "..."
            raise ValueError(f"Exchange '{exchange_id}' not found. Supported: {available}")
    return EXCHANGES[exchange_id]


def fetch_live_ohlcv(symbol: str, timeframe: str, limit: int = 100,
                     exchange_id: str = 'binance') -> pd.DataFrame:
    """Fetch the most recent candles from the exchange (for live scanner)."""
    try:
        exchange = get_exchange(exchange_id)
        raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"      [ERROR] Could not fetch live data for {symbol} {timeframe}: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════
# FVG / SWEEP / OB DETECTION
# ══════════════════════════════════════════════════════════════

def calc_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    """Detect FVGs using LuxAlgo displacement logic."""
    return detect_luxalgo_fvgs(df.copy(), wick_ratio=FVG_WICK_RATIO,
                               avg_body_lookback=FVG_AVG_BODY_LOOKBACK,
                               min_size_mult=FVG_MIN_SIZE_MULT,
                               min_volume_mult=FVG_MIN_VOLUME_MULT)


def get_liquidity_sweeps(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Detect liquidity sweeps (wicks past swing points that close back inside)."""
    return detect_liquidity_sweeps(df.copy(), lookback=10)


def find_order_blocks(df: pd.DataFrame) -> List[Dict]:
    """
    Order Block detection.
    Bullish OB = last bearish candle before an aggressive bullish impulse at a swing low.
    """
    obs = []
    avg_range = (df['high'] - df['low']).rolling(20).mean()

    for i in range(OB_SWING_LOOKBACK, len(df) - 1):
        body_val = df['close'].iloc[i] - df['open'].iloc[i]
        ar = avg_range.iloc[i]
        if pd.isna(ar) or ar == 0:
            continue
        is_impulse = abs(body_val) > (ar * OB_MOVE_MULT)

        if is_impulse:
            j = i - 1
            if j < 0:
                continue
            if body_val > 0:  # Bullish Impulse
                if df['close'].iloc[j] < df['open'].iloc[j]:
                    ob_high = df['high'].iloc[j]
                    ob_low = df['low'].iloc[j]
                    ob_open = df['open'].iloc[j]
                    ob_close = df['close'].iloc[j]

                    if df['close'].iloc[i] > ob_high:
                        lows_around = df['low'].iloc[max(0, j - OB_SWING_LOOKBACK):min(len(df), j + OB_SWING_LOOKBACK + 1)]
                        if df['low'].iloc[j] == lows_around.min():
                            obs.append({
                                "time": df.index[j],
                                "high": ob_high, "low": ob_low,
                                "median": (ob_open + ob_close) / 2,
                                "active": True,
                                "type": "bull"
                            })
            else:  # Bearish Impulse
                if df['close'].iloc[j] > df['open'].iloc[j]:
                    ob_high = df['high'].iloc[j]
                    ob_low = df['low'].iloc[j]
                    ob_open = df['open'].iloc[j]
                    ob_close = df['close'].iloc[j]

                    if df['close'].iloc[i] < ob_low:
                        highs_around = df['high'].iloc[max(0, j - OB_SWING_LOOKBACK):min(len(df), j + OB_SWING_LOOKBACK + 1)]
                        if df['high'].iloc[j] == highs_around.max():
                            obs.append({
                                "time": df.index[j],
                                "high": ob_high, "low": ob_low,
                                "median": (ob_open + ob_close) / 2,
                                "active": True,
                                "type": "bear"
                            })
    return obs


# ══════════════════════════════════════════════════════════════
# STRUCTURAL SWINGS & RANGES
# ══════════════════════════════════════════════════════════════

def find_recent_swings(df: pd.DataFrame, bar_idx: int, lookback: int = 150) -> tuple[float, float]:
    """
    Identifies the dealing range at bar_idx using 5-bar fractal swing points.

    Searches backwards from bar_idx-2 (to prevent lookahead — right-side neighbours
    at idx+1 and idx+2 are at most bar_idx-1 and bar_idx, both already closed).
    Finds the most recent fractal HIGH and fractal LOW independently.
    Their midpoint = the 50% (equilibrium) of the current dealing range.

    NOTE ON RANGE VALIDITY:
    The two fractals may not be from the same swing leg. However, this is
    intentional: we want the most recent structural high/low that price is
    oscillating between, regardless of when each formed. If the market is
    trending hard in one direction, one of the fractals will likely be
    from a much older bar — in that case the rolling fallback fires,
    which returns the actual lookback high/low, giving a conservative midpoint.

    Falls back to rolling high/low if no 5-bar fractal is found within lookback.
    """
    latest_swing_high = None
    latest_swing_low = None

    # Search backwards from bar_idx-2 (prevents lookahead)
    for idx in range(bar_idx - 2, max(2, bar_idx - lookback), -1):
        if idx < 2 or idx >= len(df) - 2:
            continue

        # 5-bar swing low: lower than 2 left + 2 right neighbours
        if latest_swing_low is None:
            if (df['low'].iloc[idx] < df['low'].iloc[idx - 1] and
                    df['low'].iloc[idx] < df['low'].iloc[idx - 2] and
                    df['low'].iloc[idx] < df['low'].iloc[idx + 1] and
                    df['low'].iloc[idx] < df['low'].iloc[idx + 2]):
                latest_swing_low = df['low'].iloc[idx]

        # 5-bar swing high: higher than 2 left + 2 right neighbours
        if latest_swing_high is None:
            if (df['high'].iloc[idx] > df['high'].iloc[idx - 1] and
                    df['high'].iloc[idx] > df['high'].iloc[idx - 2] and
                    df['high'].iloc[idx] > df['high'].iloc[idx + 1] and
                    df['high'].iloc[idx] > df['high'].iloc[idx + 2]):
                latest_swing_high = df['high'].iloc[idx]

        if latest_swing_low is not None and latest_swing_high is not None:
            break

    # Fallback: rolling high/low of the lookback window
    if latest_swing_low is None:
        latest_swing_low = df['low'].iloc[max(0, bar_idx - lookback):bar_idx + 1].min()
    if latest_swing_high is None:
        latest_swing_high = df['high'].iloc[max(0, bar_idx - lookback):bar_idx + 1].max()

    return latest_swing_high, latest_swing_low


# ══════════════════════════════════════════════════════════════
# PROTECTED SWING SL
# ══════════════════════════════════════════════════════════════

def find_protected_swing(df: pd.DataFrame, bar_idx: int, is_long: bool,
                         lookback: int = PROTECTED_SWING_LOOKBACK,
                         buffer: float = PROTECTED_SWING_BUFFER):
    """
    Enhanced Protected Stop Finder.
    Looks for:
    1. Liquidity Sweeps (SFP): Wicks that cleared previous structure.
    2. Structural Swings: Classic 3-bar fractal points.
    3. Prominent Wicks: Deepest/Highest wick in the recent lookback.
    """
    search_start = max(1, bar_idx - lookback)
    search_end   = bar_idx - 1
    
    # 1. Look for SFPs first (High Conviction)
    from core.lux_fvg import detect_liquidity_sweeps
    sweeps = detect_liquidity_sweeps(df.iloc[max(0, bar_idx - lookback - 10):bar_idx + 1].copy(), lookback=10)
    
    if is_long:
        # Check if latest bars had a Bullish Sweep (Stop Grab)
        for i in range(len(sweeps)-1, max(0, len(sweeps)-10), -1):
            if sweeps['sweep_bull'].iloc[i]:
                # The SL should be just below the sweep wick
                sweep_low = df['low'].loc[sweeps.index[i]]
                return sweep_low * (1.0 - buffer)
                
        # 2. Classic Structural Swing
        best_swing = None
        for i in range(search_end, search_start - 1, -1):
            if i < 1 or i >= len(df) - 1: continue
            lo, lo_l, lo_r = df['low'].iloc[i], df['low'].iloc[i-1], df['low'].iloc[i+1]
            if lo < lo_l and lo < lo_r:
                best_swing = lo
                break
        
        # 3. Prominent Wick (Deepest in lookback)
        deepest_wick = df['low'].iloc[search_start:search_end+1].min()
        
        # Priority: SFP > Structural Swing > Deepest Wick
        final_sl = best_swing if best_swing is not None else deepest_wick
        return final_sl * (1.0 - buffer)

    else: # SHORT
        for i in range(len(sweeps)-1, max(0, len(sweeps)-10), -1):
            if sweeps['sweep_bear'].iloc[i]:
                sweep_high = df['high'].loc[sweeps.index[i]]
                return sweep_high * (1.0 + buffer)
                
        best_swing = None
        for i in range(search_end, search_start - 1, -1):
            if i < 1 or i >= len(df) - 1: continue
            hi, hi_l, hi_r = df['high'].iloc[i], df['high'].iloc[i-1], df['high'].iloc[i+1]
            if hi > hi_l and hi > hi_r:
                best_swing = hi
                break
        
        highest_wick = df['high'].iloc[search_start:search_end+1].max()
        final_sl = best_swing if best_swing is not None else highest_wick
        return final_sl * (1.0 + buffer)


# ══════════════════════════════════════════════════════════════
# TASK 1 — 3-Candle SFP Detection System (CLEAN SWING LOGIC)
# CHANGED: C2 is now the clean 3-bar fractal swing point.
# WHY: The old code made C1 the swing and C2 just the next
# candle — meaning C2 was NOT verified as a swing low/high.
# This caused messy alerts on any random pullback.
#
# NEW STRICT DEFINITION (matches standard SMC):
#   C2 is a CLEAN SWING LOW  (bullish) : C2.low < C1.low  AND C2.low < C3.low
#   C2 is a CLEAN SWING HIGH (bearish) : C2.high > C1.high AND C2.high > C3.high
#   SFP Level = C2.low (bullish) or C2.high (bearish)
#   Invalidation: no close beyond the level between C3 and the sweep.
#   Sweep candle: wicks past the level AND closes back inside.
# ══════════════════════════════════════════════════════════════

def detect_sfp_pattern(
    df: pd.DataFrame,
    bar_idx: int,
    is_long: bool,
) -> dict:
    """
    TASK 1 — Detect a clean 3-candle SFP pattern ending at bar_idx.

    Bullish SFP (long signal — stops hunted BELOW a key low):
      C1: Any candle to the LEFT of C2.
      C2: Clean SWING LOW — C2.low < C1.low (left) AND C2.low < C3.low (right).
          SFP Level = C2.low.
      C3: Confirmed right-side neighbour of the fractal (C3.low > C2.low).
      Intermediate candles (C3+1 … bar_idx-1): None may CLOSE below C2.low.
          A close below means the level was broken, not swept.
      Sweep candle (bar_idx):
          - Wick goes BELOW C2.low (stop hunt)
          - Closes BACK ABOVE C2.low (rejection / seller trap)
          → Bullish SFP confirmed.

    Bearish SFP is the mirror image (longs trapped above C2.high).

    Returns a dict with full pattern details, or None if no pattern.
    """
    # We need at least 6 bars to find a meaningful pattern
    if bar_idx < 5 or bar_idx >= len(df):
        return None

    # Only search within a reasonable lookback window
    search_start = max(1, bar_idx - 60)

    # ── Bullish SFP: hunts stops BELOW a clean swing low (C2) ──
    if is_long:
        # Search backwards for C2 — a clean 3-bar fractal LOW.
        # C2 must have: C2.low < C1.low (left) AND C2.low < C3.low (right).
        # We need at least room for [C2, C3, …, sweep] before bar_idx.
        for c2 in range(bar_idx - 2, search_start, -1):
            c1 = c2 - 1
            c3 = c2 + 1

            if c1 < 0 or c3 >= bar_idx:
                continue

            # C2 must be a clean swing low (lower than both neighbours)
            if not (df['low'].iloc[c2] < df['low'].iloc[c1] and
                    df['low'].iloc[c2] < df['low'].iloc[c3]):
                continue

            sfp_level = df['low'].iloc[c2]  # THE level to watch

            # Check for invalidation between C3 and the sweep candle.
            # Any CLOSE below sfp_level means the level was broken, not swept.
            invalidated = False
            for mid in range(c3 + 1, bar_idx):
                if df['close'].iloc[mid] < sfp_level:
                    invalidated = True
                    break

            if invalidated:
                continue

            # Pattern expires after MAX_SWEEP_WAIT_BARS bars from C3
            if bar_idx - c3 > MAX_SWEEP_WAIT_BARS:
                continue

            # Check the sweep candle at bar_idx
            sweep = df.iloc[bar_idx]
            min_wick_distance = sfp_level * MIN_SWEEP_BUFFER_PCT
            min_close_back    = sfp_level + (sfp_level * SFP_CLOSE_BUFFER_PCT)

            # Wick must go meaningfully BELOW C2.low (stop hunt)
            wick_goes_below = sweep['low'] < (sfp_level - min_wick_distance)
            # Candle must close firmly ABOVE C2.low (buyer rejection of sellers)
            closes_above    = sweep['close'] > min_close_back

            if wick_goes_below and closes_above:
                sweep_wick = sfp_level - sweep['low']
                return {
                    "type"       : "bullish",
                    "c1_idx"     : c1,
                    "c2_idx"     : c2,
                    "c3_idx"     : c3,
                    "sweep_idx"  : bar_idx,
                    "sfp_level"  : sfp_level,
                    "sweep_wick" : round(sweep_wick, 8),
                    "close_price": sweep['close'],
                    "confirmed"  : True,
                    "invalidated": False,
                }

    # ── Bearish SFP: hunts stops ABOVE a clean swing high (C2) ──
    else:
        # Search backwards for C2 — a clean 3-bar fractal HIGH.
        # C2 must have: C2.high > C1.high (left) AND C2.high > C3.high (right).
        for c2 in range(bar_idx - 2, search_start, -1):
            c1 = c2 - 1
            c3 = c2 + 1

            if c1 < 0 or c3 >= bar_idx:
                continue

            # C2 must be a clean swing high (higher than both neighbours)
            if not (df['high'].iloc[c2] > df['high'].iloc[c1] and
                    df['high'].iloc[c2] > df['high'].iloc[c3]):
                continue

            sfp_level = df['high'].iloc[c2]  # THE level to watch

            # Invalidation: any CLOSE above sfp_level between C3 and the sweep
            invalidated = False
            for mid in range(c3 + 1, bar_idx):
                if df['close'].iloc[mid] > sfp_level:
                    invalidated = True
                    break

            if invalidated:
                continue

            # Pattern expires after MAX_SWEEP_WAIT_BARS bars from C3
            if bar_idx - c3 > MAX_SWEEP_WAIT_BARS:
                continue

            # Check the sweep candle at bar_idx
            sweep = df.iloc[bar_idx]
            min_wick_distance = sfp_level * MIN_SWEEP_BUFFER_PCT
            min_close_back    = sfp_level - (sfp_level * SFP_CLOSE_BUFFER_PCT)

            # Wick must go meaningfully ABOVE C2.high (stop hunt)
            wick_goes_above = sweep['high'] > (sfp_level + min_wick_distance)
            # Candle must close firmly BELOW C2.high (seller rejection of buyers)
            closes_below    = sweep['close'] < min_close_back

            if wick_goes_above and closes_below:
                sweep_wick = sweep['high'] - sfp_level
                return {
                    "type"       : "bearish",
                    "c1_idx"     : c1,
                    "c2_idx"     : c2,
                    "c3_idx"     : c3,
                    "sweep_idx"  : bar_idx,
                    "sfp_level"  : sfp_level,
                    "sweep_wick" : round(sweep_wick, 8),
                    "close_price": sweep['close'],
                    "confirmed"  : True,
                    "invalidated": False,
                }

    return None  # No valid SFP found at this bar


def find_latest_sfp(
    df: pd.DataFrame,
    is_long: bool,
    lookback: int = 60,
    buffer: float = PROTECTED_SWING_BUFFER
):
    """
    TASK 1 — Find the most recent confirmed 3-candle SFP.
    CHANGED: Now uses detect_sfp_pattern() (the strict 3-candle
    system) instead of the old single-candle wick check.

    WHY: The old logic just looked for any wick past a swing level.
    The new system requires C1 → C2 → C3 → Sweep structure,
    which filters out random wicks and false breakouts.

    Returns:
        (wick_price: float, timestamp: pd.Timestamp) or None if
        no valid 3-candle SFP was found within the lookback window.
    """
    if len(df) < 6:
        return None

    # Search backwards from the most recent bar within the lookback
    search_end   = len(df) - 1
    search_start = max(5, len(df) - lookback)

    for i in range(search_end, search_start - 1, -1):
        # Try to find a completed SFP with the sweep candle at bar i
        result = detect_sfp_pattern(df, i, is_long)
        if result is not None and result["confirmed"]:
            # Apply buffer so SL sits just beyond the sweep wick
            if is_long:
                # SL goes just below the sweep candle's wick (C2 low with buffer)
                wick_price = result["sfp_level"] * (1.0 - buffer)
            else:
                # SL goes just above the sweep candle's wick (C2 high with buffer)
                wick_price = result["sfp_level"] * (1.0 + buffer)

            sfp_timestamp = df.index[result["sweep_idx"]]
            return (wick_price, sfp_timestamp)

    return None  # No SFP found in lookback window


def check_confluence(fvg_top, fvg_btm, ob_high, ob_low) -> bool:
    """Checks if FVG and OB price ranges overlap."""
    return max(fvg_btm, ob_low) <= min(fvg_top, ob_high)


# ══════════════════════════════════════════════════════════════
# BIAS DETECTION — Now returns DataFrame with reason column
# ══════════════════════════════════════════════════════════════

def get_hourly_bias(df_bias: pd.DataFrame) -> pd.DataFrame:
    """
    FVG/Sweep sets PENDING bias. Confirmed by momentum candle.
    
    IMPROVED:
    - Added Invalidation: If price breaks the signal's high/low before entry, bias is killed.
    - Added Rejection Check: SFP must show some rejection (not a massive trend candle).
    """
    fvgs = calc_fvgs(df_bias)
    sweeps = get_liquidity_sweeps(df_bias)

    bias_df = pd.DataFrame({
        "bias": 0,
        "bias_reason": ""
    }, index=df_bias.index)

    bull_fvg = fvgs['fvg_bull']
    bear_fvg = fvgs['fvg_bear']
    bull_sweep = sweeps['sweep_bull']
    bear_sweep = sweeps['sweep_bear']

    body = (df_bias['close'] - df_bias['open']).abs()
    avg_body = body.rolling(10).mean()
    
    # Simple EMA for trend filter
    ema = df_bias['close'].ewm(span=20, adjust=False).mean()

    mom_bull = (df_bias['close'] > df_bias['open']) & (body > avg_body * 0.5)
    mom_bear = (df_bias['close'] < df_bias['open']) & (body > avg_body * 0.5)

    active_bias = 0
    active_reason = ""
    signal_age = 0
    pending_signal = 0
    pending_reason = ""
    invalidation_level = None

    # Infer timeframe from index difference for better logging
    tf_str = ""
    if len(df_bias) > 1:
        diff = df_bias.index[1] - df_bias.index[0]
        if diff == pd.Timedelta(hours=4): tf_str = "[4H] "
        elif diff == pd.Timedelta(hours=1): tf_str = "[1H] "
        elif diff == pd.Timedelta(minutes=15): tf_str = "[15M] "
        elif diff == pd.Timedelta(minutes=5): tf_str = "[5M] "
        elif diff == pd.Timedelta(days=1): tf_str = "[1D] "

    for i in range(len(df_bias)):
        ts = df_bias.index[i].strftime("%Y-%m-%d %H:%M")
        cur_close = df_bias['close'].iloc[i]
        cur_high = df_bias['high'].iloc[i]
        cur_low = df_bias['low'].iloc[i]

        # ── INVALIDATION CHECK ──
        # If we have an active bias but price breaks the structural level, neutralize it.
        if active_bias == 1 and invalidation_level and cur_low < invalidation_level:
            active_bias = 0
            active_reason = ""
            invalidation_level = None
        elif active_bias == -1 and invalidation_level and cur_high > invalidation_level:
            active_bias = 0
            active_reason = ""
            invalidation_level = None

        # ── NEW SIGNAL DETECTION ──
        if bull_fvg.iloc[i]:
            pending_signal = 1
            pending_reason = f"Bull FVG {tf_str}@ {ts}"
            signal_age = 0
            active_bias = 0
            invalidation_level = df_bias['low'].iloc[i-2] if i > 1 else None # Origin of FVG
        elif bull_sweep.iloc[i]:
            # Rejection check: candle shouldn't be a massive bearish trend candle
            if df_bias['close'].iloc[i] > df_bias['low'].iloc[i] + (body.iloc[i] * 0.2):
                pending_signal = 1
                pending_reason = f"Bull Sweep {tf_str}@ {ts}"
                signal_age = 0
                active_bias = 0
                invalidation_level = cur_low
        elif bear_fvg.iloc[i]:
            pending_signal = -1
            pending_reason = f"Bear FVG {tf_str}@ {ts}"
            signal_age = 0
            active_bias = 0
            invalidation_level = df_bias['high'].iloc[i-2] if i > 1 else None
        elif bear_sweep.iloc[i]:
            # Rejection check: candle shouldn't be a massive bullish trend candle
            if df_bias['close'].iloc[i] < df_bias['high'].iloc[i] - (body.iloc[i] * 0.2):
                pending_signal = -1
                pending_reason = f"Bear Sweep {tf_str}@ {ts}"
                signal_age = 0
                active_bias = 0
                invalidation_level = cur_high

        # ── CONFIRMATION LOGIC ──
        if pending_signal == 1:
            # Trend Filter: Only allow Bullish shift if not suppressed by heavy bearish EMA
            if mom_bull.iloc[i] and cur_close > ema.iloc[i] * 0.99:
                active_bias = 1
                active_reason = f"Bullish ({pending_reason}, confirmed @ {ts})"
                pending_signal = 0
            else:
                signal_age += 1
                if signal_age > BIAS_EXPIRY_BARS:
                    pending_signal = 0
                    pending_reason = ""
        elif pending_signal == -1:
            # Trend Filter: Only allow Bearish shift if not suppressed by heavy bullish EMA
            if mom_bear.iloc[i] and cur_close < ema.iloc[i] * 1.01:
                active_bias = -1
                active_reason = f"Bearish ({pending_reason}, confirmed @ {ts})"
                pending_signal = 0
            else:
                signal_age += 1
                if signal_age > BIAS_EXPIRY_BARS:
                    pending_signal = 0
                    pending_reason = ""

        bias_df.iloc[i, 0] = active_bias
        bias_df.iloc[i, 1] = active_reason

    return bias_df


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_trades(df_ltf: pd.DataFrame, df_bias: pd.DataFrame,
                    df_htf: pd.DataFrame = None,
                    symbol: str = None,
                    use_fvg_quality: bool = None,
                    use_htf_ob: bool = None,
                    use_pd_array: bool = None,
                    use_pullback: bool = None,
                    use_auto_be: bool = None,
                    use_ltf_msb: bool = None,
                    df_daily: pd.DataFrame = None) -> Dict:
    """
    SMC FVG Reaction strategy.

    Entry Logic:
    1. Bias TF must show active bias (bull or bear) via FVG/Sweep + momentum
    2. Entry TF must show FVG or OB in the direction of bias
    3. Enter on FVG touch (reaction) or candle close
    4. SL below/above FVG boundary (structural)
    5. TP at configured R:R
    """
    sym = symbol or SYMBOL
    fvg_q = use_fvg_quality if use_fvg_quality is not None else USE_FVG_QUALITY
    htf_ob = use_htf_ob if use_htf_ob is not None else USE_HTF_OB_CONFLUENCE
    pd_arr = use_pd_array if use_pd_array is not None else USE_PREMIUM_DISCOUNT
    pb_flt = use_pullback if use_pullback is not None else USE_PULLBACK_FILTER
    auto_be = use_auto_be if use_auto_be is not None else USE_AUTO_BREAKEVEN
    ltf_msb = use_ltf_msb if use_ltf_msb is not None else False

    # Get bias with reasoning
    bias_result = get_hourly_bias(df_bias)
    bias_series = bias_result['bias']
    bias_reasons = bias_result['bias_reason']

    # Align bias to entry TF
    bias_aligned = bias_series.reindex(df_ltf.index, method='ffill')
    reason_aligned = bias_reasons.reindex(df_ltf.index, method='ffill').fillna("")

    fvgs_ltf = calc_fvgs(df_ltf)

    htf_obs = find_order_blocks(df_bias)
    ltf_obs_list = find_order_blocks(df_ltf) if USE_LTF_OB_ENTRY else []

    trades = []
    account = ACCOUNT_SIZE

    funnel = {
        "Total Bars Scanned": 0, "Has Bias": 0, "FVG/OB Exists": 0,
        "FVG First Touch": 0, "Pullback Filter": 0,
        "HTF OB Confluence": 0, "Premium/Discount": 0,
        "LTF MSB": 0, "Valid SL": 0, "Final Trades": 0
    }

    in_trade = False
    current_trade = None
    visited_fvgs = set()
    active_bull_fvgs = []
    active_bear_fvgs = []

    for i in range(25, len(df_ltf)):
        funnel["Total Bars Scanned"] += 1
        row = df_ltf.iloc[i]
        timestamp = df_ltf.index[i]

        # -- Register new FVGs --
        if fvgs_ltf['fvg_bull'].iloc[i]:
            btm = fvgs_ltf['fvg_bull_btm'].iloc[i]
            top = fvgs_ltf['fvg_bull_top'].iloc[i]
            if not pd.isna(btm) and not pd.isna(top):
                active_bull_fvgs.append((btm, top, i, timestamp))
        if fvgs_ltf['fvg_bear'].iloc[i]:
            btm = fvgs_ltf['fvg_bear_btm'].iloc[i]
            top = fvgs_ltf['fvg_bear_top'].iloc[i]
            if not pd.isna(btm) and not pd.isna(top):
                active_bear_fvgs.append((btm, top, i, timestamp))

        # -- Trade management --
        if in_trade:
            direction = current_trade["direction"]
            entry_px  = current_trade["Entry $"]
            entry_idx = current_trade["_entry_idx"]

            # ── Time invalidation (Gap #2) ──
            # Close stale trades that never reached 1R within the max bar limit
            max_bars = MAX_TRADE_DURATION.get(TF_ENTRY, 96)
            bars_open = i - entry_idx
            if bars_open > max_bars:
                # Check if 1R was ever reached
                price_slice = df_ltf.iloc[entry_idx:i + 1]
                risk_px = current_trade["risk_price"]
                if direction == 1:  # LONG
                    best = price_slice["high"].max()
                    one_r_hit = (best - entry_px) >= risk_px
                else:              # SHORT
                    best = price_slice["low"].min()
                    one_r_hit = (entry_px - best) >= risk_px
                if not one_r_hit:
                    pnl = -FIXED_SL_USDT
                    account += pnl
                    current_trade.update({
                        "exit_time": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "result": "TIME_INVALID", "pnl": pnl, "account": account,
                        "duration_bars": bars_open,
                        "r_multiple": -1.0
                    })
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                    continue

            if direction == 1:  # LONG
                if auto_be and not current_trade.get("is_be", False):
                    if row['high'] >= current_trade["be_trigger"]:
                        current_trade["Stop $"] = current_trade["Entry $"]
                        current_trade["is_be"] = True
                        current_trade["be_triggered"] = True

                if not current_trade.get("partial_exit_done", False):
                    if row['high'] >= current_trade["partial_trigger"]:
                        current_trade["partial_exit_done"] = True
                        partial_pnl = (FIXED_SL_USDT * RISK_REWARD) * PARTIAL_EXIT_PCT
                        account += partial_pnl

                if row['low'] <= current_trade["Stop $"]:
                    result = "BE" if current_trade.get("be_triggered", False) else "LOSS"
                    if current_trade.get("partial_exit_done", False):
                        pnl = 0.0
                    else:
                        pnl = 0.0 if result == "BE" else -FIXED_SL_USDT
                    account += pnl
                    current_trade.update({
                        "exit_time": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "result": result, "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade["_entry_idx"],
                        "r_multiple": pnl / abs(FIXED_SL_USDT) if pnl != 0 else 0
                    })
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                elif row['high'] >= current_trade["Target $"]:
                    if current_trade.get("partial_exit_done", False):
                        pnl = (FIXED_SL_USDT * RISK_REWARD) * PARTIAL_EXIT_PCT
                    else:
                        pnl = FIXED_SL_USDT * RISK_REWARD
                    account += pnl
                    current_trade.update({
                        "exit_time": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "result": "WIN", "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade["_entry_idx"],
                        "r_multiple": RISK_REWARD
                    })
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
            else:  # SHORT
                if auto_be and not current_trade.get("is_be", False):
                    if row['low'] <= current_trade["be_trigger"]:
                        current_trade["Stop $"] = current_trade["Entry $"]
                        current_trade["is_be"] = True
                        current_trade["be_triggered"] = True

                if not current_trade.get("partial_exit_done", False):
                    if row['low'] <= current_trade["partial_trigger"]:
                        current_trade["partial_exit_done"] = True
                        partial_pnl = (FIXED_SL_USDT * RISK_REWARD) * PARTIAL_EXIT_PCT
                        account += partial_pnl

                if row['high'] >= current_trade["Stop $"]:
                    result = "BE" if current_trade.get("be_triggered", False) else "LOSS"
                    if current_trade.get("partial_exit_done", False):
                        pnl = 0.0
                    else:
                        pnl = 0.0 if result == "BE" else -FIXED_SL_USDT
                    account += pnl
                    current_trade.update({
                        "exit_time": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "result": result, "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade["_entry_idx"],
                        "r_multiple": pnl / abs(FIXED_SL_USDT) if pnl != 0 else 0
                    })
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                elif row['low'] <= current_trade["Target $"]:
                    if current_trade.get("partial_exit_done", False):
                        pnl = (FIXED_SL_USDT * RISK_REWARD) * PARTIAL_EXIT_PCT
                    else:
                        pnl = FIXED_SL_USDT * RISK_REWARD
                    account += pnl
                    current_trade.update({
                        "exit_time": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "result": "WIN", "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade["_entry_idx"],
                        "r_multiple": RISK_REWARD
                    })
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
            continue

        # ================================================================
        # ENTRY LOGIC — FVG Reaction
        # ================================================================

        b = bias_aligned.iloc[i]
        if b == 0:
            continue
        funnel["Has Bias"] += 1

        htf_reason = reason_aligned.iloc[i]

        # ── Check for FVG touch (price entering an active FVG zone) ──
        best_fvg = None
        ltf_reason_detail = ""

        if b == 1:
            for fvg in active_bull_fvgs:
                fvg_btm, fvg_top, fvg_idx, fvg_time = fvg
                if i - fvg_idx < 2:
                    continue
                if i - fvg_idx > 48:
                    continue
                if row['low'] <= fvg_top and row['close'] > fvg_btm:
                    if best_fvg is None or fvg_idx > best_fvg[2]:
                        best_fvg = fvg
                        ltf_reason_detail = f"FVG Reaction (zone {fvg_btm:.2f}-{fvg_top:.2f}, formed @ {fvg_time.strftime('%m-%d %H:%M')})"
        elif b == -1:
            for fvg in active_bear_fvgs:
                fvg_btm, fvg_top, fvg_idx, fvg_time = fvg
                if i - fvg_idx < 2:
                    continue
                if i - fvg_idx > 48:
                    continue
                if row['high'] >= fvg_btm and row['close'] < fvg_top:
                    if best_fvg is None or fvg_idx > best_fvg[2]:
                        best_fvg = fvg
                        ltf_reason_detail = f"FVG Reaction (zone {fvg_btm:.2f}-{fvg_top:.2f}, formed @ {fvg_time.strftime('%m-%d %H:%M')})"

        # Also check current-bar FVGs (immediate entry on formation)
        fvg_bull = fvgs_ltf['fvg_bull'].iloc[i]
        fvg_bear = fvgs_ltf['fvg_bear'].iloc[i]
        has_ob = False

        if USE_LTF_OB_ENTRY:
            ob_bull = any(o['time'] == timestamp and o['type'] == 'bull' for o in ltf_obs_list)
            ob_bear = any(o['time'] == timestamp and o['type'] == 'bear' for o in ltf_obs_list)
            if b == 1 and ob_bull:
                has_ob = True
                if not ltf_reason_detail:
                    ltf_reason_detail = f"OB Entry (formed @ {timestamp.strftime('%m-%d %H:%M')})"
            if b == -1 and ob_bear:
                has_ob = True
                if not ltf_reason_detail:
                    ltf_reason_detail = f"OB Entry (formed @ {timestamp.strftime('%m-%d %H:%M')})"

        if best_fvg is None and not (b == 1 and fvg_bull) and not (b == -1 and fvg_bear) and not has_ob:
            continue

        funnel["FVG/OB Exists"] += 1

        # ── Determine entry and stop ──
        is_bull = (b == 1)

        if best_fvg is not None:
            fvg_btm, fvg_top, fvg_idx, fvg_time = best_fvg
            entry_key = f"{fvg_time}_{fvg_idx}"
        elif fvg_bull and is_bull:
            fvg_btm = fvgs_ltf['fvg_bull_btm'].iloc[i]
            fvg_top = fvgs_ltf['fvg_bull_top'].iloc[i]
            entry_key = f"{timestamp}_fvg"
            if not ltf_reason_detail:
                ltf_reason_detail = f"FVG Formation (zone {fvg_btm:.2f}-{fvg_top:.2f} @ {timestamp.strftime('%m-%d %H:%M')})"
        elif fvg_bear and not is_bull:
            fvg_btm = fvgs_ltf['fvg_bear_btm'].iloc[i]
            fvg_top = fvgs_ltf['fvg_bear_top'].iloc[i]
            entry_key = f"{timestamp}_fvg"
            if not ltf_reason_detail:
                ltf_reason_detail = f"FVG Formation (zone {fvg_btm:.2f}-{fvg_top:.2f} @ {timestamp.strftime('%m-%d %H:%M')})"
        elif has_ob:
            if is_bull:
                match = [o for o in ltf_obs_list if o['time'] == timestamp and o['type'] == 'bull']
            else:
                match = [o for o in ltf_obs_list if o['time'] == timestamp and o['type'] == 'bear']
            if not match:
                continue
            fvg_btm = match[0]['low']
            fvg_top = match[0]['high']
            entry_key = f"{timestamp}_ob"
        else:
            continue

        # ── FVG first-touch filter ──
        if fvg_q:
            if entry_key in visited_fvgs:
                continue
            visited_fvgs.add(entry_key)
        funnel["FVG First Touch"] += 1

        # ── Pullback filter ──
        if pb_flt:
            pb_row = df_ltf.iloc[i - 1]
            if is_bull and not (pb_row['close'] < pb_row['open']):
                continue
            if not is_bull and not (pb_row['close'] > pb_row['open']):
                continue
        funnel["Pullback Filter"] += 1

        # ── HTF OB Confluence ──
        setup_tier = "B"
        htf_ob_reason = ""
        if htf_ob:
            confluence = False
            for ob in htf_obs:
                if not ob['active']:
                    continue
                dist_pct = min(abs(row['close'] - ob['high']), abs(row['close'] - ob['low'])) / row['close']
                if dist_pct < 0.01:
                    confluence = True
                    setup_tier = "A"
                    htf_ob_reason = f" + HTF OB near {ob['high']:.2f}"
                    break
            if not confluence:
                continue
        funnel["HTF OB Confluence"] += 1

        # ── Clamp bias to current bar (used by P/D filter AND SFP SL) ──
        # Computed once here to avoid duplicate slicing below.
        bias_slice = df_bias[df_bias.index <= timestamp]

        # ── Premium/Discount filter (1H / 4H Bias Range Focus) ──
        # Relaxed Daily HTF P/D restriction so trend continuation setups aren't blocked on Alts.
        # Enforces Premium/Discount pricing strictly on the 1H/4H Bias range.
        pd_reason = ""
        if pd_arr:
            if len(bias_slice) >= 10:
                bias_high, bias_low = find_recent_swings(bias_slice, len(bias_slice) - 1,
                                                         lookback=120)
                bias_mid = (bias_high + bias_low) / 2

                if is_bull and row['close'] > bias_mid:
                    continue
                elif not is_bull and row['close'] < bias_mid:
                    continue

            zone = "Discount" if is_bull else "Premium"
            pd_reason = f" | Zone: {zone}"
        funnel["Premium/Discount"] += 1

        # ── LTF (5m) Market Structure Break Confirmation (ASYMMETRIC) ──
        # Data-driven design:
        #   SHORT entries: require a 5m swing-low displacement close — price must
        #     close BELOW the most recent 5m fractal swing low within 30 bars.
        #     This confirms bearish momentum has flipped in the Premium zone. ✅
        #   LONG entries: MSB check is SKIPPED. In bullish pullback-to-Discount
        #     setups, the 5m swing high break fires only AFTER price bounces,
        #     forcing a later entry at a worse price with the same SL → kills R:R.
        #     The 1H Bias + 1H Discount zone together are sufficient for longs. ✅
        if ltf_msb and not is_bull:
            msb_lookback    = 30
            msb_confirmed   = False
            ltf_slice_msb   = df_ltf.iloc[:i + 1]
            n               = len(ltf_slice_msb)

            # Find most recent 5m fractal swing LOW; require close below it.
            for k in range(n - 3, max(2, n - msb_lookback), -1):
                if k < 2 or k >= n - 2:
                    continue
                l = ltf_slice_msb['low'].iloc[k]
                if (l < ltf_slice_msb['low'].iloc[k - 1] and
                        l < ltf_slice_msb['low'].iloc[k - 2] and
                        l < ltf_slice_msb['low'].iloc[k + 1] and
                        l < ltf_slice_msb['low'].iloc[k + 2]):
                    if row['close'] < l:
                        msb_confirmed = True
                    break  # Only check the most recent fractal

            if not msb_confirmed:
                continue
        funnel["LTF MSB"] += 1

        # ───────────────────────────────────────────────────────
        # NESTED SFP STOP-LOSS (mirrors live trader logic)
        # Priority: LTF SFP > HTF SFP > FVG boundary fallback
        # Slices are clamped to current bar to prevent look-ahead.
        # ───────────────────────────────────────────────────────
        entry    = row['close']          # LTF close at the moment of FVG touch
        fvg_size = abs(fvg_top - fvg_btm)

        # bias_slice already computed above; clamp LTF too
        ltf_slice  = df_ltf.iloc[:i + 1]

        htf_sfp = find_latest_sfp(bias_slice, is_long=is_bull, lookback=96)
        ltf_sfp = find_latest_sfp(ltf_slice,  is_long=is_bull, lookback=100)

        stop    = None
        sl_type = "fvg_boundary"

        if htf_sfp:
            htf_sl_price, htf_sl_time = htf_sfp
            stop    = htf_sl_price
            sl_type = "htf_sfp"

            if ltf_sfp and not PREFER_HTF_STRUCTURAL_SL:
                ltf_sl_price, ltf_sl_time = ltf_sfp
                if ltf_sl_time > htf_sl_time:   # LTF sweep is more recent → tighter SL
                    stop    = ltf_sl_price
                    sl_type = "ltf_sfp"
        elif ltf_sfp:
            # Fallback to LTF SFP if no HTF SFP exists at all
            ltf_sl_price, ltf_sl_time = ltf_sfp
            stop    = ltf_sl_price
            sl_type = "ltf_sfp"

        if stop is None:                        # No SFP found → try Protected Swing High/Low, then use FVG boundary
            if PREFER_HTF_STRUCTURAL_SL:
                # Look back 20 bars on the BIAS timeframe (much safer structure than entry TF)
                swing_sl = find_protected_swing(bias_slice, len(bias_slice)-1, is_long=is_bull, lookback=20)
            else:
                # Fallback to entry TF lookback
                swing_sl = find_protected_swing(ltf_slice, i, is_long=is_bull, lookback=20)

            if swing_sl is not None and not pd.isna(swing_sl):
                stop    = swing_sl
                sl_type = "protected_swing"
            else:
                stop    = fvg_btm * 0.999 if is_bull else fvg_top * 1.001
                sl_type = "fvg_boundary"

        if is_bull:
            risk   = entry - stop
            target = entry + (risk * RISK_REWARD)
        else:
            risk   = stop - entry
            target = entry - (risk * RISK_REWARD)

        # ── Validate risk first: stop must be on the correct side of entry ──
        # This must come BEFORE the min SL floor, otherwise the floor would
        # rescue trades with stops on the wrong side (risk <= 0) by forcing
        # a valid-looking distance, producing a massive false trade count.
        if risk <= 0:
            continue

        # ── Minimum SL distance floor (mirrors live bot, Gap #1) ──
        # Only widens stops that are valid-direction but too close to entry.
        is_btc = "BTC" in (sym or "")
        min_sl_dist = MIN_SL_PCT_BTC if is_btc else MIN_SL_PCT_ALTS
        if is_bull:
            if risk < (entry * min_sl_dist):
                stop = entry - (entry * min_sl_dist)
        else:
            if risk < (entry * min_sl_dist):
                stop = entry + (entry * min_sl_dist)
        # Recalculate risk and target with floored stop
        if is_bull:
            risk   = entry - stop
            target = entry + (risk * RISK_REWARD)
        else:
            risk   = stop - entry
            target = entry - (risk * RISK_REWARD)

        # ── Discard if stop is too wide (low RR) ──
        if risk / entry > MAX_STOP_DISTANCE:
            continue
        funnel["Valid SL"] += 1

        # ── Build the full reason string ──
        regime = classify_regime(df_daily, timestamp) if df_daily is not None else "ranging"
        train_test = "TEST" if timestamp.strftime("%Y-%m-%d") >= TEST_START else "TRAIN"

        # ── Trend-Following Filter (Counter-Trend Block) ──
        if SKIP_COUNTER_TREND:
            if regime == "strong_bull" and not is_bull:
                continue
            elif regime == "strong_bear" and is_bull:
                continue

        # ── Ranging Regime Block ──
        if SKIP_RANGING_REGIME and regime == "ranging":
            continue

        direction_str = "LONG" if is_bull else "SHORT"
        reason = (
            f"HTF: {htf_reason}{htf_ob_reason} | "
            f"LTF: {TF_ENTRY} {ltf_reason_detail} | "
            f"{direction_str} @ {entry:.2f} | "
            f"SL: {sl_type} | "
            f"Regime: {regime}{pd_reason}"
        )

        be_trigger = entry + (risk * BE_TRIGGER_R) if is_bull else entry - (risk * BE_TRIGGER_R)
        partial_trigger = entry + (risk * PARTIAL_EXIT_R) if is_bull else entry - (risk * PARTIAL_EXIT_R)

        current_trade = {
            "symbol": sym, "model": "cisd", "regime": regime, "setup_tier": setup_tier,
            "bias_type": "bullish" if is_bull else "bearish",
            "reason": reason, "sl_type": sl_type,
            "fvg_size": round(fvg_size, 4),
            "fvg_size_pct": round((fvg_size / entry) * 100, 4),
            "Entry Time": timestamp.strftime("%Y-%m-%d %H:%M"),
            "_entry_idx": i, "direction": 1 if is_bull else -1,
            "Entry $": round(entry, 8),
            "Stop $": round(stop, 8), "Target $": round(target, 8),
            "risk_price": round(risk, 8),
            "be_trigger": round(be_trigger, 8),
            "partial_trigger": round(partial_trigger, 8),
            "is_be": False, "be_triggered": False, "r_multiple": 0,
            "partial_exit_done": False,
            "Entry Type": f"FVG Reaction {'Bull' if is_bull else 'Bear'}",
            "train_test": train_test, "account": account, "Symbol": sym
        }
        funnel["Final Trades"] += 1
        in_trade = True

    # Cleanup old FVGs
    active_bull_fvgs = [f for f in active_bull_fvgs if len(df_ltf) - f[2] <= 48]
    active_bear_fvgs = [f for f in active_bear_fvgs if len(df_ltf) - f[2] <= 48]

    return {"trades": trades, "funnel": funnel}


# ══════════════════════════════════════════════════════════════
# CLI RUNNER
# ══════════════════════════════════════════════════════════════

def run_backtest():
    global SYMBOL
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--chain", default=None, help="Override ACTIVE_CHAIN (daily/weekly/intraday)")
    args = parser.parse_args()

    SYMBOL = args.symbol

    # Allow CLI chain override
    if args.chain:
        from core.config import ALIGNMENTS
        if args.chain not in ALIGNMENTS:
            print(f"  [ERROR] Unknown chain '{args.chain}'. Available: {list(ALIGNMENTS.keys())}")
            return
        chain = ALIGNMENTS[args.chain]
        tf_context = chain["context"]
        tf_bias = chain["bias"]
        tf_entry = chain["entry"]
        chain_name = args.chain
    else:
        tf_context = TF_CONTEXT
        tf_bias = TF_BIAS
        tf_entry = TF_ENTRY
        chain_name = ACTIVE_CHAIN

    print(f"\n  Chain: {chain_name.upper()} ({tf_context} -> {tf_bias} -> {tf_entry})")

    df_htf = fetch_ohlcv_full(SYMBOL, tf_context, DAYS_BACK)
    df_bias = fetch_ohlcv_full(SYMBOL, tf_bias, DAYS_BACK)
    df_ltf = fetch_ohlcv_full(SYMBOL, tf_entry, DAYS_BACK)

    try:
        df_daily = fetch_ohlcv_full(SYMBOL, "1d", DAYS_BACK)
    except FileNotFoundError:
        df_daily = None

    results = simulate_trades(df_ltf, df_bias, df_htf, symbol=SYMBOL, use_pd_array=True, use_ltf_msb=True, df_daily=df_daily)

    # Print first 5 trade reasons as a sample
    if results["trades"]:
        print(f"\n  SAMPLE TRADE REASONS (first 5):")
        for t in results["trades"][:5]:
            print(f"    [{t['Entry Time']}] {t.get('reason', 'N/A')}")

    print_enhanced_stats(results["trades"], results["funnel"],
                         f"CISD MODEL -- {SYMBOL} [{chain_name.upper()}]")


if __name__ == "__main__":
    run_backtest()
