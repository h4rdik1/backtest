import pandas as pd
import os
import argparse
import ccxt
from typing import List, Dict

from core.lux_fvg import detect_luxalgo_fvgs, detect_liquidity_sweeps

# CHANGED: Import all constants from shared config instead of hardcoding
from core.config import (
    RISK_REWARD, FIXED_SL_USDT, ACCOUNT_SIZE,
    DAYS_BACK, BIAS_EXPIRY_BARS, FVG_WICK_RATIO, FVG_AVG_BODY_LOOKBACK,
    FVG_MIN_SIZE_MULT, FVG_MIN_VOLUME_MULT, OB_MOVE_MULT, OB_SWING_LOOKBACK, CISD_BODY_MULT,
    AUTO_BREAKEVEN_R, PD_ARRAY_BARS,
    USE_FVG_QUALITY, USE_HTF_OB_CONFLUENCE, USE_PREMIUM_DISCOUNT,
    USE_PULLBACK_FILTER, USE_AUTO_BREAKEVEN, USE_LTF_OB_ENTRY,
    TRAIN_END, TEST_START, DATA_DIR
)

# CHANGED: Import analysis functions instead of duplicating stats logic
from core.analysis import (
    print_enhanced_stats, classify_regime, walk_forward_split,
    print_walk_forward
)

# Module-level symbol (set by CLI or multi-asset runner)
SYMBOL = "BTC/USDT"


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


# ── Exchange caching for live scanner ─────────────────────────
EXCHANGES = {}

def get_exchange(exchange_id: str = 'binance'):
    """Get or create a cached CCXT exchange instance with failover for Binance."""
    global EXCHANGES
    exchange_id = exchange_id.lower().replace("usdt", "")
    if exchange_id not in EXCHANGES:
        try:
            exchange_class = getattr(ccxt, exchange_id)
            if exchange_id == "binance":
                # Try standard first, but config for failover
                EXCHANGES[exchange_id] = exchange_class({
                    'enableRateLimit': True,
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
                EXCHANGES[exchange_id] = exchange_class({'enableRateLimit': True})
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
        is_impulse = abs(body_val) > (avg_range.iloc[i] * OB_MOVE_MULT)

        if is_impulse:
            j = i - 1
            if j < 0:
                continue
            if body_val > 0: # Bullish Impulse
                if df['close'].iloc[j] < df['open'].iloc[j]:
                    ob_high = df['high'].iloc[j]
                    ob_low = df['low'].iloc[j]
                    ob_open = df['open'].iloc[j]
                    ob_close = df['close'].iloc[j]

                    if df['close'].iloc[i] > ob_high:
                        lows_around = df['low'].iloc[max(0, j-OB_SWING_LOOKBACK):min(len(df), j+OB_SWING_LOOKBACK+1)]
                        if df['low'].iloc[j] == lows_around.min():
                            obs.append({
                                "time": df.index[j],
                                "high": ob_high, "low": ob_low,
                                "median": (ob_open + ob_close) / 2,
                                "active": True,
                                "type": "bull"
                            })
            else: # Bearish Impulse
                if df['close'].iloc[j] > df['open'].iloc[j]:
                    ob_high = df['high'].iloc[j]
                    ob_low = df['low'].iloc[j]
                    ob_open = df['open'].iloc[j]
                    ob_close = df['close'].iloc[j]

                    if df['close'].iloc[i] < ob_low:
                        highs_around = df['high'].iloc[max(0, j-OB_SWING_LOOKBACK):min(len(df), j+OB_SWING_LOOKBACK+1)]
                        if df['high'].iloc[j] == highs_around.max():
                            obs.append({
                                "time": df.index[j],
                                "high": ob_high, "low": ob_low,
                                "median": (ob_open + ob_close) / 2,
                                "active": True,
                                "type": "bear"
                            })
    return obs


def check_confluence(fvg_top, fvg_btm, ob_high, ob_low) -> bool:
    """Checks if FVG and OB price ranges overlap."""
    return max(fvg_btm, ob_low) <= min(fvg_top, ob_high)


def get_hourly_bias(df_1h: pd.DataFrame) -> pd.Series:
    """Logic: FVG/Sweep sets PENDING bias. Confirmed by momentum candle."""
    fvgs = calc_fvgs(df_1h)
    sweeps = get_liquidity_sweeps(df_1h)
    bias = pd.Series(0, index=df_1h.index)

    bull_signal = fvgs['fvg_bull'] | sweeps['sweep_bull']
    bear_signal = fvgs['fvg_bear'] | sweeps['sweep_bear']

    body = (df_1h['close'] - df_1h['open']).abs()
    avg_body = body.rolling(10).mean()
    mom_bull = (df_1h['close'] > df_1h['open']) & (body > avg_body)
    mom_bear = (df_1h['close'] < df_1h['open']) & (body > avg_body)

    active_bias = 0
    signal_age = 0
    pending_signal = 0

    for i in range(len(df_1h)):
        if bull_signal.iloc[i]:
            pending_signal = 1
            signal_age = 0
            active_bias = 0
        elif bear_signal.iloc[i]:
            pending_signal = -1
            signal_age = 0
            active_bias = 0

        if pending_signal == 1:
            if mom_bull.iloc[i]:
                active_bias = 1
                pending_signal = 0
            else:
                signal_age += 1
                if signal_age > BIAS_EXPIRY_BARS:
                    pending_signal = 0
        elif pending_signal == -1:
            if mom_bear.iloc[i]:
                active_bias = -1
                pending_signal = 0
            else:
                signal_age += 1
                if signal_age > BIAS_EXPIRY_BARS:
                    pending_signal = 0

        bias.iloc[i] = active_bias
    return bias


def simulate_trades(df_ltf: pd.DataFrame, df_bias: pd.DataFrame,
                    df_htf: pd.DataFrame = None,
                    symbol: str = None,
                    use_fvg_quality: bool = None,
                    use_htf_ob: bool = None,
                    use_pd_array: bool = None,
                    use_pullback: bool = None,
                    use_auto_be: bool = None,
                    df_daily: pd.DataFrame = None) -> Dict:
    sym = symbol or SYMBOL
    fvg_q = use_fvg_quality if use_fvg_quality is not None else USE_FVG_QUALITY
    htf_ob = use_htf_ob if use_htf_ob is not None else USE_HTF_OB_CONFLUENCE
    pd_arr = use_pd_array if use_pd_array is not None else USE_PREMIUM_DISCOUNT
    pb_flt = use_pullback if use_pullback is not None else USE_PULLBACK_FILTER
    auto_be = use_auto_be if use_auto_be is not None else USE_AUTO_BREAKEVEN

    bias_series = get_hourly_bias(df_bias)
    bias_aligned = bias_series.reindex(df_ltf.index, method='ffill')
    fvgs_ltf = calc_fvgs(df_ltf)
    avg_range_ltf = (df_ltf['high'] - df_ltf['low']).rolling(20).mean()

    htf_obs = find_order_blocks(df_bias)
    ltf_obs_list = find_order_blocks(df_ltf) if USE_LTF_OB_ENTRY else []

    trades = []
    account = ACCOUNT_SIZE

    funnel = {
        "Total Signals": 0, "Bias Filter": 0, "FVG/OB Exists": 0,
        "FVG First Touch": 0, "CISD Pullback": 0, "CISD Impulsive": 0,
        "HTF OB Confluence": 0, "Premium/Discount": 0,
        "CISD Triggered": 0, "Max SL Logic": 0, "Final Trades": 0
    }

    in_trade = False
    current_trade = None
    visited_fvgs = set()
    # Removed global regime classification to fix TypeError
    
    for i in range(25, len(df_ltf)):
        funnel["Total Signals"] += 1
        row = df_ltf.iloc[i]
        timestamp = df_ltf.index[i]

        if in_trade:
            if current_trade["direction"] == 1:
                if auto_be and not current_trade.get("is_be", False):
                    if row['high'] >= current_trade["be_trigger"]:
                        current_trade["Stop $"] = current_trade["Entry $"]
                        current_trade["is_be"] = True
                        current_trade["be_triggered"] = True
                if row['low'] <= current_trade["Stop $"]:
                    result = "BE" if current_trade.get("is_be", False) else "LOSS"
                    pnl = 0.0 if result == "BE" else -FIXED_SL_USDT
                    account += pnl
                    current_trade.update({
                        "exit_time": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "result": result, "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade["_entry_idx"],
                        "r_multiple": pnl / abs(FIXED_SL_USDT) if pnl != 0 else 0
                    })
                    trades.append(current_trade); in_trade = False; current_trade = None
                elif row['high'] >= current_trade["Target $"]:
                    pnl = FIXED_SL_USDT * RISK_REWARD
                    account += pnl
                    current_trade.update({
                        "exit_time": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "result": "WIN", "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade["_entry_idx"],
                        "r_multiple": RISK_REWARD
                    })
                    trades.append(current_trade); in_trade = False; current_trade = None
            else: # BEARISH trade
                if auto_be and not current_trade.get("is_be", False):
                    if row['low'] <= current_trade["be_trigger"]:
                        current_trade["Stop $"] = current_trade["Entry $"]
                        current_trade["is_be"] = True
                        current_trade["be_triggered"] = True
                if row['high'] >= current_trade["Stop $"]:
                    result = "BE" if current_trade.get("is_be", False) else "LOSS"
                    pnl = 0.0 if result == "BE" else -FIXED_SL_USDT
                    account += pnl
                    current_trade.update({
                        "exit_time": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "result": result, "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade["_entry_idx"],
                        "r_multiple": pnl / abs(FIXED_SL_USDT) if pnl != 0 else 0
                    })
                    trades.append(current_trade); in_trade = False; current_trade = None
                elif row['low'] <= current_trade["Target $"]:
                    pnl = FIXED_SL_USDT * RISK_REWARD
                    account += pnl
                    current_trade.update({
                        "exit_time": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "result": "WIN", "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade["_entry_idx"],
                        "r_multiple": RISK_REWARD
                    })
                    trades.append(current_trade); in_trade = False; current_trade = None
            continue

        b = bias_aligned.iloc[i]
        if b == 0: continue
        funnel["Bias Filter"] += 1

        fvg_bull = fvgs_ltf['fvg_bull'].iloc[i]
        fvg_bear = fvgs_ltf['fvg_bear'].iloc[i]
        ob_bull = any(o['time'] == timestamp and o['type'] == 'bull' for o in ltf_obs_list) if USE_LTF_OB_ENTRY else False
        ob_bear = any(o['time'] == timestamp and o['type'] == 'bear' for o in ltf_obs_list) if USE_LTF_OB_ENTRY else False

        is_bull_setup = (b == 1 and (fvg_bull or ob_bull))
        is_bear_setup = (b == -1 and (fvg_bear or ob_bear))
        if not (is_bull_setup or is_bear_setup): continue
        funnel["FVG/OB Exists"] += 1

        setup_tier = "B"
        curr_fvg = (fvgs_ltf['fvg_bull_btm'].iloc[i], fvgs_ltf['fvg_bull_top'].iloc[i]) if fvg_bull else ((fvgs_ltf['fvg_bear_btm'].iloc[i], fvgs_ltf['fvg_bear_top'].iloc[i]) if fvg_bear else None)
        curr_ob = None
        if ob_bull or ob_bear:
            match = [o for o in ltf_obs_list if o['time'] == timestamp and o['type'] == ('bull' if ob_bull else 'bear')]
            if match: curr_ob = (match[0]['low'], match[0]['high'])

        if curr_fvg and curr_ob:
            if check_confluence(curr_fvg[1], curr_fvg[0], curr_ob[1], curr_ob[0]):
                setup_tier = "A++"
        
        if setup_tier == "B" and htf_ob:
            for o in htf_obs:
                if not o['active']: continue
                dist_pct = min(abs(row['close'] - o['high']), abs(row['close'] - o['low'])) / row['close']
                if dist_pct < 0.0025:
                    setup_tier = "A"; break

        if fvg_q and (fvg_bull or fvg_bear):
            if timestamp in visited_fvgs: continue
            visited_fvgs.add(timestamp)
        funnel["FVG First Touch"] += 1

        if pb_flt:
            pb_row = df_ltf.iloc[i-2]
            if is_bull_setup and not (pb_row['close'] < pb_row['open']): continue
            if is_bear_setup and not (pb_row['close'] > pb_row['open']): continue
        funnel["CISD Pullback"] += 1

        entry = row['close']
        if is_bull_setup:
            if entry <= df_ltf['high'].iloc[i-3:i].max(): continue
            stop = df_ltf['low'].iloc[i-3:i+1].min()
            risk = entry - stop
        else:
            if entry >= df_ltf['low'].iloc[i-3:i].min(): continue
            stop = df_ltf['high'].iloc[i-3:i+1].max()
            risk = stop - entry
        funnel["CISD Triggered"] += 1

        if fvg_q and (abs(row['close'] - row['open']) < (avg_range_ltf.iloc[i] * CISD_BODY_MULT)): continue
        funnel["CISD Impulsive"] += 1

        if htf_ob and setup_tier == "B":
            confluence = False
            for ob in htf_obs:
                if not ob['active']: continue
                if min(abs(entry - ob['high']), abs(entry - ob['low'])) / entry < 0.005:
                    confluence = True; break
            if not confluence: continue
        funnel["HTF OB Confluence"] += 1

        if pd_arr:
            lookback = 100
            hi = df_ltf['high'].iloc[max(0, i-lookback):i].max()
            lo = df_ltf['low'].iloc[max(0, i-lookback):i].min()
            mid = (hi + lo) / 2
            if is_bull_setup and entry > mid: continue
            if is_bear_setup and entry < mid: continue
        funnel["Premium/Discount"] += 1

        max_risk = entry * 0.08 if setup_tier == "A++" else entry * 0.05
        if risk <= 0 or risk > max_risk: continue
        funnel["Max SL Logic"] += 1

        target = entry + (risk * RISK_REWARD) if is_bull_setup else entry - (risk * RISK_REWARD)
        be_trigger = entry + (risk * AUTO_BREAKEVEN_R) if is_bull_setup else entry - (risk * AUTO_BREAKEVEN_R)
        fvg_size = (curr_fvg[1] - curr_fvg[0]) if curr_fvg else 0
        fvg_size_pct = (fvg_size / entry) * 100
        regime = classify_regime(df_daily, timestamp) if df_daily is not None else "ranging"
        train_test = "TEST" if timestamp.strftime("%Y-%m-%d") >= TEST_START else "TRAIN"

        current_trade = {
            "symbol": sym, "model": "cisd", "regime": regime, "setup_tier": setup_tier,
            "bias_type": "bullish" if is_bull_setup else "bearish", "fvg_size": round(fvg_size, 4),
            "fvg_size_pct": round(fvg_size_pct, 4), "Entry Time": timestamp.strftime("%Y-%m-%d %H:%M"),
            "_entry_idx": i, "direction": 1 if is_bull_setup else -1, "Entry $": round(entry, 8),
            "Stop $": round(stop, 8), "Target $": round(target, 8), "risk_price": round(risk, 8),
            "be_trigger": round(be_trigger, 8), "is_be": False, "be_triggered": False, "r_multiple": 0,
            "Entry Type": f"Phase 4 {'Bull' if is_bull_setup else 'Bear'} CISD",
            "train_test": train_test, "account": account, "Symbol": sym
        }
        funnel["Final Trades"] += 1
        in_trade = True

    return {"trades": trades, "funnel": funnel}


def run_backtest():
    global SYMBOL
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--walk-forward", action="store_true")
    args = parser.parse_args()
    
    SYMBOL = args.symbol

    tf_context, tf_bias, tf_ltf = "1d", "1h", "5m"
    df_htf = fetch_ohlcv_full(SYMBOL, tf_context, DAYS_BACK)
    df_bias = fetch_ohlcv_full(SYMBOL, tf_bias, DAYS_BACK)
    df_ltf = fetch_ohlcv_full(SYMBOL, tf_ltf, DAYS_BACK)

    try:
        df_daily = fetch_ohlcv_full(SYMBOL, "1d", DAYS_BACK)
    except FileNotFoundError:
        df_daily = None

    results = simulate_trades(df_ltf, df_bias, df_htf, symbol=SYMBOL, df_daily=df_daily)
    print_enhanced_stats(results["trades"], results["funnel"], f"CISD MODEL — {SYMBOL}")

if __name__ == "__main__":
    run_backtest()
