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
                # Check for alternative endpoints if needed
                # (api.binance.com is often blocked; api1, api2, api3 are alternatives)
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
        body = df['close'].iloc[i] - df['open'].iloc[i]
        is_impulse = body > (avg_range.iloc[i] * OB_MOVE_MULT)

        if is_impulse:
            j = i - 1
            if j < 0:
                continue
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
            elif body < -(avg_range.iloc[i] * OB_MOVE_MULT): # Bearish Impulse
                j = i - 1
                if j >= 0 and df['close'].iloc[j] > df['open'].iloc[j]:
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
    """
    Calculates trading bias on the HTF (1H/4H).

    Logic: A signal (FVG or Sweep) sets a PENDING bias. It only ACTIVATES
    when a momentum candle (large body close) confirms it within 10 bars.
    """
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
    """
    Core CISD (Change in State of Delivery) simulation engine.

    CHANGED vs old version:
    - Filter toggles are now parameters (for filter contribution analysis)
    - Enhanced trade dict with regime, fvg_size, duration, r_multiple, etc.
    - Removed unreachable duplicate elif block (old line 370-391)
    - Uses shared config constants instead of hardcoded values

    Parameters:
        df_ltf: Lower timeframe candles (5m/15m)
        df_bias: Bias timeframe candles (1h/4h)
        df_htf: Context timeframe candles (1d/1w) — optional
        symbol: Override symbol name for trade log
        use_fvg_quality/use_htf_ob/use_pd_array/use_pullback/use_auto_be:
            Filter toggles. If None, uses config.py defaults.
        df_daily: Daily candles for regime classification (optional)
    """
    # Use config defaults if not explicitly provided
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

    # Priority 3: HTF OBs for confluence
    htf_obs = find_order_blocks(df_bias)
    
    # NEW: LTF OBs for alternative entry
    ltf_obs_list = find_order_blocks(df_ltf) if USE_LTF_OB_ENTRY else []

    trades = []
    account = ACCOUNT_SIZE

    # Funnel Tracking — shows how many signals pass each filter stage
    funnel = {
        "Total Signals": 0, "Bias Filter": 0, "FVG/OB Exists": 0,
        "FVG First Touch": 0, "CISD Pullback": 0, "CISD Impulsive": 0,
        "HTF OB Confluence": 0, "Premium/Discount": 0,
        "CISD Triggered": 0, "Max SL Logic": 0, "Final Trades": 0
    }

    in_trade = False
    current_trade = None
    visited_fvgs = set()  # Track used FVG timestamps (first-touch only)

    for i in range(25, len(df_ltf)):
        funnel["Total Signals"] += 1
        row = df_ltf.iloc[i]
        timestamp = df_ltf.index[i]

        # ── TRADE MANAGEMENT (if already in a trade) ──
        if in_trade:
            if current_trade["direction"] == 1:  # BULLISH trade
                # Auto Break-Even check
                if auto_be and not current_trade.get("is_be", False):
                    if row['high'] >= current_trade["be_trigger"]:
                        current_trade["Stop $"] = current_trade["Entry $"]
                        current_trade["is_be"] = True
                        current_trade["be_triggered"] = True

                if row['low'] <= current_trade["Stop $"]:
                    # Hit stop loss (or break-even)
                    result = "BE" if current_trade.get("is_be", False) else "LOSS"
                    pnl = 0.0 if result == "BE" else -FIXED_SL_USDT
                    account += pnl
                    current_trade.update({
                        "result": result, "exit_time": timestamp,
                        "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade["_entry_idx"],
                        "r_multiple": 0.0 if result == "BE" else -1.0
                    })
                    trades.append(current_trade)
                    in_trade = False
                elif row['high'] >= current_trade["Target $"]:
                    # Hit target
                    pnl = FIXED_SL_USDT * RISK_REWARD
                    account += pnl
                    current_trade.update({
                        "result": "WIN", "exit_time": timestamp,
                        "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade["_entry_idx"],
                        "r_multiple": RISK_REWARD
                    })
                    trades.append(current_trade)
                    in_trade = False
            else:  # BEARISH trade
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
                        "result": result, "exit_time": timestamp,
                        "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade["_entry_idx"],
                        "r_multiple": 0.0 if result == "BE" else -1.0
                    })
                    trades.append(current_trade)
                    in_trade = False
                elif row['low'] <= current_trade["Target $"]:
                    pnl = FIXED_SL_USDT * RISK_REWARD
                    account += pnl
                    current_trade.update({
                        "result": "WIN", "exit_time": timestamp,
                        "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade["_entry_idx"],
                        "r_multiple": RISK_REWARD
                    })
                    trades.append(current_trade)
                    in_trade = False
            continue

        # ── ENTRY LOGIC ──
        b = bias_aligned.iloc[i]
        if b == 0:
            continue
        funnel["Bias Filter"] += 1

        # Check for FVG setups
        fvg_bull = fvgs_ltf['fvg_bull'].iloc[i]
        fvg_bear = fvgs_ltf['fvg_bear'].iloc[i]
        
        # Check for LTF OB setups
        ob_bull = any(o['time'] == timestamp and o['type'] == 'bull' for o in ltf_obs_list) if USE_LTF_OB_ENTRY else False
        ob_bear = any(o['time'] == timestamp and o['type'] == 'bear' for o in ltf_obs_list) if USE_LTF_OB_ENTRY else False

        is_bull_setup = (b == 1 and (fvg_bull or ob_bull))
        is_bear_setup = (b == -1 and (fvg_bear or ob_bear))

        if not (is_bull_setup or is_bear_setup):
            continue
        funnel["FVG/OB Exists"] += 1

        # Tiering Logic (Priority: A++ > A > B)
        setup_tier = "B"
        
        # Get actual levels for confluence check
        curr_fvg = None
        if fvg_bull:
            curr_fvg = (fvgs_ltf['fvg_bull_btm'].iloc[i], fvgs_ltf['fvg_bull_top'].iloc[i])
        elif fvg_bear:
            curr_fvg = (fvgs_ltf['fvg_bear_btm'].iloc[i], fvgs_ltf['fvg_bear_top'].iloc[i])
        
        curr_ob = None
        if ob_bull:
            match = [o for o in ltf_obs_list if o['time'] == timestamp and o['type'] == 'bull']
            if match:
                curr_ob = (match[0]['low'], match[0]['high'])
        elif ob_bear:
            match = [o for o in ltf_obs_list if o['time'] == timestamp and o['type'] == 'bear']
            if match:
                curr_ob = (match[0]['low'], match[0]['high'])

        # A++: FVG + OB overlap
        if curr_fvg and curr_ob:
            if check_confluence(curr_fvg[1], curr_fvg[0], curr_ob[1], curr_ob[0]):
                setup_tier = "A++"
        
        # A: HTF OB Confluence
        if setup_tier == "B" and htf_ob:
            has_htf_confluence = False
            for o in htf_obs:
                if not o['active']:
                    continue
                # Look for overlap with current range
                target_range = curr_fvg if curr_fvg else curr_ob
                if target_range and check_confluence(target_range[1], target_range[0], o['high'], o['low']):
                    has_htf_confluence = True
                    break
            if has_htf_confluence:
                setup_tier = "A"

        # Filter: FVG First Touch Only (if using FVG)
        if fvg_q and (fvg_bull or fvg_bear):
            fvg_id = timestamp
            if fvg_id in visited_fvgs:
                continue
            visited_fvgs.add(fvg_id)
        funnel["FVG First Touch"] += 1

        # Filter: Pullback before FVG formation
        if pb_flt:
            pullback_row = df_ltf.iloc[i-2]
            is_bull_pb = pullback_row['close'] < pullback_row['open']
            is_bear_pb = pullback_row['close'] > pullback_row['open']
            if is_bull_setup and not is_bull_pb:
                continue
            if is_bear_setup and not is_bear_pb:
                continue
        funnel["CISD Pullback"] += 1

        # CISD trigger check
        entry_price = row['close']
        if is_bull_setup:
            if entry_price > df_ltf['high'].iloc[i-3:i].max():
                funnel["CISD Triggered"] += 1
        else:
            if entry_price < df_ltf['low'].iloc[i-3:i].min():
                funnel["CISD Triggered"] += 1

        # Filter: CISD candle must be impulsive
        if fvg_q:
            body_size = abs(row['close'] - row['open'])
            if body_size < (avg_range_ltf.iloc[i] * CISD_BODY_MULT):
                continue
        funnel["CISD Impulsive"] += 1

        # Filter: HTF OB Confluence
        if htf_ob:
            confluence = False
            for ob in htf_obs:
                if not ob['active']:
                    continue
                margin = avg_range_ltf.iloc[i] * 0.5
                if is_bull_setup:
                    if row['close'] >= (ob['low'] - margin) and row['close'] <= (ob['high'] + margin):
                        confluence = True
                        break
                else:
                    if row['close'] <= (ob['high'] + margin) and row['close'] >= (ob['low'] - margin):
                        confluence = True
                        break
            if not confluence:
                continue
        funnel["HTF OB Confluence"] += 1

        # Filter: Premium/Discount Array Validation
        pd_valid = False
        if pd_arr:
            recent_1h = df_bias.loc[:timestamp].tail(PD_ARRAY_BARS)
            if len(recent_1h) >= 2:
                r_high = recent_1h['high'].max()
                r_low = recent_1h['low'].min()
                htf_50_level = r_low + ((r_high - r_low) / 2)

                entry = row['close']
                if is_bull_setup and entry <= htf_50_level:
                    pd_valid = True
                elif is_bear_setup and entry >= htf_50_level:
                    pd_valid = True
            if not pd_valid:
                continue
        else:
            pd_valid = True
        funnel["Premium/Discount"] += 1

        # ── Calculate FVG size for trade log ──
        if is_bull_setup:
            fvg_top_val = fvgs_ltf['fvg_bull_top'].iloc[i]
            fvg_btm_val = fvgs_ltf['fvg_bull_btm'].iloc[i]
        else:
            fvg_top_val = fvgs_ltf['fvg_bear_top'].iloc[i]
            fvg_btm_val = fvgs_ltf['fvg_bear_btm'].iloc[i]
        fvg_size = abs(fvg_top_val - fvg_btm_val) if not (pd.isna(fvg_top_val) or pd.isna(fvg_btm_val)) else 0
        fvg_size_pct = (fvg_size / row['close'] * 100) if row['close'] > 0 else 0

        # ── Regime classification (if daily data available) ──
        regime = "unknown"
        if df_daily is not None:
            regime = classify_regime(df_daily, timestamp)

        # ── Train/Test label ──
        train_test = "train" if timestamp <= pd.Timestamp(TRAIN_END, tz="UTC") else "test"

        # ── ENTRY CALCULATIONS ──
        if is_bull_setup:
            entry = row['close']
            if entry <= df_ltf['high'].iloc[i-3:i].max():
                continue

            stop = df_ltf['low'].iloc[i-3:i+1].min()
            risk = entry - stop
            # Relax SL for A++ setups (allow 8% instead of 5%)
            max_risk = entry * 0.08 if setup_tier == "A++" else entry * 0.05
            if risk <= 0 or risk > max_risk:
                continue
            funnel["Max SL Logic"] += 1

            target = entry + (risk * RISK_REWARD)
            be_trigger = entry + (risk * AUTO_BREAKEVEN_R)
            in_trade = True
            current_trade = {
                "symbol": sym, "model": "cisd", "regime": regime,
                "setup_tier": setup_tier,
                "bias_type": "bullish", "fvg_size": round(fvg_size, 4),
                "fvg_size_pct": round(fvg_size_pct, 4),
                "Entry Time": timestamp.strftime("%Y-%m-%d %H:%M"),
                "_entry_idx": i,
                "direction": 1, "Entry $": round(entry, 8),
                "Stop $": round(stop, 8), "Target $": round(target, 8),
                "risk_price": round(risk, 8),
                "be_trigger": round(be_trigger, 8), "is_be": False,
                "be_triggered": False,
                "Entry Type": "Phase 4 Bull CISD",
                "train_test": train_test,
                "account": account, "Symbol": sym
            }
            funnel["Final Trades"] += 1

        elif is_bear_setup:
            entry = row['close']
            if entry >= df_ltf['low'].iloc[i-3:i].min():
                continue

            stop = df_ltf['high'].iloc[i-3:i+1].max()
            risk = stop - entry
            # Relax SL for A++ setups (allow 8% instead of 5%)
            max_risk = entry * 0.08 if setup_tier == "A++" else entry * 0.05
            if risk <= 0 or risk > max_risk:
                continue
            funnel["Max SL Logic"] += 1

            target = entry - (risk * RISK_REWARD)
            be_trigger = entry - (risk * AUTO_BREAKEVEN_R)
            in_trade = True
            current_trade = {
                "symbol": sym, "model": "cisd", "regime": regime,
                "setup_tier": setup_tier,
                "bias_type": "bearish", "fvg_size": round(fvg_size, 4),
                "fvg_size_pct": round(fvg_size_pct, 4),
                "Entry Time": timestamp.strftime("%Y-%m-%d %H:%M"),
                "_entry_idx": i,
                "direction": -1, "Entry $": round(entry, 8),
                "Stop $": round(stop, 8), "Target $": round(target, 8),
                "risk_price": round(risk, 8),
                "be_trigger": round(be_trigger, 8), "is_be": False,
                "be_triggered": False,
                "Entry Type": "Phase 4 Bear CISD",
                "train_test": train_test,
                "account": account, "Symbol": sym
            }
            funnel["Final Trades"] += 1

    # CHANGED: Removed unreachable duplicate elif is_bear_setup block
    # that existed in the old version (old lines 370-391)

    return {"trades": trades, "funnel": funnel}


def run_backtest():
    """CLI entry point for running the CISD backtest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--alignment", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward train/test validation")
    args = parser.parse_args()

    global SYMBOL
    SYMBOL = args.symbol

    print("=" * 62)
    print(f"  TTrades CISD Model (Phase 4) - {SYMBOL}")
    print(f"  Alignment: {args.alignment.upper()}")
    print("=" * 62)

    if args.alignment == "daily":
        tf_context, tf_bias, tf_ltf = "1d", "1h", "5m"
    else:
        tf_context, tf_bias, tf_ltf = "1w", "4h", "15m"

    df_htf = fetch_ohlcv_full(SYMBOL, tf_context, DAYS_BACK)
    df_bias = fetch_ohlcv_full(SYMBOL, tf_bias, DAYS_BACK)
    df_ltf = fetch_ohlcv_full(SYMBOL, tf_ltf, DAYS_BACK)

    # Load daily data for regime classification
    try:
        df_daily = fetch_ohlcv_full(SYMBOL, "1d", DAYS_BACK)
    except FileNotFoundError:
        df_daily = None
        print("   [WARN] No 1d data found — regime classification disabled")

    # Run with all filters ON
    results = simulate_trades(df_ltf, df_bias, df_htf, symbol=SYMBOL, df_daily=df_daily)
    print_enhanced_stats(results["trades"], results["funnel"], f"CISD MODEL — {SYMBOL}")

    # Walk-forward (if requested)
    if args.walk_forward:
        wf = walk_forward_split(df_ltf, df_bias, df_htf, simulate_trades,
                                TRAIN_END, TEST_START, symbol=SYMBOL)
        print_walk_forward(wf, f"({SYMBOL} CISD)")

    # Excel export
    if results["trades"]:
        df_t = pd.DataFrame(results["trades"])
        export_cols = {
            "Entry Time": "Entry Time", "exit_time": "Exit Time",
            "Entry Type": "Strategy", "pnl": "Net Profit/Loss",
            "Entry $": "Entry Price", "Stop $": "Stop Loss",
            "Target $": "Target Price", "result": "Result",
            "regime": "Regime", "bias_type": "Bias",
            "fvg_size": "FVG Size", "duration_bars": "Duration (Bars)",
            "be_triggered": "BE Triggered", "r_multiple": "R Multiple",
            "train_test": "Period"
        }
        available_cols = [c for c in export_cols.keys() if c in df_t.columns]
        df_export = df_t[available_cols].rename(
            columns={k: v for k, v in export_cols.items() if k in available_cols})

        for col in ["Entry Time", "Exit Time"]:
            if col in df_export.columns:
                df_export[col] = pd.to_datetime(df_export[col]).dt.tz_localize(None)

        safe_name = SYMBOL.replace("/", "_")
        filename_xlsx = f"trades_cisd_{safe_name}.xlsx"
        df_export.to_excel(filename_xlsx, index=False)
        print(f"\n  Excel report saved: {filename_xlsx}")


if __name__ == "__main__":
    run_backtest()
