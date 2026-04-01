import pandas as pd
import os
import argparse
import ccxt
from typing import List, Dict

from core.lux_fvg import detect_luxalgo_fvgs, detect_liquidity_sweeps

from core.config import (
    RISK_REWARD, FIXED_SL_USDT, ACCOUNT_SIZE,
    DAYS_BACK, BIAS_EXPIRY_BARS, FVG_WICK_RATIO, FVG_AVG_BODY_LOOKBACK,
    FVG_MIN_SIZE_MULT, FVG_MIN_VOLUME_MULT, OB_MOVE_MULT, OB_SWING_LOOKBACK, CISD_BODY_MULT,
    AUTO_BREAKEVEN_R,
    USE_FVG_QUALITY, USE_HTF_OB_CONFLUENCE, USE_PREMIUM_DISCOUNT,
    USE_PULLBACK_FILTER, USE_AUTO_BREAKEVEN, USE_LTF_OB_ENTRY,
    TEST_START, DATA_DIR, DEFAULT_LTF
)

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


# -- Exchange caching for live scanner --
EXCHANGES = {}

def get_exchange(exchange_id: str = 'binance'):
    """Get or create a cached CCXT exchange instance with failover for Binance."""
    global EXCHANGES
    exchange_id = exchange_id.lower().replace("usdt", "")
    if exchange_id not in EXCHANGES:
        try:
            exchange_class = getattr(ccxt, exchange_id)
            if exchange_id == "binance":
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
    """
    REWRITTEN simulate_trades with proper SMC logic:

    Entry Logic:
    1. HTF bias must be active (bull or bear)
    2. LTF FVG or OB must form in the direction of bias
    3. Entry at FVG midpoint (limit-style) or candle close
    4. SL below/above the FVG boundaries (structural SL)
    5. TP at configured R:R ratio

    Key fix: Old CISD logic required price to break above 3-bar high
    which is actually ANTI-SMC. Real SMC enters on the pullback INTO
    the FVG, not on the breakout away from it. The old code required
    the entry candle to be a breakout candle, which contradicts the
    FVG-based entry.
    """
    sym = symbol or SYMBOL
    fvg_q = use_fvg_quality if use_fvg_quality is not None else USE_FVG_QUALITY
    htf_ob = use_htf_ob if use_htf_ob is not None else USE_HTF_OB_CONFLUENCE
    pd_arr = use_pd_array if use_pd_array is not None else USE_PREMIUM_DISCOUNT
    pb_flt = use_pullback if use_pullback is not None else USE_PULLBACK_FILTER
    auto_be = use_auto_be if use_auto_be is not None else USE_AUTO_BREAKEVEN

    bias_series = get_hourly_bias(df_bias)
    bias_aligned = bias_series.reindex(df_ltf.index, method='ffill')
    fvgs_ltf = calc_fvgs(df_ltf)

    htf_obs = find_order_blocks(df_bias)
    ltf_obs_list = find_order_blocks(df_ltf) if USE_LTF_OB_ENTRY else []

    trades = []
    account = ACCOUNT_SIZE

    funnel = {
        "Total Bars Scanned": 0,
        "Has Bias": 0,
        "FVG/OB Exists": 0,
        "FVG First Touch": 0,
        "Pullback Filter": 0,
        "HTF OB Confluence": 0,
        "Premium/Discount": 0,
        "Valid SL": 0,
        "Final Trades": 0
    }

    in_trade = False
    current_trade = None
    visited_fvgs = set()
    # Track active FVG zones for "touch" detection
    active_bull_fvgs = []  # list of (btm, top, creation_idx, creation_time)
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
            if current_trade["direction"] == 1:  # LONG
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
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                elif row['high'] >= current_trade["Target $"]:
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
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                elif row['low'] <= current_trade["Target $"]:
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
        # ENTRY LOGIC — Proper SMC FVG Reaction
        # ================================================================

        b = bias_aligned.iloc[i]
        if b == 0:
            continue
        funnel["Has Bias"] += 1

        # ── Check for FVG touch (price entering an active FVG zone) ──
        # This is the CORRECT SMC approach: we wait for price to RETRACE
        # into an FVG, not break out of a range.

        best_fvg = None

        if b == 1:  # Bullish bias — look for price touching bull FVGs
            for fvg in active_bull_fvgs:
                fvg_btm, fvg_top, fvg_idx, fvg_time = fvg
                # Only react to FVGs formed at least 2 bars ago
                if i - fvg_idx < 2:
                    continue
                # Expire FVGs older than 48 bars (~12h on 15m)
                if i - fvg_idx > 48:
                    continue
                # Price must touch into the FVG zone (low dips into zone)
                if row['low'] <= fvg_top and row['close'] > fvg_btm:
                    # Pick the most recent valid FVG
                    if best_fvg is None or fvg_idx > best_fvg[2]:
                        best_fvg = fvg
        elif b == -1:  # Bearish bias — look for price touching bear FVGs
            for fvg in active_bear_fvgs:
                fvg_btm, fvg_top, fvg_idx, fvg_time = fvg
                if i - fvg_idx < 2:
                    continue
                if i - fvg_idx > 48:
                    continue
                # Price must touch into the FVG zone (high reaches into zone)
                if row['high'] >= fvg_btm and row['close'] < fvg_top:
                    if best_fvg is None or fvg_idx > best_fvg[2]:
                        best_fvg = fvg

        # Also check current-bar FVGs (immediate entry on formation)
        fvg_bull = fvgs_ltf['fvg_bull'].iloc[i]
        fvg_bear = fvgs_ltf['fvg_bear'].iloc[i]
        has_ob = False

        if USE_LTF_OB_ENTRY:
            ob_bull = any(o['time'] == timestamp and o['type'] == 'bull' for o in ltf_obs_list)
            ob_bear = any(o['time'] == timestamp and o['type'] == 'bear' for o in ltf_obs_list)
            if b == 1 and ob_bull:
                has_ob = True
            if b == -1 and ob_bear:
                has_ob = True

        if best_fvg is None and not (b == 1 and fvg_bull) and not (b == -1 and fvg_bear) and not has_ob:
            continue

        funnel["FVG/OB Exists"] += 1

        # ── Determine entry and stop ──
        is_bull = (b == 1)

        if best_fvg is not None:
            fvg_btm, fvg_top, fvg_idx, fvg_time = best_fvg
            fvg_mid = (fvg_btm + fvg_top) / 2
            entry_key = f"{fvg_time}_{fvg_idx}"
        elif fvg_bull and is_bull:
            fvg_btm = fvgs_ltf['fvg_bull_btm'].iloc[i]
            fvg_top = fvgs_ltf['fvg_bull_top'].iloc[i]
            fvg_mid = (fvg_btm + fvg_top) / 2
            entry_key = f"{timestamp}_fvg"
        elif fvg_bear and not is_bull:
            fvg_btm = fvgs_ltf['fvg_bear_btm'].iloc[i]
            fvg_top = fvgs_ltf['fvg_bear_top'].iloc[i]
            fvg_mid = (fvg_btm + fvg_top) / 2
            entry_key = f"{timestamp}_fvg"
        elif has_ob:
            # Use OB levels
            if is_bull:
                match = [o for o in ltf_obs_list if o['time'] == timestamp and o['type'] == 'bull']
            else:
                match = [o for o in ltf_obs_list if o['time'] == timestamp and o['type'] == 'bear']
            if not match:
                continue
            fvg_btm = match[0]['low']
            fvg_top = match[0]['high']
            fvg_mid = match[0]['median']
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
        if htf_ob:
            confluence = False
            for ob in htf_obs:
                if not ob['active']:
                    continue
                dist_pct = min(abs(row['close'] - ob['high']), abs(row['close'] - ob['low'])) / row['close']
                if dist_pct < 0.01:  # Within 1% of an HTF OB
                    confluence = True
                    setup_tier = "A"
                    break
            if not confluence:
                continue
        funnel["HTF OB Confluence"] += 1

        # ── Premium/Discount filter ──
        if pd_arr:
            lookback = min(200, i)  # Use more lookback for better range detection
            hi = df_ltf['high'].iloc[max(0, i - lookback):i].max()
            lo = df_ltf['low'].iloc[max(0, i - lookback):i].min()
            mid = (hi + lo) / 2
            if is_bull and row['close'] > mid:
                continue
            if not is_bull and row['close'] < mid:
                continue
        funnel["Premium/Discount"] += 1

        # ── Calculate entry, stop, target ──
        entry = row['close']  # Market entry on candle close
        fvg_size = abs(fvg_top - fvg_btm)

        if is_bull:
            stop = fvg_btm - (fvg_size * 0.2)  # SL just below FVG with small buffer
            risk = entry - stop
            target = entry + (risk * RISK_REWARD)
        else:
            stop = fvg_top + (fvg_size * 0.2)   # SL just above FVG with small buffer
            risk = stop - entry
            target = entry - (risk * RISK_REWARD)

        # ── Validate risk ──
        if risk <= 0:
            continue
        max_risk_pct = 0.03  # Max 3% risk per trade
        if risk / entry > max_risk_pct:
            continue
        funnel["Valid SL"] += 1

        # ── Regime & metadata ──
        regime = classify_regime(df_daily, timestamp) if df_daily is not None else "ranging"
        train_test = "TEST" if timestamp.strftime("%Y-%m-%d") >= TEST_START else "TRAIN"

        be_trigger = entry + (risk * AUTO_BREAKEVEN_R) if is_bull else entry - (risk * AUTO_BREAKEVEN_R)

        current_trade = {
            "symbol": sym, "model": "cisd", "regime": regime, "setup_tier": setup_tier,
            "bias_type": "bullish" if is_bull else "bearish",
            "fvg_size": round(fvg_size, 4),
            "fvg_size_pct": round((fvg_size / entry) * 100, 4),
            "Entry Time": timestamp.strftime("%Y-%m-%d %H:%M"),
            "_entry_idx": i, "direction": 1 if is_bull else -1,
            "Entry $": round(entry, 8),
            "Stop $": round(stop, 8), "Target $": round(target, 8),
            "risk_price": round(risk, 8),
            "be_trigger": round(be_trigger, 8),
            "is_be": False, "be_triggered": False, "r_multiple": 0,
            "Entry Type": f"FVG Reaction {'Bull' if is_bull else 'Bear'}",
            "train_test": train_test, "account": account, "Symbol": sym
        }
        funnel["Final Trades"] += 1
        in_trade = True

    # Cleanup old FVGs to prevent memory leak in long runs
    active_bull_fvgs = [f for f in active_bull_fvgs if len(df_ltf) - f[2] <= 48]
    active_bear_fvgs = [f for f in active_bear_fvgs if len(df_ltf) - f[2] <= 48]

    return {"trades": trades, "funnel": funnel}


def run_backtest():
    global SYMBOL
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--walk-forward", action="store_true")
    args = parser.parse_args()

    SYMBOL = args.symbol

    tf_context, tf_bias, tf_ltf = "1d", "1h", DEFAULT_LTF
    df_htf = fetch_ohlcv_full(SYMBOL, tf_context, DAYS_BACK)
    df_bias = fetch_ohlcv_full(SYMBOL, tf_bias, DAYS_BACK)
    df_ltf = fetch_ohlcv_full(SYMBOL, tf_ltf, DAYS_BACK)

    try:
        df_daily = fetch_ohlcv_full(SYMBOL, "1d", DAYS_BACK)
    except FileNotFoundError:
        df_daily = None

    results = simulate_trades(df_ltf, df_bias, df_htf, symbol=SYMBOL, df_daily=df_daily)
    print_enhanced_stats(results["trades"], results["funnel"], f"CISD MODEL -- {SYMBOL}")

if __name__ == "__main__":
    run_backtest()
