import pandas as pd
import os
import argparse
from typing import List, Dict

from core.lux_fvg import detect_luxalgo_fvgs, detect_liquidity_sweeps

from core.config import (
    RISK_REWARD, FIXED_SL_USDT, ACCOUNT_SIZE,
    DAYS_BACK, BIAS_EXPIRY_BARS, FVG_WICK_RATIO, FVG_AVG_BODY_LOOKBACK,
    FVG_MIN_SIZE_MULT, FVG_MIN_VOLUME_MULT, OB_MOVE_MULT, OB_SWING_LOOKBACK,
    AUTO_BREAKEVEN_R,
    USE_FVG_QUALITY, USE_HTF_OB_CONFLUENCE, USE_PREMIUM_DISCOUNT,
    USE_PULLBACK_FILTER, USE_AUTO_BREAKEVEN,
    TEST_START, DATA_DIR, DEFAULT_LTF
)

from core.analysis import (
    print_enhanced_stats, classify_regime
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


def calc_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    """Detect FVGs using LuxAlgo displacement logic."""
    return detect_luxalgo_fvgs(df.copy(), wick_ratio=FVG_WICK_RATIO,
                               avg_body_lookback=FVG_AVG_BODY_LOOKBACK,
                               min_size_mult=FVG_MIN_SIZE_MULT,
                               min_volume_mult=FVG_MIN_VOLUME_MULT)


def get_liquidity_sweeps(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Detect liquidity sweeps."""
    return detect_liquidity_sweeps(df.copy(), lookback=10)


def find_htf_obs(df: pd.DataFrame) -> List[Dict]:
    """HTF Order Block detection."""
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
                    if df['close'].iloc[i] > ob_high:
                        lows_around = df['low'].iloc[max(0, j - OB_SWING_LOOKBACK):min(len(df), j + OB_SWING_LOOKBACK + 1)]
                        if df['low'].iloc[j] == lows_around.min():
                            obs.append({
                                "time": df.index[j],
                                "high": ob_high, "low": ob_low,
                                "type": "bull", "active": True
                            })
            else:  # Bearish Impulse
                if df['close'].iloc[j] > df['open'].iloc[j]:
                    ob_high = df['high'].iloc[j]
                    ob_low = df['low'].iloc[j]
                    if df['close'].iloc[i] < ob_low:
                        highs_around = df['high'].iloc[max(0, j - OB_SWING_LOOKBACK):min(len(df), j + OB_SWING_LOOKBACK + 1)]
                        if df['high'].iloc[j] == highs_around.max():
                            obs.append({
                                "time": df.index[j],
                                "high": ob_high, "low": ob_low,
                                "type": "bear", "active": True
                            })
    return obs


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
    LIMIT MODEL — Places limit orders at FVG midpoint.

    Logic:
    1. Wait for FVG to form on LTF with HTF bias confirmation
    2. Place limit entry at FVG midpoint
    3. SL below/above FVG boundary
    4. TP at R:R ratio from entry

    This model WAITS for price to retrace to FVG midpoint (limit fill)
    instead of entering at market on candle close.
    """
    sym = symbol or SYMBOL
    fvg_q = use_fvg_quality if use_fvg_quality is not None else USE_FVG_QUALITY
    htf_ob_flt = use_htf_ob if use_htf_ob is not None else USE_HTF_OB_CONFLUENCE
    pd_arr = use_pd_array if use_pd_array is not None else USE_PREMIUM_DISCOUNT
    pb_flt = use_pullback if use_pullback is not None else USE_PULLBACK_FILTER
    auto_be = use_auto_be if use_auto_be is not None else USE_AUTO_BREAKEVEN

    bias_series = get_hourly_bias(df_bias)
    bias_aligned = bias_series.reindex(df_ltf.index, method='ffill')
    fvgs_ltf = calc_fvgs(df_ltf)
    htf_obs = find_htf_obs(df_bias)

    trades = []
    account = ACCOUNT_SIZE

    funnel = {
        "Total Bars": 0, "Has Bias": 0, "FVG Exists": 0,
        "FVG First Touch": 0, "Pullback Filter": 0,
        "HTF OB Confluence": 0, "Premium/Discount": 0,
        "Limit Order Set": 0, "Limit Filled": 0,
        "Valid SL": 0, "Final Trades": 0
    }

    in_trade = False
    current_trade = None
    visited_fvgs = set()

    # Pending limit orders: list of dicts with entry, stop, etc.
    pending_limits = []

    for i in range(25, len(df_ltf)):
        funnel["Total Bars"] += 1
        row = df_ltf.iloc[i]
        timestamp = df_ltf.index[i]

        # -- Check if any pending limit orders are filled --
        if not in_trade and pending_limits:
            filled = None
            for idx, lim in enumerate(pending_limits):
                # Expire orders older than 24 bars
                if i - lim["_set_idx"] > 24:
                    continue
                if lim["direction"] == 1:  # Bull: price dips to entry
                    if row['low'] <= lim["entry"]:
                        filled = idx
                        break
                else:  # Bear: price rises to entry
                    if row['high'] >= lim["entry"]:
                        filled = idx
                        break

            if filled is not None:
                lim = pending_limits.pop(filled)
                funnel["Limit Filled"] += 1
                regime = classify_regime(df_daily, timestamp) if df_daily is not None else "ranging"
                train_test = "TEST" if timestamp.strftime("%Y-%m-%d") >= TEST_START else "TRAIN"

                current_trade = {
                    "symbol": sym, "model": "limit", "regime": regime,
                    "setup_tier": lim.get("setup_tier", "B"),
                    "bias_type": "bullish" if lim["direction"] == 1 else "bearish",
                    "Entry Time": timestamp.strftime("%Y-%m-%d %H:%M"),
                    "_entry_idx": i, "direction": lim["direction"],
                    "Entry $": round(lim["entry"], 8),
                    "Stop $": round(lim["stop"], 8),
                    "Target $": round(lim["target"], 8),
                    "is_be": False,
                    "be_trigger": round(lim["be_trigger"], 8),
                    "Entry Type": f"Limit {'Bull' if lim['direction'] == 1 else 'Bear'}",
                    "train_test": train_test, "account": account, "Symbol": sym
                }
                funnel["Final Trades"] += 1
                in_trade = True

        # -- Trade management --
        if in_trade:
            if current_trade["direction"] == 1:
                if auto_be and not current_trade["is_be"]:
                    if row['high'] >= current_trade["be_trigger"]:
                        current_trade["Stop $"] = current_trade["Entry $"]
                        current_trade["is_be"] = True
                if row['low'] <= current_trade["Stop $"]:
                    res = "BE" if current_trade["is_be"] else "LOSS"
                    pnl = 0 if res == "BE" else -FIXED_SL_USDT
                    account += pnl
                    current_trade.update({
                        "exit_time": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "result": res, "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade["_entry_idx"]
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
                        "duration_bars": i - current_trade["_entry_idx"]
                    })
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
            else:  # BEARISH
                if auto_be and not current_trade["is_be"]:
                    if row['low'] <= current_trade["be_trigger"]:
                        current_trade["Stop $"] = current_trade["Entry $"]
                        current_trade["is_be"] = True
                if row['high'] >= current_trade["Stop $"]:
                    res = "BE" if current_trade["is_be"] else "LOSS"
                    pnl = 0 if res == "BE" else -FIXED_SL_USDT
                    account += pnl
                    current_trade.update({
                        "exit_time": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "result": res, "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade["_entry_idx"]
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
                        "duration_bars": i - current_trade["_entry_idx"]
                    })
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
            continue

        # ── Signal Detection: FVG Formation ──
        b = bias_aligned.iloc[i]
        if b == 0:
            continue
        funnel["Has Bias"] += 1

        fvg_bull = fvgs_ltf['fvg_bull'].iloc[i]
        fvg_bear = fvgs_ltf['fvg_bear'].iloc[i]

        if not ((b == 1 and fvg_bull) or (b == -1 and fvg_bear)):
            continue
        funnel["FVG Exists"] += 1

        if fvg_q:
            if timestamp in visited_fvgs:
                continue
            visited_fvgs.add(timestamp)
        funnel["FVG First Touch"] += 1

        if pb_flt:
            pb_row = df_ltf.iloc[i - 1]
            if fvg_bull and not (pb_row['close'] < pb_row['open']):
                continue
            if fvg_bear and not (pb_row['close'] > pb_row['open']):
                continue
        funnel["Pullback Filter"] += 1

        # HTF OB Confluence
        setup_tier = "B"
        if htf_ob_flt:
            confluence = False
            for ob in htf_obs:
                if not ob['active']:
                    continue
                dist_pct = min(abs(row['close'] - ob['high']), abs(row['close'] - ob['low'])) / row['close']
                if dist_pct < 0.01:
                    confluence = True
                    setup_tier = "A"
                    break
            if not confluence:
                continue
        funnel["HTF OB Confluence"] += 1

        # Premium/Discount
        if pd_arr:
            lookback = min(200, i)
            hi = df_ltf['high'].iloc[max(0, i - lookback):i].max()
            lo = df_ltf['low'].iloc[max(0, i - lookback):i].min()
            mid = (hi + lo) / 2
            if fvg_bull and row['close'] > mid:
                continue
            if fvg_bear and row['close'] < mid:
                continue
        funnel["Premium/Discount"] += 1

        # ── Place Limit Order at FVG midpoint ──
        if fvg_bull:
            fvg_btm = fvgs_ltf['fvg_bull_btm'].iloc[i]
            fvg_top = fvgs_ltf['fvg_bull_top'].iloc[i]
        else:
            fvg_btm = fvgs_ltf['fvg_bear_btm'].iloc[i]
            fvg_top = fvgs_ltf['fvg_bear_top'].iloc[i]

        if pd.isna(fvg_btm) or pd.isna(fvg_top):
            continue

        fvg_mid = (fvg_btm + fvg_top) / 2
        fvg_size = abs(fvg_top - fvg_btm)

        if fvg_bull:
            entry = fvg_mid
            stop = fvg_btm - (fvg_size * 0.2)
            risk = entry - stop
            target = entry + (risk * RISK_REWARD)
        else:
            entry = fvg_mid
            stop = fvg_top + (fvg_size * 0.2)
            risk = stop - entry
            target = entry - (risk * RISK_REWARD)

        if risk <= 0:
            continue
        if risk / entry > 0.03:  # Max 3% risk
            continue
        funnel["Valid SL"] += 1
        funnel["Limit Order Set"] += 1

        be_trigger = entry + (risk * AUTO_BREAKEVEN_R) if fvg_bull else entry - (risk * AUTO_BREAKEVEN_R)

        pending_limits.append({
            "entry": entry, "stop": stop, "target": target,
            "direction": 1 if fvg_bull else -1,
            "be_trigger": be_trigger,
            "_set_idx": i,
            "setup_tier": setup_tier
        })

    # Expire stale pending orders
    pending_limits = [p for p in pending_limits if len(df_ltf) - p["_set_idx"] <= 24]

    return {"trades": trades, "funnel": funnel}


def run_backtest():
    global SYMBOL
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    args = parser.parse_args()

    SYMBOL = args.symbol
    df_htf = fetch_ohlcv_full(SYMBOL, "1d", DAYS_BACK)
    df_bias = fetch_ohlcv_full(SYMBOL, "1h", DAYS_BACK)
    df_ltf = fetch_ohlcv_full(SYMBOL, DEFAULT_LTF, DAYS_BACK)

    try:
        df_daily = fetch_ohlcv_full(SYMBOL, "1d", DAYS_BACK)
    except FileNotFoundError:
        df_daily = None

    res = simulate_trades(df_ltf, df_bias, df_htf, symbol=SYMBOL, df_daily=df_daily)
    print_enhanced_stats(res["trades"], res["funnel"], f"LIMIT MODEL -- {SYMBOL}")

if __name__ == "__main__":
    run_backtest()
