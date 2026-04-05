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
    TEST_START, DATA_DIR,
    TF_CONTEXT, TF_BIAS, TF_ENTRY, ACTIVE_CHAIN
)

from core.analysis import (
    print_enhanced_stats, classify_regime
)

SYMBOL = "BTC/USDT"


def fetch_ohlcv_full(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
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
    return detect_luxalgo_fvgs(df.copy(), wick_ratio=FVG_WICK_RATIO,
                               avg_body_lookback=FVG_AVG_BODY_LOOKBACK,
                               min_size_mult=FVG_MIN_SIZE_MULT,
                               min_volume_mult=FVG_MIN_VOLUME_MULT)


def get_liquidity_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    return detect_liquidity_sweeps(df.copy(), lookback=10)


def find_htf_obs(df: pd.DataFrame) -> List[Dict]:
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
            if body_val > 0:
                if df['close'].iloc[j] < df['open'].iloc[j]:
                    ob_high = df['high'].iloc[j]
                    ob_low = df['low'].iloc[j]
                    if df['close'].iloc[i] > ob_high:
                        lows = df['low'].iloc[max(0, j - OB_SWING_LOOKBACK):min(len(df), j + OB_SWING_LOOKBACK + 1)]
                        if df['low'].iloc[j] == lows.min():
                            obs.append({"time": df.index[j], "high": ob_high, "low": ob_low, "type": "bull", "active": True})
            else:
                if df['close'].iloc[j] > df['open'].iloc[j]:
                    ob_high = df['high'].iloc[j]
                    ob_low = df['low'].iloc[j]
                    if df['close'].iloc[i] < ob_low:
                        highs = df['high'].iloc[max(0, j - OB_SWING_LOOKBACK):min(len(df), j + OB_SWING_LOOKBACK + 1)]
                        if df['high'].iloc[j] == highs.max():
                            obs.append({"time": df.index[j], "high": ob_high, "low": ob_low, "type": "bear", "active": True})
    return obs


def get_hourly_bias(df_bias: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame with 'bias' and 'bias_reason' columns."""
    fvgs = calc_fvgs(df_bias)
    sweeps = get_liquidity_sweeps(df_bias)

    bias_df = pd.DataFrame({"bias": 0, "bias_reason": ""}, index=df_bias.index)

    bull_fvg = fvgs['fvg_bull']
    bear_fvg = fvgs['fvg_bear']
    bull_sweep = sweeps['sweep_bull']
    bear_sweep = sweeps['sweep_bear']

    body = (df_bias['close'] - df_bias['open']).abs()
    avg_body = body.rolling(10).mean()
    mom_bull = (df_bias['close'] > df_bias['open']) & (body > avg_body)
    mom_bear = (df_bias['close'] < df_bias['open']) & (body > avg_body)

    active_bias = 0
    active_reason = ""
    signal_age = 0
    pending_signal = 0
    pending_reason = ""

    for i in range(len(df_bias)):
        ts = df_bias.index[i].strftime("%Y-%m-%d %H:%M")

        if bull_fvg.iloc[i]:
            pending_signal, pending_reason, signal_age = 1, f"Bull FVG @ {ts}", 0
            active_bias, active_reason = 0, ""
        elif bull_sweep.iloc[i]:
            pending_signal, pending_reason, signal_age = 1, f"Bull Sweep @ {ts}", 0
            active_bias, active_reason = 0, ""
        elif bear_fvg.iloc[i]:
            pending_signal, pending_reason, signal_age = -1, f"Bear FVG @ {ts}", 0
            active_bias, active_reason = 0, ""
        elif bear_sweep.iloc[i]:
            pending_signal, pending_reason, signal_age = -1, f"Bear Sweep @ {ts}", 0
            active_bias, active_reason = 0, ""

        if pending_signal == 1:
            if mom_bull.iloc[i]:
                active_bias = 1
                active_reason = f"Bullish ({pending_reason}, confirmed @ {ts})"
                pending_signal = 0
            else:
                signal_age += 1
                if signal_age > BIAS_EXPIRY_BARS:
                    pending_signal, pending_reason = 0, ""
        elif pending_signal == -1:
            if mom_bear.iloc[i]:
                active_bias = -1
                active_reason = f"Bearish ({pending_reason}, confirmed @ {ts})"
                pending_signal = 0
            else:
                signal_age += 1
                if signal_age > BIAS_EXPIRY_BARS:
                    pending_signal, pending_reason = 0, ""

        bias_df.iloc[i, 0] = active_bias
        bias_df.iloc[i, 1] = active_reason

    return bias_df


def simulate_trades(df_ltf: pd.DataFrame, df_bias: pd.DataFrame,
                    df_htf: pd.DataFrame = None,
                    symbol: str = None,
                    use_fvg_quality: bool = None,
                    use_htf_ob: bool = None,
                    use_pd_array: bool = None,
                    use_pullback: bool = None,
                    use_auto_be: bool = None,
                    df_daily: pd.DataFrame = None) -> Dict:
    """Limit Order Model — places limits at FVG midpoint, waits for fill."""
    sym = symbol or SYMBOL
    fvg_q = use_fvg_quality if use_fvg_quality is not None else USE_FVG_QUALITY
    htf_ob_flt = use_htf_ob if use_htf_ob is not None else USE_HTF_OB_CONFLUENCE
    pd_arr = use_pd_array if use_pd_array is not None else USE_PREMIUM_DISCOUNT
    pb_flt = use_pullback if use_pullback is not None else USE_PULLBACK_FILTER
    auto_be = use_auto_be if use_auto_be is not None else USE_AUTO_BREAKEVEN

    bias_result = get_hourly_bias(df_bias)
    bias_series = bias_result['bias']
    bias_reasons = bias_result['bias_reason']

    bias_aligned = bias_series.reindex(df_ltf.index, method='ffill')
    reason_aligned = bias_reasons.reindex(df_ltf.index, method='ffill').fillna("")
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
    pending_limits = []

    for i in range(25, len(df_ltf)):
        funnel["Total Bars"] += 1
        row = df_ltf.iloc[i]
        timestamp = df_ltf.index[i]

        # -- Check pending limit fills --
        if not in_trade and pending_limits:
            filled = None
            for idx, lim in enumerate(pending_limits):
                if i - lim["_set_idx"] > 24:
                    continue
                if lim["direction"] == 1:
                    if row['low'] <= lim["entry"]:
                        filled = idx
                        break
                else:
                    if row['high'] >= lim["entry"]:
                        filled = idx
                        break

            if filled is not None:
                lim = pending_limits.pop(filled)
                funnel["Limit Filled"] += 1
                regime = classify_regime(df_daily, timestamp) if df_daily is not None else "ranging"
                train_test = "TEST" if timestamp.strftime("%Y-%m-%d") >= TEST_START else "TRAIN"

                direction_str = "LONG" if lim["direction"] == 1 else "SHORT"
                reason = (
                    f"HTF: {lim['htf_reason']} | "
                    f"LTF: {TF_ENTRY} {lim['ltf_reason']} | "
                    f"{direction_str} Limit Fill @ {lim['entry']:.2f} | "
                    f"Regime: {regime}"
                )

                current_trade = {
                    "symbol": sym, "model": "limit", "regime": regime,
                    "setup_tier": lim.get("setup_tier", "B"),
                    "bias_type": "bullish" if lim["direction"] == 1 else "bearish",
                    "reason": reason,
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
                    in_trade, current_trade = False, None
                elif row['high'] >= current_trade["Target $"]:
                    pnl = FIXED_SL_USDT * RISK_REWARD
                    account += pnl
                    current_trade.update({
                        "exit_time": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "result": "WIN", "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade["_entry_idx"]
                    })
                    trades.append(current_trade)
                    in_trade, current_trade = False, None
            else:
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
                    in_trade, current_trade = False, None
                elif row['low'] <= current_trade["Target $"]:
                    pnl = FIXED_SL_USDT * RISK_REWARD
                    account += pnl
                    current_trade.update({
                        "exit_time": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "result": "WIN", "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade["_entry_idx"]
                    })
                    trades.append(current_trade)
                    in_trade, current_trade = False, None
            continue

        # ── Signal: FVG formation on LTF with bias ──
        b = bias_aligned.iloc[i]
        if b == 0:
            continue
        funnel["Has Bias"] += 1

        htf_reason = reason_aligned.iloc[i]

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
        if risk / entry > 0.03:
            continue
        funnel["Valid SL"] += 1
        funnel["Limit Order Set"] += 1

        be_trigger = entry + (risk * AUTO_BREAKEVEN_R) if fvg_bull else entry - (risk * AUTO_BREAKEVEN_R)

        ltf_reason = f"FVG Limit (zone {fvg_btm:.2f}-{fvg_top:.2f}, entry @ midpoint {fvg_mid:.2f})"

        pending_limits.append({
            "entry": entry, "stop": stop, "target": target,
            "direction": 1 if fvg_bull else -1,
            "be_trigger": be_trigger, "_set_idx": i,
            "setup_tier": setup_tier,
            "htf_reason": htf_reason, "ltf_reason": ltf_reason
        })

    pending_limits = [p for p in pending_limits if len(df_ltf) - p["_set_idx"] <= 24]
    return {"trades": trades, "funnel": funnel}


def run_backtest():
    global SYMBOL
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--chain", default=None, help="Override ACTIVE_CHAIN")
    args = parser.parse_args()

    SYMBOL = args.symbol

    if args.chain:
        from core.config import ALIGNMENTS
        if args.chain not in ALIGNMENTS:
            print(f"  [ERROR] Unknown chain '{args.chain}'. Available: {list(ALIGNMENTS.keys())}")
            return
        chain = ALIGNMENTS[args.chain]
        tf_context, tf_bias, tf_entry = chain["context"], chain["bias"], chain["entry"]
        chain_name = args.chain
    else:
        tf_context, tf_bias, tf_entry = TF_CONTEXT, TF_BIAS, TF_ENTRY
        chain_name = ACTIVE_CHAIN

    print(f"\n  Chain: {chain_name.upper()} ({tf_context} -> {tf_bias} -> {tf_entry})")

    df_htf = fetch_ohlcv_full(SYMBOL, tf_context, DAYS_BACK)
    df_bias = fetch_ohlcv_full(SYMBOL, tf_bias, DAYS_BACK)
    df_ltf = fetch_ohlcv_full(SYMBOL, tf_entry, DAYS_BACK)

    try:
        df_daily = fetch_ohlcv_full(SYMBOL, "1d", DAYS_BACK)
    except FileNotFoundError:
        df_daily = None

    res = simulate_trades(df_ltf, df_bias, df_htf, symbol=SYMBOL, df_daily=df_daily)

    if res["trades"]:
        print(f"\n  SAMPLE TRADE REASONS (first 5):")
        for t in res["trades"][:5]:
            print(f"    [{t['Entry Time']}] {t.get('reason', 'N/A')}")

    print_enhanced_stats(res["trades"], res["funnel"],
                         f"LIMIT MODEL -- {SYMBOL} [{chain_name.upper()}]")

if __name__ == "__main__":
    run_backtest()
