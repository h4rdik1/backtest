import pandas as pd
import os
import argparse
from typing import List, Dict

from core.lux_fvg import detect_luxalgo_fvgs, detect_liquidity_sweeps

# CHANGED: Import all constants from shared config instead of hardcoding
from core.config import (
    RISK_REWARD, FIXED_SL_USDT, ACCOUNT_SIZE,
    DAYS_BACK, BIAS_EXPIRY_BARS, FVG_WICK_RATIO, FVG_AVG_BODY_LOOKBACK,
    FVG_MIN_SIZE_MULT, FVG_MIN_VOLUME_MULT, OB_MOVE_MULT, OB_SWING_LOOKBACK,
    AUTO_BREAKEVEN_R, PD_ARRAY_BARS,
    USE_FVG_QUALITY, USE_HTF_OB_CONFLUENCE, USE_PREMIUM_DISCOUNT,
    USE_PULLBACK_FILTER, USE_AUTO_BREAKEVEN,
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


def calc_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    """Detect FVGs using LuxAlgo displacement logic."""
    return detect_luxalgo_fvgs(df.copy(), wick_ratio=FVG_WICK_RATIO,
                               avg_body_lookback=FVG_AVG_BODY_LOOKBACK,
                               min_size_mult=FVG_MIN_SIZE_MULT,
                               min_volume_mult=FVG_MIN_VOLUME_MULT)


def get_liquidity_sweeps(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Detect liquidity sweeps (wicks past swing points that close back inside)."""
    return detect_liquidity_sweeps(df.copy(), lookback=10)


def find_htf_obs(df: pd.DataFrame) -> List[Dict]:
    """
    HTF Order Block detection.
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
                            "active": True
                        })
    return obs


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
    Core LIMIT ENTRY simulation engine — places limit orders at FVG 50% CE.

    CHANGED vs old version:
    - Filter toggles are now parameters (for filter contribution analysis)
    - Enhanced trade dict with regime, fvg_size, duration, r_multiple, etc.
    - Removed unreachable duplicate elif block (old lines 359-387)
    - Removed duplicated print_stats() function (old lines 450-497)
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

    htf_obs = find_htf_obs(df_bias)

    trades = []
    account = ACCOUNT_SIZE

    # Funnel Tracking
    funnel = {
        "Total Signals": 0, "Bias Filter": 0, "FVG Exists": 0,
        "FVG First Touch": 0, "CISD Pullback": 0,
        "HTF OB Confluence": 0, "Premium/Discount": 0,
        "Limit Order Set": 0, "Max SL Logic": 0, "Final Trades": 0
    }

    in_trade = False
    current_trade = None
    visited_fvgs = set()
    active_limit = None  # CHANGED: Initialize properly instead of checking with locals()

    for i in range(25, len(df_ltf)):
        funnel["Total Signals"] += 1
        row = df_ltf.iloc[i]
        timestamp = df_ltf.index[i]

        # ── CHECK PENDING LIMIT ORDER ──
        if not in_trade and active_limit is not None:
            # Expire limit after N bars
            if i - active_limit['index'] > BIAS_EXPIRY_BARS:
                active_limit = None
            else:
                # Check for fill
                filled = False
                if active_limit['direction'] == 1 and row['low'] <= active_limit['entry']:
                    filled = True
                elif active_limit['direction'] == -1 and row['high'] >= active_limit['entry']:
                    filled = True

                if filled:
                    in_trade = True
                    current_trade = active_limit['trade_template'].copy()
                    current_trade['Entry Time'] = timestamp.strftime("%Y-%m-%d %H:%M")
                    current_trade['_entry_idx'] = i
                    active_limit = None
                    funnel["Final Trades"] += 1

        # ── TRADE MANAGEMENT ──
        if in_trade:
            if current_trade["direction"] == 1:  # BULLISH trade
                # Auto Break-Even
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
                        "result": result, "exit_time": timestamp,
                        "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade.get("_entry_idx", i),
                        "r_multiple": 0.0 if result == "BE" else -1.0
                    })
                    trades.append(current_trade)
                    in_trade = False
                elif row['high'] >= current_trade["Target $"]:
                    pnl = FIXED_SL_USDT * RISK_REWARD
                    account += pnl
                    current_trade.update({
                        "result": "WIN", "exit_time": timestamp,
                        "pnl": pnl, "account": account,
                        "duration_bars": i - current_trade.get("_entry_idx", i),
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
                        "duration_bars": i - current_trade.get("_entry_idx", i),
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
                        "duration_bars": i - current_trade.get("_entry_idx", i),
                        "r_multiple": RISK_REWARD
                    })
                    trades.append(current_trade)
                    in_trade = False
            continue

        # ── ENTRY SIGNAL LOGIC ──
        b = bias_aligned.iloc[i]
        if b == 0:
            continue
        funnel["Bias Filter"] += 1

        is_bull_setup = (b == 1 and fvgs_ltf['fvg_bull'].iloc[i])
        is_bear_setup = (b == -1 and fvgs_ltf['fvg_bear'].iloc[i])

        if not (is_bull_setup or is_bear_setup):
            continue
        funnel["FVG Exists"] += 1

        # Filter: FVG First Touch Only
        if fvg_q:
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

        # Filter: Premium/Discount Array Validation
        # Get FVG boundaries for the limit entry
        fvg_top = fvgs_ltf['fvg_bull_top'].iloc[i] if is_bull_setup else fvgs_ltf['fvg_bear_top'].iloc[i]
        fvg_btm = fvgs_ltf['fvg_bull_btm'].iloc[i] if is_bull_setup else fvgs_ltf['fvg_bear_btm'].iloc[i]
        if pd.isna(fvg_top) or pd.isna(fvg_btm):
            continue

        # Limit entry at 50% CE (midpoint of FVG)
        entry = (fvg_top + fvg_btm) / 2

        pd_valid = False
        if pd_arr:
            recent_1h = df_bias.loc[:timestamp].tail(PD_ARRAY_BARS)
            if len(recent_1h) >= 2:
                r_high = recent_1h['high'].max()
                r_low = recent_1h['low'].min()
                htf_50_level = r_low + ((r_high - r_low) / 2)

                if is_bull_setup and entry <= htf_50_level:
                    pd_valid = True
                elif is_bear_setup and entry >= htf_50_level:
                    pd_valid = True
            if not pd_valid:
                continue
        else:
            pd_valid = True
        funnel["Premium/Discount"] += 1

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

        # ── Calculate FVG size for trade log ──
        fvg_size = abs(fvg_top - fvg_btm)
        fvg_size_pct = (fvg_size / row['close'] * 100) if row['close'] > 0 else 0

        # ── Regime classification ──
        regime = "unknown"
        if df_daily is not None:
            regime = classify_regime(df_daily, timestamp)

        # ── Train/Test label ──
        train_test = "train" if timestamp <= pd.Timestamp(TRAIN_END, tz="UTC") else "test"

        # ── PLACE LIMIT ORDER ──
        if is_bull_setup:
            stop = fvg_btm - (avg_range_ltf.iloc[i] * 0.1)
            risk = entry - stop
            if risk <= 0 or risk > (entry * 0.05):
                continue
            funnel["Max SL Logic"] += 1

            target = entry + (risk * RISK_REWARD)
            be_trigger = entry + (risk * AUTO_BREAKEVEN_R)

            active_limit = {
                'index': i, 'direction': 1, 'entry': entry,
                'trade_template': {
                    "symbol": sym, "model": "limit", "regime": regime,
                    "bias_type": "bullish", "fvg_size": round(fvg_size, 4),
                    "fvg_size_pct": round(fvg_size_pct, 4),
                    "direction": 1, "Entry $": round(entry, 8),
                    "Stop $": round(stop, 8), "Target $": round(target, 8),
                    "risk_price": round(risk, 8),
                    "be_trigger": round(be_trigger, 8), "is_be": False,
                    "be_triggered": False,
                    "Entry Type": "Phase 4 Bull Limit (50%)",
                    "train_test": train_test,
                    "account": account, "Symbol": sym
                }
            }
            funnel["Limit Order Set"] += 1

        elif is_bear_setup:
            stop = fvg_top + (avg_range_ltf.iloc[i] * 0.1)
            risk = stop - entry
            if risk <= 0 or risk > (entry * 0.05):
                continue
            funnel["Max SL Logic"] += 1

            target = entry - (risk * RISK_REWARD)
            be_trigger = entry - (risk * AUTO_BREAKEVEN_R)

            active_limit = {
                'index': i, 'direction': -1, 'entry': entry,
                'trade_template': {
                    "symbol": sym, "model": "limit", "regime": regime,
                    "bias_type": "bearish", "fvg_size": round(fvg_size, 4),
                    "fvg_size_pct": round(fvg_size_pct, 4),
                    "direction": -1, "Entry $": round(entry, 8),
                    "Stop $": round(stop, 8), "Target $": round(target, 8),
                    "risk_price": round(risk, 8),
                    "be_trigger": round(be_trigger, 8), "is_be": False,
                    "be_triggered": False,
                    "Entry Type": "Phase 4 Bear Limit (50%)",
                    "train_test": train_test,
                    "account": account, "Symbol": sym
                }
            }
            funnel["Limit Order Set"] += 1

    # CHANGED: Removed unreachable duplicate elif is_bear_setup block
    # and removed duplicated print_stats() function
    return {"trades": trades, "funnel": funnel}


def run_backtest():
    """CLI entry point for running the Limit model backtest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--alignment", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward train/test validation")
    args = parser.parse_args()

    global SYMBOL
    SYMBOL = args.symbol

    print("=" * 62)
    print(f"  TTrades Limit Model (Phase 4) - {SYMBOL}")
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
    print_enhanced_stats(results["trades"], results["funnel"], f"LIMIT MODEL — {SYMBOL}")

    # Walk-forward (if requested)
    if args.walk_forward:
        wf = walk_forward_split(df_ltf, df_bias, df_htf, simulate_trades,
                                TRAIN_END, TEST_START, symbol=SYMBOL)
        print_walk_forward(wf, f"({SYMBOL} Limit)")

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
        filename_xlsx = f"trades_limit_{safe_name}.xlsx"
        df_export.to_excel(filename_xlsx, index=False)
        print(f"\n  Excel report saved: {filename_xlsx}")


if __name__ == "__main__":
    run_backtest()
