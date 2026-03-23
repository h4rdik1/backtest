import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta, timezone
import os
import argparse
import copy
from typing import List, Dict, Any, Optional, Tuple

from lux_fvg import detect_luxalgo_fvgs, detect_liquidity_sweeps

SETUP                    = "daily"   # "weekly","daily","4h","1h"
SYMBOL                   = "ETH/USDT"
EXCHANGE                 = "binance"
RISK_REWARD              = 3.0
FIXED_SL_USDT            = 25.0
ACCOUNT_SIZE             = 1000.0
DAYS_BACK                = 730       # extend to 2 years for sample size
BIAS_EXPIRY_BARS         = 10
FVG_WICK_RATIO           = 0.36      # LuxAlgo hardcoded value
FVG_AVG_BODY_LOOKBACK    = 10
FVG_MIN_SIZE_MULT        = 0.5
OB_MOVE_MULT             = 1.5
OB_SWING_LOOKBACK        = 5
CISD_BODY_MULT           = 0.5
REQUIRE_HTF_OB_CONFLUENCE= True
AUTO_BREAKEVEN_R         = 1.5       # move SL to BE after 1.5R
PREMIUM_DISCOUNT_BARS    = 20

def fetch_ohlcv_full(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    safe_symbol = symbol.replace("/", "_")
    filename = os.path.join("prev_candles", safe_symbol, f"{timeframe}.csv")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing {filename}.")
    df = pd.read_csv(filename)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    print(f"   {timeframe}: {len(df):,} candles loaded from {filename}")
    return df

def calc_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    return detect_luxalgo_fvgs(df.copy(), wick_ratio=FVG_WICK_RATIO, avg_body_lookback=FVG_AVG_BODY_LOOKBACK, min_size_mult=FVG_MIN_SIZE_MULT)

def get_liquidity_sweeps(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return detect_liquidity_sweeps(df.copy(), lookback=10)

def find_htf_obs(df: pd.DataFrame) -> List[Dict]:
    """
    Fix 6: HTF OB confluence check.
    Bullish OB = last consecutive bearish candles before an aggressive bullish impulse.
    """
    obs = []
    avg_range = (df['high'] - df['low']).rolling(20).mean()
    
    for i in range(SWING_LOOKBACK, len(df) - 1):
        # 1. Check for aggressive bullish impulse at index i
        body = df['close'].iloc[i] - df['open'].iloc[i]
        is_impulse = body > (avg_range.iloc[i] * OB_MOVE_MULT)
        
        if is_impulse:
            # 2. Look backwards for the 'OB' (last bearish candles)
            # Find the last bearish candle before the impulse
            j = i - 1
            if j < 0: continue
            if df['close'].iloc[j] < df['open'].iloc[j]:
                ob_high = df['high'].iloc[j]
                ob_low = df['low'].iloc[j]
                ob_open = df['open'].iloc[j]
                ob_close = df['close'].iloc[j]
                
                # Close of impulse must be above OB high
                if df['close'].iloc[i] > ob_high:
                    # 3. Must be at/near a swing low
                    lows_around = df['low'].iloc[max(0, j-SWING_LOOKBACK):min(len(df), j+SWING_LOOKBACK+1)]
                    if df['low'].iloc[j] == lows_around.min():
                        obs.append({
                            "time": df.index[j],
                            "high": ob_high,
                            "low": ob_low,
                            "median": (ob_open + ob_close) / 2, # Mean threshold = 50% of body
                            "active": True
                        })
    return obs

def get_hourly_bias(df_1h: pd.DataFrame) -> pd.DataFrame:
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
    pending_signal = 0 # 1=bull, -1=bear
    
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
                if signal_age > BIAS_EXPIRY_BARS: pending_signal = 0
        elif pending_signal == -1:
            if mom_bear.iloc[i]:
                active_bias = -1
                pending_signal = 0
            else:
                signal_age += 1
                if signal_age > BIAS_EXPIRY_BARS: pending_signal = 0
                    
        bias.iloc[i] = active_bias
    return bias

def simulate_trades(df_ltf: pd.DataFrame, df_bias: pd.DataFrame, df_htf: pd.DataFrame = None) -> Dict:
    """
    Core simulation with Phase 2 filters.
    """
    bias_series = get_hourly_bias(df_bias)
    bias_aligned = bias_series.reindex(df_ltf.index, method='ffill')
    fvgs_ltf = calc_fvgs(df_ltf)
    avg_range_ltf = (df_ltf['high'] - df_ltf['low']).rolling(20).mean()
    
    # Fix 6: HTF OBs
    htf_obs = find_htf_obs(df_bias) # Use bias TF as HTF for context
    
    trades = []
    account = ACCOUNT_SIZE
    
    # Funnel Tracking
    funnel = {
        "Total Signals": 0,
        "Bias Filter": 0,
        "FVG Exists": 0,
        "FVG First Touch": 0,
        "CISD Pullback": 0,
        "CISD Impulsive": 0,
        "HTF OB Confluence": 0,
        "Premium/Discount": 0,
        "CISD Triggered": 0,
        "Max SL Logic": 0,
        "Final Trades": 0
    }
    
    # State tracking
    in_trade = False
    current_trade = None
    visited_fvgs = set() # Fix 3: Track used FVG timestamps
    
    for i in range(25, len(df_ltf)):
        funnel["Total Signals"] += 1
        row = df_ltf.iloc[i]
        timestamp = df_ltf.index[i]
        
if in_trade:
            if current_trade["direction"] == 1:
                # BE Check
                if not current_trade.get("is_be", False) and row['high'] >= current_trade["be_trigger"]:
                    current_trade["Stop $"] = current_trade["Entry $"]
                    current_trade["is_be"] = True

                if row['low'] <= current_trade["Stop $"]:
                    current_trade["result"] = "BE" if current_trade.get("is_be", False) else "LOSS"
                    current_trade["exit_time"] = timestamp
                    current_trade["pnl"] = 0.0 if current_trade["result"] == "BE" else -FIXED_SL_USDT
                    account += current_trade["pnl"]
                    current_trade["account"] = account
                    trades.append(current_trade)
                    in_trade = False
                elif row['high'] >= current_trade["Target $"]:
                    current_trade["result"] = "WIN"
                    current_trade["exit_time"] = timestamp
                    current_trade["pnl"] = FIXED_SL_USDT * RISK_REWARD
                    account += current_trade["pnl"]
                    current_trade["account"] = account
                    trades.append(current_trade)
                    in_trade = False
            else:
                # BE Check
                if not current_trade.get("is_be", False) and row['low'] <= current_trade["be_trigger"]:
                    current_trade["Stop $"] = current_trade["Entry $"]
                    current_trade["is_be"] = True

                if row['high'] >= current_trade["Stop $"]:
                    current_trade["result"] = "BE" if current_trade.get("is_be", False) else "LOSS"
                    current_trade["exit_time"] = timestamp
                    current_trade["pnl"] = 0.0 if current_trade["result"] == "BE" else -FIXED_SL_USDT
                    account += current_trade["pnl"]
                    current_trade["account"] = account
                    trades.append(current_trade)
                    in_trade = False
                elif row['low'] <= current_trade["Target $"]:
                    current_trade["result"] = "WIN"
                    current_trade["exit_time"] = timestamp
                    current_trade["pnl"] = FIXED_SL_USDT * RISK_REWARD
                    account += current_trade["pnl"]
                    current_trade["account"] = account
                    trades.append(current_trade)
                    in_trade = False
            continue
            
        b = bias_aligned.iloc[i]
        if b == 0: continue
        funnel["Bias Filter"] += 1
        
        is_bull_setup = (b == 1 and fvgs_ltf['fvg_bull'].iloc[i])
        is_bear_setup = (b == -1 and fvgs_ltf['fvg_bear'].iloc[i])
        
        if not (is_bull_setup or is_bear_setup): continue
        funnel["FVG Exists"] += 1
        
        # Fix 3: FVG First Touch Only
        if REQUIRE_PHASE2_FILTERS:
            fvg_id = timestamp 
            if fvg_id in visited_fvgs: continue
            visited_fvgs.add(fvg_id)
        funnel["FVG First Touch"] += 1
        
        # Fix 4: CISD must follow a real pullback
        if REQUIRE_PHASE2_FILTERS:
            # The impulse candle for the FVG is i-1. 
            # The candle BEFORE the impulse is i-2. This is the 'pullback'.
            pullback_row = df_ltf.iloc[i-2]
            is_bull_pb = pullback_row['close'] < pullback_row['open']
            is_bear_pb = pullback_row['close'] > pullback_row['open']
            
            if is_bull_setup and not is_bull_pb: continue
            if is_bear_setup and not is_bear_pb: continue
        funnel["CISD Pullback"] += 1
        
        # Check if CISD is even possible (Basic trigger)
        entry_price = row['close']
        if is_bull_setup:
            if entry_price > df_ltf['high'].iloc[i-3:i].max(): 
                funnel["CISD Triggered"] += 1
        else:
            if entry_price < df_ltf['low'].iloc[i-3:i].min():
                funnel["CISD Triggered"] += 1

        # Fix 5: CISD candle must be impulsive
        if REQUIRE_PHASE2_FILTERS:
            body_size = abs(row['close'] - row['open'])
            if body_size < (avg_range_ltf.iloc[i] * CISD_BODY_MULT): continue
        funnel["CISD Impulsive"] += 1
        
        # Fix 6: HTF OB Confluence
        if REQUIRE_PHASE2_FILTERS and REQUIRE_HTF_OB_CONFLUENCE:
            confluence = False
            for ob in htf_obs:
                if not ob['active']: continue
                margin = avg_range_ltf.iloc[i] * 0.5
                if is_bull_setup:
                    if row['close'] >= (ob['low'] - margin) and row['close'] <= (ob['high'] + margin):
                        confluence = True; break
                else:
                    if row['close'] <= (ob['high'] + margin) and row['close'] >= (ob['low'] - margin):
                        confluence = True; break
            if not confluence: continue
        funnel["HTF OB Confluence"] += 1
        
# Phase 4: Premium / Discount Array Validation (1H timeframe)
        pd_valid = False
        recent_1h = df_bias.loc[:timestamp].tail(PREMIUM_DISCOUNT_BARS)
        if len(recent_1h) >= 2:
            r_high = recent_1h['high'].max()
            r_low = recent_1h['low'].min()
            htf_50_level = r_low + ((r_high - r_low) / 2)
            
            entry = row['close']
            if is_bull_setup:
                if entry <= htf_50_level: pd_valid = True
            else:
                if entry >= htf_50_level: pd_valid = True
                
        if not pd_valid: continue
        funnel["Premium/Discount"] += 1
        
        # Entry Calcs
        if is_bull_setup:
            entry = row['close']
            if entry <= df_ltf['high'].iloc[i-3:i].max(): continue
            
            stop = df_ltf['low'].iloc[i-3:i+1].min()
            risk = entry - stop
            
            if risk <= 0 or risk > FIXED_SL_USDT: continue
            funnel["Max SL Logic"] += 1
            
            target = entry + (risk * RISK_REWARD)
            be_trigger = entry + (risk * AUTO_BREAKEVEN_R)
            in_trade = True
            current_trade = {
                "Symbol": SYMBOL, "Entry Time": timestamp.strftime("%Y-%m-%d %H:%M"),
                "direction": 1, "Entry $": round(entry, 4), "Stop $": round(stop, 4),
                "Target $": round(target, 4), "be_trigger": round(be_trigger, 4), "is_be": False,
                "Entry Type": "Phase 4 Bull CISD", "account": account
            }
            funnel["Final Trades"] += 1
            
        elif is_bear_setup:
            entry = row['close']
            if entry >= df_ltf['low'].iloc[i-3:i].min(): continue
            
            stop = df_ltf['high'].iloc[i-3:i+1].max()
            risk = stop - entry
            
            if risk <= 0 or risk > FIXED_SL_USDT: continue
            funnel["Max SL Logic"] += 1
            
            target = entry - (risk * RISK_REWARD)
            be_trigger = entry - (risk * AUTO_BREAKEVEN_R)
            in_trade = True
            current_trade = {
                "Symbol": SYMBOL, "Entry Time": timestamp.strftime("%Y-%m-%d %H:%M"),
                "direction": -1, "Entry $": round(entry, 4), "Stop $": round(stop, 4),
                "Target $": round(target, 4), "be_trigger": round(be_trigger, 4), "is_be": False,
                "Entry Type": "Phase 4 Bear CISD", "account": account
            }
            funnel["Final Trades"] += 1
            
        elif is_bear_setup:
            entry = row['close']
            # CISD trigger: close below lowest low of last 3 bars
            if entry >= df_ltf['low'].iloc[i-3:i].min(): continue
            
            stop = df_ltf['high'].iloc[i-2:i+1].max()
            risk = stop - entry
            
            # Fix 7: Skip if SL is too wide
            if risk <= 0: continue
            if risk > (entry * 0.05): continue
            funnel["Max SL Logic"] += 1
            
            target = entry - (risk * RISK_REWARD)
            in_trade = True
            current_trade = {
                "Symbol": SYMBOL, "Entry Time": timestamp.strftime("%Y-%m-%d %H:%M"),
                "direction": -1, "Entry $": round(entry, 4), "Stop $": round(stop, 4),
                "Target $": round(target, 4), "Entry Type": "Phase 2 Bearish", "account": account
            }
            funnel["Final Trades"] += 1
                
    return {"trades": trades, "funnel": funnel}

def print_stats(trades: List[Dict], funnel: Dict = None, label: str = "BACKTEST"):
    print("
" + "=" * 62)
    print(f"  {label} RESULTS")
    print("=" * 62)
    
    if funnel:
        print("
  FILTER FUNNEL")
        for stage, count in funnel.items():
            print(f"  {stage:<20}: {count}")

    if not trades:
        print(f"
  No trades completed for {label}.")
        return

    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["result"] == "WIN"]
    losses = df_t[df_t["result"] == "LOSS"]
    bes = df_t[df_t["result"] == "BE"]
    
    total_effective = len(wins) + len(losses)
    wr = (len(wins) / total_effective * 100) if total_effective > 0 else 0
    gross_wins = wins["pnl"].sum()
    gross_losses = abs(losses["pnl"].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    
    results = df_t["result"].tolist()
    max_win_streak = curr_win = 0
    max_loss_streak = curr_loss = 0
    for r in results:
        if r == "WIN":
            curr_win += 1
            curr_loss = 0
            max_win_streak = max(max_win_streak, curr_win)
        elif r == "LOSS":
            curr_loss += 1
            curr_win = 0
            max_loss_streak = max(max_loss_streak, curr_loss)

    print("
" + "=" * 62)
    print(f"  Total Trades    : {len(df_t)} (Break-Evens: {len(bes)})")
    print(f"  Wins / Losses   : {len(wins)} / {len(losses)}")
    print(f"  Win Rate        : {wr:.1f}%")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Max Win Streak  : {max_win_streak}")
    print(f"  Max Loss Streak : {max_loss_streak}")
    print(f"  Total PnL       : ${df_t['pnl'].sum():.2f}")
    
    print("
  MONTHLY BREAKDOWN:")
    df_t['Month'] = pd.to_datetime(df_t['Entry Time']).dt.to_period('M')
    monthly = df_t.groupby('Month').agg(
        trades=('Symbol', 'count'),
        wins=('result', lambda x: (x == 'WIN').sum()),
        losses=('result', lambda x: (x == 'LOSS').sum()),
        pnl=('pnl', 'sum')
    )
    for month, data in monthly.iterrows():
        print(f"  {month} | Trades: {data['trades']:<3} | W: {data['wins']:<2} L: {data['losses']:<2} | PnL: ${data['pnl']:<7.2f}")

    if not trades:
        print(f"\n  No trades completed for {label}.")
        return

    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["result"] == "WIN"]
    losses = df_t[df_t["result"] == "LOSS"]
    
    wr = len(wins) / len(df_t) * 100
    gross_wins = wins["pnl"].sum()
    gross_losses = abs(losses["pnl"].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    
    # Streaks
    results = df_t["result"].apply(lambda x: 1 if x == "WIN" else -1).tolist()
    max_win_streak = 0
    max_loss_streak = 0
    curr_win = 0
    curr_loss = 0
    for r in results:
        if r == 1:
            curr_win += 1
            curr_loss = 0
            max_win_streak = max(max_win_streak, curr_win)
        else:
            curr_loss += 1
            curr_win = 0
            max_loss_streak = max(max_loss_streak, curr_loss)

    print("\n" + "=" * 62)
    print(f"  {label} RESULTS")
    print("=" * 62)
    print(f"  Total Trades    : {len(df_t)}")
    print(f"  Wins / Losses   : {len(wins)} / {len(losses)}")
    print(f"  Win Rate        : {wr:.1f}%")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Max Win Streak  : {max_win_streak}")
    print(f"  Max Loss Streak : {max_loss_streak}")
    print(f"  Total PnL       : ${df_t['pnl'].sum():.2f}")
    
    if funnel:
        print("\n  FILTER FUNNEL")
        for stage, count in funnel.items():
            print(f"  {stage:<20}: {count}")

def run_backtest():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--alignment", choices=["daily", "weekly"], default="daily")
    args = parser.parse_args()
    
    global SYMBOL, REQUIRE_HTF_OB_CONFLUENCE
    SYMBOL = args.symbol
    
    print("=" * 62)
    print(f"  TTrades Scalping Model v2.0 - {SYMBOL}")
    print(f"  Alignment: {args.alignment.upper()}")
    print("=" * 62)
    
    # Define TFs based on alignment
    if args.alignment == "daily":
        tf_context, tf_bias, tf_ltf = "1d", "1h", "5m"
    else:
        tf_context, tf_bias, tf_ltf = "1w", "4h", "15m"
        
    df_htf = fetch_ohlcv_full(SYMBOL, tf_context, DAYS_BACK)
    df_bias = fetch_ohlcv_full(SYMBOL, tf_bias, DAYS_BACK)
    df_ltf = fetch_ohlcv_full(SYMBOL, tf_ltf, DAYS_BACK)
    
    # Run A/B Check
    print("\n[STEP 1] Running WITHOUT new filters...")
    orig_f6 = REQUIRE_HTF_OB_CONFLUENCE
    REQUIRE_PHASE2_FILTERS = False
    REQUIRE_HTF_OB_CONFLUENCE = False
    results_a = simulate_trades(df_ltf, df_bias, df_htf)
    print_stats(results_a["trades"], results_a["funnel"], "BASE (NO FILTERS)")
    
    print("\n[STEP 2] Running WITH new Phase 2 filters...")
    REQUIRE_PHASE2_FILTERS = True
    REQUIRE_HTF_OB_CONFLUENCE = orig_f6
    results_b = simulate_trades(df_ltf, df_bias, df_htf)
    print_stats(results_b["trades"], results_b["funnel"], "PHASE 2 ENHANCED")
    
    if results_b["trades"]:
        df_t = pd.DataFrame(results_b["trades"])
        df_t.to_csv("trade_log_v5.csv", index=False)
        print("\nFinal enhanced log saved to trade_log_v5.csv")

if __name__ == "__main__":
    run_backtest()
