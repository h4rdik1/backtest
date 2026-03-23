import sys
import re

def rewrite_cisd_file():
    with open('backtest_cisd.py', 'r') as f:
        code = f.read()

    # 1. Config Replacement
    config_str = r"""from lux_fvg import detect_luxalgo_fvgs, detect_liquidity_sweeps

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
"""
    code = re.sub(r'SYMBOL.*?SHOW_PLOTS\s*=\s*False', config_str.strip(), code, flags=re.DOTALL)

    # 2. calc_fvgs and get_liquidity_sweeps
    fvg_funcs = r"""def calc_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    return detect_luxalgo_fvgs(df.copy(), wick_ratio=FVG_WICK_RATIO, avg_body_lookback=FVG_AVG_BODY_LOOKBACK, min_size_mult=FVG_MIN_SIZE_MULT)

def get_liquidity_sweeps(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return detect_liquidity_sweeps(df.copy(), lookback=10)
"""
    code = re.sub(r'def calc_fvgs.*?return out\n\ndef get_liquidity_sweeps.*?return out', fvg_funcs.strip(), code, flags=re.DOTALL)

    # 3. get_hourly_bias
    bias_logic = r"""def get_hourly_bias(df_1h: pd.DataFrame) -> pd.DataFrame:
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
"""
    code = re.sub(r'def get_hourly_bias.*?return bias', bias_logic.strip(), code, flags=re.DOTALL)

    # 4. BE Logic inside simulate_trades
    be_logic = r"""        if in_trade:
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
            continue"""
    code = re.sub(r'        if in_trade:.*?continue', be_logic.strip(), code, flags=re.DOTALL)
    
    # 5. Funnel setup
    funnel_str = r""""Premium/Discount": 0,
        "CISD Triggered": 0,
        "Max SL Logic": 0,
        "Final Trades": 0"""
    code = re.sub(r'"CISD Triggered": 0,\s*"Max SL Logic": 0,\s*"Final Trades": 0', funnel_str, code)
    
    # 6. Premium/Discount and CISD setup
    cisd_logic = r"""        # Phase 4: Premium / Discount Array Validation (1H timeframe)
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
            funnel["Final Trades"] += 1"""
    code = re.sub(r'        # Entry Calcs.*?funnel\["Final Trades"\] \+= 1', cisd_logic.strip(), code, flags=re.DOTALL)

    # 7. Print Stats (RAW String)
    stats_logic = r"""def print_stats(trades: List[Dict], funnel: Dict = None, label: str = "BACKTEST"):
    print("\n" + "=" * 62)
    print(f"  {label} RESULTS")
    print("=" * 62)
    
    if funnel:
        print("\n  FILTER FUNNEL")
        for stage, count in funnel.items():
            print(f"  {stage:<20}: {count}")

    if not trades:
        print(f"\n  No trades completed for {label}.")
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

    print("\n" + "=" * 62)
    print(f"  Total Trades    : {len(df_t)} (Break-Evens: {len(bes)})")
    print(f"  Wins / Losses   : {len(wins)} / {len(losses)}")
    print(f"  Win Rate        : {wr:.1f}%")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Max Win Streak  : {max_win_streak}")
    print(f"  Max Loss Streak : {max_loss_streak}")
    print(f"  Total PnL       : ${df_t['pnl'].sum():.2f}")
    
    print("\n  MONTHLY BREAKDOWN:")
    df_t['Month'] = pd.to_datetime(df_t['Entry Time']).dt.to_period('M')
    monthly = df_t.groupby('Month').agg(
        trades=('Symbol', 'count'),
        wins=('result', lambda x: (x == 'WIN').sum()),
        losses=('result', lambda x: (x == 'LOSS').sum()),
        pnl=('pnl', 'sum')
    )
    for month, data in monthly.iterrows():
        print(f"  {month} | Trades: {data['trades']:<3} | W: {data['wins']:<2} L: {data['losses']:<2} | PnL: ${data['pnl']:<7.2f}")
"""
    code = re.sub(r'def print_stats.*?print\(f"  {stage:<20}: {count}"\)', stats_logic.strip(), code, flags=re.DOTALL)

    with open('backtest_cisd.py', 'w') as f:
        f.write(code)

if __name__ == "__main__":
    rewrite_cisd_file()
