import sys

def rewrite_limit_file():
    with open('backtest_limit.py', 'r') as f:
        code = f.read()

    # 1. Config Replacement
    config_str = """
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
"""
    import re
    code = re.sub(r'SYMBOL.*?SHOW_PLOTS\s*=\s*False', config_str.strip(), code, flags=re.DOTALL)

    # 2. calc_fvgs and get_liquidity_sweeps
    fvg_funcs = """
def calc_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    # Phase 4: Import LuxAlgo Pine Script explicit logic
    return detect_luxalgo_fvgs(df.copy(), wick_ratio=FVG_WICK_RATIO, avg_body_lookback=FVG_AVG_BODY_LOOKBACK, min_size_mult=FVG_MIN_SIZE_MULT)

def get_liquidity_sweeps(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    # Phase 4: Valid sweeps
    return detect_liquidity_sweeps(df.copy(), lookback=10)
"""
    code = re.sub(r'def calc_fvgs.*?return out\n\ndef get_liquidity_sweeps.*?return out', fvg_funcs.strip(), code, flags=re.DOTALL)

    # 3. get_hourly_bias
    bias_logic = """
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
        # Check newly formed signals
        if bull_signal.iloc[i]:
            pending_signal = 1
            signal_age = 0
            active_bias = 0
        elif bear_signal.iloc[i]:
            pending_signal = -1
            signal_age = 0
            active_bias = 0
            
        # Confirm pending signal
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
    
    # 4. In simulate_trades: Premium/Discount and Limit setup
    limit_logic = """
        # Phase 4: Premium / Discount Array Validation (1H timeframe)
        # Range = swing high to swing low of last 20 bars on 1H
        pd_valid = False
        recent_1h = df_bias.loc[:timestamp].tail(PREMIUM_DISCOUNT_BARS)
        if len(recent_1h) >= 2:
            r_high = recent_1h['high'].max()
            r_low = recent_1h['low'].min()
            htf_50_level = r_low + ((r_high - r_low) / 2)
            
            fvg_top = fvgs_ltf['fvg_bull_top'].iloc[i] if is_bull_setup else fvgs_ltf['fvg_bear_top'].iloc[i]
            fvg_btm = fvgs_ltf['fvg_bull_btm'].iloc[i] if is_bull_setup else fvgs_ltf['fvg_bear_btm'].iloc[i]
            if pd.isna(fvg_top) or pd.isna(fvg_btm): continue
            
            entry = (fvg_top + fvg_btm) / 2
            
            if is_bull_setup:
                if entry <= htf_50_level: pd_valid = True # Discount
            else:
                if entry >= htf_50_level: pd_valid = True # Premium
                
        if not pd_valid: continue
        funnel["Premium/Discount"] += 1
        
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
        
        # Setup Limit Order Profile
        if is_bull_setup:
            stop = fvg_btm - (avg_range_ltf.iloc[i] * 0.1) # strictly below FVG gap bottom
            
            risk = entry - stop
            if risk <= 0 or risk > (entry * 0.05): continue 
            
            funnel["Max SL Logic"] += 1
            target = entry + (risk * RISK_REWARD)
            be_trigger = entry + (risk * AUTO_BREAKEVEN_R)
            
            active_limit = {
                'index': i,
                'direction': 1,
                'entry': entry,
                'trade_template': {
                    "Symbol": SYMBOL, "direction": 1, "Entry $": round(entry, 4), 
                    "Stop $": round(stop, 4), "Target $": round(target, 4), 
                    "be_trigger": round(be_trigger, 4), "is_be": False,
                    "Entry Type": "Phase 4 Bull Limit (50%)", "account": account
                }
            }
            funnel["CISD Triggered"] += 1 # Repurposed as Valid Limit Set
            
        elif is_bear_setup:
            stop = fvg_top + (avg_range_ltf.iloc[i] * 0.1)
            
            risk = stop - entry
            if risk <= 0 or risk > (entry * 0.05): continue
            
            funnel["Max SL Logic"] += 1
            target = entry - (risk * RISK_REWARD)
            be_trigger = entry - (risk * AUTO_BREAKEVEN_R)
            
            active_limit = {
                'index': i,
                'direction': -1,
                'entry': entry,
                'trade_template': {
                    "Symbol": SYMBOL, "direction": -1, "Entry $": round(entry, 4), 
                    "Stop $": round(stop, 4), "Target $": round(target, 4), 
                    "be_trigger": round(be_trigger, 4), "is_be": False,
                    "Entry Type": "Phase 4 Bear Limit (50%)", "account": account
                }
            }
            funnel["CISD Triggered"] += 1 
"""
    code = re.sub(r'# Phase 3: Premium / Discount Array Validation.*?funnel\["CISD Triggered"\] \+= 1 # Repurposed as Valid Limit Set', limit_logic.strip(), code, flags=re.DOTALL)

    # 5. Month Grouping Stats
    stats_logic = """
def print_stats(trades: List[Dict], funnel: Dict = None, label: str = "BACKTEST"):
    print("\\n" + "=" * 62)
    print(f"  {label} RESULTS")
    print("=" * 62)
    
    if funnel:
        print("\\n  FILTER FUNNEL")
        for stage, count in funnel.items():
            print(f"  {stage:<20}: {count}")

    if not trades:
        print(f"\\n  No trades completed for {label}.")
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
    
    # Streaks
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

    print("\\n" + "=" * 62)
    print(f"  Total Trades    : {len(df_t)} (Break-Evens: {len(bes)})")
    print(f"  Wins / Losses   : {len(wins)} / {len(losses)}")
    print(f"  Win Rate        : {wr:.1f}%")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Max Win Streak  : {max_win_streak}")
    print(f"  Max Loss Streak : {max_loss_streak}")
    print(f"  Total PnL       : ${df_t['pnl'].sum():.2f}")
    
    # Phase 4: Monthly breakdown
    print("\\n  MONTHLY BREAKDOWN:")
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

    with open('backtest_limit.py', 'w') as f:
        f.write(code)

if __name__ == "__main__":
    rewrite_limit_file()
