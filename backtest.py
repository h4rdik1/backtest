import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta, timezone
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

SYMBOL = "SOL/USDT"
RISK_REWARD = 3.0
FIXED_SL_USDT = 25.0
ACCOUNT_SIZE = 1000.0
DAYS_BACK = 365
SHOW_PLOTS = False

# --- Refinement Configuration ---
BIAS_EXPIRY_BARS          = 10     # Fix 1: bars before active bias resets
FVG_SIZE_MULT             = 0.5    # Fix 2: FVG must be >= this x avg range
CISD_BODY_MULT            = 0.5    # Fix 5: CISD candle body minimum size
REQUIRE_HTF_OB_CONFLUENCE = True   # Fix 6: skip trades not near HTF OB
OB_MOVE_MULT              = 1.5    # Fix 6: impulse size for OB detection
SWING_LOOKBACK            = 5      # Fix 6: candles each side for swing highs/lows
# --------------------------------

def fetch_ohlcv_full(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    safe_symbol = symbol.replace("/", "_")
    filename = os.path.join("prev_candles", safe_symbol, f"{timeframe}.csv")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing {filename}. Run fetch_candles.py first!")
    df = pd.read_csv(filename)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    print(f"   {timeframe}: {len(df):,} candles loaded from {filename}")
    return df

def calc_fvgs(df: pd.DataFrame, min_body_perc: float = 0.36) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    body = (df['close'] - df['open']).abs()
    mx = np.maximum(df['close'], df['open'])
    mn = np.minimum(df['close'], df['open'])
    L_body = ((df['high'] - mx) < body * min_body_perc) & ((mn - df['low']) < body * min_body_perc)
    meanBody = body.rolling(5).mean()
    L_bodyUP = (body > meanBody.shift(1)) & L_body & (df['close'] > df['open'])
    L_bodyDN = (body > meanBody.shift(1)) & L_body & (df['close'] < df['open'])
    # Fix 2: FVG size filter mask
    # Calculate 20-period average candle range
    avg_range = (df['high'] - df['low']).rolling(20).mean().shift(1)
    
    # Gap sizes
    gap_bull = df['low'] - df['high'].shift(2)
    gap_bear = df['low'].shift(2) - df['high']
    
    fvg_bull_raw = L_bodyUP.shift(1) & (gap_bull > 0)
    fvg_bear_raw = L_bodyDN.shift(1) & (gap_bear > 0)
    
    # Apply FVG_SIZE_MULT filter
    out['fvg_bull'] = fvg_bull_raw & (gap_bull >= (avg_range * FVG_SIZE_MULT))
    out['fvg_bear'] = fvg_bear_raw & (gap_bear >= (avg_range * FVG_SIZE_MULT))
    
    for col in ['fvg_bull', 'fvg_bear']:
        out[col] = out[col].fillna(False)
    out['fvg_bull_top'] = np.where(out['fvg_bull'], df['low'], np.nan)
    out['fvg_bull_btm'] = np.where(out['fvg_bull'], df['high'].shift(2), np.nan)
    out['fvg_bear_top'] = np.where(out['fvg_bear'], df['low'].shift(2), np.nan)
    out['fvg_bear_btm'] = np.where(out['fvg_bear'], df['high'], np.nan)
    return out

# Fix 6: HTF OB Confluence
def find_htf_obs(df: pd.DataFrame) -> list:
    obs = []
    avg_range = (df['high'] - df['low']).rolling(20).mean().shift(1)
    
    for i in range(20 + SWING_LOOKBACK, len(df) - SWING_LOOKBACK):
        # Bullish OB Check
        bull_body = df['close'].iloc[i] - df['open'].iloc[i]
        if bull_body > (avg_range.iloc[i] * OB_MOVE_MULT):
            if df['close'].iloc[i-1] < df['open'].iloc[i-1]:
                ob_idx = i - 1
                if df['close'].iloc[i] > df['high'].iloc[ob_idx]:
                    # Swing check
                    local_min = df['low'].iloc[ob_idx - SWING_LOOKBACK : ob_idx + SWING_LOOKBACK + 1].min()
                    if df['low'].iloc[ob_idx] == local_min:
                        ob_body_hi = df['open'].iloc[ob_idx]
                        ob_body_lo = df['close'].iloc[ob_idx]
                        obs.append({
                            "dir": 1,
                            "time": df.index[i],
                            "top": ob_body_hi,
                            "bottom": ob_body_lo,
                            "mean": ob_body_lo + (ob_body_hi - ob_body_lo) / 2,
                            "invalidated": False
                        })
                        
        # Bearish OB Check
        bear_body = df['open'].iloc[i] - df['close'].iloc[i]
        if bear_body > (avg_range.iloc[i] * OB_MOVE_MULT):
            if df['close'].iloc[i-1] > df['open'].iloc[i-1]:
                ob_idx = i - 1
                if df['close'].iloc[i] < df['low'].iloc[ob_idx]:
                    # Swing check
                    local_max = df['high'].iloc[ob_idx - SWING_LOOKBACK : ob_idx + SWING_LOOKBACK + 1].min() # Wait, max() for swing high
                    if df['high'].iloc[ob_idx] == df['high'].iloc[ob_idx - SWING_LOOKBACK : ob_idx + SWING_LOOKBACK + 1].max():
                        ob_body_hi = df['close'].iloc[ob_idx]
                        ob_body_lo = df['open'].iloc[ob_idx]
                        obs.append({
                            "dir": -1,
                            "time": df.index[i],
                            "top": ob_body_hi,
                            "bottom": ob_body_lo,
                            "mean": ob_body_lo + (ob_body_hi - ob_body_lo) / 2,
                            "invalidated": False
                        })
    return obs

def get_liquidity_sweeps(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    highs = df['high'].rolling(n*2+1, center=True).max()
    lows = df['low'].rolling(n*2+1, center=True).min()
    pivot_highs = (df['high'] == highs)
    pivot_lows = (df['low'] == lows)
    
    out['pivot_high'] = np.where(pivot_highs, df['high'], np.nan)
    out['pivot_low'] = np.where(pivot_lows, df['low'], np.nan)
    out['pivot_high'] = out['pivot_high'].ffill()
    out['pivot_low'] = out['pivot_low'].ffill()
    
    out['sweep_bull'] = (df['low'] < out['pivot_low'].shift(1)) & (df['close'] > out['pivot_low'].shift(1))
    out['sweep_bear'] = (df['high'] > out['pivot_high'].shift(1)) & (df['close'] < out['pivot_high'].shift(1))
    return out

def get_hourly_bias(df_1h: pd.DataFrame) -> pd.DataFrame:
    fvgs = calc_fvgs(df_1h)
    sweeps = get_liquidity_sweeps(df_1h)
    bias = pd.Series(0, index=df_1h.index)
    
    bull_signal = fvgs['fvg_bull'] | sweeps['sweep_bull']
    bear_signal = fvgs['fvg_bear'] | sweeps['sweep_bear']
    
    active_bull = bull_signal.rolling(3).max().fillna(0) > 0
    active_bear = bear_signal.rolling(3).max().fillna(0) > 0
    
    bull_close = df_1h['close'] > df_1h['open']
    bear_close = df_1h['close'] < df_1h['open']
    
    # Fix 1: Bias expires if no active confirmation within BIAS_EXPIRY_BARS
    for i in range(len(df_1h)):
        # Calculate trailing window for signals (Sweep or FVG)
        start_idx = max(0, i - BIAS_EXPIRY_BARS)
        recent_bull_signal = bull_signal.iloc[start_idx:i+1].any()
        recent_bear_signal = bear_signal.iloc[start_idx:i+1].any()
        
        # Require both the signal to be within expiry window, and momentum closure
        if recent_bull_signal and bull_close.iloc[i]:
            # Also ensure the signal happened within the last 3 bars of the moment for initial active state
            # Or if already active, allow it to persist up to BIAS_EXPIRY_BARS from the signal
            if active_bull.iloc[i]:
                bias.iloc[i] = 1
        elif recent_bear_signal and bear_close.iloc[i]:
            if active_bear.iloc[i]:
                bias.iloc[i] = -1

    # Carry forward active bias unless it expires
    active_b = 0
    signal_idx = 0
    
    # Better implementation of Expiry:
    bias_series = pd.Series(0, index=df_1h.index)
    for i in range(len(df_1h)):
        # Check for new signals
        if bull_signal.iloc[i]:
            signal_idx = i
        elif bear_signal.iloc[i]:
            signal_idx = i
            
        # Check for momentum close activating the bias (within 3 bars of signal per original logic)
        if bull_close.iloc[i] and active_bull.iloc[i]:
            active_b = 1
        elif bear_close.iloc[i] and active_bear.iloc[i]:
            active_b = -1
            
        # Check for expiry: reset if bars since last signal > BIAS_EXPIRY_BARS
        if (i - signal_idx) > BIAS_EXPIRY_BARS:
            active_b = 0
            
        bias_series.iloc[i] = active_b
        
    return bias_series

def simulate_trades(df_entry: pd.DataFrame, df_bias: pd.DataFrame, entry_tf: str = "15m") -> list:
    bias_higher = get_hourly_bias(df_bias)
    bias_aligned = bias_higher.reindex(df_entry.index, method='ffill')
    fvgs_entry = calc_fvgs(df_entry)
    
    # Fix 6: Get active HTF OBs
    htf_obs = find_htf_obs(df_bias) if REQUIRE_HTF_OB_CONFLUENCE else []
    
    # Precalculate for Fix 5 (CISD Impulsive check)
    avg_range_entry = (df_entry['high'] - df_entry['low']).rolling(20).mean().shift(1)
    
    trades = []
    account = ACCOUNT_SIZE
    eq_curve = [account]
    
    in_trade = False
    current_trade = None
    
    # Funnel tracking tracking
    funnel = {
        "Initial Bias & FVG Taps": 0,
        "Pullback Failed (Fix 4)": 0,
        "Non-Impulsive CISD (Fix 5)": 0,
        "No HTF OB (Fix 6)": 0,
        "Wide SL (Fix 7)": 0,
        "Valid Entries": 0
    }
    
    # Fix 3: Keep track of FVGs
    active_bull_fvgs = []
    active_bear_fvgs = []
    
    for i in range(10, len(df_entry)):
        row = df_entry.iloc[i]
        timestamp = df_entry.index[i]
        prev_row = df_entry.iloc[i-1]
        
        # Track newly formed FVGs
        if fvgs_entry['fvg_bull'].iloc[i]:
            active_bull_fvgs.append({
                "top": fvgs_entry['fvg_bull_top'].iloc[i],
                "btm": fvgs_entry['fvg_bull_btm'].iloc[i],
                "used": False
            })
        if fvgs_entry['fvg_bear'].iloc[i]:
            active_bear_fvgs.append({
                "top": fvgs_entry['fvg_bear_top'].iloc[i],
                "btm": fvgs_entry['fvg_bear_btm'].iloc[i],
                "used": False
            })
        
        if in_trade:
            if current_trade["direction"] == 1:
                if row['low'] <= current_trade["Stop $"]:
                    current_trade["Result"] = "LOSS"
                    current_trade["Exit Time"] = timestamp.strftime("%Y-%m-%d %H:%M")
                    current_trade["PnL"] = -FIXED_SL_USDT
                    account += current_trade["PnL"]
                    current_trade["Account"] = account
                    current_trade["exit_ts"] = timestamp
                    trades.append(current_trade)
                    eq_curve.append(account)
                    in_trade = False
                elif row['high'] >= current_trade["Target $"]:
                    current_trade["Result"] = "WIN"
                    current_trade["Exit Time"] = timestamp.strftime("%Y-%m-%d %H:%M")
                    current_trade["PnL"] = FIXED_SL_USDT * RISK_REWARD
                    account += current_trade["PnL"]
                    current_trade["Account"] = account
                    current_trade["exit_ts"] = timestamp
                    trades.append(current_trade)
                    eq_curve.append(account)
                    in_trade = False
            else:
                if row['high'] >= current_trade["Stop $"]:
                    current_trade["Result"] = "LOSS"
                    current_trade["Exit Time"] = timestamp.strftime("%Y-%m-%d %H:%M")
                    current_trade["PnL"] = -FIXED_SL_USDT
                    account += current_trade["PnL"]
                    current_trade["Account"] = account
                    current_trade["exit_ts"] = timestamp
                    trades.append(current_trade)
                    eq_curve.append(account)
                    in_trade = False
                elif row['low'] <= current_trade["Target $"]:
                    current_trade["Result"] = "WIN"
                    current_trade["Exit Time"] = timestamp.strftime("%Y-%m-%d %H:%M")
                    current_trade["PnL"] = FIXED_SL_USDT * RISK_REWARD
                    account += current_trade["PnL"]
                    current_trade["Account"] = account
                    current_trade["exit_ts"] = timestamp
                    trades.append(current_trade)
                    eq_curve.append(account)
                    in_trade = False
            continue
            
        b = bias_aligned.iloc[i]
        if b == 0: continue
        
        # Bullish Bias Execution
        if b == 1:
            # Check active FVGs (Fix 3: First touch tracking)
            in_bull_fvg = False
            for f in active_bull_fvgs:
                if not f["used"] and row['low'] <= f["top"] and row['high'] >= f["btm"]:
                    in_bull_fvg = True
                    f["used"] = True # Mark used on touch
                    break
                    
            if in_bull_fvg:
                funnel["Initial Bias & FVG Taps"] += 1
                
                # Check CISD
                if row['close'] > df_entry['high'].iloc[i-3:i].max():
                    # Fix 4: Pullback check - previous candle must be bearish (close < open)
                    if prev_row['close'] >= prev_row['open']:
                        funnel["Pullback Failed (Fix 4)"] += 1
                        continue
                        
                    # Fix 5: Impulsive CISD check
                    cisd_body = row['close'] - row['open']
                    if cisd_body < (avg_range_entry.iloc[i] * CISD_BODY_MULT):
                        funnel["Non-Impulsive CISD (Fix 5)"] += 1
                        continue
                        
                    # Fix 6: HTF OB Confluence
                    if REQUIRE_HTF_OB_CONFLUENCE:
                        has_ob = False
                        entry_price = row['close']
                        for ob in htf_obs:
                            if ob["dir"] == 1 and not ob["invalidated"] and ob["time"] <= timestamp:
                                # Check proximity (within 0.5x avg range of zone)
                                dist_to_zone = min(abs(entry_price - ob["top"]), abs(entry_price - ob["bottom"]))
                                if dist_to_zone <= (avg_range_entry.iloc[i] * 0.5) or (ob["bottom"] <= entry_price <= ob["top"]):
                                    has_ob = True
                                    break
                        if not has_ob:
                            funnel["No HTF OB (Fix 6)"] += 1
                            continue
                    
                    entry = row['close']
                    stop = df_entry['low'].iloc[i-2:i+1].min()
                    risk = entry - stop
                    if risk <= 0: continue
                    
                    # Fix 7: SL Width sanity check
                    if risk > entry * 0.10 or (FIXED_SL_USDT / risk * entry > ACCOUNT_SIZE * 0.5):
                        funnel["Wide SL (Fix 7)"] += 1
                        continue
                        
                    funnel["Valid Entries"] += 1
                    target = entry + (risk * RISK_REWARD)
                    in_trade = True
                    current_trade = {
                        "Symbol": SYMBOL,
                        "Entry Time": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "direction": 1,
                        "Entry $": round(entry, 4),
                        "Stop $": round(stop, 4),
                        "Target $": round(target, 4),
                        "Entry Type": f"{entry_tf.upper()} Bullish FVG + CISD",
                        "entry_ts": timestamp
                    }
                    
        # Bearish Bias Execution
        elif b == -1:
            # Check active FVGs (Fix 3: First touch tracking)
            in_bear_fvg = False
            for f in active_bear_fvgs:
                if not f["used"] and row['high'] >= f["btm"] and row['low'] <= f["top"]:
                    in_bear_fvg = True
                    f["used"] = True # Mark used on touch
                    break
                    
            if in_bear_fvg:
                funnel["Initial Bias & FVG Taps"] += 1
                
                if row['close'] < df_entry['low'].iloc[i-3:i].min():
                    # Fix 4: Pullback check - previous candle must be bullish
                    if prev_row['close'] <= prev_row['open']:
                        funnel["Pullback Failed (Fix 4)"] += 1
                        continue
                        
                    # Fix 5: Impulsive CISD check
                    cisd_body = row['open'] - row['close']
                    if cisd_body < (avg_range_entry.iloc[i] * CISD_BODY_MULT):
                        funnel["Non-Impulsive CISD (Fix 5)"] += 1
                        continue
                        
                    # Fix 6: HTF OB Confluence
                    if REQUIRE_HTF_OB_CONFLUENCE:
                        has_ob = False
                        entry_price = row['close']
                        for ob in htf_obs:
                            if ob["dir"] == -1 and not ob["invalidated"] and ob["time"] <= timestamp:
                                dist_to_zone = min(abs(entry_price - ob["top"]), abs(entry_price - ob["bottom"]))
                                if dist_to_zone <= (avg_range_entry.iloc[i] * 0.5) or (ob["bottom"] <= entry_price <= ob["top"]):
                                    has_ob = True
                                    break
                        if not has_ob:
                            funnel["No HTF OB (Fix 6)"] += 1
                            continue
                    
                    entry = row['close']
                    stop = df_entry['high'].iloc[i-2:i+1].max()
                    risk = stop - entry
                    if risk <= 0: continue
                    
                    # Fix 7: SL Width sanity check
                    if risk > entry * 0.10 or (FIXED_SL_USDT / risk * entry > ACCOUNT_SIZE * 0.5):
                        funnel["Wide SL (Fix 7)"] += 1
                        continue
                        
                    funnel["Valid Entries"] += 1
                    target = entry - (risk * RISK_REWARD)
                    in_trade = True
                    current_trade = {
                        "Symbol": SYMBOL,
                        "Entry Time": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "direction": -1,
                        "Entry $": round(entry, 4),
                        "Stop $": round(stop, 4),
                        "Target $": round(target, 4),
                        "Entry Type": f"{entry_tf.upper()} Bearish FVG + CISD",
                        "entry_ts": timestamp
                    }
                
    return trades, eq_curve, funnel

def plot_equity_curve(trades, eq_curve, entry_tf, bias_tf, ctx_tf):
    if not trades: return
    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["Result"] == "WIN"]
    losses = df_t[df_t["Result"] == "LOSS"]
    wr = len(wins) / len(df_t) * 100
    breakeven = 1 / (1 + RISK_REWARD) * 100
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#161616")
        ax.tick_params(colors="#aaa")
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_color("#aaa")
        ax.yaxis.label.set_color("#ccc")
        ax.xaxis.label.set_color("#ccc")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    fig.suptitle(
        f"TTrades Scalping Model v5.0 - {SYMBOL}  ({DAYS_BACK} days)\n"
        f"Alignment: {ctx_tf} -> {bias_tf} -> {entry_tf}  |  Win Rate: {wr:.1f}%\n"
        f"Break-even: {breakeven:.0f}%",
        fontsize=11, fontweight="bold", color="white"
    )

    ax1 = axes[0]
    ax1.plot(eq_curve, color="#00C896", linewidth=2)
    ax1.axhline(ACCOUNT_SIZE, color="#555", linestyle="--", linewidth=0.8)
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Account (USDT)")
    ax1.grid(alpha=0.08, color="white")

    ax2 = axes[1]
    colors = ["#00C896" if r == "WIN" else "#FF4C4C" for r in df_t["Result"]]
    ax2.bar(range(len(df_t)), df_t["PnL"], color=colors, edgecolor="none")
    ax2.axhline(0, color="#555", linewidth=0.8)
    ax2.set_title("PnL Per Trade")
    ax2.set_ylabel("PnL (USDT)")
    ax2.grid(alpha=0.08, color="white")

    ax3 = axes[2]
    cum = df_t["PnL"].cumsum()
    ax3.plot(cum.values, color="#4CA3FF", linewidth=2)
    ax3.axhline(0, color="#555", linestyle="--", linewidth=0.8)
    ax3.set_title("Cumulative PnL")
    ax3.set_ylabel("USDT")
    ax3.set_xlabel("Trade #")
    ax3.grid(alpha=0.08, color="white")

    plt.tight_layout()
    plt.savefig("backtest_results_v5.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved: backtest_results_v5.png")


def run_backtest(entry_tf: str) -> None:
    if entry_tf == "15m":
        bias_tf = "4h"
        ctx_tf = "1w"
    elif entry_tf == "5m":
        bias_tf = "1h"
        ctx_tf = "1d"
    else:
        bias_tf = "1h"
        ctx_tf = "1d"

    print("=" * 62)
    print(f"  TTrades Scalping Model Backtester v5.0")
    print(f"  Alignment: {ctx_tf} -> {bias_tf} -> {entry_tf}")
    print(f"  RR: 1:{int(RISK_REWARD)}  |  SL: ${FIXED_SL_USDT}")
    print("=" * 62)
    
    df_bias = fetch_ohlcv_full(SYMBOL, bias_tf, DAYS_BACK)
    df_entry = fetch_ohlcv_full(SYMBOL, entry_tf, DAYS_BACK)
    
    trades, eq_curve, funnel = simulate_trades(df_entry, df_bias, entry_tf)
    
    # Print Funnel
    print("\n" + "=" * 62)
    print("  TRADE FILTER FUNNEL")
    print("=" * 62)
    for k, v in funnel.items():
        print(f"  {k:.<30}: {v}")
    
    if not trades:
        print("\nNo trades found.")
        return
        
    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["Result"] == "WIN"]
    losses = df_t[df_t["Result"] == "LOSS"]
    wr = len(wins) / len(df_t) * 100
    
    # Calculate Streaks & Profit Factor
    max_win_streak = win_streak = 0
    max_loss_streak = loss_streak = 0
    
    for r in df_t["Result"]:
        if r == "WIN":
            win_streak += 1
            loss_streak = 0
            max_win_streak = max(max_win_streak, win_streak)
        else:
            loss_streak += 1
            win_streak = 0
            max_loss_streak = max(max_loss_streak, loss_streak)
            
    gross_wins = wins["PnL"].sum() if len(wins) > 0 else 0
    gross_losses = abs(losses["PnL"].sum()) if len(losses) > 0 else 0
    pf = (gross_wins / gross_losses) if gross_losses > 0 else 0
    
    print("\n" + "=" * 62)
    print("  RESULTS")
    print("=" * 62)
    print(f"  Total Trades    : {len(df_t)}")
    print(f"  Wins / Losses   : {len(wins)} / {len(losses)}")
    print(f"  Win Rate        : {wr:.1f}%")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Max Win Streak  : {max_win_streak}")
    print(f"  Max Loss Streak : {max_loss_streak}")
    print(f"  Total PnL       : ${df_t['PnL'].sum():.2f}")
    
    # Save log
    df_t.drop(["direction", "entry_ts", "exit_ts"], axis=1, inplace=True, errors="ignore")
    out_csv = f"trade_log_v5_{entry_tf}.csv"
    df_t.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    
    plot_equity_curve(trades, eq_curve, entry_tf, bias_tf, ctx_tf)


def parse_args():
    parser = argparse.ArgumentParser(description="TTrades Scalping Backtester v5.0")
    parser.add_argument("--symbol", default=SYMBOL, help="Trading symbol e.g BTC/USDT")
    parser.add_argument("--days", type=int, default=DAYS_BACK, help="Lookback days")
    parser.add_argument("--entry-tf", default="15m", choices=["5m", "15m"], help="Entry timeframe")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    SYMBOL = args.symbol
    DAYS_BACK = args.days
    run_backtest(args.entry_tf)
