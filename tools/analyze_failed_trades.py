import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set PYTHONPATH to project root
sys.path.insert(0, 'e:/backtest')

from models.backtest_cisd import simulate_trades, fetch_ohlcv_full, detect_sfp_pattern
from core.config import DAYS_BACK, ALIGNMENTS, ACTIVE_CHAIN

# Set artifact folder path for saving images
ARTIFACT_DIR = r"C:\Users\hardi\.gemini\antigravity\brain\69fd3186-4cee-4d0c-b13d-731efec73aa0"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def generate_failed_trade_plots():
    symbol = "BTC/USDT"
    chain = "intraday"
    
    tf_context = ALIGNMENTS[chain]["context"] # 1d
    tf_bias = ALIGNMENTS[chain]["bias"]       # 1h
    tf_entry = ALIGNMENTS[chain]["entry"]     # 5m
    
    print(f"Loading data for {symbol}...")
    df_htf = fetch_ohlcv_full(symbol, tf_context, DAYS_BACK)
    df_bias = fetch_ohlcv_full(symbol, tf_bias, DAYS_BACK)
    df_ltf = fetch_ohlcv_full(symbol, tf_entry, DAYS_BACK)
    
    try:
        df_daily = fetch_ohlcv_full(symbol, "1d", DAYS_BACK)
    except FileNotFoundError:
        df_daily = None

    print("Running backtest simulation...")
    # Simulate trades with premium/discount disabled first to find the trades we actually took
    results = simulate_trades(df_ltf, df_bias, df_htf, symbol=symbol, use_pd_array=False, df_daily=df_daily)
    trades = results["trades"]
    
    # Filter for losses
    failed_trades = [t for t in trades if t["result"] == "LOSS"]
    print(f"Total trades: {len(trades)}, Failed trades: {len(failed_trades)}")
    
    # Let's inspect a few failed trades
    # We will pick 3 representative failed trades:
    # 1. A Long that failed in April 2025
    # 2. A Short that failed in June/July 2025
    # 3. A Long/Short that failed in Sept/Oct 2025
    
    selected_trades = []
    for t in failed_trades:
        entry_time = pd.to_datetime(t["Entry Time"])
        if entry_time.year == 2025:
            if entry_time.month == 4 and len(selected_trades) == 0:
                selected_trades.append(t)
            elif entry_time.month == 6 and len(selected_trades) == 1:
                selected_trades.append(t)
            elif entry_time.month == 9 and len(selected_trades) == 2:
                selected_trades.append(t)
                
    if len(selected_trades) < 3:
        # Fallback to first 3 failed trades
        selected_trades = failed_trades[:3]
        
    for idx, t in enumerate(selected_trades):
        entry_time_str = t["Entry Time"]
        entry_time = pd.to_datetime(entry_time_str, utc=True)
        exit_time = pd.to_datetime(t["exit_time"], utc=True)
        direction = t["direction"] # 1=Long, -1=Short
        entry_px = t["Entry $"]
        stop_px = t["Stop $"]
        target_px = t["Target $"]
        reason = t["reason"]
        
        print(f"\nAnalyzing trade {idx+1}: {entry_time_str} | Direction: {'LONG' if direction == 1 else 'SHORT'}")
        
        # 1. Calculate HTF (Daily) range at entry_time
        htf_slice = df_htf[df_htf.index <= entry_time]
        htf_high, htf_low, htf_mid = np.nan, np.nan, np.nan
        if len(htf_slice) >= 40:
            recent_htf = htf_slice.tail(40)
            htf_high = recent_htf['high'].max()
            htf_low = recent_htf['low'].min()
            htf_mid = (htf_high + htf_low) / 2
            
        # 2. Calculate Local LTF (5m) range at entry_time
        # Let's find index in ltf
        ltf_idx_list = df_ltf.index.get_indexer([entry_time], method='pad')
        ltf_i = ltf_idx_list[0]
        
        lookback = min(200, ltf_i)
        ltf_high_val = df_ltf['high'].iloc[max(0, ltf_i - lookback):ltf_i].max()
        ltf_low_val = df_ltf['low'].iloc[max(0, ltf_i - lookback):ltf_i].min()
        ltf_mid = (ltf_high_val + ltf_low_val) / 2
        
        # 3. Calculate Bias (1h) range at entry_time
        bias_slice = df_bias[df_bias.index <= entry_time]
        bias_high, bias_low, bias_mid = np.nan, np.nan, np.nan
        if len(bias_slice) >= 40:
            recent_bias = bias_slice.tail(40)
            bias_high = recent_bias['high'].max()
            bias_low = recent_bias['low'].min()
            bias_mid = (bias_high + bias_low) / 2
            
        print(f"  HTF (1d) Range (40d): Low={htf_low:.2f}, High={htf_high:.2f}, Mid={htf_mid:.2f} | Entry close is at {entry_px:.2f} ({'Premium' if entry_px > htf_mid else 'Discount'})")
        print(f"  Bias (1h) Range (40h): Low={bias_low:.2f}, High={bias_high:.2f}, Mid={bias_mid:.2f} | Entry close is at {entry_px:.2f} ({'Premium' if entry_px > bias_mid else 'Discount'})")
        print(f"  LTF (5m) Range (200b): Low={ltf_low_val:.2f}, High={ltf_high_val:.2f}, Mid={ltf_mid:.2f} | Entry close is at {entry_px:.2f} ({'Premium' if entry_px > ltf_mid else 'Discount'})")
        
        # Plotting the trade surroundings on LTF (5m) and Bias (1h) timeframes
        # Find exit index
        exit_idx_list = df_ltf.index.get_indexer([exit_time], method='pad')
        exit_i = exit_idx_list[0]
        
        plot_slice = df_ltf.iloc[max(0, ltf_i - 50):min(len(df_ltf), exit_i + 30)]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False, gridspec_kw={'height_ratios': [2, 1]})
        
        # Draw candlesticks on LTF (ax1)
        for t_idx, r in plot_slice.iterrows():
            color = 'green' if r['close'] >= r['open'] else 'red'
            # wick
            ax1.plot([t_idx, t_idx], [r['low'], r['high']], color='black', linewidth=1)
            # body
            ax1.bar(t_idx, r['close'] - r['open'], bottom=r['open'] if r['close'] >= r['open'] else r['close'], 
                    color=color, width=pd.Timedelta(minutes=4), align='center')
                    
        # Plot Entry, SL, TP levels
        ax1.axhline(y=entry_px, color='blue', linestyle='--', label=f'Entry: {entry_px:.2f}')
        ax1.axhline(y=stop_px, color='red', linestyle='-', label=f'Stop Loss: {stop_px:.2f}')
        ax1.axhline(y=target_px, color='green', linestyle='-', label=f'Target: {target_px:.2f}')
        
        # Plot local LTF range
        ax1.axhline(y=ltf_high_val, color='orange', linestyle=':', label=f'LTF High: {ltf_high_val:.2f}')
        ax1.axhline(y=ltf_low_val, color='purple', linestyle=':', label=f'LTF Low: {ltf_low_val:.2f}')
        ax1.axhline(y=ltf_mid, color='gray', linestyle='-.', label=f'LTF Mid (50%): {ltf_mid:.2f}')
        
        # Highlight entry candle
        ax1.axvline(x=entry_time, color='blue', alpha=0.3, label='Entry Time')
        ax1.axvline(x=exit_time, color='red', alpha=0.3, label='Exit Time')
        
        # Style ax1
        ax1.set_title(f"Trade #{idx+1} ({'LONG' if direction == 1 else 'SHORT'}) - 5m Timeframe\n{entry_time_str} to {t['exit_time']}", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Price (USDT)")
        ax1.legend(loc='best', framealpha=0.8)
        ax1.grid(True, alpha=0.3)
        
        # Draw volume on ax2
        colors_vol = ['green' if r['close'] >= r['open'] else 'red' for _, r in plot_slice.iterrows()]
        ax2.bar(plot_slice.index, plot_slice['volume'], color=colors_vol, width=pd.Timedelta(minutes=4), alpha=0.8)
        ax2.set_ylabel("Volume")
        ax2.set_xlabel("Time")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_name = f"failed_trade_{idx+1}.png"
        fig_path = os.path.join(ARTIFACT_DIR, fig_name)
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Saved plot to {fig_path}")

if __name__ == "__main__":
    generate_failed_trade_plots()
