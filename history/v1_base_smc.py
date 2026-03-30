import pandas as pd
import numpy as np
import os

# CONFIGURATION
SYMBOL = "BTC/USDT"
RISK_REWARD = 2.0
ACCOUNT_SIZE = 1000.0
DAYS_BACK = 365
FIXED_SL_USDT = 25.0

def calc_fvgs(df):
    # Historical V1 logic: Simple 3-candle gap, no wick filters
    out = pd.DataFrame(index=df.index)
    out['fvg_bull'] = (df['low'] > df['high'].shift(2)) & (df['close'].shift(1) > df['open'].shift(1))
    out['fvg_bear'] = (df['high'] < df['low'].shift(2)) & (df['close'].shift(1) < df['open'].shift(1))
    return out

def get_hourly_bias(df_1h):
    fvgs = calc_fvgs(df_1h)
    bias = pd.Series(0, index=df_1h.index)
    bias[fvgs['fvg_bull']] = 1
    bias[fvgs['fvg_bear']] = -1
    return bias.ffill().fillna(0)

def simulate_trades(df_ltf, df_bias):
    bias_series = get_hourly_bias(df_bias)
    bias_aligned = bias_series.reindex(df_ltf.index, method='ffill')
    fvgs_ltf = calc_fvgs(df_ltf)
    
    trades = []
    account = ACCOUNT_SIZE
    in_trade = False
    
    for i in range(2, len(df_ltf)):
        if in_trade:
            # Simple TP/SL check
            row = df_ltf.iloc[i]
            if current_trade["direction"] == 1:
                if row['low'] <= current_trade["Stop $"]:
                    current_trade["result"] = "LOSS"; account -= FIXED_SL_USDT
                    trades.append(current_trade); in_trade = False
                elif row['high'] >= current_trade["Target $"]:
                    current_trade["result"] = "WIN"; account += FIXED_SL_USDT * RISK_REWARD
                    trades.append(current_trade); in_trade = False
            else:
                if row['high'] >= current_trade["Stop $"]:
                    current_trade["result"] = "LOSS"; account -= FIXED_SL_USDT
                    trades.append(current_trade); in_trade = False
                elif row['low'] <= current_trade["Target $"]:
                    current_trade["result"] = "WIN"; account += FIXED_SL_USDT * RISK_REWARD
                    trades.append(current_trade); in_trade = False
            continue

        b = bias_aligned.iloc[i]
        if b == 1 and fvgs_ltf['fvg_bull'].iloc[i]:
            entry = df_ltf['close'].iloc[i]
            stop = df_ltf['low'].iloc[i-2:i+1].min()
            risk = entry - stop
            if risk > 0:
                in_trade = True
                current_trade = {"direction": 1, "Entry $": entry, "Stop $": stop, "Target $": entry + risk*RISK_REWARD}
        elif b == -1 and fvgs_ltf['fvg_bear'].iloc[i]:
            entry = df_ltf['close'].iloc[i]
            stop = df_ltf['high'].iloc[i-2:i+1].max()
            risk = stop - entry
            if risk > 0:
                in_trade = True
                current_trade = {"direction": -1, "Entry $": entry, "Stop $": stop, "Target $": entry - risk*RISK_REWARD}
    return trades

print("Milestone V1: The first functional SMC prototype.")
