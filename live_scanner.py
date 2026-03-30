import pandas as pd
import os
import models.backtest_cisd as cisd
import core.lux_fvg as lux
from datetime import datetime
import time
import argparse

# CHANGED: Import signal logging from analysis module
from core.analysis import log_signal, compare_live_vs_backtest, print_live_comparison
from core.config import (
    LIVE_WATCHLIST, USE_LTF_OB_ENTRY, USE_HTF_OB_CONFLUENCE, format_price,
    FVG_WICK_RATIO, FVG_AVG_BODY_LOOKBACK, FVG_MIN_SIZE_MULT, FVG_MIN_VOLUME_MULT
)


def scan_symbol(symbol: str, alignment: str = "daily", exchange_id: str = "binance"):
    """
    Scans a single symbol for active or pending trade setups.

    CHANGED vs old version: Now logs every detected signal to signal_log.csv
    for live vs backtest comparison tracking.
    """
    print(f"\n--- SCANNING {symbol} ({alignment.upper()}) on {exchange_id.upper()} ---")

    if alignment == "daily":
        tf_context, tf_bias, tf_ltf = "1d", "1h", "5m"
    else:
        tf_context, tf_bias, tf_ltf = "1w", "4h", "15m"

    try:
        _ = cisd.fetch_live_ohlcv(symbol, tf_context, 50, exchange_id)
        df_bias = cisd.fetch_live_ohlcv(symbol, tf_bias, 100, exchange_id)
        df_ltf = cisd.fetch_live_ohlcv(symbol, tf_ltf, 200, exchange_id)

        if df_ltf.empty:
            print(f"   [SKIP] No live data returned for {symbol} on {exchange_id}")
            return
    except Exception as e:
        print(f"   [SKIP] ERROR fetching live data for {symbol} on {exchange_id}: {e}")
        return

    # Calculate bias on 1H/4H
    df_bias["bias"] = cisd.get_hourly_bias(df_bias)

    # Detect FVGs on LTF
    fvgs_ltf = lux.detect_luxalgo_fvgs(df_ltf, 
                                      wick_ratio=FVG_WICK_RATIO,
                                      avg_body_lookback=FVG_AVG_BODY_LOOKBACK,
                                      min_size_mult=FVG_MIN_SIZE_MULT,
                                      min_volume_mult=FVG_MIN_VOLUME_MULT)

    # Align bias to LTF
    df_bias.index = pd.to_datetime(df_bias.index, utc=True)
    df_ltf.index = pd.to_datetime(df_ltf.index, utc=True)
    bias_aligned = df_bias["bias"].reindex(df_ltf.index, method="ffill").fillna(0)

    # NEW: Detect OBs on LTF
    ltf_obs = cisd.find_order_blocks(df_ltf) if USE_LTF_OB_ENTRY else []
    
    # Priority 3: HTF OBs for confluence
    htf_obs = cisd.find_order_blocks(df_bias)

    # Check latest state
    last_idx = len(df_ltf) - 1
    current_bias = bias_aligned.iloc[last_idx]

    if current_bias == 0:
        print("   [STATE] No Bias. Waiting for HTF FVG or Liquidity Sweep.")
        return

    bias_dir = "BULLISH" if current_bias == 1 else "BEARISH"
    bias_type = "bullish" if current_bias == 1 else "bearish"
    print(f"   [STATE] Bias is {bias_dir}")

    # Find latest FVG or OB on LTF (last 30 bars)
    anchor_type = None  # "FVG", "OB", or "CONFLUENCE"
    latest_fvg_idx = -1
    latest_ob_idx = -1
    
    for j in range(last_idx, last_idx - 30, -1):
        # Check FVG
        has_fvg = (current_bias == 1 and fvgs_ltf["fvg_bull"].iloc[j]) or (
            current_bias == -1 and fvgs_ltf["fvg_bear"].iloc[j]
        )
        if has_fvg and latest_fvg_idx == -1:
            latest_fvg_idx = j
            
        # Check OB
        ts = df_ltf.index[j]
        has_ob = any(o['time'] == ts and o['type'] == ('bull' if current_bias == 1 else 'bear') for o in ltf_obs)
        if has_ob and latest_ob_idx == -1:
            latest_ob_idx = j

        if latest_fvg_idx != -1 and (not USE_LTF_OB_ENTRY or latest_ob_idx != -1):
            break

    if latest_fvg_idx == -1 and latest_ob_idx == -1:
        print("   [STATE] Bias exists, but no valid Entry Anchor (FVG/OB) found in the last 30 bars.")
        return

    # Determine Tiering and Anchor Levels
    setup_tier = "B"
    fvg_range = None
    ob_range = None
    
    if latest_fvg_idx != -1:
        if current_bias == 1:
            fvg_range = (fvgs_ltf["fvg_bull_btm"].iloc[latest_fvg_idx], fvgs_ltf["fvg_bull_top"].iloc[latest_fvg_idx])
        else:
            fvg_range = (fvgs_ltf["fvg_bear_btm"].iloc[latest_fvg_idx], fvgs_ltf["fvg_bear_top"].iloc[latest_fvg_idx])
            
    if latest_ob_idx != -1:
        match = [o for o in ltf_obs if o['time'] == df_ltf.index[latest_ob_idx] and o['type'] == ('bull' if current_bias == 1 else 'bear')]
        if match:
            ob_range = (match[0]['low'], match[0]['high'])

        if fvg_range and ob_range:
            if cisd.check_confluence(fvg_range[1], fvg_range[0], ob_range[1], ob_range[0]):
                setup_tier = "A++"
                anchor_type = "CONFLUENCE (FVG+OB)"
            else:
                # Use the most recent one
                if latest_fvg_idx >= latest_ob_idx:
                    anchor_type = "FVG"
                else:
                    anchor_type = "OB"
        elif fvg_range:
            anchor_type = "FVG"
        else:
            anchor_type = "OB"

    # A: HTF OB Confluence (if not already A++)
    if setup_tier == "B" and USE_HTF_OB_CONFLUENCE:
        target_range = fvg_range if fvg_range else ob_range
        has_htf = False
        for o in htf_obs:
            if not o['active']:
                continue
            if cisd.check_confluence(target_range[1], target_range[0], o['high'], o['low']):
                has_htf = True
                break
        if has_htf:
            setup_tier = "A"

    anchor_time = df_ltf.index[latest_fvg_idx if anchor_type in ["FVG", "CONFLUENCE (FVG+OB)"] else latest_ob_idx]
    print(f"   [SIGNAL] Found {bias_dir} {anchor_type} at {anchor_time} [TIER {setup_tier}]")

    # Get dominant boundaries for tap detection
    # Use FVG if available, otherwise OB
    if fvg_range:
        btm, top = fvg_range
    else:
        btm, top = ob_range

    # Check if price has touched the anchor
    touched = False
    start_idx = latest_fvg_idx if anchor_type in ["FVG", "CONFLUENCE (FVG+OB)"] else latest_ob_idx
    for k in range(start_idx + 1, last_idx + 1):
        low, high = df_ltf["low"].iloc[k], df_ltf["high"].iloc[k]
        if (current_bias == 1 and low <= top) or (
            current_bias == -1 and high >= btm
        ):
            touched = True
            break

    midpoint = (top + btm) / 2

    if not touched:
        print(f"   [PENDING] Waiting for price to tap into {anchor_type} ({format_price(btm)} - {format_price(top)})")

        # Log the pending signal
        log_signal({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol, "bias_type": bias_type,
            "fvg_top": round(top, 2), "fvg_bottom": round(btm, 2),
            "fvg_midpoint": round(midpoint, 2),
            "entry_price": "", "stop_price": "", "target_price": "",
            "triggered": False, "outcome": "PENDING"
        })
    else:
        # Calculate entry levels
        price_now = df_ltf["close"].iloc[last_idx]
        
        # Calculate Manual CISD Trigger (10-bar swing extreme before/during tap)
        lookback = 10
        tap_idx = -1
        anchor_idx_for_tap = latest_fvg_idx if anchor_type in ["FVG", "CONFLUENCE (FVG+OB)"] else latest_ob_idx
        for k in range(anchor_idx_for_tap + 1, last_idx + 1):
            low, high = df_ltf["low"].iloc[k], df_ltf["high"].iloc[k]
            if (current_bias == 1 and low <= top) or (current_bias == -1 and high >= btm):
                tap_idx = k
                break
        
        cisd_trigger = 0
        if tap_idx != -1:
            start_lb = max(0, tap_idx - lookback)
            if current_bias == 1: # Bullish: Break above swing high
                cisd_trigger = df_ltf["high"].iloc[start_lb:tap_idx+1].max()
            else: # Bearish: Break below swing low
                cisd_trigger = df_ltf["low"].iloc[start_lb:tap_idx+1].min()

        if current_bias == -1:
            sl = top * 1.001
            entry = midpoint
            target = entry - (sl - entry) * 2.0
        else:
            sl = btm * 0.999
            entry = midpoint
            target = entry + (entry - sl) * 2.0

        print(f"   [ACTIVE] Price has tapped the {anchor_type}!")
        print(f"      Setup Tier:    {setup_tier}")
        print(f"      Current Price: {format_price(price_now)}")
        print(f"      Entry Zone:    {format_price(btm)} - {format_price(top)}")
        print(f"      Stop Loss:     {format_price(sl)}")
        print(f"      Target(2R):    {format_price(target)}")
        
        if cisd_trigger > 0:
            direction = "ABOVE" if current_bias == 1 else "BELOW"
            label = "Swing High" if current_bias == 1 else "Swing Low"
            print(f"      CISD Trigger:  Close {direction} {format_price(cisd_trigger)} ({label})")
        
        print("      Action:        Watch for CISD locally to confirm Entry.")

        # Log the active signal
        log_signal({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol, "bias_type": bias_type,
            "fvg_top": round(top, 2), "fvg_bottom": round(btm, 2),
            "fvg_midpoint": round(midpoint, 2),
            "entry_price": round(entry, 2), "stop_price": round(sl, 2),
            "target_price": round(target, 2),
            "triggered": True, "outcome": "OPEN"
        })


def run_scanner():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols", nargs="+",
        default=LIVE_WATCHLIST,
    )
    parser.add_argument("--alignment", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--interval", type=int, default=60, help="Scan interval in seconds")
    # CHANGED: Added --compare flag for live vs backtest comparison
    parser.add_argument("--compare", action="store_true",
                        help="Compare live signal log vs backtest win rate")
    args = parser.parse_args()

    # If compare mode, just print the comparison and exit
    if args.compare:
        result = compare_live_vs_backtest(backtest_wr=36.7)  # Using CISD baseline WR
        print_live_comparison(result)
        return

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("=" * 62)
        print(f"  TTrades Live Signal Scanner - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Alignment: {args.alignment.upper()} | Next scan in {args.interval}s")
        print("=" * 62)

        for sym_raw in args.symbols:
            if ":" in sym_raw:
                sym, ex_id = sym_raw.split(":")
            else:
                sym, ex_id = sym_raw, "binance"
            scan_symbol(sym, args.alignment, ex_id)

        time.sleep(args.interval)


if __name__ == "__main__":
    run_scanner()
