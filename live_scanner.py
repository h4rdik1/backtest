import pandas as pd
import os
import models.backtest_cisd as cisd
import core.lux_fvg as lux
from datetime import datetime
import time
import argparse

from core.analysis import log_signal, compare_live_vs_backtest, print_live_comparison, classify_regime
from core.config import (
    LIVE_WATCHLIST, USE_LTF_OB_ENTRY, USE_HTF_OB_CONFLUENCE, format_price,
    FVG_WICK_RATIO, FVG_AVG_BODY_LOOKBACK, FVG_MIN_SIZE_MULT, FVG_MIN_VOLUME_MULT,
    ACTIVE_CHAIN, ALIGNMENTS, chain_label
)


def scan_symbol(symbol: str, chain_name: str = None, exchange_id: str = "binance"):
    """
    Scans a single symbol for active or pending trade setups.
    Now prints a full [REASON] block for every signal.
    """
    chain = ALIGNMENTS.get(chain_name or ACTIVE_CHAIN)
    tf_context = chain["context"]
    tf_bias = chain["bias"]
    tf_ltf = chain["entry"]

    print(f"\n--- SCANNING {symbol} ({tf_context}/{tf_bias}/{tf_ltf}) on {exchange_id.upper()} ---")

    try:
        df_context = cisd.fetch_live_ohlcv(symbol, tf_context, 100, exchange_id)
        df_bias_data = cisd.fetch_live_ohlcv(symbol, tf_bias, 250, exchange_id)
        df_ltf_data = cisd.fetch_live_ohlcv(symbol, tf_ltf, 500, exchange_id)

        if df_ltf_data.empty:
            print(f"   [SKIP] No live data returned for {symbol} on {exchange_id}")
            return
    except Exception as e:
        print(f"   [SKIP] ERROR fetching live data for {symbol} on {exchange_id}: {e}")
        return

    # ── Layer 1: Regime Classification ──
    regime = "N/A"
    if not df_context.empty and len(df_context) >= 20:
        try:
            regime = classify_regime(df_context, df_context.index[-1])
        except Exception:
            regime = "unknown"

    # ── Layer 2: Bias Detection (with reason) ──
    bias_result = cisd.get_hourly_bias(df_bias_data)
    bias_series = bias_result['bias']
    bias_reasons = bias_result['bias_reason']

    # Align bias to LTF
    df_bias_data.index = pd.to_datetime(df_bias_data.index, utc=True)
    df_ltf_data.index = pd.to_datetime(df_ltf_data.index, utc=True)
    bias_aligned = bias_series.reindex(df_ltf_data.index, method="ffill").fillna(0)
    reason_aligned = bias_reasons.reindex(df_ltf_data.index, method="ffill").fillna("")

    # Detect FVGs and OBs
    fvgs_ltf = lux.detect_luxalgo_fvgs(df_ltf_data.copy(),
                                       wick_ratio=FVG_WICK_RATIO,
                                       avg_body_lookback=FVG_AVG_BODY_LOOKBACK,
                                       min_size_mult=FVG_MIN_SIZE_MULT,
                                       min_volume_mult=FVG_MIN_VOLUME_MULT)
    ltf_obs = cisd.find_order_blocks(df_ltf_data) if USE_LTF_OB_ENTRY else []
    htf_obs = cisd.find_order_blocks(df_bias_data)

    # Check latest state
    last_idx = len(df_ltf_data) - 1
    current_bias = bias_aligned.iloc[last_idx]
    current_bias_reason = reason_aligned.iloc[last_idx]

    if current_bias == 0:
        print(f"   [STATE] No Bias. Waiting for {tf_bias.upper()} FVG or Liquidity Sweep.")
        return

    bias_dir = "BULLISH" if current_bias == 1 else "BEARISH"
    bias_type = "bullish" if current_bias == 1 else "bearish"
    print(f"   [STATE] Bias is {bias_dir}")

    # ── Layer 3: Find Entry Anchor (FVG or OB) ──
    anchor_type = None
    latest_fvg_idx = -1
    latest_ob_idx = -1
    fvg_range = None
    ob_range = None

    for j in range(last_idx, max(last_idx - 30, 0), -1):
        has_fvg = (current_bias == 1 and fvgs_ltf["fvg_bull"].iloc[j]) or (
            current_bias == -1 and fvgs_ltf["fvg_bear"].iloc[j]
        )
        if has_fvg and latest_fvg_idx == -1:
            latest_fvg_idx = j

        ts = df_ltf_data.index[j]
        has_ob = any(o['time'] == ts and o['type'] == ('bull' if current_bias == 1 else 'bear') for o in ltf_obs)
        if has_ob and latest_ob_idx == -1:
            latest_ob_idx = j

        if latest_fvg_idx != -1 and (not USE_LTF_OB_ENTRY or latest_ob_idx != -1):
            break

    if latest_fvg_idx == -1 and latest_ob_idx == -1:
        print(f"   [STATE] Bias exists, but no valid {tf_ltf.upper()} Entry Anchor (FVG/OB) in last 30 bars.")
        return

    # Determine anchor details
    setup_tier = "B"
    ltf_anchor_reason = ""

    if latest_fvg_idx != -1:
        if current_bias == 1:
            fvg_range = (fvgs_ltf["fvg_bull_btm"].iloc[latest_fvg_idx], fvgs_ltf["fvg_bull_top"].iloc[latest_fvg_idx])
        else:
            fvg_range = (fvgs_ltf["fvg_bear_btm"].iloc[latest_fvg_idx], fvgs_ltf["fvg_bear_top"].iloc[latest_fvg_idx])
        ltf_anchor_reason = f"{tf_ltf.upper()} FVG (zone: {format_price(fvg_range[0])} - {format_price(fvg_range[1])}) formed @ {df_ltf_data.index[latest_fvg_idx].strftime('%H:%M')}"
        anchor_type = "FVG"

    if latest_ob_idx != -1:
        match = [o for o in ltf_obs if o['time'] == df_ltf_data.index[latest_ob_idx] and o['type'] == ('bull' if current_bias == 1 else 'bear')]
        if match:
            ob_range = (match[0]['low'], match[0]['high'])

        if fvg_range and ob_range:
            if cisd.check_confluence(fvg_range[1], fvg_range[0], ob_range[1], ob_range[0]):
                setup_tier = "A++"
                anchor_type = "CONFLUENCE (FVG+OB)"
                ltf_anchor_reason += f" + OB overlap"
            else:
                if latest_fvg_idx >= latest_ob_idx:
                    anchor_type = "FVG"
                else:
                    anchor_type = "OB"
                    ltf_anchor_reason = f"{tf_ltf.upper()} OB (zone: {format_price(ob_range[0])} - {format_price(ob_range[1])}) formed @ {df_ltf_data.index[latest_ob_idx].strftime('%H:%M')}"
        elif not fvg_range and ob_range:
            anchor_type = "OB"
            ltf_anchor_reason = f"{tf_ltf.upper()} OB (zone: {format_price(ob_range[0])} - {format_price(ob_range[1])}) formed @ {df_ltf_data.index[latest_ob_idx].strftime('%H:%M')}"

    # HTF OB Confluence check
    if setup_tier == "B" and USE_HTF_OB_CONFLUENCE:
        target_range = fvg_range if fvg_range else ob_range
        if target_range:
            for o in htf_obs:
                if not o['active']:
                    continue
                if cisd.check_confluence(target_range[1], target_range[0], o['high'], o['low']):
                    setup_tier = "A"
                    break

    # Price tap detection
    if fvg_range:
        btm, top = fvg_range
    else:
        btm, top = ob_range

    touched = False
    anchor_idx = latest_fvg_idx if anchor_type in ["FVG", "CONFLUENCE (FVG+OB)"] else latest_ob_idx
    tap_time = ""
    for k in range(anchor_idx + 1, last_idx + 1):
        low, high = df_ltf_data["low"].iloc[k], df_ltf_data["high"].iloc[k]
        if (current_bias == 1 and low <= top) or (current_bias == -1 and high >= btm):
            touched = True
            tap_time = df_ltf_data.index[k].strftime("%H:%M")
            break

    midpoint = (top + btm) / 2

    # ── Entry levels ──
    price_now = df_ltf_data["close"].iloc[last_idx]
    if current_bias == -1:
        sl = top * 1.001
        entry = midpoint
        target = entry - (sl - entry) * 2.0
    else:
        sl = btm * 0.999
        entry = midpoint
        target = entry + (entry - sl) * 2.0

    # ══════════════════════════════════════════
    # PRINT THE FULL REASONING BLOCK
    # ══════════════════════════════════════════
    print(f"\n   [REASON] Top-to-Bottom Analysis:")
    print(f"      Chain:          {chain_label(chain_name or ACTIVE_CHAIN)}")
    print(f"      Context Layer:  {regime.replace('_', ' ').title()} market ({symbol})")
    print(f"      Bias Layer:     {current_bias_reason or 'Active ' + bias_dir}")
    print(f"      Entry Anchor:   {ltf_anchor_reason}")

    if not touched:
        print(f"      Entry Signal:   PENDING - Waiting for price to tap zone ({format_price(btm)} - {format_price(top)})")
        print(f"      Setup Tier:     {setup_tier}")
        print(f"      Current Price:  {format_price(price_now)}")

        log_signal({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol, "bias_type": bias_type,
            "reason": f"HTF: {current_bias_reason} | LTF: {ltf_anchor_reason} | PENDING",
            "fvg_top": round(top, 2), "fvg_bottom": round(btm, 2),
            "fvg_midpoint": round(midpoint, 2),
            "entry_price": "", "stop_price": "", "target_price": "",
            "triggered": False, "outcome": "PENDING"
        })
    else:
        print(f"      Entry Signal:   ACTIVE - Price tapped zone @ {tap_time}")
        print(f"      Risk Layout:    Entry ~{format_price(entry)} | SL: {format_price(sl)} | TP: {format_price(target)} (2R)")
        print(f"      Setup Tier:     {setup_tier}")
        print(f"      Current Price:  {format_price(price_now)}")

        log_signal({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol, "bias_type": bias_type,
            "reason": f"HTF: {current_bias_reason} | LTF: {ltf_anchor_reason} | Tap @ {tap_time}",
            "fvg_top": round(top, 2), "fvg_bottom": round(btm, 2),
            "fvg_midpoint": round(midpoint, 2),
            "entry_price": round(entry, 2), "stop_price": round(sl, 2),
            "target_price": round(target, 2),
            "triggered": True, "outcome": "OPEN"
        })


def run_scanner():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=LIVE_WATCHLIST)
    parser.add_argument("--chain", default=None, help="Override chain (daily/weekly/intraday)")
    parser.add_argument("--interval", type=int, default=60, help="Scan interval in seconds")
    parser.add_argument("--compare", action="store_true", help="Compare live vs backtest")
    args = parser.parse_args()

    chain_name = args.chain or ACTIVE_CHAIN

    if args.compare:
        result = compare_live_vs_backtest(backtest_wr=43.0)
        print_live_comparison(result)
        return

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("=" * 62)
        print(f"  TTrades Live Signal Scanner - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Chain: {chain_label(chain_name)} | Next scan in {args.interval}s")
        print("=" * 62)

        for sym_raw in args.symbols:
            if ":" in sym_raw:
                sym, ex_id = sym_raw.split(":")
            else:
                sym, ex_id = sym_raw, "binance"
            scan_symbol(sym, chain_name, ex_id)

        time.sleep(args.interval)


if __name__ == "__main__":
    run_scanner()
