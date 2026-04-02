#!/usr/bin/env python
# ==============================================================
# challenge_trader.py  --  Paper Trading Challenge Engine
#
# RULES:
#   Start balance : $5,000
#   Floor         : $4,700  (stop if we lose $300)
#   Target        : $5,500  (pass when we gain $500)
#   Risk per trade: $50     (1% of starting capital)
#   Max concurrent: 2 open trades at a time
#   Scan interval : every 10 minutes (default)
#   Reward ratio  : 2R  =  +$100 on win, -$50 on loss
#
# Run:  python challenge_trader.py
#       python challenge_trader.py --interval 600
#       python challenge_trader.py --once            (single scan)
# ==============================================================

import os
import json
import time
import argparse
import csv
from datetime import datetime
from typing import List, Dict, Optional

import models.backtest_cisd as cisd
import core.lux_fvg as lux

from core.config import (
    LIVE_WATCHLIST,
    ALIGNMENTS,
    ACTIVE_CHAIN,
    chain_label,
    FVG_WICK_RATIO,
    FVG_AVG_BODY_LOOKBACK,
    FVG_MIN_SIZE_MULT,
    FVG_MIN_VOLUME_MULT,
    USE_LTF_OB_ENTRY,
    USE_HTF_OB_CONFLUENCE,
    format_price,
)
from core.analysis import classify_regime, log_signal

# -----------------------------------------------------------
# CHALLENGE PARAMETERS
# -----------------------------------------------------------
CHALLENGE_START = 5000.00
CHALLENGE_FLOOR = 4700.00  # stop trading if balance drops here
CHALLENGE_TARGET = 5500.00  # challenge passed
RISK_PER_TRADE = 50.00  # fixed dollar risk per trade (1%)
REWARD_PER_TRADE = 100.00  # 2R = $100 win
MAX_OPEN_TRADES = 2  # max concurrent positions
DEFAULT_INTERVAL = 600  # 10 minutes

# -----------------------------------------------------------
# FILE PATHS
# -----------------------------------------------------------
STATE_FILE = "exports/challenge_state.json"
TRADE_LOG = "exports/challenge_trades.csv"
EVENT_LOG = "exports/challenge_events.txt"

TRADE_LOG_HEADERS = [
    "trade_id",
    "opened_at",
    "closed_at",
    "symbol",
    "chain",
    "direction",
    "entry_price",
    "sl_price",
    "tp_price",
    "exit_price",
    "result",
    "pnl_usd",
    "balance_after",
    "regime",
    "bias_reason",
    "anchor_type",
    "setup_tier",
    "fvg_zone_low",
    "fvg_zone_high",
    "scan_number",
    "notes",
]

# -----------------------------------------------------------
# STATE MANAGEMENT
# -----------------------------------------------------------


def load_state() -> Dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        # Migrate older state files that lack visited_signals
        if "visited_signals" not in state:
            state["visited_signals"] = []
        return state
    # Fresh start
    return {
        "balance": CHALLENGE_START,
        "peak_balance": CHALLENGE_START,
        "open_trades": [],
        "pending_trades": [],
        "closed_trades": [],
        "visited_signals": [],  # FVG fingerprints already traded this session
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "scan_count": 0,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ACTIVE",  # ACTIVE | PASSED | STOPPED
        "next_trade_id": 1,
    }


def save_state(state: Dict):
    os.makedirs("exports", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def reset_state():
    """Wipe state and start fresh. Called if user wants a clean run."""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    state = load_state()
    save_state(state)
    event(state, "RESET", "Challenge state reset. Starting fresh $5,000.")
    return state


# -----------------------------------------------------------
# LOGGING HELPERS
# -----------------------------------------------------------


def event(state: Dict, tag: str, msg: str):
    """Append a timestamped event to the event log and print it."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{tag}] {msg}"
    print(f"  {line}")
    os.makedirs("exports", exist_ok=True)
    with open(EVENT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_trade_to_csv(trade: Dict):
    """Append a completed trade row to the CSV log."""
    os.makedirs("exports", exist_ok=True)
    write_header = not os.path.exists(TRADE_LOG)
    with open(TRADE_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_LOG_HEADERS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(trade)


def print_dashboard(state: Dict):
    """Print current challenge status to terminal."""
    bal = state["balance"]
    peak = state.get("peak_balance", CHALLENGE_START)
    dd = CHALLENGE_START - bal
    progress = ((bal - CHALLENGE_START) / (CHALLENGE_TARGET - CHALLENGE_START)) * 100
    open_count = len(state["open_trades"])
    total = state["total_trades"]
    w = state["wins"]
    rem_profit = CHALLENGE_TARGET - bal
    pnl = bal - CHALLENGE_START
    pnl_sign = "+" if pnl >= 0 else ""

    print("\n" + "=" * 62)
    print(f"  TTrades Challenge Trader  --  Scan #{state['scan_count']}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    open_count = len(state.get("open_trades", []))
    pending_count = len(state.get("pending_trades", []))
    total = state.get("total_trades", 0)
    w = state.get("wins", 0)
    l = state.get("losses", 0)
    wr = (w / total * 100) if total > 0 else 0

    print(f"  Balance     : ${bal:>8.2f}   (Start: ${CHALLENGE_START:.2f})")
    print(f"  P&L         : ${pnl:>+8.2f}   (Peak: ${peak:.2f})")
    print(
        f"  Drawdown    : ${dd:>8.2f}   (Floor: ${CHALLENGE_FLOOR:.2f}  [{CHALLENGE_FLOOR - bal:+.2f}])"
    )
    print(
        f"  Progress    : {max(0, progress):>5.1f}%     (Target: ${CHALLENGE_TARGET:.2f}  [need ${CHALLENGE_TARGET - bal:.2f}])"
    )
    print(f"  Trades      : {total} total  |  {w}W / {l}L  |  WR: {wr:.1f}%")
    print(f"  Open trades : {open_count}/{MAX_OPEN_TRADES}")
    print(f"  Pending     : {pending_count} waiting for entry")
    print()
    if state["open_trades"]:
        print("\n  OPEN POSITIONS:")
        for t in state["open_trades"]:
            print(
                f"    [{t['trade_id']}] {t['symbol']} {t['direction']}  "
                f"Entry: {format_price(t['entry_price'])}  "
                f"SL: {format_price(t['sl_price'])}  "
                f"TP: {format_price(t['tp_price'])}  "
                f"(opened {t['opened_at'][11:16]})"
            )
    print("=" * 62)


# -----------------------------------------------------------
# SIGNAL SCANNER (adapted from live_scanner.scan_symbol)
# -----------------------------------------------------------


def scan_for_signal(
    symbol: str, chain_name: str, exchange_id: str = "binance"
) -> Optional[Dict]:
    """
    Scan a single symbol on a given chain for an ACTIVE entry signal.
    Returns a signal dict if found, else None.
    Only returns ACTIVE signals (price already tapped the zone).
    """
    chain = ALIGNMENTS.get(chain_name)
    tf_context = chain["context"]
    tf_bias = chain["bias"]
    tf_ltf = chain["entry"]

    try:
        df_context = cisd.fetch_live_ohlcv(symbol, tf_context, 100, exchange_id)
        df_bias_df = cisd.fetch_live_ohlcv(symbol, tf_bias, 250, exchange_id)
        df_ltf_df = cisd.fetch_live_ohlcv(symbol, tf_ltf, 500, exchange_id)
    except Exception as e:
        event({}, "FETCH_ERR", f"{symbol} {chain_name}: {e}")
        return None

    if df_ltf_df.empty or df_bias_df.empty:
        return None

    import pandas as pd

    # Regime
    regime = "ranging"
    if len(df_context) >= 20:
        try:
            regime = classify_regime(df_context, df_context.index[-1])
        except Exception:
            pass

    # Bias
    bias_result = cisd.get_hourly_bias(df_bias_df)
    bias_series = bias_result["bias"]
    bias_reasons = bias_result["bias_reason"]

    df_bias_df.index = pd.to_datetime(df_bias_df.index, utc=True)
    df_ltf_df.index = pd.to_datetime(df_ltf_df.index, utc=True)

    bias_aligned = bias_series.reindex(df_ltf_df.index, method="ffill").fillna(0)
    reason_aligned = bias_reasons.reindex(df_ltf_df.index, method="ffill").fillna("")

    last_idx = len(df_ltf_df) - 1
    current_bias = bias_aligned.iloc[last_idx]
    if current_bias == 0:
        return None

    bias_reason = reason_aligned.iloc[last_idx]
    bias_dir = 1 if current_bias == 1 else -1

    # FVGs
    fvgs = lux.detect_luxalgo_fvgs(
        df_ltf_df.copy(),
        wick_ratio=FVG_WICK_RATIO,
        avg_body_lookback=FVG_AVG_BODY_LOOKBACK,
        min_size_mult=FVG_MIN_SIZE_MULT,
        min_volume_mult=FVG_MIN_VOLUME_MULT,
    )
    ltf_obs = cisd.find_order_blocks(df_ltf_df) if USE_LTF_OB_ENTRY else []
    htf_obs = cisd.find_order_blocks(df_bias_df)

    # Find most recent relevant FVG / OB
    latest_fvg_idx = -1
    fvg_range = None
    ob_range = None
    anchor_type = None

    for j in range(last_idx, max(last_idx - 30, 0), -1):
        has_fvg = (bias_dir == 1 and fvgs["fvg_bull"].iloc[j]) or (
            bias_dir == -1 and fvgs["fvg_bear"].iloc[j]
        )
        if has_fvg and latest_fvg_idx == -1:
            latest_fvg_idx = j
            if bias_dir == 1:
                fvg_range = (fvgs["fvg_bull_btm"].iloc[j], fvgs["fvg_bull_top"].iloc[j])
            else:
                fvg_range = (fvgs["fvg_bear_btm"].iloc[j], fvgs["fvg_bear_top"].iloc[j])
            anchor_type = "FVG"
            break

    if latest_fvg_idx == -1:
        # try OB
        for j in range(last_idx, max(last_idx - 30, 0), -1):
            ts = df_ltf_df.index[j]
            kind = "bull" if bias_dir == 1 else "bear"
            match = [o for o in ltf_obs if o["time"] == ts and o["type"] == kind]
            if match:
                ob_range = (match[0]["low"], match[0]["high"])
                anchor_type = "OB"
                break

    if fvg_range is None and ob_range is None:
        return None

    btm, top = fvg_range if fvg_range else ob_range

    # Has price actually tapped the zone yet?
    anchor_bar = latest_fvg_idx if latest_fvg_idx != -1 else last_idx
    touched = False
    for k in range(anchor_bar + 1, last_idx + 1):
        lo = df_ltf_df["low"].iloc[k]
        hi = df_ltf_df["high"].iloc[k]
        if bias_dir == 1 and lo <= top:
            touched = True
            break
        if bias_dir == -1 and hi >= btm:
            touched = True
            break
    # We now return signals whether touched or not, so they can be queued as pending.
    # The run_scan loop handles realistic fills vs pending queue.
    # Setup tier
    setup_tier = "B"
    if (
        fvg_range
        and ob_range
        and cisd.check_confluence(top, btm, ob_range[1], ob_range[0])
    ):
        setup_tier = "A++"
        anchor_type = "CONFLUENCE"
    elif USE_HTF_OB_CONFLUENCE:
        target_range = fvg_range if fvg_range else ob_range
        for ob in htf_obs:
            if ob.get("active") and cisd.check_confluence(
                target_range[1], target_range[0], ob["high"], ob["low"]
            ):
                setup_tier = "A"
                break

    # Entry levels  (entry at midpoint of zone)
    midpoint = (top + btm) / 2
    price_now = df_ltf_df["close"].iloc[last_idx]

    if bias_dir == 1:  # LONG
        entry = midpoint
        sl = btm * 0.999
        tp = entry + (entry - sl) * 2.0
    else:  # SHORT
        entry = midpoint
        sl = top * 1.001
        tp = entry - (sl - entry) * 2.0

    # Unique fingerprint for this specific FVG zone.
    # Rounded to 2 dp so minor float drift doesn't create fake-new signals.
    # We omit chain_name here so that if the same FVG exists across daily/intraday, it's only traded once.
    signal_key = f"{symbol}|{round(btm, 2)}|{round(top, 2)}"

    return {
        "symbol": symbol,
        "chain": chain_name,
        "direction": "LONG" if bias_dir == 1 else "SHORT",
        "entry_price": round(entry, 8),
        "sl_price": round(sl, 8),
        "tp_price": round(tp, 8),
        "current_price": round(price_now, 8),
        "fvg_zone_low": round(btm, 8),
        "fvg_zone_high": round(top, 8),
        "anchor_type": anchor_type,
        "setup_tier": setup_tier,
        "regime": regime,
        "bias_reason": bias_reason,
        "exchange_id": exchange_id,
        "signal_key": signal_key,
    }


# -----------------------------------------------------------
# TRADE LIFECYCLE
# -----------------------------------------------------------


def open_trade(state: Dict, signal: Dict) -> Dict:
    """Record a new paper trade entry."""
    tid = f"T{state['next_trade_id']:04d}"
    trade = {
        "trade_id": tid,
        "opened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "closed_at": None,
        "symbol": signal["symbol"],
        "chain": signal["chain"],
        "direction": signal["direction"],
        "entry_price": signal["entry_price"],
        "sl_price": signal["sl_price"],
        "tp_price": signal["tp_price"],
        "exit_price": None,
        "result": "OPEN",
        "pnl_usd": 0.0,
        "balance_after": state["balance"],
        "regime": signal["regime"],
        "bias_reason": signal["bias_reason"],
        "anchor_type": signal["anchor_type"],
        "setup_tier": signal["setup_tier"],
        "fvg_zone_low": signal["fvg_zone_low"],
        "fvg_zone_high": signal["fvg_zone_high"],
        "scan_number": state["scan_count"],
        "exchange_id": signal.get("exchange_id", "binance"),
        "notes": "",
    }
    state["open_trades"].append(trade)
    state["next_trade_id"] += 1
    state["total_trades"] += 1
    save_state(state)

    event(
        state,
        "OPEN",
        f"{tid} {signal['symbol']} {signal['direction']}  "
        f"Entry: {format_price(signal['entry_price'])}  "
        f"SL: {format_price(signal['sl_price'])}  "
        f"TP: {format_price(signal['tp_price'])}  "
        f"Tier: {signal['setup_tier']}  "
        f"Chain: {signal['chain'].upper()}  "
        f"Regime: {signal['regime']}",
    )
    return trade


def check_and_close_trades(state: Dict):
    """
    For each open trade, fetch latest candle and check if SL or TP was hit.
    Paper fill: if current candle's range includes SL or TP, close it.
    """
    still_open = []
    for trade in state["open_trades"]:
        sym = trade["symbol"]
        chain = ALIGNMENTS[trade["chain"]]
        tf_ltf = chain["entry"]
        ex_id = trade.get("exchange_id", "binance")

        try:
            df = cisd.fetch_live_ohlcv(sym, tf_ltf, 5, ex_id)
        except Exception:
            still_open.append(trade)
            continue

        if df.empty:
            still_open.append(trade)
            continue

        latest = df.iloc[-1]
        hi = latest["high"]
        lo = latest["low"]
        close = latest["close"]

        sl = trade["sl_price"]
        tp = trade["tp_price"]
        direction = trade["direction"]

        result = None
        exit_price = close

        if direction == "LONG":
            if lo <= sl:  # SL hit
                result = "LOSS"
                exit_price = sl
                pnl = -RISK_PER_TRADE
            elif hi >= tp:  # TP hit
                result = "WIN"
                exit_price = tp
                pnl = REWARD_PER_TRADE
        else:  # SHORT
            if hi >= sl:  # SL hit
                result = "LOSS"
                exit_price = sl
                pnl = -RISK_PER_TRADE
            elif lo <= tp:  # TP hit
                result = "WIN"
                exit_price = tp
                pnl = REWARD_PER_TRADE

        if result is None:
            still_open.append(trade)
            continue

        # Close the trade
        state["balance"] += pnl
        if state["balance"] > state.get("peak_balance", CHALLENGE_START):
            state["peak_balance"] = state["balance"]

        trade.update(
            {
                "closed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "exit_price": round(exit_price, 8),
                "result": result,
                "pnl_usd": pnl,
                "balance_after": round(state["balance"], 2),
                "notes": f"SL={format_price(sl)} | TP={format_price(tp)} | Exit={format_price(exit_price)}",
            }
        )

        if result == "WIN":
            state["wins"] += 1
        else:
            state["losses"] += 1

        state["closed_trades"].append(trade)
        log_trade_to_csv(trade)

        marker = "WIN  +$100" if result == "WIN" else "LOSS  -$50"
        event(
            state,
            result,
            f"{trade['trade_id']} {sym} {direction}  "
            f"{marker}  Balance: ${state['balance']:.2f}  "
            f"Bias: {trade['bias_reason'][:60]}",
        )

    state["open_trades"] = still_open


def check_challenge_status(state: Dict) -> str:
    """Returns ACTIVE | PASSED | STOPPED."""
    bal = state["balance"]
    if bal >= CHALLENGE_TARGET:
        return "PASSED"
    if bal <= CHALLENGE_FLOOR:
        return "STOPPED"
    return "ACTIVE"


def already_trading(state: Dict, symbol: str, chain: str) -> bool:
    """Avoid opening duplicate positions on the same symbol."""
    for t in state["open_trades"]:
        if t["symbol"] == symbol:
            return True
    return False


# -----------------------------------------------------------
# MAIN SCAN LOOP
# -----------------------------------------------------------

def check_pending_trades(state: Dict):
    """Iterate pending trades, cancel if expired or SL hit, open if entry hit."""
    from datetime import datetime
    import models.backtest_cisd as cisd

    if "pending_trades" not in state:
        state["pending_trades"] = []
    
    still_pending = []
    
    for pt in state["pending_trades"]:
        try:
            now_dt = datetime.now()
            exp_dt = datetime.strptime(pt["expires_at"], "%Y-%m-%d %H:%M:%S")
            if now_dt >= exp_dt:
                event(state, "EXPIRED", f"Pending {pt['symbol']} {pt['direction']} expired.")
                continue
                
            ex_id = pt.get("exchange_id", "binance")
            df = cisd.fetch_live_ohlcv(pt["symbol"], "5m", 5, ex_id)
            if df.empty:
                still_pending.append(pt)
                continue
                
            # --- deduplication fix ---
            if already_trading(state, pt["symbol"], pt["chain"]):
                event(state, "CANCELLED", f"Pending {pt['symbol']} cancelled (already trading this coin)")
                continue
                
            cur_price = df["close"].iloc[-1]
            high_price = df["high"].iloc[-1]
            low_price = df["low"].iloc[-1]
            
            dire = pt["direction"]
            ent = pt["entry_price"]
            sl = pt["sl_price"]
            
            # Check invalidation
            invalidated = False
            if dire == "SHORT" and cur_price >= sl:
                invalidated = True
            elif dire == "LONG" and cur_price <= sl:
                invalidated = True
                
            if invalidated:
                event(state, "CANCELLED", f"Pending {pt['symbol']} {dire} hit SL prior to entry.")
                continue
                
            # Check fill
            triggered = False
            if dire == "SHORT" and high_price >= ent:
                triggered = True
            elif dire == "LONG" and low_price <= ent:
                triggered = True
                
            if triggered:
                if len(state["open_trades"]) < MAX_OPEN_TRADES:
                    pt["current_price"] = cur_price
                    pt["entry_price"] = cur_price
                    if dire == "LONG":
                        actual_risk = cur_price - sl
                        pt["tp_price"] = round(cur_price + actual_risk * 2.0, 8)
                    else:
                        actual_risk = sl - cur_price
                        pt["tp_price"] = round(cur_price - actual_risk * 2.0, 8)
                        
                    open_trade(state, pt)
                    key = pt.get("signal_key")
                    if key and key not in state.get("visited_signals", []):
                        state.setdefault("visited_signals", []).append(key)
                        event(state, "VISITED", f"Marked FVG as traded: {key}")
                else:
                    still_pending.append(pt)
            else:
                still_pending.append(pt)
                
        except Exception as e:
            print(f"Error checking pending {pt['symbol']}: {e}")
            still_pending.append(pt)

    state["pending_trades"] = still_pending



def run_scan(state: Dict, chains_to_scan: List[str]):
    """One full scan: close open trades, then hunt for new entries."""
    state["scan_count"] += 1

    # 0. Check pending trades to see if they filled or expired
    if state.get("pending_trades"):
        event(state, "CHECK", f"Checking {len(state['pending_trades'])} pending trade(s)...")
        check_pending_trades(state)

    # 1. Check / close existing open trades first
    if state["open_trades"]:
        event(state, "CHECK", f"Checking {len(state['open_trades'])} open trade(s)...")
        check_and_close_trades(state)

    # 2. Check challenge status after closes
    state["status"] = check_challenge_status(state)
    save_state(state)
    if state["status"] != "ACTIVE":
        return

    # 3. Can we open more trades?
    slots = MAX_OPEN_TRADES - len(state["open_trades"])
    if slots <= 0:
        event(
            state,
            "SCAN",
            f"Max positions ({MAX_OPEN_TRADES}) open. Waiting for closes.",
        )
        return

    # 4. Scan all coins across all chains for fresh signals
    signals_found = []
    for chain_name in chains_to_scan:
        for sym_raw in LIVE_WATCHLIST:
            if ":" in sym_raw:
                sym, ex_id = sym_raw.split(":")
            else:
                sym, ex_id = sym_raw, "binance"

            if already_trading(state, sym, chain_name):
                continue

            print(f"    Scanning {sym} [{chain_name.upper()}]...", end=" ", flush=True)
            try:
                sig = scan_for_signal(sym, chain_name, ex_id)
                if sig:
                    # Skip this FVG if we already traded it this session
                    if sig["signal_key"] in state.get("visited_signals", []):
                        print(f"SKIP (already traded this FVG zone)")
                    else:
                        cur  = sig["current_price"]
                        ent  = sig["entry_price"]
                        dire = sig["direction"]
                        sl   = sig["sl_price"]
                        
                        from datetime import datetime, timedelta
                        sig["expires_at"] = (datetime.now() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")

                        is_active = False
                        if dire == "SHORT":
                            if cur >= ent and cur < sl:
                                is_active = True
                            elif cur >= sl:
                                print(f"SKIP (price {cur:.4f} at/above SHORT SL {sl:.4f})")
                                continue
                        else:  # LONG
                            if cur <= ent and cur > sl:
                                is_active = True
                            elif cur <= sl:
                                print(f"SKIP (price {cur:.4f} at/below LONG SL {sl:.4f})")
                                continue
                                
                        if is_active:
                            sig["entry_price"] = cur
                            if dire == "LONG":
                                actual_risk = cur - sl
                                sig["tp_price"] = round(cur + actual_risk * 2.0, 8)
                            else:
                                actual_risk = sl - cur
                                sig["tp_price"] = round(cur - actual_risk * 2.0, 8)
                                
                            signals_found.append(sig)
                            print(f"SIGNAL ACTIVE! {dire} @ {cur:.4f} Tier:{sig['setup_tier']}")
                        else:
                            already_pending = False
                            for pt in state.get("pending_trades", []):
                                if pt.get("signal_key") == sig["signal_key"] or pt.get("symbol") == sig["symbol"]:
                                    already_pending = True
                                    break
                            if not already_pending:
                                state.setdefault("pending_trades", []).append(sig)
                                print(f"QUEUED PENDING {dire} @ {ent:.4f} (Cur: {cur:.4f})")
                            else:
                                print(f"SKIP (already pending on symbol/zone)")
                else:
                    print("no setup")
            except Exception as e:
                print(f"ERROR: {e}")

    if not signals_found:
        event(state, "SCAN", "No active signals this scan.")
        return

    # 5. Rank signals: A++ > A > B, then by chain priority (daily > weekly > intraday)
    chain_priority = {"daily": 0, "weekly": 1, "intraday": 2}
    tier_priority = {"A++": 0, "A": 1, "B": 2}
    signals_found.sort(
        key=lambda s: (
            tier_priority.get(s["setup_tier"], 9),
            chain_priority.get(s["chain"], 9),
        )
    )

    # NEW: Filter to keep only one signal per symbol (top priority chain)
    deduped = []
    seen_syms = set()
    for s in signals_found:
        if s["symbol"] not in seen_syms:
            deduped.append(s)
            seen_syms.add(s["symbol"])
    signals_found = deduped

    event(
        state, "SIGNALS", f"{len(signals_found)} signal(s) found. Taking top {min(len(signals_found), slots)}."
    )

    for sig in signals_found[:slots]:
        open_trade(state, sig)
        # Mark this FVG zone as visited so we never re-enter it
        key = sig.get("signal_key")
        if key and key not in state.get("visited_signals", []):
            state.setdefault("visited_signals", []).append(key)
            event(state, "VISITED", f"Marked FVG as traded: {key}")

    save_state(state)


def print_final_report(state: Dict):
    """Print full session summary at end."""
    bal = state["balance"]
    pnl = bal - CHALLENGE_START
    w = state["wins"]
    l = state["losses"]
    total = state["total_trades"]
    wr = (w / (w + l) * 100) if (w + l) > 0 else 0.0
    dd = CHALLENGE_START - state.get("peak_balance", CHALLENGE_START)

    print("\n" + "=" * 62)
    print(f"  CHALLENGE SESSION REPORT")
    print(f"  Status  : {state['status']}")
    print(f"  Started : {state['started_at']}")
    print(f"  Ended   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Scans   : {state['scan_count']}")
    print("=" * 62)
    print(f"  Start   : ${CHALLENGE_START:.2f}")
    print(f"  End     : ${bal:.2f}")
    print(f"  P&L     : ${pnl:+.2f}")
    print(f"  Trades  : {total}  ({w}W / {l}L)  WR: {wr:.1f}%")
    print(f"  LogFile : {TRADE_LOG}")
    print("=" * 62)

    if state["status"] == "PASSED":
        print("  *** CHALLENGE PASSED! ***  Target $5,500 reached!")
    elif state["status"] == "STOPPED":
        print("  *** CHALLENGE STOPPED ***  Floor $4,700 hit. Review needed.")
    print()


# -----------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Challenge Engine")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help="Scan interval in seconds (default: 600 = 10min)",
    )
    parser.add_argument(
        "--chains",
        nargs="+",
        default=list(ALIGNMENTS.keys()),
        help="Chains to scan (daily weekly intraday)",
    )
    parser.add_argument(
        "--once", action="store_true", help="Run one scan only then exit"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset challenge state (fresh start)"
    )
    args = parser.parse_args()

    os.makedirs("exports", exist_ok=True)

    if args.reset:
        state = reset_state()
        print("  Challenge state reset. Starting fresh.")
    else:
        state = load_state()

    if state["status"] != "ACTIVE":
        print(f"\n  Challenge already {state['status']}. Use --reset to start fresh.")
        print_final_report(state)
        return

    chains = args.chains

    print("\n" + "=" * 62)
    print(f"  TTrades Challenge Trader")
    print(f"  Start balance : ${CHALLENGE_START:.2f}")
    print(
        f"  Floor         : ${CHALLENGE_FLOOR:.2f}  (max loss: ${CHALLENGE_START - CHALLENGE_FLOOR:.2f})"
    )
    print(
        f"  Target        : ${CHALLENGE_TARGET:.2f}  (need: ${CHALLENGE_TARGET - CHALLENGE_START:.2f})"
    )
    print(
        f"  Risk/trade    : ${RISK_PER_TRADE:.2f}  |  Reward: ${REWARD_PER_TRADE:.2f}  (2R)"
    )
    print(f"  Max trades    : {MAX_OPEN_TRADES} open at a time")
    print(f"  Chains        : {', '.join(chains)}")
    print(f"  Watchlist     : {', '.join(LIVE_WATCHLIST)}")
    print(f"  Interval      : {args.interval}s  ({args.interval // 60}min)")
    print(
        f"  Resume from   : Balance ${state['balance']:.2f}  "
        f"| Scans: {state['scan_count']}  "
        f"| Trades: {state['total_trades']}"
    )
    print(f"  Logs          : {TRADE_LOG}")
    print(f"  Press Ctrl+C to pause")
    print("=" * 62)

    if args.once:
        print_dashboard(state)
        run_scan(state, chains)
        save_state(state)
        print_dashboard(state)
        return

    try:
        while True:
            print_dashboard(state)
            run_scan(state, chains)
            save_state(state)

            state["status"] = check_challenge_status(state)
            if state["status"] != "ACTIVE":
                break

            print(
                f"\n  Sleeping {args.interval}s... "
                f"Next scan at {datetime.now().strftime('%H:%M:%S')}"
            )
            print(f"  (Ctrl+C to stop)")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\n  Paused by user. State saved.")
        save_state(state)

    print_final_report(state)


if __name__ == "__main__":
    main()
