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
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import pandas as pd

import models.backtest_cisd as cisd
from models.backtest_cisd import find_protected_swing, find_latest_sfp, detect_sfp_pattern
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
    USE_PREMIUM_DISCOUNT,
    format_price,
    MIN_DISPLACEMENT_STRENGTH,
    MIN_SETUP_TIER,
    SKIP_RANGING_REGIME,
    SKIP_COUNTER_TREND,
    MAX_TRADES_PER_DAY,
    RETIRED_ALIGNMENTS,
    RISK_REWARD,
    MIN_SL_PCT_BTC,
    MIN_SL_PCT_ALTS,
    MAX_STOP_DISTANCE,
    # Task 3 — Sessions
    SESSIONS, SESSION_MIN_TIER,
    # Task 4 — Break-even & trailing
    BE_TRIGGER_R, PARTIAL_EXIT_R, PARTIAL_EXIT_PCT,
    # Task 5 — Correlation
    CORRELATION_GROUPS,
    # Task 6 — Scoring / sizing
    SCORE_THRESHOLDS, BASE_RISK_USDT,
    # Task 8 — Drawdown halt
    TRAILING_HALT_PCT, MAX_WEEKLY_LOSS_PCT,
    # Task 9 — Time invalidation
    MAX_TRADE_DURATION,
    # Task 10 — Live vs backtest
    LIVE_BACKTEST_CHECK_N,
    # Health check
    run_system_health_check,
    ACCOUNT_SIZE,
)
from core.analysis import classify_regime, log_signal, get_current_session, compare_live_to_backtest

# -----------------------------------------------------------
# CHALLENGE PARAMETERS
# -----------------------------------------------------------
CHALLENGE_START  = 5000.00
CHALLENGE_FLOOR  = 4700.00  # hard floor — kept for backwards compat
CHALLENGE_TARGET = 5500.00  # challenge passed
RISK_PER_TRADE   = BASE_RISK_USDT  # base dollar risk per trade (score-adjusted dynamically)
REWARD_PER_TRADE = 100.00   # 2R = $100 win (at full size)
MAX_OPEN_TRADES  = 2        # max concurrent positions
DEFAULT_INTERVAL = 600      # 10 minutes

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


def load_state():
    """Load challenge data from JSON, with self-healing for missing keys."""
    if not os.path.exists(STATE_FILE):
        return reset_state()
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            
            # --- SELF-HEALING: Ensure all required keys exist ---
            defaults = {
                "balance": CHALLENGE_START,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "scan_count": 0,
                "status": "ACTIVE",
                "closed_trades": [],
                "open_trades": [],
                "visited_signals": [],
                "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "next_trade_id": 1,
            }
            fixed = False
            for k, v in defaults.items():
                if k not in state:
                    state[k] = v
                    fixed = True
            
            # Cleanup old key name if it exists
            if "trades" in state and not state["closed_trades"]:
                state["closed_trades"] = state.pop("trades")
                fixed = True
                
            if fixed:
                print("   [SYSTEM] State file self-healed (added missing keys).")
                save_state(state)
                
            return state
    except Exception:
        return reset_state()


def save_state(state: Dict):
    os.makedirs("exports", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)



def reset_state():
    """Wipe state and start fresh, but ARCHIVE the old data first."""
    import shutil
    # Ensure archive folder exists
    os.makedirs("archive", exist_ok=True)
    
    # Check if we have something to archive
    if os.path.exists(STATE_FILE):
        try:
            # Prepare metadata from existing file
            with open(STATE_FILE, 'r') as f:
                old_state = json.load(f)
                status = old_state.get('status', 'RESET')
                ts = datetime.now().strftime("%Y%m%d_%H%M")
                
            # Archive state (Use copy instead of rename to avoid file lock issues)
            arc_state = f"archive/challenge_{status}_{ts}.json"
            shutil.copy(STATE_FILE, arc_state)
            print(f"   [ARCHIVE] Saved state backup to {arc_state}")
            
            # Archive logs
            if os.path.exists(TRADE_LOG):
                arc_log = f"archive/challenge_{status}_{ts}.csv"
                try:
                    shutil.copy(TRADE_LOG, arc_log)
                    os.remove(TRADE_LOG)
                    print(f"   [ARCHIVE] Moved logs to {arc_log}")
                except Exception as e:
                    print(f"   [WARN] Could not move logs (locked): {e}")

        except Exception as e:
             print(f"   [WARN] Archive failed: {e}")

    # Fresh Start - Ensure ALL expected keys for current dashboard are present
    state = {
        "balance": CHALLENGE_START,
        "peak_balance": CHALLENGE_START,  # Task 8 — trail from peak
        "weekly_open_balance": CHALLENGE_START,  # Task 8 — weekly loss calc
        "weekly_open_date": datetime.now().strftime("%Y-%m-%d"),
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "scan_count": 0,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ACTIVE",
        "next_trade_id": 1,
        "closed_trades": [],       # History of closed trades
        "open_trades": [],         # Active positions
        "visited_signals": [],     # Deduplication memory
        "halt_reason": None,       # Task 8 — why trading was halted
    }
    save_state(state)
    event(state, "RESET", f"Challenge reset to ${CHALLENGE_START:.2f}")
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
# TASK 3 — Session Filter
# WHY: The best setups happen during London Open and NY Open.
# Asian session has low liquidity and frequently reverses London.
# We skip lower-tier setups in the Asian session entirely.
# -----------------------------------------------------------

def check_session_allows_trade(setup_tier: str) -> tuple:
    """
    TASK 3 — Determine if the current session allows a trade for this tier.

    Rules:
      - S-tier and A++ tier  : allowed in ANY session
      - A-tier               : london_open or ny_open only
      - B-tier               : london_open or ny_open only
      - Asian session        : only S-tier setups allowed (all others skipped)
      - off_hours            : only S-tier setups allowed

    Returns:
        (allowed: bool, session_name: str)
    """
    session = get_current_session()  # UTC-based session lookup
    tier_rank = {"S": 0, "A++": 1, "A": 2, "B": 3}  # lower = better
    min_tier_rank = tier_rank.get(SESSION_MIN_TIER.get("asian", "S"), 0)

    if session == "asian" or session == "off_hours":
        # Only S-tier or better allowed in Asian/off-hours
        allowed = tier_rank.get(setup_tier, 9) <= min_tier_rank
        return allowed, session

    # London Open, NY Open, London Close — A and above allowed
    min_other_rank = tier_rank.get(SESSION_MIN_TIER.get("other", "A"), 2)
    allowed = tier_rank.get(setup_tier, 9) <= min_other_rank
    return allowed, session


# -----------------------------------------------------------
# TASK 5 — Correlation Check
# WHY: BTC, ETH, SOL are highly correlated. Taking both BTC long
# and ETH long is not diversification — it doubles the risk from
# a single macro move. We allow at most one trade per group.
# Counter-direction trades are allowed (natural hedge).
# -----------------------------------------------------------

def is_correlated_trade_open(state: Dict, symbol: str, direction: str) -> bool:
    """
    TASK 5 — Check if we already have an open trade in the same
    correlation group AND the same direction as the new signal.

    Rules:
      - Same group + SAME direction  → skip (over-exposed)
      - Same group + OPPOSITE direction → allow (natural hedge)
      - Different group → always allow

    Args:
        state     : Challenge state dict with 'open_trades' list.
        symbol    : e.g., 'ETH/USDT'
        direction : 'LONG' or 'SHORT'

    Returns:
        True if a conflicting correlated trade is open (should skip).
        False if the new trade is safe to enter.
    """
    # Find which correlation group this symbol belongs to
    my_group = None
    for group in CORRELATION_GROUPS:
        if symbol in group:
            my_group = group
            break

    if my_group is None:
        return False  # Symbol not in any group — no correlation risk

    # Check every open trade against the group
    for trade in state.get("open_trades", []):
        if trade["symbol"] in my_group and trade["symbol"] != symbol:
            # Only block if SAME direction (different direction = hedge)
            if trade["direction"] == direction:
                return True  # Block: same group, same direction

    return False  # No conflict found


# -----------------------------------------------------------
# TASK 6 — Setup Quality Score for Variable Sizing
# WHY: Not all setups deserve the same risk. A perfect S-tier
# setup with HTF SFP, session confluence, and trend alignment
# should get full size ($50). A marginal A-tier setup gets half.
# -----------------------------------------------------------

def score_setup(signal: Dict) -> int:
    """
    TASK 6 — Score a setup from 0 to 13 based on confluence factors.
    Higher score = larger position size.

    Scoring breakdown:
      +3 : HTF SFP confirmed (the sl_type contains 'sfp')
      +2 : LTF structure break is impulsive (displacement_strength > 1.5)
      +2 : FVG and Order Block overlap (anchor_type == CONFLUENCE)
      +1 : Trade is in london_open or ny_open session
      +1 : Regime is trending (strong_bull or strong_bear)
      +2 : Second SFP confluences exist (sl_type is ltf_sfp)
      +2 : Tier is S (highest quality)

    Returns:
        Integer score from 0 to 13.
    """
    score = 0

    # +3 if HTF SFP was used for the stop-loss (SFP at HTF = institutional footprint)
    sl_type = signal.get("sl_type", "")
    if "sfp" in sl_type:
        score += 3

    # +2 if LTF displacement was strong (impulse candle, not just a drift)
    disp = signal.get("metadata", {}).get("displacement_strength", 0)
    if disp >= 1.5:
        score += 2

    # +2 if setup has FVG + OB overlap (strongest confluence possible)
    if signal.get("anchor_type") == "CONFLUENCE":
        score += 2

    # +1 if we are in a prime session (London Open or NY Open)
    session = get_current_session()
    if session in ("london_open", "ny_open"):
        score += 1

    # +1 if the market regime is strongly trending (not ranging)
    regime = signal.get("regime", "ranging")
    if regime in ("strong_bull", "strong_bear", "trending_up", "trending_down"):
        score += 1

    # +2 if we have a nested LTF SFP (more precise SL anchor = less noise)
    if sl_type == "ltf_sfp":
        score += 2

    # +2 if this is an S-tier setup (the best of the best)
    if signal.get("setup_tier") == "S":
        score += 2

    return score


def get_risk_for_score(score: int) -> float:
    """
    TASK 6 — Convert a quality score to a dollar risk amount.

    Based on SCORE_THRESHOLDS in config:
      Score >= 9  → full $50 risk
      Score 6-8   → $35 risk (70%)
      Score 4-5   → $25 risk (50%)
      Score < 4   → skip entirely (returns 0)
    """
    # Check tiers from best to worst
    for tier_name, tier_cfg in sorted(
        SCORE_THRESHOLDS.items(),
        key=lambda x: x[1]["min_score"],
        reverse=True
    ):
        if score >= tier_cfg["min_score"]:
            return round(BASE_RISK_USDT * tier_cfg["risk_pct"], 2)
    return 0.0  # Score below minimum — skip this setup


# -----------------------------------------------------------
# TASK 8 — Trailing Drawdown Halt
# WHY: A fixed floor ($4700) doesn’t protect profits. If we grow
# to $5500 then lose $300, we should halt at $5500 * 0.94 = $5170,
# not wait until $4700. We also add a weekly loss limit.
# -----------------------------------------------------------

def update_peak_balance(state: Dict):
    """
    TASK 8 — Update the peak balance tracker whenever balance rises.
    This is the reference point for the trailing halt calculation.
    Must be called every time the balance changes.
    """
    current = state["balance"]
    peak    = state.get("peak_balance", CHALLENGE_START)
    if current > peak:
        state["peak_balance"] = current  # New high water mark!


def check_drawdown_halt(state: Dict) -> bool:
    """
    TASK 8 — Check if the trailing drawdown limit has been hit.

    Halt rules:
      1. Trailing halt: if balance <= peak_balance * (1 - TRAILING_HALT_PCT)
         i.e., 6% drop from the highest point we have ever reached.
      2. Weekly loss limit: if (weekly_open_balance - balance) / weekly_open_balance
         exceeds MAX_WEEKLY_LOSS_PCT (8%), pause until next Monday.

    Returns:
        True if trading should be halted now.
        False if we are within acceptable limits.
    """
    balance = state["balance"]
    peak    = state.get("peak_balance", CHALLENGE_START)

    # ── Rule 1: Trailing halt from peak ──
    halt_level = peak * (1.0 - TRAILING_HALT_PCT)
    if balance <= halt_level:
        msg = f"Drawdown halt triggered! Peak: ${peak:.2f} | Current: ${balance:.2f} | Halt level: ${halt_level:.2f}"
        event(state, "HALT", msg)
        state["status"] = "STOPPED"
        state["halt_reason"] = "trailing_drawdown"
        return True

    # ── Rule 2: Weekly loss limit ──
    # Check if it's a new week (Monday) and reset the weekly reference
    today = datetime.now().strftime("%Y-%m-%d")
    weekly_open_date = state.get("weekly_open_date", today)

    # Parse the stored date and check if we're in a new Monday
    try:
        stored_dt  = datetime.strptime(weekly_open_date, "%Y-%m-%d")
        today_dt   = datetime.strptime(today, "%Y-%m-%d")
        days_since = (today_dt - stored_dt).days
        # Reset weekly tracking every Monday
        if today_dt.weekday() == 0 and days_since >= 7:
            state["weekly_open_balance"] = balance
            state["weekly_open_date"]    = today
    except Exception:
        pass  # If date parsing fails, just skip weekly check

    weekly_open = state.get("weekly_open_balance", CHALLENGE_START)
    if weekly_open > 0:
        weekly_loss_pct = (weekly_open - balance) / weekly_open
        if weekly_loss_pct >= MAX_WEEKLY_LOSS_PCT:
            msg = (
                f"Weekly loss limit hit! Open: ${weekly_open:.2f} | "
                f"Current: ${balance:.2f} | "
                f"Loss: {weekly_loss_pct:.1%} (limit: {MAX_WEEKLY_LOSS_PCT:.0%}). "
                f"Pausing until next Monday."
            )
            event(state, "HALT", msg)
            state["halt_reason"] = "weekly_loss_limit"
            # Don't permanently stop — mark as halted until Monday
            state["status"] = "WEEKLY_HALT"
            return True

    return False  # All limits are OK, continue trading


# -----------------------------------------------------------
# TASK 9 — Time-Based Trade Invalidation
# WHY: A setup that hasn’t reached 1R in our favour after the
# maximum allowed bars has failed to play out. Holding it longer
# just ties up margin and risks a late adverse move.
# -----------------------------------------------------------

def check_time_invalidation(
    trade: Dict,
    current_bar_idx: int,
    df_ltf: pd.DataFrame,
) -> bool:
    """
    TASK 9 — Check if a trade has been open too long without reaching 1R.

    Logic:
      1. Look up the max duration for this trade’s timeframe
         from MAX_TRADE_DURATION in config.
      2. Count bars since entry.
      3. If bars_open > max and the trade never hit 1R → return True (close it).

    Args:
        trade           : Open trade dict from state['open_trades'].
        current_bar_idx : Current bar index in df_ltf.
        df_ltf          : The LTF OHLCV DataFrame used for this trade.

    Returns:
        True if the trade should be closed (time expired, no 1R hit).
        False if the trade is still within its time window OR has hit 1R.
    """
    # Find which chain/timeframe this trade is using
    chain_name = trade.get("chain", "intraday")
    all_chains = {**ALIGNMENTS, **RETIRED_ALIGNMENTS}
    tf_entry   = all_chains.get(chain_name, {}).get("entry", "15m")

    max_bars = MAX_TRADE_DURATION.get(tf_entry, 48)  # default 48 bars

    # entry_bar_index is stored when we open the trade
    entry_bar_idx = trade.get("entry_bar_idx", None)
    if entry_bar_idx is None:
        return False  # No index stored — can't check

    bars_open = current_bar_idx - entry_bar_idx
    if bars_open <= max_bars:
        return False  # Still within allowed window

    # Check if price ever reached 1R in our direction
    entry    = trade["entry_price"]
    sl       = trade["sl_price"]
    risk     = abs(entry - sl)
    one_r    = risk  # 1R distance

    direction = trade["direction"]
    # Look through candles since entry
    slice_start = max(0, entry_bar_idx)
    slice_end   = current_bar_idx + 1
    price_slice = df_ltf.iloc[slice_start:slice_end]

    max_r_reached = 0.0
    if direction == "LONG":
        best_high = price_slice["high"].max()
        max_r_reached = (best_high - entry) / one_r if one_r > 0 else 0.0
    else:
        best_low  = price_slice["low"].min()
        max_r_reached = (entry - best_low) / one_r if one_r > 0 else 0.0

    # If price never reached 1R in our direction → time invalidation
    if max_r_reached < 1.0:
        return True  # Close this stale trade

    return False  # Price reached 1R at some point, let it run


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

    # ── Trend-Following Filter (Counter-Trend Block) ──
    if SKIP_COUNTER_TREND:
        if regime == "trending_up" and bias_dir == -1:
            print(f"SKIP (Counter-Trend SHORT in context trending_up)", end=" ")
            return None
        elif regime == "trending_down" and bias_dir == 1:
            print(f"SKIP (Counter-Trend LONG in context trending_down)", end=" ")
            return None

    # ── Ranging Regime Block ──
    if SKIP_RANGING_REGIME and regime == "ranging":
        print(f"SKIP (Ranging context regime)", end=" ")
        return None

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

    for j in range(last_idx, max(last_idx - 100, 0), -1):
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
        for j in range(last_idx, max(last_idx - 100, 0), -1):
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
    midpoint  = (top + btm) / 2
    price_now = df_ltf_df["close"].iloc[last_idx]

    # ── HTF Premium/Discount Filter ──
    if USE_PREMIUM_DISCOUNT:
        lookback_htf = 40
        if len(df_context) >= lookback_htf:
            recent_htf = df_context.tail(lookback_htf)
            htf_high = recent_htf['high'].max()
            htf_low = recent_htf['low'].min()
            htf_mid = (htf_high + htf_low) / 2
            
            if bias_dir == 1 and price_now > htf_mid:
                print(f"SKIP (HTF Premium: {format_price(price_now)} > {format_price(htf_mid)})", end=" ")
                return None
            elif bias_dir == -1 and price_now < htf_mid:
                print(f"SKIP (HTF Discount: {format_price(price_now)} < {format_price(htf_mid)})", end=" ")
                return None

    # ─────────────────────────────────────────────────────────────
    # NESTED SFP STOP-LOSS LOGIC
    # Priority: LTF SFP (tighter/higher RR) > HTF SFP (structural) > FVG boundary
    # Discard if final stop is wider than MAX_STOP_DISTANCE (protects RR).
    # ─────────────────────────────────────────────────────────────
    is_long  = (bias_dir == 1)
    is_btc   = "BTC" in symbol
    min_sl_dist = MIN_SL_PCT_BTC if is_btc else MIN_SL_PCT_ALTS
    entry    = midpoint  # Always enter at FVG midpoint (institutional equilibrium)

    # 1. HTF SFP — from the bias timeframe (1H or 4H)
    #    lookback = 96 bars covers ~4 sessions on 1H, ~24 sessions on 4H
    htf_sfp = find_latest_sfp(df_bias_df, is_long=is_long, lookback=96)

    # 2. LTF SFP — from the entry timeframe (5m or 15m)
    #    lookback = 100 bars covers ~8 hours on 5m, ~25 hours on 15m
    ltf_sfp = find_latest_sfp(df_ltf_df,  is_long=is_long, lookback=100)

    # 3. Choose SL by priority
    sl       = None
    sl_type  = "fvg_boundary"  # fallback label

    if htf_sfp:
        htf_sl_price, htf_sl_time = htf_sfp
        sl       = htf_sl_price
        sl_type  = "htf_sfp"

        # Check if there is an LTF SFP that formed AFTER the HTF sweep
        if ltf_sfp:
            ltf_sl_price, ltf_sl_time = ltf_sfp
            if ltf_sl_time > htf_sl_time:  # only valid if LTF sweep is more recent
                sl      = ltf_sl_price
                sl_type = "ltf_sfp"  # tighter stop = higher RR

    # 4. Fallback: no SFP found — try Protected Swing High/Low, then use FVG boundary
    if sl is None:
        swing_sl = find_protected_swing(df_ltf_df, last_idx, is_long=is_long, lookback=20)
        if swing_sl is not None and not pd.isna(swing_sl):
            sl = swing_sl
            sl_type = "protected_swing"
        else:
            if is_long:
                sl = btm * 0.998
            else:
                sl = top * 1.002
            sl_type = "fvg_boundary"

    # 5. Minimum distance floor (survive noise)
    if is_long:
        if (entry - sl) < (entry * min_sl_dist):
            sl = entry - (entry * min_sl_dist)
    else:
        if (sl - entry) < (entry * min_sl_dist):
            sl = entry + (entry * min_sl_dist)

    # 6. DISCARD if stop is too wide (low RR) — no trade is better than a bad trade
    stop_pct = abs(entry - sl) / entry
    if stop_pct > MAX_STOP_DISTANCE:
        return None   # Logged by the caller as skipped

    # 7. Take-profit at 2R
    if is_long:
        tp = entry + (entry - sl) * RISK_REWARD
    else:
        tp = entry - (sl - entry) * RISK_REWARD

    # Unique fingerprint for this specific FVG zone + timeframe chain.
    # This allow taking the same zone on Daily and Scalp, but blocks duplicates on the same TF.
    signal_key = f"{chain_name}|{symbol}|{round(btm, 1)}|{round(top, 1)}"

    return {
        "symbol": symbol,
        "chain": chain_name,
        "direction": "LONG" if bias_dir == 1 else "SHORT",
        "entry_price": round(entry, 8),
        "sl_price": round(sl, 8),
        "tp_price": round(tp, 8),
        "sl_type": sl_type,             # "protected_swing" or "fvg_boundary"
        "current_price": round(price_now, 8),
        "fvg_zone_low": round(btm, 8),
        "fvg_zone_high": round(top, 8),
        "anchor_type": anchor_type,
        "setup_tier": setup_tier,
        "regime": regime,
        "bias_reason": bias_reason,
        "exchange_id": exchange_id,
        "signal_key": signal_key,
        "is_counter_trend": (bias_dir == 1 and regime == "trending_down") or (bias_dir == -1 and regime == "trending_up"),
        "metadata": get_market_metadata(df_ltf_df) # Training context/Quality meter
    }

def get_market_metadata(df: pd.DataFrame) -> Dict:
    """SMC-native quality data for later training."""
    if len(df) < 20: return {}
    
    # Displacement Strength (Avg body size of last 3 bars vs previous 10)
    bodies = (df['close'] - df['open']).abs()
    avg_recent = bodies.tail(3).mean()
    avg_hist = bodies.tail(13).head(10).mean()
    displacement_mult = round(avg_recent / avg_hist, 2) if avg_hist > 0 else 1.0
    
    return {
        "displacement_strength": displacement_mult,
    }


# -----------------------------------------------------------
# TRADE LIFECYCLE
# -----------------------------------------------------------


def open_trade(state: Dict, signal: Dict) -> Dict:
    """Record a new paper trade entry."""
    tid = f"T{state['next_trade_id']:04d}"

    # ── TASK 6: Score the setup and compute risk-adjusted size ──
    setup_score = score_setup(signal)
    trade_risk  = get_risk_for_score(setup_score)
    if trade_risk <= 0:
        event(state, "SKIP", f"Score {setup_score}/13 below minimum — setup skipped.")
        return None  # Score too low — don't take this trade

    # Scale the reward proportionally (same RR ratio, smaller size)
    trade_reward = round(trade_risk * RISK_REWARD, 2)

    # ── TASK 3: Record which session we entered in ──
    session = get_current_session()

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
        # Task 3: Session tracking
        "session": session,
        # Task 4: Trade management state
        "be_triggered": False,          # True once SL moved to breakeven (at 1.5R)
        "partial_exit_done": False,     # True once 50% was closed at 2R
        "trailing_active": False,       # True once trailing stop is in use
        "trailing_sl": None,            # Current trailing stop level
        "max_r_reached": 0.0,           # Highest R multiple hit so far
        # Task 6: Sizing info
        "setup_score": setup_score,
        "trade_risk": trade_risk,
        "trade_reward": trade_reward,
        # Task 9: Bar index at entry (for time invalidation)
        "entry_bar_idx": None,          # set after fetch in check_and_close
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
        f"Score: {setup_score}/13  Risk: ${trade_risk}  "
        f"Chain: {signal['chain'].upper()}  "
        f"Session: {session}  "
        f"Regime: {signal['regime']}",
    )
    return trade


def check_and_close_trades(state: Dict):
    """
    TASK 4 UPDATED: For each open trade, fetch latest candle and check:
    1. Break-even trigger at 1.5R (CHANGED from 1.8R — earlier protection)
    2. Partial exit at 2R (close 50% of position)
    3. Trailing stop after 2R using LTF swing lows/highs
    4. SL or TP hit at full position
    5. TASK 9: Time-based invalidation (close if no 1R after max bars)

    Paper fill: if current candle’s range includes SL or TP, close it.
    """
    still_open = []
    for trade in state["open_trades"]:
        if trade.get("disabled"):
            still_open.append(trade)
            continue
        sym = trade["symbol"]

        # Safe chain lookup (check active first, then retired)
        chain_name  = trade["chain"]
        all_alignments = {**ALIGNMENTS, **RETIRED_ALIGNMENTS}

        if chain_name not in all_alignments:
            event(state, "WARN", f"Unknown chain '{chain_name}' for {sym} [{trade['trade_id']}]. Using daily as fallback.")
            chain = ALIGNMENTS.get("intraday", list(ALIGNMENTS.values())[0])
        else:
            chain = all_alignments[chain_name]

        tf_ltf = chain["entry"]
        ex_id  = trade.get("exchange_id", "binance")

        try:
            df = cisd.fetch_live_ohlcv(sym, tf_ltf, 20, ex_id)  # fetch last 20 bars
        except Exception:
            still_open.append(trade)
            continue

        if df.empty:
            still_open.append(trade)
            continue

        # ── Set entry bar index if not already stored (Task 9) ──
        if trade.get("entry_bar_idx") is None:
            trade["entry_bar_idx"] = max(0, len(df) - 5)  # approx: within last 5 bars

        latest    = df.iloc[-1]
        hi        = latest["high"]
        lo        = latest["low"]
        close_now = latest["close"]

        sl        = trade["sl_price"]
        tp        = trade["tp_price"]
        direction = trade["direction"]
        entry     = trade["entry_price"]
        risk      = abs(entry - sl)

        # ── TASK 4: Calculate current R multiple ──
        if direction == "LONG":
            current_r = (close_now - entry) / risk if risk > 0 else 0
        else:
            current_r = (entry - close_now) / risk if risk > 0 else 0

        # Update max R reached (for time invalidation check)
        trade["max_r_reached"] = max(trade.get("max_r_reached", 0), current_r)

        # ── TASK 9: Time-based invalidation (skip for manual trades) ──
        if not trade.get("manual") and check_time_invalidation(trade, len(df) - 1, df):
            result     = "TIME_EXPIRED"
            exit_price = close_now
            pnl        = 0.0  # close at market (approx breakeven for paper trading)
            state["balance"] += pnl
            update_peak_balance(state)
            trade.update({
                "closed_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "exit_price": round(exit_price, 8),
                "result"    : result,
                "pnl_usd"   : pnl,
                "balance_after": round(state["balance"], 2),
                "notes"     : f"Time expired after {len(df)} bars with max_r={trade['max_r_reached']:.2f}R",
            })
            state["closed_trades"].append(trade)
            log_trade_to_csv(trade)
            event(state, "TIME_EXP",
                  f"{trade['trade_id']} {sym} {direction} closed (time invalidation, {trade['max_r_reached']:.2f}R reached)")
            continue

        result     = None
        exit_price = close_now

        # ── TASK 4: Break-even trigger at 1.5R (CHANGED from 1.8R) ──
        if not trade.get("be_triggered", False):
            be_r_hit = (
                (direction == "LONG"  and hi >= entry + risk * BE_TRIGGER_R) or
                (direction == "SHORT" and lo <= entry - risk * BE_TRIGGER_R)
            )
            if be_r_hit:
                trade["sl_price"]    = entry  # move stop to entry price (breakeven)
                trade["be_triggered"] = True
                sl = entry  # update local variable too
                event(state, "BE",
                      f"{trade['trade_id']} {sym} — SL moved to breakeven at {format_price(entry)} (1.5R hit)")

        # ── TASK 4: Partial exit at 2R ──
        # Note: In paper trading we simulate this by halving the pnl on a full TP hit.
        # In a real system you would close half the position here.
        if not trade.get("partial_exit_done", False):
            partial_r_hit = (
                (direction == "LONG"  and hi >= entry + risk * PARTIAL_EXIT_R) or
                (direction == "SHORT" and lo <= entry - risk * PARTIAL_EXIT_R)
            )
            if partial_r_hit:
                # Record that partial exit happened
                trade["partial_exit_done"] = True
                trade["trailing_active"]   = True  # start trailing after partial
                partial_pnl = trade.get("trade_reward", REWARD_PER_TRADE) * PARTIAL_EXIT_PCT
                state["balance"] += partial_pnl
                update_peak_balance(state)
                event(state, "PARTIAL",
                      f"{trade['trade_id']} {sym} — 50% closed at 2R (+${partial_pnl:.2f}). "
                      f"Trailing now active. Balance: ${state['balance']:.2f}")

        # ── TASK 4: Trailing stop after 2R (move to each new swing point) ──
        if trade.get("trailing_active", False):
            # Use the last 5 bars to find the most recent LTF swing low (longs) / swing high (shorts)
            lookback_bars = df.tail(5)
            if direction == "LONG" and len(lookback_bars) >= 3:
                # Find new swing lows in recent bars
                for k in range(len(lookback_bars) - 1, 0, -1):
                    lo_val = lookback_bars["low"].iloc[k]
                    lo_l   = lookback_bars["low"].iloc[k - 1]
                    lo_r   = lookback_bars["low"].iloc[-1] if k < len(lookback_bars) - 1 else close_now
                    if lo_val < lo_l:  # potential swing low
                        new_trail_sl = lo_val
                        current_trail = trade.get("trailing_sl") if trade.get("trailing_sl") is not None else sl
                        # NEVER move the trailing stop backwards (only tighten)
                        if new_trail_sl > current_trail:
                            trade["trailing_sl"] = new_trail_sl
                            trade["sl_price"]    = new_trail_sl
                            sl = new_trail_sl
                            event(state, "TRAIL",
                                  f"{trade['trade_id']} {sym} trailing SL -> {format_price(new_trail_sl)}")
                        break
            elif direction == "SHORT" and len(lookback_bars) >= 3:
                # Find new swing highs in recent bars
                for k in range(len(lookback_bars) - 1, 0, -1):
                    hi_val = lookback_bars["high"].iloc[k]
                    hi_l   = lookback_bars["high"].iloc[k - 1]
                    if hi_val > hi_l:  # potential swing high
                        new_trail_sl = hi_val
                        current_trail = trade.get("trailing_sl") if trade.get("trailing_sl") is not None else sl
                        # NEVER move trailing stop backwards
                        if new_trail_sl < current_trail:
                            trade["trailing_sl"] = new_trail_sl
                            trade["sl_price"]    = new_trail_sl
                            sl = new_trail_sl
                            event(state, "TRAIL",
                                  f"{trade['trade_id']} {sym} trailing SL -> {format_price(new_trail_sl)}")
                        break

        # ── Standard SL / TP check ──
        # Use trade-specific risk/reward (Task 6: variable sizing)
        # Falls back to global RISK_PER_TRADE if not stored
        trade_risk_amt   = trade.get("trade_risk",   RISK_PER_TRADE)
        trade_reward_amt = trade.get("trade_reward", REWARD_PER_TRADE)

        if direction == "LONG":
            if lo <= sl:      # SL hit
                result     = "BE" if trade.get("be_triggered") else "LOSS"
                exit_price = sl
                # If partial exit was done, we already locked in 50% profit
                # The remaining 50% is lost at SL
                if trade.get("partial_exit_done"):
                    pnl = -(trade_risk_amt * PARTIAL_EXIT_PCT)  # lose 50% of risk on remaining
                else:
                    pnl = 0.0 if result == "BE" else -trade_risk_amt
            elif hi >= tp:    # TP hit
                result     = "WIN"
                exit_price = tp
                if trade.get("partial_exit_done"):
                    # Already got 50% at 2R; collect remaining 50% at TP
                    pnl = trade_reward_amt * PARTIAL_EXIT_PCT
                else:
                    pnl = trade_reward_amt
        else:  # SHORT
            if hi >= sl:      # SL hit
                result     = "BE" if trade.get("be_triggered") else "LOSS"
                exit_price = sl
                if trade.get("partial_exit_done"):
                    pnl = -(trade_risk_amt * PARTIAL_EXIT_PCT)
                else:
                    pnl = 0.0 if result == "BE" else -trade_risk_amt
            elif lo <= tp:    # TP hit
                result     = "WIN"
                exit_price = tp
                if trade.get("partial_exit_done"):
                    pnl = trade_reward_amt * PARTIAL_EXIT_PCT
                else:
                    pnl = trade_reward_amt

        if result is None:
            still_open.append(trade)
            continue

        # Close the trade
        state["balance"] += pnl
        update_peak_balance(state)  # Task 8: update peak on every balance change

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
        elif result == "LOSS":
            state["losses"] += 1

        state["closed_trades"].append(trade)
        log_trade_to_csv(trade)

        pnl_label = f"+${abs(pnl):.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        event(
            state,
            result,
            f"{trade['trade_id']} {sym} {direction}  "
            f"{pnl_label}  Balance: ${state['balance']:.2f}  "
            f"Session: {trade.get('session','?')}  "
            f"Bias: {trade['bias_reason'][:60]}",
        )

        # Task 10: After every N closed trades, compare live to backtest
        n_closed = len(state["closed_trades"])
        if n_closed > 0 and n_closed % LIVE_BACKTEST_CHECK_N == 0:
            _run_live_backtest_check(state)

    state["open_trades"] = still_open


def _run_live_backtest_check(state: Dict):
    """
    TASK 10 — Internal helper: run the live vs backtest comparison
    and halt if deviation is too large.
    These backtest stats should ideally come from your last backtest run.
    For now we use conservative reference values.
    """
    # Reference backtest stats (update these after each backtest run)
    REFERENCE_BACKTEST_STATS = {
        "win_rate"      : 0.45,   # 45% win rate from last backtest
        "profit_factor" : 1.50,   # 1.5 PF from last backtest
    }

    def halt_trading():
        state["status"] = "STOPPED"
        state["halt_reason"] = "live_backtest_deviation"
        save_state(state)

    result = compare_live_to_backtest(
        live_trades    = state["closed_trades"],
        backtest_stats = REFERENCE_BACKTEST_STATS,
        halt_fn        = halt_trading,
    )
    if result.get("checked"):
        event(state, "LIVE_CHECK",
              f"Live vs Backtest: WR={result['live_wr']:.1%} "
              f"| PF={result['live_pf']} "
              f"| Deviation={result['max_deviation']:.1%} "
              f"| Action={result['action']}")



def check_challenge_status(state: Dict) -> str:
    """Returns ACTIVE | PASSED | STOPPED | WEEKLY_HALT."""
    bal = state["balance"]
    if bal >= CHALLENGE_TARGET:
        return "PASSED"
    # Task 8: trailing halt check (in addition to fixed floor)
    if check_drawdown_halt(state):
        return state["status"]  # already set inside check_drawdown_halt
    if bal <= CHALLENGE_FLOOR:
        return "STOPPED"
    if state.get("status") == "WEEKLY_HALT":
        return "WEEKLY_HALT"
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
        if pt.get("disabled"):
            still_pending.append(pt)
            continue
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
                    # FILL LOGIC: We assume fill at the intended LIMIT price 'ent'
                    # unless the candle closed even BETTER (cur_price), in which
                    # case we take the better price but KEEP the structural SL.
                    # This fixes the "tight SL" bug where entry=close was too near SL.
                    
                    fill_price = ent # default to limit
                    if dire == "LONG" and cur_price < ent:
                        fill_price = cur_price
                    elif dire == "SHORT" and cur_price > ent:
                        fill_price = cur_price
                    
                    # ── MINIMUM STOP DISTANCE ENFORCEMENT ──
                    is_btc = "BTC" in pt["symbol"]
                    min_sl_dist = MIN_SL_PCT_BTC if is_btc else MIN_SL_PCT_ALTS
                    min_dist_price = fill_price * min_sl_dist
                    
                    actual_dist = abs(fill_price - sl)
                    
                    # If SL is too tight (< 0.3%/0.5%), we shift it slightly to survive noise
                    # OR we can discard the trade if preferred. Here we shift it.
                    if actual_dist < min_dist_price:
                        if dire == "LONG":
                            sl = fill_price - min_dist_price
                        else:
                            sl = fill_price + min_dist_price
                        actual_dist = min_dist_price
                    
                    pt["entry_price"] = round(fill_price, 8)
                    pt["sl_price"] = round(sl, 8)
                    
                    # Recalculate TP based on the NEW entry and 2R risk
                    if dire == "LONG":
                        pt["tp_price"] = round(fill_price + (actual_dist * RISK_REWARD), 8)
                    else:
                        pt["tp_price"] = round(fill_price - (actual_dist * RISK_REWARD), 8)
                        
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

    # 1. ALWAYS Manage existing positions first! (Even if cap is hit)
    if state.get("pending_trades"):
        event(state, "CHECK", f"Checking {len(state['pending_trades'])} pending trade(s)...")
        check_pending_trades(state)

    if state["open_trades"]:
        event(state, "CHECK", f"Checking {len(state['open_trades'])} open trade(s)...")
        check_and_close_trades(state)

    # 2. Check challenge status after closes
    state["status"] = check_challenge_status(state)
    save_state(state)
    if state["status"] != "ACTIVE":
        return

    # ── Daily trade cap ————————————————————————————————
    # (Only prevents STARTING new trades or processing new signals)
    today_str = datetime.now().strftime("%Y-%m-%d")
    trades_today = sum(
        1 for t in state.get("closed_trades", []) + state.get("open_trades", [])
        if t.get("opened_at", "").startswith(today_str)
    ) + sum(
        1 for t in state.get("pending_trades", [])
        if t.get("expires_at", "")[:10] == today_str
    )
    if trades_today >= MAX_TRADES_PER_DAY:
        event(state, "SCAN", f"Daily trade cap hit ({trades_today}/{MAX_TRADES_PER_DAY}). Waiting for tomorrow.")
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
                    # ── QUALITY GATE 1: Minimum Setup Tier ──
                    tier_rank = {"S": -1, "A++": 0, "A": 1, "B": 2}
                    min_rank  = {"S": -1, "A++": 0, "A": 1, "B": 2}.get(MIN_SETUP_TIER, 1)
                    sig_rank  = tier_rank.get(sig.get("setup_tier", "B"), 2)
                    if sig_rank > min_rank:
                        print(f"SKIP (tier {sig['setup_tier']} below minimum {MIN_SETUP_TIER})")
                        continue

                    # ── QUALITY GATE 2: Displacement Strength ──
                    disp = sig.get("metadata", {}).get("displacement_strength", 0)
                    if disp < MIN_DISPLACEMENT_STRENGTH:
                        print(f"SKIP (displacement {disp:.2f}x below {MIN_DISPLACEMENT_STRENGTH}x)")
                        continue

                    # Skip this FVG if we already traded it this session
                    if sig["signal_key"] in state.get("visited_signals", []):
                        print(f"SKIP (already traded this FVG zone)")
                    else:
                        cur  = sig["current_price"]
                        ent  = sig["entry_price"]
                        dire = sig["direction"]
                        sl   = sig["sl_price"]
                        
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


def _print_startup_summary(state: Dict):
    """
    Print a summary of current system state on startup.
    Shows: regime (from last known), active session, open trades,
    current drawdown from peak.
    WHY: Gives an instant snapshot before the first scan runs.
    """
    bal    = state["balance"]
    peak   = state.get("peak_balance", CHALLENGE_START)
    dd_pct = ((peak - bal) / peak * 100) if peak > 0 else 0
    halt_lvl = peak * (1.0 - TRAILING_HALT_PCT)

    session  = get_current_session()
    n_open   = len(state.get("open_trades", []))
    n_closed = len(state.get("closed_trades", []))
    w        = state.get("wins", 0)
    l        = state.get("losses", 0)

    print("\n" + "=" * 62)
    print("  SMC TRADING ENGINE — STARTUP SUMMARY")
    print("=" * 62)
    print(f"  Balance         : ${bal:.2f}  (Peak: ${peak:.2f})")
    print(f"  Drawdown        : {dd_pct:.1f}% from peak  (halt at {TRAILING_HALT_PCT*100:.0f}% = ${halt_lvl:.2f})")
    print(f"  Session (UTC)   : {session}")
    print(f"  Open Trades     : {n_open}/{MAX_OPEN_TRADES}")
    print(f"  Closed Trades   : {n_closed}  ({w}W / {l}L)")
    print(f"  Weekly balance  : ${state.get('weekly_open_balance', bal):.2f} " +
          f"(limit: -{MAX_WEEKLY_LOSS_PCT*100:.0f}% = ${state.get('weekly_open_balance', bal) * (1 - MAX_WEEKLY_LOSS_PCT):.2f})")
    print(f"  BE trigger      : {BE_TRIGGER_R}R  |  Partial exit: {PARTIAL_EXIT_R}R ({PARTIAL_EXIT_PCT*100:.0f}%)")
    print(f"  Halt reason     : {state.get('halt_reason', 'None')}")
    if n_open > 0:
        print("  Open positions  :")
        for t in state["open_trades"]:
            print(f"    [{t['trade_id']}] {t['symbol']} {t['direction']} "
                  f"@ {format_price(t['entry_price'])}  "
                  f"SL:{format_price(t['sl_price'])}  "
                  f"TP:{format_price(t['tp_price'])}  "
                  f"Score:{t.get('setup_score','?')}/13")
    print("=" * 62)


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

    # ── STARTUP: Health check (validates all config) ──
    is_healthy = run_system_health_check()
    if not is_healthy:
        print("\n  [ERROR] Config health check failed. Fix the above errors before running.")
        return

    if args.reset:
        state = reset_state()
        print("  Challenge state reset. Starting fresh.")
    else:
        state = load_state()

    # ── STARTUP: Self-heal state for new keys ──
    defaults = {
        "peak_balance": state.get("balance", CHALLENGE_START),
        "weekly_open_balance": state.get("balance", CHALLENGE_START),
        "weekly_open_date": datetime.now().strftime("%Y-%m-%d"),
        "halt_reason": None,
    }
    for k, v in defaults.items():
        if k not in state:
            state[k] = v

    if state["status"] not in ("ACTIVE", "WEEKLY_HALT"):
        print(f"\n  Challenge already {state['status']}. Use --reset to start fresh.")
        print_final_report(state)
        return

    # ── STARTUP: Print system summary ──
    _print_startup_summary(state)

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
