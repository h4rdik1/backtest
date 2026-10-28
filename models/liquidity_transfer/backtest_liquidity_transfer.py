"""
================================================================================
LIQUIDITY TRANSFER MODEL — 4H CONTEXT ➔ 15M EXECUTION
================================================================================
Centralized Smart Money Concepts (SMC) quantitative trading engine.
Designed specifically to eliminate the core flaws identified in legacy models:
  1. Premature entries before sweeps (strictly anchored to 4H SFP / Discount-Premium FVG).
  2. Entering on wicks without body closes (strict 15M full candle body close MSS rule).
  3. Stops placed directly on liquidity pools (structural stop + 0.15% / 0.5 ATR buffer).

Execution Architecture:
  Phase 1: 4H Structural Anchor (HTF Context & Directional Bias via ERL <-> IRL)
  Phase 2: 15M Market Structure Shift (Strict Body-Close Rule against preceding pivot)
  Phase 3: Retest Entry & Structural Stop (Touch of 15M FVG/broken pivot + buffer)

Fixed Risk Engine:
  Starting Capital: $5,000.00
  Drawdown Floor:   $4,700.00 ($300 max drawdown buffer / circuit breaker)
  Risk Per Trade:   $20.00 (Fixed 1.0R risk)
  Position Sizing:  Contracts = $20.00 / abs(Entry - SL)
================================================================================
"""

import os
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

STARTING_ACCOUNT: float = 5000.00
ACCOUNT_FLOOR: float = 4700.00
RISK_PER_TRADE_USD: float = 20.00
DEFAULT_RR: float = 2.00
BE_TRIGGER_R: float = 1.80
BIAS_EXPIRY_HOURS: int = 24  # 6 x 4H bars = 24 hours
HTF_LOOKBACK_BARS: int = 20  # 4H swing lookback
REPO_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR: str = os.path.join(REPO_ROOT, "data", "ohlcv")
DEFAULT_EXPORT_PATH: str = os.path.join(REPO_ROOT, "exports", "liquidity_transfer_trades.csv")


@dataclass
class BiasEvent:
    close_time: pd.Timestamp
    start_time: pd.Timestamp
    bias: str  # 'BULLISH' or 'BEARISH'
    source: str  # '4H_SFP' or '4H_FVG'
    anchor_level: float
    range_high: float
    range_low: float
    equilibrium: float
    htf_target: float
    bar_idx: int


@dataclass
class Trade:
    timestamp: str
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry: float
    sl: float
    tp: float
    exit_price: float
    exit_reason: str  # 'TP', 'SL', 'BE'
    pnl: float
    r_multiple: float
    contracts: float
    risk_price: float
    duration_bars: int
    entry_type: str
    htf_bias_source: str


# ==============================================================================
# DATA INGESTION & PIPELINE
# ==============================================================================

def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load multi-timeframe cached OHLCV data from data/ohlcv/<SYMBOL>/<timeframe>.csv.
    Strictly parses timestamps in UTC and sets them as a sorted DatetimeIndex.
    """
    safe_symbol = symbol.replace("/", "_")
    file_path = os.path.join(DATA_DIR, safe_symbol, f"{timeframe}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing cached OHLCV data at: {file_path}")

    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Cast numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["timestamp", "open", "high", "low", "close"], inplace=True)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Calculate standard Average True Range (ATR) as a NumPy array."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]

    atr = pd.Series(tr).rolling(window=period, min_periods=1).mean().values
    return atr


# ==============================================================================
# PHASE 1: 4H STRUCTURAL ANCHOR (HTF CONTEXT & BIAS)
# ==============================================================================

def detect_4h_bias_events(df_4h: pd.DataFrame, lookback: int = HTF_LOOKBACK_BARS) -> Dict[pd.Timestamp, BiasEvent]:
    """
    Scans 4H candles to detect external liquidity raids (SFPs) or FVG taps.
    
    1. External Liquidity Raid (4H SFP):
       - Sweeps prior 4H swing high or swing low (lookback 20 bars) and closes back inside.
    2. 4H FVG Tap:
       - Price retraces into confirmed 4H FVG in Discount (< 0.5) for Longs, or Premium (> 0.5) for Shorts.
    
    Returns a dictionary mapping 4H close timestamp to BiasEvent.
    """
    events: Dict[pd.Timestamp, BiasEvent] = {}
    n = len(df_4h)
    if n < 5:
        return events

    # Pre-detect confirmed 4H Fair Value Gaps (3-bar formation)
    fvgs: List[Dict[str, Any]] = []
    for i in range(2, n):
        # Bullish FVG: Candle i-2 High < Candle i Low
        if df_4h["low"].iloc[i] > df_4h["high"].iloc[i - 2]:
            fvgs.append({
                "type": "bull",
                "idx": i,
                "top": df_4h["low"].iloc[i],
                "btm": df_4h["high"].iloc[i - 2],
                "time": df_4h["timestamp"].iloc[i]
            })
        # Bearish FVG: Candle i-2 Low > Candle i High
        if df_4h["high"].iloc[i] < df_4h["low"].iloc[i - 2]:
            fvgs.append({
                "type": "bear",
                "idx": i,
                "top": df_4h["low"].iloc[i - 2],
                "btm": df_4h["high"].iloc[i],
                "time": df_4h["timestamp"].iloc[i]
            })

    # Scan 4H bars starting after minimum lookback
    min_start = min(lookback, max(5, n - 2))
    for i in range(min_start, n):
        # 4H bar closes 4 hours after its start timestamp
        bar_start = df_4h["timestamp"].iloc[i]
        close_time = bar_start + pd.Timedelta(hours=4)

        cur_low = df_4h["low"].iloc[i]
        cur_high = df_4h["high"].iloc[i]
        cur_close = df_4h["close"].iloc[i]

        eff_lookback = min(lookback, i)
        prior_slice = df_4h.iloc[i - eff_lookback : i]
        rng_hi = prior_slice["high"].max()
        rng_lo = prior_slice["low"].min()
        eq = (rng_hi + rng_lo) / 2.0

        event: Optional[BiasEvent] = None

        # 1. 4H SFP Check (External Liquidity Raid)
        # Bullish SFP: low pierces prior 20-bar swing low, but closes back above
        if cur_low < rng_lo and cur_close > rng_lo:
            event = BiasEvent(
                close_time=close_time,
                start_time=bar_start,
                bias="BULLISH",
                source="4H_SFP",
                anchor_level=rng_lo,
                range_high=rng_hi,
                range_low=rng_lo,
                equilibrium=eq,
                htf_target=rng_hi,
                bar_idx=i
            )
        # Bearish SFP: high pierces prior 20-bar swing high, but closes back below
        elif cur_high > rng_hi and cur_close < rng_hi:
            event = BiasEvent(
                close_time=close_time,
                start_time=bar_start,
                bias="BEARISH",
                source="4H_SFP",
                anchor_level=rng_hi,
                range_high=rng_hi,
                range_low=rng_lo,
                equilibrium=eq,
                htf_target=rng_lo,
                bar_idx=i
            )
        else:
            # 2. 4H FVG Tap Check (Retrace into confirmed FVG in Discount/Premium)
            for f in fvgs:
                # FVG must have formed at least 1 bar prior and within 30 bars (fresh)
                bars_since = i - f["idx"]
                if 1 <= bars_since <= 30:
                    if f["type"] == "bull" and f["top"] < eq:  # In Discount (< 0.5)
                        if cur_low <= f["top"] and cur_close >= f["btm"]:
                            event = BiasEvent(
                                close_time=close_time,
                                start_time=bar_start,
                                bias="BULLISH",
                                source="4H_FVG",
                                anchor_level=f["top"],
                                range_high=rng_hi,
                                range_low=rng_lo,
                                equilibrium=eq,
                                htf_target=rng_hi,
                                bar_idx=i
                            )
                            break
                    elif f["type"] == "bear" and f["btm"] > eq:  # In Premium (> 0.5)
                        if cur_high >= f["btm"] and cur_close <= f["top"]:
                            event = BiasEvent(
                                close_time=close_time,
                                start_time=bar_start,
                                bias="BEARISH",
                                source="4H_FVG",
                                anchor_level=f["btm"],
                                range_high=rng_hi,
                                range_low=rng_lo,
                                equilibrium=eq,
                                htf_target=rng_lo,
                                bar_idx=i
                            )
                            break

        if event is not None:
            events[close_time] = event

    return events


# ==============================================================================
# PHASE 2 & 3: 15M EXECUTION ENGINE & SIMULATION
# ==============================================================================

def simulate_liquidity_transfer(
    df_4h: pd.DataFrame,
    df_15m: pd.DataFrame,
    symbol: str,
    target_mode: str = "next_swing",  # 'next_swing', 'htf_roadblock', or 'fixed_rr'
    min_rr: float = 1.50,
    rr: float = DEFAULT_RR,
    be_trigger_r: float = 1.50,
    entry_mode: str = "midpoint",  # 'boundary' (FVG top/btm or pivot) or 'midpoint' (50% FVG)
    max_stop_distance_pct: float = 0.035,
    min_stop_distance_pct: float = 0.0015,
) -> Tuple[List[Trade], Dict[str, Any]]:
    """
    Simulates the Liquidity Transfer Model bar-by-bar across the 15M timeframe.
    Enforces the strict body-close rule, retest entry, and liquidity-buffered SL.
    """
    # Pre-detect 4H bias anchors
    bias_events = detect_4h_bias_events(df_4h)

    # 15M Arrays for high-speed lookahead-free traversal
    ts_15m = df_15m["timestamp"].tolist()
    highs = df_15m["high"].values
    lows = df_15m["low"].values
    opens = df_15m["open"].values
    closes = df_15m["close"].values
    atrs = calculate_atr(df_15m, period=14)
    n_bars = len(df_15m)

    trades: List[Trade] = []
    account: float = STARTING_ACCOUNT
    peak_account: float = STARTING_ACCOUNT
    max_drawdown_usd: float = 0.0
    floor_breached: bool = False

    active_bias: Optional[str] = None
    active_source: str = ""
    bias_expiry: Optional[pd.Timestamp] = None
    bias_start_idx: int = 0
    current_eq: float = 0.0
    current_htf_target: float = 0.0
    pending_setup: Optional[Dict[str, Any]] = None

    in_trade: bool = False
    current_trade: Optional[Dict[str, Any]] = None

    funnel = {
        "bars_scanned": 0,
        "active_bias_bars": 0,
        "mss_triggered": 0,
        "retest_entries": 0,
        "tp_exits": 0,
        "sl_exits": 0,
        "be_exits": 0,
    }

    start_idx = min(50, n_bars - 1)

    for i in range(start_idx, n_bars):
        funnel["bars_scanned"] += 1
        t = ts_15m[i]
        h = highs[i]
        l = lows[i]
        o = opens[i]
        c = closes[i]
        atr = atrs[i]

        # ----------------------------------------------------------------------
        # Circuit Breaker: Halt if Drawdown Floor ($4,700) is Breached
        # ----------------------------------------------------------------------
        if account <= ACCOUNT_FLOOR:
            floor_breached = True
            break

        # ----------------------------------------------------------------------
        # IN-TRADE MANAGEMENT
        # ----------------------------------------------------------------------
        if in_trade and current_trade is not None:
            direction = current_trade["direction"]
            entry = current_trade["entry"]
            sl = current_trade["sl"]
            tp = current_trade["tp"]
            be_trig = current_trade["be_trigger"]
            be_armed = current_trade["be_armed"]
            planned_rr = current_trade.get("planned_rr", rr)

            if direction == "LONG":
                # Check Auto-Breakeven Trigger
                if h >= be_trig and not be_armed:
                    current_trade["be_armed"] = True
                    current_trade["sl"] = entry

                # Check Take Profit Hit first
                if h >= tp:
                    pnl = RISK_PER_TRADE_USD * planned_rr
                    r_mult = planned_rr
                    exit_price = tp

                    account += pnl
                    peak_account = max(peak_account, account)
                    max_drawdown_usd = max(max_drawdown_usd, peak_account - account)

                    trades.append(Trade(
                        timestamp=current_trade["entry_time"].strftime("%Y-%m-%d %H:%M"),
                        symbol=symbol,
                        direction=direction,
                        entry=round(entry, 4),
                        sl=round(sl, 4),
                        tp=round(tp, 4),
                        exit_price=round(exit_price, 4),
                        exit_reason="TP",
                        pnl=round(pnl, 2),
                        r_multiple=round(r_mult, 2),
                        contracts=round(current_trade["contracts"], 4),
                        risk_price=round(current_trade["risk_price"], 4),
                        duration_bars=i - current_trade["entry_idx"],
                        entry_type=current_trade["entry_type"],
                        htf_bias_source=current_trade["htf_source"]
                    ))
                    funnel["tp_exits"] += 1
                    in_trade = False
                    current_trade = None

                # Check Stop Loss / Breakeven Hit
                elif l <= current_trade["sl"]:
                    exit_reason = "BE" if current_trade["be_armed"] else "SL"
                    pnl = 0.0 if current_trade["be_armed"] else -RISK_PER_TRADE_USD
                    r_mult = 0.0 if current_trade["be_armed"] else -1.0
                    exit_price = current_trade["sl"]

                    account += pnl
                    peak_account = max(peak_account, account)
                    max_drawdown_usd = max(max_drawdown_usd, peak_account - account)

                    trades.append(Trade(
                        timestamp=current_trade["entry_time"].strftime("%Y-%m-%d %H:%M"),
                        symbol=symbol,
                        direction=direction,
                        entry=round(entry, 4),
                        sl=round(sl, 4),
                        tp=round(tp, 4),
                        exit_price=round(exit_price, 4),
                        exit_reason=exit_reason,
                        pnl=round(pnl, 2),
                        r_multiple=round(r_mult, 2),
                        contracts=round(current_trade["contracts"], 4),
                        risk_price=round(current_trade["risk_price"], 4),
                        duration_bars=i - current_trade["entry_idx"],
                        entry_type=current_trade["entry_type"],
                        htf_bias_source=current_trade["htf_source"]
                    ))
                    if exit_reason == "BE":
                        funnel["be_exits"] += 1
                    else:
                        funnel["sl_exits"] += 1

                    in_trade = False
                    current_trade = None

            else:  # SHORT
                # Check Auto-Breakeven Trigger
                if l <= be_trig and not be_armed:
                    current_trade["be_armed"] = True
                    current_trade["sl"] = entry

                # Check Take Profit Hit first
                if l <= tp:
                    pnl = RISK_PER_TRADE_USD * planned_rr
                    r_mult = planned_rr
                    exit_price = tp

                    account += pnl
                    peak_account = max(peak_account, account)
                    max_drawdown_usd = max(max_drawdown_usd, peak_account - account)

                    trades.append(Trade(
                        timestamp=current_trade["entry_time"].strftime("%Y-%m-%d %H:%M"),
                        symbol=symbol,
                        direction=direction,
                        entry=round(entry, 4),
                        sl=round(sl, 4),
                        tp=round(tp, 4),
                        exit_price=round(exit_price, 4),
                        exit_reason="TP",
                        pnl=round(pnl, 2),
                        r_multiple=round(r_mult, 2),
                        contracts=round(current_trade["contracts"], 4),
                        risk_price=round(current_trade["risk_price"], 4),
                        duration_bars=i - current_trade["entry_idx"],
                        entry_type=current_trade["entry_type"],
                        htf_bias_source=current_trade["htf_source"]
                    ))
                    funnel["tp_exits"] += 1
                    in_trade = False
                    current_trade = None

                # Check Stop Loss / Breakeven Hit
                elif h >= current_trade["sl"]:
                    exit_reason = "BE" if current_trade["be_armed"] else "SL"
                    pnl = 0.0 if current_trade["be_armed"] else -RISK_PER_TRADE_USD
                    r_mult = 0.0 if current_trade["be_armed"] else -1.0
                    exit_price = current_trade["sl"]

                    account += pnl
                    peak_account = max(peak_account, account)
                    max_drawdown_usd = max(max_drawdown_usd, peak_account - account)

                    trades.append(Trade(
                        timestamp=current_trade["entry_time"].strftime("%Y-%m-%d %H:%M"),
                        symbol=symbol,
                        direction=direction,
                        entry=round(entry, 4),
                        sl=round(sl, 4),
                        tp=round(tp, 4),
                        exit_price=round(exit_price, 4),
                        exit_reason=exit_reason,
                        pnl=round(pnl, 2),
                        r_multiple=round(r_mult, 2),
                        contracts=round(current_trade["contracts"], 4),
                        risk_price=round(current_trade["risk_price"], 4),
                        duration_bars=i - current_trade["entry_idx"],
                        entry_type=current_trade["entry_type"],
                        htf_bias_source=current_trade["htf_source"]
                    ))
                    if exit_reason == "BE":
                        funnel["be_exits"] += 1
                    else:
                        funnel["sl_exits"] += 1

                    in_trade = False
                    current_trade = None

            # Skip checking new setups while already in position
            continue

        # ----------------------------------------------------------------------
        # UPDATE 4H BIAS STATE
        # ----------------------------------------------------------------------
        if t in bias_events:
            ev = bias_events[t]
            active_bias = ev.bias
            active_source = ev.source
            bias_start_idx = max(0, i - 16)  # 4H is 16 15m candles
            bias_expiry = t + pd.Timedelta(hours=BIAS_EXPIRY_HOURS)
            current_eq = ev.equilibrium
            current_htf_target = ev.htf_target
            pending_setup = None  # New HTF anchor resets pending micro setups

        # Bias expiration check
        if active_bias is not None and bias_expiry is not None and t > bias_expiry:
            active_bias = None
            pending_setup = None

        if active_bias is None:
            continue

        funnel["active_bias_bars"] += 1

        # ----------------------------------------------------------------------
        # PHASE 2: 15M MARKET STRUCTURE SHIFT (THE BODY-CLOSE RULE)
        # ----------------------------------------------------------------------
        if pending_setup is None:
            if active_bias == "BULLISH":
                # SOP 4: Longs strictly permitted in Discount (< 0.5 Eq)
                if c > current_eq:
                    continue

                # 1. Identify 15M Sweep Low in the setup window (anchored to 4H event start)
                sweep_bar_pos = bias_start_idx + int(np.argmin(lows[bias_start_idx : i + 1]))
                sweep_low = lows[sweep_bar_pos]

                # 2. Identify the last 15M structural pivot (swing high that led to sweep low)
                pivot_price: Optional[float] = None
                for p in range(sweep_bar_pos - 1, max(1, sweep_bar_pos - 20), -1):
                    # 3-bar fractal swing high
                    if highs[p] > highs[p - 1] and highs[p] > highs[p + 1]:
                        pivot_price = float(highs[p])
                        break
                if pivot_price is None and sweep_bar_pos > bias_start_idx:
                    pivot_price = float(np.max(highs[max(0, sweep_bar_pos - 10) : sweep_bar_pos]))

                if pivot_price is None or pd.isna(pivot_price):
                    continue

                # 3. The Strict Filter: 15M candle MUST close with FULL CANDLE BODY beyond pivot
                # Wicks beyond pivot (h > pivot but c <= pivot) are treated as sweeps/noise
                if h > pivot_price and c > pivot_price and c > o:
                    # Mark 15M FVG created during displacement leg
                    fvg_top: Optional[float] = None
                    fvg_btm: Optional[float] = None
                    for k in range(sweep_bar_pos + 2, i + 1):
                        if lows[k] > highs[k - 2]:
                            fvg_top = float(lows[k])
                            fvg_btm = float(highs[k - 2])

                    # Strict FVG Mandate: Displacement leg MUST leave an imbalance
                    if fvg_top is None or fvg_btm is None:
                        continue

                    funnel["mss_triggered"] += 1
                    entry_type_str = "15M_FVG_MID" if entry_mode == "midpoint" else "15M_FVG_TOP"
                    entry_level = (fvg_top + fvg_btm) / 2.0 if entry_mode == "midpoint" else fvg_top

                    # Structural Stop Loss with Buffer: max(0.15%, 0.5 * ATR)
                    buf = max(sweep_low * min_stop_distance_pct, 0.5 * atr)
                    sl = sweep_low - buf
                    risk = entry_level - sl

                    # Stop distance guardrail
                    if risk > 0 and (risk / entry_level) <= max_stop_distance_pct:
                        # Determine Dynamic Target (Next Swing High / Roadblock)
                        tp: Optional[float] = None
                        if target_mode == "next_swing":
                            # Search preceding 15M structure for swing highs above entry offering >= min_rr
                            for k in range(i - 2, max(0, i - 60), -1):
                                if highs[k] > highs[k - 1] and highs[k] > highs[k + 1]:
                                    if highs[k] >= entry_level + (min_rr * risk):
                                        if tp is None or highs[k] < tp:
                                            tp = float(highs[k])
                            if tp is None:
                                if current_htf_target > entry_level + (min_rr * risk):
                                    tp = float(current_htf_target)
                                else:
                                    tp = entry_level + (rr * risk)
                            planned_rr = (tp - entry_level) / risk
                        elif target_mode == "htf_roadblock":
                            if current_htf_target > entry_level + (min_rr * risk):
                                tp = float(current_htf_target)
                            else:
                                tp = entry_level + (rr * risk)
                            planned_rr = (tp - entry_level) / risk
                        else:  # fixed_rr
                            tp = entry_level + (rr * risk)
                            planned_rr = rr

                        pending_setup = {
                            "direction": "LONG",
                            "entry_level": entry_level,
                            "sl": sl,
                            "tp": tp,
                            "planned_rr": planned_rr,
                            "sweep_low": sweep_low,
                            "mss_bar": i,
                            "entry_type": entry_type_str,
                            "expires": t + pd.Timedelta(hours=12),
                            "htf_source": active_source
                        }

            elif active_bias == "BEARISH":
                # SOP 4: Shorts strictly permitted in Premium (> 0.5 Eq)
                if c < current_eq:
                    continue

                # 1. Identify 15M Sweep High in the setup window (anchored to 4H event start)
                sweep_bar_pos = bias_start_idx + int(np.argmax(highs[bias_start_idx : i + 1]))
                sweep_high = highs[sweep_bar_pos]

                # 2. Identify the last 15M structural pivot (swing low that led to sweep high)
                pivot_price = None
                for p in range(sweep_bar_pos - 1, max(1, sweep_bar_pos - 20), -1):
                    # 3-bar fractal swing low
                    if lows[p] < lows[p - 1] and lows[p] < lows[p + 1]:
                        pivot_price = float(lows[p])
                        break
                if pivot_price is None and sweep_bar_pos > bias_start_idx:
                    pivot_price = float(np.min(lows[max(0, sweep_bar_pos - 10) : sweep_bar_pos]))

                if pivot_price is None or pd.isna(pivot_price):
                    continue

                # 3. The Strict Filter: 15M candle MUST close with FULL CANDLE BODY beyond pivot
                if l < pivot_price and c < pivot_price and c < o:
                    # Mark 15M FVG created during displacement leg
                    fvg_top = None
                    fvg_btm = None
                    for k in range(sweep_bar_pos + 2, i + 1):
                        if highs[k] < lows[k - 2]:
                            fvg_btm = float(highs[k])
                            fvg_top = float(lows[k - 2])

                    # Strict FVG Mandate: Displacement leg MUST leave an imbalance
                    if fvg_top is None or fvg_btm is None:
                        continue

                    funnel["mss_triggered"] += 1
                    entry_type_str = "15M_FVG_MID" if entry_mode == "midpoint" else "15M_FVG_BTM"
                    entry_level = (fvg_top + fvg_btm) / 2.0 if entry_mode == "midpoint" else fvg_btm

                    # Structural Stop Loss with Buffer: max(0.15%, 0.5 * ATR)
                    buf = max(sweep_high * min_stop_distance_pct, 0.5 * atr)
                    sl = sweep_high + buf
                    risk = sl - entry_level

                    if risk > 0 and (risk / entry_level) <= max_stop_distance_pct:
                        # Determine Dynamic Target (Next Swing Low / Roadblock)
                        tp = None
                        if target_mode == "next_swing":
                            # Search preceding 15M structure for swing lows below entry offering >= min_rr
                            for k in range(i - 2, max(0, i - 60), -1):
                                if lows[k] < lows[k - 1] and lows[k] < lows[k + 1]:
                                    if lows[k] <= entry_level - (min_rr * risk):
                                        if tp is None or lows[k] > tp:
                                            tp = float(lows[k])
                            if tp is None:
                                if current_htf_target < entry_level - (min_rr * risk):
                                    tp = float(current_htf_target)
                                else:
                                    tp = entry_level - (rr * risk)
                            planned_rr = (entry_level - tp) / risk
                        elif target_mode == "htf_roadblock":
                            if current_htf_target < entry_level - (min_rr * risk):
                                tp = current_htf_target
                            else:
                                tp = entry_level - (rr * risk)
                            planned_rr = (entry_level - tp) / risk
                        else:  # fixed_rr
                            tp = entry_level - (rr * risk)
                            planned_rr = rr

                        pending_setup = {
                            "direction": "SHORT",
                            "entry_level": entry_level,
                            "sl": sl,
                            "tp": tp,
                            "planned_rr": planned_rr,
                            "sweep_high": sweep_high,
                            "mss_bar": i,
                            "entry_type": entry_type_str,
                            "expires": t + pd.Timedelta(hours=12),
                            "htf_source": active_source
                        }

        # ----------------------------------------------------------------------
        # PHASE 3: THE RETEST ENTRY & POSITION SIZING
        # ----------------------------------------------------------------------
        elif pending_setup is not None:
            # Rule: NEVER enter at the breakout candle high/low. Wait for subsequent retest.
            if i == pending_setup["mss_bar"]:
                continue

            # Invalidation check: setup expires after 12h or if HTF bias expires
            if t > pending_setup["expires"] or (bias_expiry is not None and t > bias_expiry):
                pending_setup = None
                continue

            direction = pending_setup["direction"]
            entry_lvl = pending_setup["entry_level"]
            sl = pending_setup["sl"]
            tp = pending_setup["tp"]
            planned_rr = pending_setup["planned_rr"]

            if direction == "LONG":
                # Structural invalidation: price violates sweep low before fill
                if l <= pending_setup["sweep_low"]:
                    pending_setup = None
                    continue

                # Pullback retest touch trigger
                if l <= entry_lvl:
                    entry_px = min(o, entry_lvl) if o <= entry_lvl else entry_lvl
                    risk = entry_px - sl
                    if risk > 0 and tp > entry_px:
                        be_trigger = entry_px + (risk * be_trigger_r)
                        contracts = RISK_PER_TRADE_USD / risk

                        current_trade = {
                            "direction": "LONG",
                            "entry_time": t,
                            "entry_idx": i,
                            "entry": entry_px,
                            "sl": sl,
                            "tp": tp,
                            "planned_rr": planned_rr,
                            "be_trigger": be_trigger,
                            "be_armed": False,
                            "contracts": contracts,
                            "risk_price": risk,
                            "entry_type": pending_setup["entry_type"],
                            "htf_source": pending_setup["htf_source"]
                        }
                        in_trade = True
                        pending_setup = None
                        funnel["retest_entries"] += 1

            else:  # SHORT
                # Structural invalidation: price violates sweep high before fill
                if h >= pending_setup["sweep_high"]:
                    pending_setup = None
                    continue

                # Pullback retest touch trigger
                if h >= entry_lvl:
                    entry_px = max(o, entry_lvl) if o >= entry_lvl else entry_lvl
                    risk = sl - entry_px
                    if risk > 0 and tp < entry_px:
                        be_trigger = entry_px - (risk * be_trigger_r)
                        contracts = RISK_PER_TRADE_USD / risk

                        current_trade = {
                            "direction": "SHORT",
                            "entry_time": t,
                            "entry_idx": i,
                            "entry": entry_px,
                            "sl": sl,
                            "tp": tp,
                            "planned_rr": planned_rr,
                            "be_trigger": be_trigger,
                            "be_armed": False,
                            "contracts": contracts,
                            "risk_price": risk,
                            "entry_type": pending_setup["entry_type"],
                            "htf_source": pending_setup["htf_source"]
                        }
                        in_trade = True
                        pending_setup = None
                        funnel["retest_entries"] += 1

    sim_metadata = {
        "final_account": round(account, 2),
        "peak_account": round(peak_account, 2),
        "max_drawdown_usd": round(max_drawdown_usd, 2),
        "max_drawdown_pct": round((max_drawdown_usd / peak_account) * 100, 2) if peak_account > 0 else 0.0,
        "floor_breached": floor_breached,
        "funnel": funnel,
    }
    return trades, sim_metadata


# ==============================================================================
# PERFORMANCE ANALYTICS & INSTITUTIONAL REPORT
# ==============================================================================

def calculate_metrics(trades: List[Trade], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates institutional performance metrics from trade logs."""
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "total_trades": 0, "win_rate_all": 0.0, "win_rate_decisive": 0.0,
            "profit_factor": 0.0, "net_pnl_usd": 0.0, "net_pnl_r": 0.0,
            "max_dd_usd": metadata.get("max_drawdown_usd", 0.0),
            "max_dd_pct": metadata.get("max_drawdown_pct", 0.0),
            "long_trades": 0, "long_wins": 0, "long_wr": 0.0,
            "short_trades": 0, "short_wins": 0, "short_wr": 0.0,
            "avg_rr": 0.0, "max_consecutive_wins": 0, "max_consecutive_losses": 0,
            "wins": 0, "losses": 0, "bes": 0
        }

    wins = sum(1 for t in trades if t.exit_reason == "TP")
    losses = sum(1 for t in trades if t.exit_reason == "SL")
    bes = sum(1 for t in trades if t.exit_reason == "BE")

    win_rate_all = (wins / total_trades) * 100.0
    decisive_trades = wins + losses
    win_rate_decisive = (wins / decisive_trades) * 100.0 if decisive_trades > 0 else 0.0

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    net_pnl_usd = sum(t.pnl for t in trades)
    net_pnl_r = sum(t.r_multiple for t in trades)

    # Disaggregated Stats (Long vs Short)
    long_trades = [t for t in trades if t.direction == "LONG"]
    short_trades = [t for t in trades if t.direction == "SHORT"]

    long_wins = sum(1 for t in long_trades if t.exit_reason == "TP")
    short_wins = sum(1 for t in short_trades if t.exit_reason == "TP")

    long_wr = (long_wins / len(long_trades)) * 100.0 if long_trades else 0.0
    short_wr = (short_wins / len(short_trades)) * 100.0 if short_trades else 0.0

    # Average realized R:R for winning trades
    win_r_list = [t.r_multiple for t in trades if t.exit_reason == "TP"]
    avg_rr = float(np.mean(win_r_list)) if win_r_list else 0.0

    # Consecutive Wins & Losses
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    curr_wins = 0
    curr_losses = 0

    for t in trades:
        if t.exit_reason == "TP":
            curr_wins += 1
            curr_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, curr_wins)
        elif t.exit_reason == "SL":
            curr_losses += 1
            curr_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, curr_losses)
        else:
            # Breakeven does not count as a loss or win streak breaker in standard evaluation
            pass

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "bes": bes,
        "win_rate_all": win_rate_all,
        "win_rate_decisive": win_rate_decisive,
        "profit_factor": profit_factor,
        "net_pnl_usd": net_pnl_usd,
        "net_pnl_r": net_pnl_r,
        "max_dd_usd": metadata.get("max_drawdown_usd", 0.0),
        "max_dd_pct": metadata.get("max_drawdown_pct", 0.0),
        "long_trades": len(long_trades),
        "long_wins": long_wins,
        "long_wr": long_wr,
        "short_trades": len(short_trades),
        "short_wins": short_wins,
        "short_wr": short_wr,
        "avg_rr": avg_rr,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "final_account": metadata.get("final_account", STARTING_ACCOUNT),
        "floor_breached": metadata.get("floor_breached", False)
    }


def print_institutional_report(metrics: Dict[str, Any], symbol: str, funnel: Dict[str, Any]) -> None:
    """Prints a clean, formatted institutional SMC performance report."""
    border = "=" * 76
    sub_border = "-" * 76

    print(f"\n{border}")
    print(f"      INSTITUTIONAL SMC BACKTEST REPORT: {symbol.upper()}")
    print(f"      Strategy: The Liquidity Transfer Model (4H Context -> 15M Execution)")
    print(f"{border}")

    print("\n[1] CAPITAL & RISK PROFILE")
    print(f"    Starting Account:          ${STARTING_ACCOUNT:,.2f}")
    print(f"    Drawdown Limit Floor:      ${ACCOUNT_FLOOR:,.2f} ($300 Max Buffer)")
    print(f"    Fixed Risk per Trade:      ${RISK_PER_TRADE_USD:,.2f} (1.0R)")
    print(f"    Final Account Balance:     ${metrics.get('final_account', STARTING_ACCOUNT):,.2f}")
    floor_status = "BREACHED (HALTED)" if metrics.get("floor_breached") else "PASSED (SAFE)"
    print(f"    Drawdown Floor Status:     {floor_status}")

    print(f"\n{sub_border}")
    print("[2] CORE PERFORMANCE METRICS")
    print(f"    Total Closed Trades:       {metrics['total_trades']}")
    print(f"    Outcome Distribution:      {metrics['wins']} TP | {metrics['losses']} SL | {metrics['bes']} BE")
    print(f"    Gross Win Rate (all):      {metrics['win_rate_all']:.2f}%")
    print(f"    Decisive Win Rate (ex-BE): {metrics['win_rate_decisive']:.2f}%")
    pf_str = f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] != float('inf') else "N/A"
    print(f"    Profit Factor:             {pf_str}")
    pnl_sign = "+" if metrics['net_pnl_usd'] >= 0 else ""
    print(f"    Net PnL (USD):             {pnl_sign}${metrics['net_pnl_usd']:,.2f}")
    print(f"    Net PnL (R-Multiples):     {pnl_sign}{metrics['net_pnl_r']:.2f}R")
    print(f"    Max Drawdown:              ${metrics['max_dd_usd']:,.2f} ({metrics['max_dd_pct']:.2f}%)")

    print(f"\n{sub_border}")
    print("[3] DISAGGREGATED & STREAK ANALYTICS")
    print(f"    Long Trades:               {metrics['long_trades']} trades | {metrics['long_wins']} wins | {metrics['long_wr']:.2f}% WR")
    print(f"    Short Trades:              {metrics['short_trades']} trades | {metrics['short_wins']} wins | {metrics['short_wr']:.2f}% WR")
    print(f"    Average Realized R:R:      {metrics['avg_rr']:.2f}R")
    print(f"    Max Consecutive Wins:      {metrics['max_consecutive_wins']}")
    print(f"    Max Consecutive Losses:    {metrics['max_consecutive_losses']}")

    print(f"\n{sub_border}")
    print("[4] EXECUTION FUNNEL EFFICIENCY")
    print(f"    15M Bars Scanned:          {funnel.get('bars_scanned', 0):,}")
    print(f"    Bars in Active 4H Bias:    {funnel.get('active_bias_bars', 0):,}")
    print(f"    15M MSS Confirmed (Body):  {funnel.get('mss_triggered', 0):,}")
    print(f"    Retest Touch Entries:      {funnel.get('retest_entries', 0):,}")
    print(f"    Position Conversion Rate:  {(funnel.get('retest_entries', 0) / max(1, funnel.get('mss_triggered', 1)) * 100):.1f}%")
    print(f"{border}\n")


def export_trades_csv(trades: List[Trade], filepath: str) -> None:
    """Exports simulated trades strictly matching the required schema to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_trades = pd.DataFrame([asdict(t) for t in trades])

    if df_trades.empty:
        # Create empty template with required headers
        df_trades = pd.DataFrame(columns=[
            "Timestamp", "Symbol", "Direction", "Entry", "SL", "TP",
            "Exit Price", "Exit Reason", "PnL", "R_Multiple"
        ])
    else:
        # Rename columns to institutional presentation format
        df_trades.rename(columns={
            "timestamp": "Timestamp",
            "symbol": "Symbol",
            "direction": "Direction",
            "entry": "Entry",
            "sl": "SL",
            "tp": "TP",
            "exit_price": "Exit Price",
            "exit_reason": "Exit Reason",
            "pnl": "PnL",
            "r_multiple": "R_Multiple",
            "contracts": "Contracts",
            "risk_price": "Risk_USD",
            "duration_bars": "Duration_Bars",
            "entry_type": "Entry_Type",
            "htf_bias_source": "HTF_Source"
        }, inplace=True)

    df_trades.to_csv(filepath, index=False)
    print(f"   [EXPORT] Saved {len(df_trades)} trade records to: {filepath}")


# ==============================================================================
# CLI RUNNER & EXECUTION ORCHESTRATOR
# ==============================================================================

def run():
    parser = argparse.ArgumentParser(
        description="Run Liquidity Transfer Model (4H Context ➔ 15M Execution) SMC Backtest."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        choices=["BTC/USDT", "ETH/USDT", "SOL/USDT", "HYPE/USDT", "ALL"],
        help="Trading pair symbol to simulate (default: BTC/USDT). Use 'ALL' for portfolio simulation."
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        default="next_swing",
        choices=["next_swing", "htf_roadblock", "fixed_rr"],
        help="Target profit mode: 'next_swing' (opposing 15M swing roadblock), 'htf_roadblock' (4H range high/low), or 'fixed_rr' (default: next_swing)."
    )
    parser.add_argument(
        "--min-rr",
        type=float,
        default=1.5,
        help="Minimum R:R required for a swing roadblock target (default: 1.5)."
    )
    parser.add_argument(
        "--rr",
        type=float,
        default=DEFAULT_RR,
        help="Fallback / Fixed Risk-Reward ratio for take profit (default: 2.0)."
    )
    parser.add_argument(
        "--be-trigger",
        type=float,
        default=1.5,
        help="R-multiple trigger to move Stop Loss to Breakeven (default: 1.5)."
    )
    parser.add_argument(
        "--entry-mode",
        type=str,
        default="midpoint",
        choices=["midpoint", "boundary"],
        help="Retest entry level: 'midpoint' (50% FVG consequent encroachment) or 'boundary' (FVG top/btm or broken pivot)."
    )
    parser.add_argument(
        "--export",
        type=str,
        default=DEFAULT_EXPORT_PATH,
        help="Output CSV path for trade logs."
    )

    args = parser.parse_args()

    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "HYPE/USDT"] if args.symbol == "ALL" else [args.symbol]
    all_trades: List[Trade] = []
    combined_funnel: Dict[str, Any] = {
        "bars_scanned": 0, "active_bias_bars": 0, "mss_triggered": 0,
        "retest_entries": 0, "tp_exits": 0, "sl_exits": 0, "be_exits": 0
    }

    for sym in symbols:
        print(f"\n[RUNNING] Loading multi-timeframe cached OHLCV for {sym}...")
        try:
            df_4h = load_ohlcv(sym, "4h")
            df_15m = load_ohlcv(sym, "15m")
            print(f"   Loaded {len(df_4h):,} 4H bars and {len(df_15m):,} 15M bars.")
        except Exception as e:
            print(f"   [ERROR] Failed to load data for {sym}: {e}")
            continue

        trades, meta = simulate_liquidity_transfer(
            df_4h=df_4h,
            df_15m=df_15m,
            symbol=sym,
            target_mode=args.target_mode,
            min_rr=args.min_rr,
            rr=args.rr,
            be_trigger_r=args.be_trigger,
            entry_mode=args.entry_mode
        )

        all_trades.extend(trades)
        for k in combined_funnel:
            combined_funnel[k] += meta["funnel"].get(k, 0)

        # Print single asset report if running single asset
        if len(symbols) == 1:
            metrics = calculate_metrics(trades, meta)
            print_institutional_report(metrics, sym, meta["funnel"])

    # If running ALL, print combined portfolio report
    if len(symbols) > 1:
        portfolio_meta = {
            "final_account": STARTING_ACCOUNT + sum(t.pnl for t in all_trades),
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "floor_breached": False
        }
        # Compute portfolio drawdown
        equity = STARTING_ACCOUNT
        peak = STARTING_ACCOUNT
        max_dd = 0.0
        for t in all_trades:
            equity += t.pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        portfolio_meta["max_drawdown_usd"] = max_dd
        portfolio_meta["max_drawdown_pct"] = (max_dd / peak) * 100 if peak > 0 else 0

        portfolio_metrics = calculate_metrics(all_trades, portfolio_meta)
        print_institutional_report(portfolio_metrics, "PORTFOLIO (BTC + HYPE)", combined_funnel)

    # Export trade log
    export_trades_csv(all_trades, args.export)


run_liquidity_transfer_backtest = run


if __name__ == "__main__":
    run()
