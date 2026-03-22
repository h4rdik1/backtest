"""
CHANGELOG:
- Added type hints and comprehensive docstrings across all functions.
- Implemented `debug_mode` for detailed failure diagnostics.
- Updated 4H MSB logic: Now strictly enforces a close above the *most recent lower high* that drove price into the OB (fallback logic removed).
- Refined OB invalidation: A close below 50% of the OB body invalidates the setup, evaluated *only after* the OB has been touched.
- Explained position sizing mapping formulas in code comments where $25 SL is used.
- Handled plt.show() to prevent blocking in headless runs.

TTrades Order Block Strategy Backtester v4.1
==========================================
Settings:
  - Risk:Reward  = 1:3
  - Stop Loss    = $25 fixed per trade
  - Data         = 1 year (~365 days)
  - Batched fetch to bypass Binance 1000 candle limit

Full Logic:
  STEP 1 — 4H : Bullish OB at swing low
  STEP 2A— 4H : MSB (strict close above last lower high)
                Invalidation: close below 50% of OB body after touch.
  STEP 2B— 1H : Valid bullish OB zone after 4H MSB
  STEP 3 — 15M / 5M: Entry inside 1H OB zone
                Stop   = below 4H/1D OB body low
                Target = entry + 3 * risk_in_price
"""

import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta, timezone
import os
import time
import warnings
import argparse
import copy
from typing import List, Dict, Any, Optional, Tuple

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIG (easy-to-tweak strategy parameters)
# ─────────────────────────────────────────────
SYMBOL: str = "SOL/USDT"
RISK_REWARD: float = 3.0
FIXED_SL_USDT: float = 25.0
ACCOUNT_SIZE: float = 1000.0
DAYS_BACK: int = 365
SHOW_PLOTS: bool = False

BASE_PARAMS: Dict[str, Any] = {
    "debug_mode": True,  # Print specific reasons why setups invalidate
    "swing_lookback": 3,
    "max_candles_wait_4h": 60,
    "impulse_candles_4h": 2,
    "impulse_candles_1h": 2,
    "ob_impulse_mult_4h": 1.15,
    "ob_impulse_mult_1h": 1.10,
    "min_body_to_range": 0.35,
    "atr_lookback": 14,
    "msb_break_mult": 0.05,
    "ob_retest_lookback_4h": 14,
    "ob_retest_lookback_1h": 12,
    "max_1h_search_bars": 240,
    "entry_timeframe": "5m",  # "15m" or "5m"
    "max_entry_search_bars": 720,
    "fvg_lookback_1h": 20,
    "fvg_lookback_entry": 40,
    "max_risk_pct_price": 0.10,
    "stop_buffer": 0.999,
    "use_ob_mean_invalidation": True, # Invalidate if touched OB closes < 50% body
    "resolve_open_with_time_stop": True,
    "time_stop_bars": 288,
    "allow_msb_fallback_break": False, # Strict MSB: must break actual lower high
}
PARAMS: Dict[str, Any] = copy.deepcopy(BASE_PARAMS)

PARAM_PRESETS: Dict[str, Dict[str, Any]] = {
    "strict": {
        "ob_impulse_mult_4h": 1.25,
        "ob_impulse_mult_1h": 1.20,
        "msb_break_mult": 0.10,
        "use_ob_mean_invalidation": True,
        "allow_msb_fallback_break": False,
        "max_entry_search_bars": 480,
    },
    "balanced": {
        "ob_impulse_mult_4h": 1.15,
        "ob_impulse_mult_1h": 1.10,
        "msb_break_mult": 0.05,
        "use_ob_mean_invalidation": True,
        "allow_msb_fallback_break": False,
        "max_entry_search_bars": 720,
    },
    "relaxed": {
        "ob_impulse_mult_4h": 1.05,
        "ob_impulse_mult_1h": 1.00,
        "msb_break_mult": 0.02,
        "use_ob_mean_invalidation": True,
        "allow_msb_fallback_break": True,
        "max_entry_search_bars": 960,
    },
    "aggressive": {
        "ob_impulse_mult_4h": 0.90,
        "ob_impulse_mult_1h": 0.90,
        "msb_break_mult": 0.00,
        "use_ob_mean_invalidation": False,
        "allow_msb_fallback_break": True,
        "max_entry_search_bars": 1200,
        "min_body_to_range": 0.25,
    },
}


def _coerce_param_value(raw: str) -> Any:
    """Parse CLI string values into bool/int/float/str."""
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def build_params(profile: str, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build runtime params from base + preset + key=value overrides."""
    p = copy.deepcopy(BASE_PARAMS)
    preset = PARAM_PRESETS.get(profile, {})
    p.update(preset)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Invalid --set '{item}'. Use key=value.")
        key, val = item.split("=", 1)
        key = key.strip()
        if key not in p:
            raise ValueError(f"Unknown param key '{key}'.")
        p[key] = _coerce_param_value(val.strip())
    return p


# ─────────────────────────────────────────────
# BATCHED DATA FETCHING
# ─────────────────────────────────────────────
def fetch_ohlcv_full(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Load local CSV and handle index/type conversion instead of fetching from API."""
    import os
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


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def is_swing_low(df: pd.DataFrame, i: int, n: int) -> bool:
    """Check if candle i is a swing low over +/- n candles."""
    if i < n or i >= len(df) - n:
        return False
    return df["low"].iloc[i] == df["low"].iloc[i-n:i+n+1].min()


def is_swing_high(df: pd.DataFrame, i: int, n: int) -> bool:
    """Check if candle i is a swing high over +/- n candles."""
    if i < n or i >= len(df) - n:
        return False
    return df["high"].iloc[i] == df["high"].iloc[i-n:i+n+1].max()


def calc_atr(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """Calculate the Average True Range."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(lookback).mean()


def body_size(row: pd.Series) -> float:
    """Return the absolute body size of a candle."""
    return abs(row["close"] - row["open"])


def candle_range(row: pd.Series) -> float:
    """Return the high-low range of a candle."""
    return row["high"] - row["low"]


def is_bearish(row: pd.Series) -> bool:
    """Return True if candle closed lower than it opened."""
    return row["close"] < row["open"]


def is_bullish(row: pd.Series) -> bool:
    """Return True if candle closed higher than it opened."""
    return row["close"] > row["open"]


def is_aggressive_bullish_move(df: pd.DataFrame, atr: pd.Series, start_idx: int, move_candles: int, impulse_mult: float, min_body_to_range: float) -> bool:
    """
    Robust impulse filter to avoid noise:
    - at least 2 mostly bullish candles
    - meaningful body/expansion relative to ATR
    """
    end_idx = start_idx + move_candles
    if end_idx >= len(df):
        return False

    segment = df.iloc[start_idx + 1 : end_idx + 1]
    bull_count = (segment["close"] > segment["open"]).sum()
    if bull_count < max(1, move_candles - 1):
        return False

    net_move = segment["close"].iloc[-1] - segment["open"].iloc[0]
    seg_range = segment["high"].max() - segment["low"].min()
    atr_val = atr.iloc[start_idx]
    if pd.isna(atr_val) or atr_val <= 0:
        return False

    if net_move < impulse_mult * atr_val:
        return False
    if seg_range < impulse_mult * atr_val:
        return False

    avg_body_ratio = np.mean(
        [body_size(r) / candle_range(r) if candle_range(r) > 0 else 0 for _, r in segment.iterrows()]
    )
    if avg_body_ratio < min_body_to_range:
        return False

    return True


def has_recent_bullish_fvg_confluence(df: pd.DataFrame, i: int, lookback: int) -> bool:
    """Check if the OB has structural confluence with a recent bullish FVG."""
    start = max(2, i - lookback)
    ob_mid = (df["open"].iloc[i] + df["close"].iloc[i]) / 2
    for j in range(start, i + 1):
        if df["low"].iloc[j] > df["high"].iloc[j - 2]:
            zone_low = df["high"].iloc[j - 2]
            zone_high = df["low"].iloc[j]
            if zone_low <= ob_mid <= zone_high:
                return True
    return False


# ─────────────────────────────────────────────
# STEP 1 — HTF BULLISH OB (4H/1D)
# ─────────────────────────────────────────────
def find_4h_bullish_obs(df_4h: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Find bullish Order Blocks on the structural HTF (1D or 4H).
    Returns a list of dicts with OB properties.
    """
    obs = []
    atr = calc_atr(df_4h, PARAMS["atr_lookback"])
    swing_n = PARAMS["swing_lookback"]
    move_candles = PARAMS["impulse_candles_4h"]

    for i in range(swing_n + 1, len(df_4h) - move_candles - 1):
        candle = df_4h.iloc[i]

        if not is_bearish(candle):
            continue

        if not is_aggressive_bullish_move(
            df=df_4h,
            atr=atr,
            start_idx=i,
            move_candles=move_candles,
            impulse_mult=PARAMS["ob_impulse_mult_4h"],
            min_body_to_range=PARAMS["min_body_to_range"],
        ):
            continue

        # Ensure displacement actually closes above OB high.
        if df_4h["close"].iloc[i + move_candles] <= candle["high"]:
            continue

        nearby_swing = any(
            is_swing_low(df_4h, j, swing_n)
            for j in range(max(0, i - PARAMS["ob_retest_lookback_4h"]), i + 1)
        )
        if not nearby_swing:
            continue

        ob_body_hi = max(candle["open"], candle["close"])
        ob_body_lo = min(candle["open"], candle["close"])

        obs.append({
            "idx"        : i,
            "time"       : df_4h.index[i],
            "ob_high"    : candle["high"],
            "ob_low"     : candle["low"],
            "ob_body_hi" : ob_body_hi,
            "ob_body_lo" : ob_body_lo,
            "mean_thresh": ob_body_lo + (ob_body_hi - ob_body_lo) / 2, # 50% of OB body
            "used"       : False,
        })

    # Note: 50% OB mean invalidation is now evaluated dynamically in the MSB check 
    # (after the OB has been touched) to simulate realistic live logic.

    return obs


# ─────────────────────────────────────────────
# STEP 2A — HTF MSB (4H/1D)
# ─────────────────────────────────────────────
def detect_4h_msb(df_4h: pd.DataFrame, ob: Dict[str, Any], ob_touch_time: pd.Timestamp) -> Optional[Dict[str, Any]]:
    """
    Bullish MSB after OB touch:
    - Track invalidation: if any close < mean_thresh after touch, invalidate.
    - Identify the most recent lower high extending from OB creation to post-touch.
    - Confirm break with close above that lower high by an ATR margin.
    """
    swing_n = PARAMS["swing_lookback"]
    
    # Window to search for MSB *after* touch
    window = df_4h[df_4h.index >= ob_touch_time].head(PARAMS["max_candles_wait_4h"])
    if len(window) < swing_n * 2 + 2:
        return None

    atr = calc_atr(df_4h, PARAMS["atr_lookback"])
    
    # Find all swing highs between OB formation and the end of our MSB wait window
    # This precisely captures the descent *into* the OB.
    search_start = ob["time"]
    search_window = df_4h[(df_4h.index >= search_start) & (df_4h.index <= window.index[-1])]
    
    swing_highs = []
    for i in range(swing_n, len(search_window) - swing_n):
        idx_full = df_4h.index.get_loc(search_window.index[i])
        if is_swing_high(df_4h, idx_full, swing_n):
            swing_highs.append({"time": search_window.index[i], "price": search_window["high"].iloc[i]})

    if len(swing_highs) < 1:
        if PARAMS.get("debug_mode"): print(f"   [DEBUG] MSB fail: No swing highs found around OB {ob['time']}")
        return None

    # Find the most recent strictly lower high
    lower_high = None
    for k in range(1, len(swing_highs)):
        if swing_highs[k]["price"] < swing_highs[k - 1]["price"]:
            lower_high = swing_highs[k]

    if lower_high is None:
        if not PARAMS["allow_msb_fallback_break"]:
            if PARAMS.get("debug_mode"): print(f"   [DEBUG] MSB fail: Strict lower high not found for OB {ob['time']}")
            return None
        lower_high = swing_highs[-1]

    # Look for a close above this lower_high inside our post-touch window
    for ts, row in window.iterrows():
        # 1. Check OB invalidation: if we closed below 50% of the body after touching it
        if PARAMS["use_ob_mean_invalidation"] and row["close"] < ob["mean_thresh"]:
            if PARAMS.get("debug_mode"): print(f"   [DEBUG] MSB fail: OB {ob['time']} invalidated (< 50% body) at {ts}")
            return None
            
        # 2. Can't break the high before it actually formed
        if ts <= lower_high["time"]:
            continue
            
        atr_val = atr.loc[ts] if ts in atr.index else np.nan
        if pd.isna(atr_val) or atr_val <= 0:
            continue
            
        break_margin = PARAMS["msb_break_mult"] * atr_val
        if row["close"] > (lower_high["price"] + break_margin):
            return {
                "time": ts,
                "broken_level": lower_high["price"],
                "msb_close": row["close"],
                "break_margin": break_margin,
            }

    if PARAMS.get("debug_mode"): print(f"   [DEBUG] MSB fail: No close above lower high {lower_high['price']} for OB {ob['time']}")
    return None


# ─────────────────────────────────────────────
# STEP 2B — 1H OB ZONE
# ─────────────────────────────────────────────
def find_1h_ob_after_msb(df_1h: pd.DataFrame, msb_time: pd.Timestamp, ob_4h: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find a valid 1H bullish OB that formed after the HTF MSB.
    Must have confluence (near a swing low or FVG) and respect HTF stop level.
    """
    swing_n = PARAMS["swing_lookback"]
    window = df_1h[df_1h.index >= msb_time].head(PARAMS["max_1h_search_bars"])
    atr = calc_atr(window, PARAMS["atr_lookback"])

    for i in range(swing_n + 1, len(window) - PARAMS["impulse_candles_1h"] - 1):
        candle = window.iloc[i]

        if not is_bearish(candle):
            continue

        if not is_aggressive_bullish_move(
            df=window,
            atr=atr,
            start_idx=i,
            move_candles=PARAMS["impulse_candles_1h"],
            impulse_mult=PARAMS["ob_impulse_mult_1h"],
            min_body_to_range=PARAMS["min_body_to_range"],
        ):
            continue

        if window["close"].iloc[i + PARAMS["impulse_candles_1h"]] <= candle["high"]:
            continue
            
        # 1H OB must not violate the primary HTF stop
        if candle["low"] < ob_4h["ob_low"]:
            continue

        idx_full = df_1h.index.get_loc(window.index[i])
        near_swing_low = any(
            is_swing_low(df_1h, j, swing_n)
            for j in range(max(0, idx_full - PARAMS["ob_retest_lookback_1h"]), idx_full + 1)
        )
        near_fvg = has_recent_bullish_fvg_confluence(window, i, PARAMS["fvg_lookback_1h"])
        
        if not (near_swing_low or near_fvg):
            continue

        ob_body_hi = max(candle["open"], candle["close"])
        ob_body_lo = min(candle["open"], candle["close"])

        return {
            "time"       : window.index[i],
            "ob_high"    : candle["high"],
            "ob_low"     : candle["low"],
            "ob_body_hi" : ob_body_hi,
            "ob_body_lo" : ob_body_lo,
            "mean_thresh": ob_body_lo + (ob_body_hi - ob_body_lo) / 2,
            "confluence" : "swing_low" if near_swing_low else "fvg",
        }
        
    if PARAMS.get("debug_mode"): print(f"   [DEBUG] 1H OB fail: No fresh 1H OB found after MSB {msb_time}")
    return None


# ─────────────────────────────────────────────
# STEP 3 — LTF ENTRY (5M/15M)
# ─────────────────────────────────────────────
def find_ltf_entry(df_ltf: pd.DataFrame, ob_1h: Dict[str, Any], ob_4h: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Entry triggers inside the 1H OB zone:
    1) OB trigger: first bullish close above prior candle high.
    2) FVG trigger: reclaim above bullish LTF FVG upper bound after pullback.
    """
    window = df_ltf[df_ltf.index >= ob_1h["time"]].head(PARAMS["max_entry_search_bars"])
    stop_ref = ob_4h["ob_body_lo"] * PARAMS["stop_buffer"]
    zone_touched = False
    fvgs = []

    for i in range(1, len(window) - 1):
        row = window.iloc[i]
        prev = window.iloc[i - 1]

        if i >= 2:
            # Track new lower-timeframe bullish FVGs
            if window["low"].iloc[i] > window["high"].iloc[i - 2]:
                fvgs.append(
                    {
                        "created_idx": i,
                        "low": window["high"].iloc[i - 2],
                        "high": window["low"].iloc[i],
                        "touched": False,
                    }
                )

        if not zone_touched:
            if row["low"] <= ob_1h["ob_body_hi"] and row["high"] >= ob_1h["ob_body_lo"]:
                zone_touched = True
            continue

        # Trigger A: OB continuation candle
        ob_trigger = (
            row["close"] > row["open"]
            and row["close"] > prev["high"]
            and row["low"] >= stop_ref
        )

        # Trigger B: FVG reclaim after tap into recent bullish FVG
        fvg_trigger = False
        for fvg in fvgs[-PARAMS["fvg_lookback_entry"] :]:
            in_gap = row["low"] <= fvg["high"] and row["high"] >= fvg["low"]
            if in_gap:
                fvg["touched"] = True
            if fvg["touched"] and row["close"] > fvg["high"] and row["close"] > prev["high"]:
                fvg_trigger = True
                break

        if ob_trigger or fvg_trigger:
            entry = row["close"]
            risk_price = entry - stop_ref

            if risk_price <= 0 or risk_price > entry * PARAMS["max_risk_pct_price"]:
                continue

            return {
                "entry_time": window.index[i],
                "entry": entry,
                "stop": stop_ref,
                "target": entry + RISK_REWARD * risk_price,
                "risk_price": risk_price,
                "entry_type": "FVG" if fvg_trigger else "OB",
            }
            
    if PARAMS.get("debug_mode"): print(f"   [DEBUG] LTF Entry fail: No trigger inside 1H OB {ob_1h['time']}")
    return None


# ─────────────────────────────────────────────
# SIMULATE
# ─────────────────────────────────────────────
def simulate_trade(trade: Dict[str, Any], df_ltf: pd.DataFrame) -> Tuple[str, Optional[float], Optional[pd.Timestamp]]:
    """
    Simulates the trade forward to SL, TP, or Time Stop.
    
    Position Sizing Math:
      - Risk per trade = $25 (FIXED_SL_USDT)
      - Risk in price = trade["entry"] - trade["stop"]
      - Tokens to buy = $25 / risk_in_price
    
      Win PnL  = Tokens * (target - entry) = 3 * $25 = $75
      Loss PnL = Tokens * (stop - entry)   = -$25
    """
    bars = 0
    future = df_ltf[df_ltf.index > trade["entry_time"]]
    for ts, row in future.iterrows():
        bars += 1
        if row["low"] <= trade["stop"]:
            return "LOSS", trade["stop"], ts
        if row["high"] >= trade["target"]:
            return "WIN", trade["target"], ts
        if PARAMS["resolve_open_with_time_stop"] and bars >= PARAMS["time_stop_bars"]:
            if row["close"] >= trade["entry"]:
                return "WIN", row["close"], ts
            return "LOSS", row["close"], ts
    return "OPEN", None, None


def compute_streaks(results: List[str]) -> Tuple[int, int]:
    """Calculate the maximum win and loss streaks."""
    max_win_streak = max_loss_streak = win_streak = loss_streak = 0
    for r in results:
        if r == "WIN":
            win_streak += 1
            loss_streak = 0
        elif r == "LOSS":
            loss_streak += 1
            win_streak = 0
        max_win_streak = max(max_win_streak, win_streak)
        max_loss_streak = max(max_loss_streak, loss_streak)
    return max_win_streak, max_loss_streak


def compute_longest_drawdown_period(eq_curve: List[float]) -> int:
    """Calculate the longest period (in trades) in a drawdown."""
    peak = eq_curve[0]
    dd_start = None
    longest = 0
    current = 0
    for i, v in enumerate(eq_curve):
        if v >= peak:
            peak = v
            dd_start = None
            current = 0
        else:
            if dd_start is None:
                dd_start = i
            current = i - dd_start + 1
            longest = max(longest, current)
    return longest


def compute_sharpe_like(pnl_series: pd.Series) -> float:
    """Simple trade-based Sharpe-like score using trade returns."""
    trade_returns = pnl_series / FIXED_SL_USDT
    if len(trade_returns) < 2:
        return 0.0
    std = trade_returns.std(ddof=1)
    if std == 0 or pd.isna(std):
        return 0.0
    return trade_returns.mean() / std * np.sqrt(len(trade_returns))


def plot_trade(df_ltf: pd.DataFrame, trade: Dict[str, Any], symbol: str, timeframe: str, filename: str) -> None:
    """Save a detailed chart for a specific trade with structural zones."""
    import os
    os.makedirs("trades", exist_ok=True)
    
    entry_time = pd.to_datetime(trade["Entry Time"], utc=True)
    exit_ts = trade.get("exit_ts", entry_time + timedelta(hours=24))
    
    entry = trade["Entry $"]
    stop = trade["Stop $"]
    target = trade["Target $"]
    result = trade["Result"]

    # Window for visualization
    pre = entry_time - timedelta(hours=48)
    post = exit_ts + timedelta(hours=48)
    view = df_ltf[(df_ltf.index >= pre) & (df_ltf.index <= post)].copy()
    if view.empty:
        return

    x = np.arange(len(view))
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("#0b0b0e")
    ax.set_facecolor("#111114")

    # Plot price
    ax.plot(x, view["close"].values, color="#4CA3FF", linewidth=1.5, alpha=0.9, label="Price")
    
    # Highlight zones if available in trade dict
    if "ob_4h" in trade:
        ob = trade["ob_4h"]
        ob_x_start = np.argmin(np.abs((view.index - ob["time"]).total_seconds()))
        ax.add_patch(mpatches.Rectangle((ob_x_start, ob["ob_low"]), len(view)-ob_x_start, ob["ob_high"]-ob["ob_low"], 
                                       color="#FFD166", alpha=0.1, label="HTF OB Zone"))
        ax.axhline(ob["mean_thresh"], color="#FFD166", linestyle=":", alpha=0.3, label="OB 50% Mean")

    if "ob_1h" in trade:
        ob1 = trade["ob_1h"]
        ob1_x_start = np.argmin(np.abs((view.index - ob1["time"]).total_seconds()))
        ax.add_patch(mpatches.Rectangle((ob1_x_start, ob1["ob_low"]), len(view)-ob1_x_start, ob1["ob_high"]-ob1["ob_low"], 
                                       color="#4CA3FF", alpha=0.15, label="1H Setup Zone"))

    # Entry/Exit markers
    entry_idx = np.argmin(np.abs((view.index - entry_time).total_seconds()))
    exit_idx = np.argmin(np.abs((view.index - exit_ts).total_seconds()))
    
    # Shade RR Zone
    ax.fill_between(x[entry_idx:exit_idx+1], stop, target, color="#22c55e" if result=="WIN" else "#ef4444", alpha=0.08)

    ax.scatter(entry_idx, entry, color="#00C896", s=120, marker="^", zorder=5, label=f"Entry: {entry:.2f}")
    ax.scatter(exit_idx, view["close"].iloc[exit_idx], color="#FFD166", s=120, marker="o", zorder=5, label=f"Exit: {view['close'].iloc[exit_idx]:.2f}")

    # Lines
    ax.axhline(stop, color="#FF4C4C", linestyle="--", linewidth=1.5, alpha=0.8, label=f"SL: {stop:.2f}")
    ax.axhline(target, color="#00C896", linestyle="--", linewidth=1.5, alpha=0.8, label=f"TP: {target:.2f}")

    # Markers for MSB
    if "msb" in trade:
        msb_time = trade["msb"]["time"]
        msb_idx = np.argmin(np.abs((view.index - msb_time).total_seconds()))
        ax.annotate("MSB", xy=(msb_idx, trade["msb"]["broken_level"]), xytext=(0, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color="white"), color="white", fontsize=9, ha='center')

    title = f"{symbol} | {result} | Entry: {entry_time.strftime('%Y-%m-%d %H:%M')} | {trade['Entry Type']} Trigger"
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=20)
    ax.set_ylabel("Price (USDT)", color="#94a3b8")
    ax.grid(alpha=0.05, color="white")
    ax.tick_params(colors="#64748b", labelsize=9)

    xticks = np.linspace(0, len(view) - 1, 10).astype(int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([view.index[i].strftime("%m-%d %H:%M") for i in xticks], rotation=25, ha="right")

    leg = ax.legend(facecolor="#111114", edgecolor="#333", loc="upper left", fontsize=9)
    for text in leg.get_texts(): text.set_color("white")

    plt.tight_layout()
    plt.savefig(os.path.join("trades", filename), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_backtest(profile_name: str = "balanced") -> None:
    print("=" * 62)
    print(f"  TTrades OB Strategy v4.1 - {SYMBOL}")
    print(f"  RR: 1:{int(RISK_REWARD)}  |  SL: ${FIXED_SL_USDT}  |  TP: ${FIXED_SL_USDT*RISK_REWARD:.0f}  |  {DAYS_BACK} days")
    print(f"  Top-Down TFs: 1d -> 1h -> {PARAMS['entry_timeframe']}")
    print(f"  Profile: {profile_name}")
    print("=" * 62)

    print("\n[DATA] Fetching 1 year of OHLCV data (~30s)...")
    try:
        df_4h  = fetch_ohlcv_full(SYMBOL, "1d",  DAYS_BACK)
        df_1h  = fetch_ohlcv_full(SYMBOL, "1h",  DAYS_BACK)
        df_entry = fetch_ohlcv_full(SYMBOL, PARAMS["entry_timeframe"], DAYS_BACK)
        print(f"\n   Date range: {df_4h.index[0].date()} -> {df_4h.index[-1].date()}")
    except Exception as e:
        print(f"\n[ERROR] {e}\n   Run: pip install ccxt")
        return

    print("\n[SCAN] Finding 1D Bullish Order Blocks...")
    obs_4h = find_4h_bullish_obs(df_4h)
    
    # We no longer pre-invalidate them in step 1, because 50% threshold is checked later
    valid_obs = obs_4h 
    print(f"   Total: {len(obs_4h)}")

    trades   = []
    account  = ACCOUNT_SIZE
    eq_curve = [account]
    f = {"touch": 0, "msb": 0, "ob1h": 0, "entry": 0, "entry_ob": 0, "entry_fvg": 0}

    print("\n[SIM] Simulating trades...")

    for ob in valid_obs:
        if ob["used"]:
            continue

        # 1. Price touches HTF OB zone?
        ob_touch_time = None
        for ts, row in df_4h[df_4h.index > ob["time"]].iterrows():
            if row["low"] <= ob["ob_high"] and row["close"] >= ob["ob_low"]:
                ob_touch_time = ts
                break
                
        if not ob_touch_time:
            if PARAMS.get("debug_mode"): print(f"   [DEBUG] Touch fail: OB {ob['time']} never visited.")
            continue
            
        f["touch"] += 1

        # 2. Daily MSB validation (including 50% OB invalidation logic)
        msb_4h = detect_4h_msb(df_4h, ob, ob_touch_time)
        if not msb_4h:
            continue
        f["msb"] += 1

        # 3. 1H Valid zone 
        ob_1h = find_1h_ob_after_msb(df_1h, msb_4h["time"], ob)
        if not ob_1h:
            continue
        f["ob1h"] += 1

        # 4. LTF Entry
        trade_setup = find_ltf_entry(df_entry, ob_1h, ob)
        if not trade_setup:
            continue
            
        f["entry"] += 1
        if trade_setup["entry_type"] == "FVG":
            f["entry_fvg"] += 1
        else:
            f["entry_ob"] += 1

        # 5. Simulate Open Trade
        result, _, exit_time = simulate_trade(trade_setup, df_entry)
        if result == "OPEN":
            continue

        pnl = FIXED_SL_USDT * RISK_REWARD if result == "WIN" else -FIXED_SL_USDT
        account += pnl
        eq_curve.append(account)

        trade_record = {
            "Symbol"    : SYMBOL,
            "Entry Time": trade_setup["entry_time"].strftime("%Y-%m-%d %H:%M"),
            "Exit Time" : exit_time.strftime("%Y-%m-%d %H:%M") if exit_time else "-",
            "Entry $"   : round(trade_setup["entry"], 4),
            "Stop $"    : round(trade_setup["stop"], 4),
            "Target $"  : round(trade_setup["target"], 4),
            "Result"    : result,
            "Entry Type": trade_setup["entry_type"],
            "PnL"       : round(pnl, 2),
            "Account"   : round(account, 2),
            "exit_ts"   : exit_time,
        }
        trades.append(trade_record)
        
        # Save visualization for this trade
        fname = f"trade_{len(trades)}_{trade_setup['entry_time'].strftime('%m%d_%H%M')}.png"
        plot_trade(df_entry, trade_record, SYMBOL, PARAMS["entry_timeframe"], fname)
        
        ob["used"] = True

    # Always show funnel stats
    if not trades:
        print("\n" + "=" * 62)
        print("  FILTER FUNNEL")
        print("=" * 62)
        print(f"  Initial 1D OBs        : {len(valid_obs)}")
        print(f"  -> Touched zone       : {f['touch']}")
        print(f"  -> 1D MSB confirmed   : {f['msb']}")
        print(f"  -> 1H OB found        : {f['ob1h']}")
        print(f"  -> {PARAMS['entry_timeframe'].upper()} entry fired : {f['entry']}")
        print("  -> Completed trades   : 0")
        print("\n[WARN] No completed trades found under strict parameters.")
        return

    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["Result"] == "WIN"]
    losses = df_t[df_t["Result"] == "LOSS"]
    wr = len(wins) / len(df_t) * 100
    breakeven = 1 / (1 + RISK_REWARD) * 100
    exp = (wr / 100 * FIXED_SL_USDT * RISK_REWARD) - ((1 - wr / 100) * FIXED_SL_USDT)
    gross_w = wins["PnL"].sum()
    gross_l  = abs(losses["PnL"].sum()) if len(losses) > 0 else 1
    pf = gross_w / gross_l

    peak = ACCOUNT_SIZE
    max_dd = 0
    for v in eq_curve:
        if v > peak: peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd: max_dd = dd

    max_win_streak, max_loss_streak = compute_streaks(df_t["Result"].tolist())
    longest_dd_trades = compute_longest_drawdown_period(eq_curve)
    sharpe_like = compute_sharpe_like(df_t["PnL"])

    print("\n" + "=" * 62)
    print("  FILTER FUNNEL")
    print("=" * 62)
    print(f"  Initial 1D OBs        : {len(valid_obs)}")
    print(f"  -> Touched zone       : {f['touch']}")
    print(f"  -> 1D MSB confirmed   : {f['msb']}")
    print(f"  -> 1H OB found        : {f['ob1h']}")
    print(f"  -> {PARAMS['entry_timeframe'].upper()} entry fired : {f['entry']}")
    print(f"     - OB entries       : {f['entry_ob']}")
    print(f"     - FVG entries      : {f['entry_fvg']}")
    print(f"  -> Completed trades   : {len(trades)}")

    print("\n" + "=" * 62)
    print("  RESULTS")
    print("=" * 62)
    print(f"  Total Trades    : {len(df_t)}")
    print(f"  Wins / Losses   : {len(wins)} / {len(losses)}")
    print(f"  Win Rate        : {wr:.1f}%")
    print(f"  Break-even WR   : {breakeven:.1f}%  <- must beat this at 1:{int(RISK_REWARD)}")
    print(f"  Expectancy      : ${exp:.2f} per trade")
    print(f"  Profit Factor   : {pf:.2f}  (>1.5 is good)")
    print(f"  Sharpe-like     : {sharpe_like:.2f}")
    print(f"  SL / TP         : -${FIXED_SL_USDT} / +${FIXED_SL_USDT*RISK_REWARD:.0f}")
    print(f"  Total PnL       : ${df_t['PnL'].sum():.2f}")
    print(f"  Final Account   : ${account:.2f}")
    print(f"  Max Drawdown    : {max_dd:.1f}%")
    print(f"  Max Win Streak  : {max_win_streak}")
    print(f"  Max Loss Streak : {max_loss_streak}")
    print(f"  Longest DD Len  : {longest_dd_trades} trades")
    print("=" * 62)

    verdict = "PROFITABLE" if wr > breakeven else "BELOW BREAK-EVEN"
    print(f"\n  {verdict}  ({wr:.1f}% vs {breakeven:.1f}% needed)")

    print("\nTrade Log:")
    print(df_t.to_string(index=False))

    if os.path.exists("trade_log_v4.csv"):
        try:
            existing_df = pd.read_csv("trade_log_v4.csv")
            df_t = pd.concat([existing_df, df_t], ignore_index=True)
            df_t.drop_duplicates(subset=["Symbol", "Entry Time"], keep="last", inplace=True)
        except Exception:
            pass
            
    df_t.to_csv("trade_log_v4.csv", index=False)
    print("\nSaved: trade_log_v4.csv (Cumulative)")
    print(f"Saved: {len(trades)} detailed trade charts in 'trades/' folder.")

    # ── Charts ──
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
        f"TTrades OB Strategy v4.1 - XRP/USDT  ({DAYS_BACK} days)\n"
        f"Win Rate: {wr:.1f}%  |  Break-even: {breakeven:.0f}%  |  "
        f"Expectancy: ${exp:.2f}/trade  |  Max DD: {max_dd:.1f}%  |  PF: {pf:.2f}  |  Sharpe-like: {sharpe_like:.2f}",
        fontsize=11, fontweight="bold", color="white"
    )

    # Equity curve
    ax1 = axes[0]
    ax1.plot(eq_curve, color="#00C896", linewidth=2)
    ax1.axhline(ACCOUNT_SIZE, color="#555", linestyle="--", linewidth=0.8)
    ax1.fill_between(range(len(eq_curve)), ACCOUNT_SIZE, eq_curve,
                     where=[v >= ACCOUNT_SIZE for v in eq_curve], alpha=0.2, color="#00C896")
    ax1.fill_between(range(len(eq_curve)), ACCOUNT_SIZE, eq_curve,
                     where=[v < ACCOUNT_SIZE  for v in eq_curve], alpha=0.2, color="#FF4C4C")
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Account (USDT)")
    ax1.grid(alpha=0.08, color="white")

    # PnL bars
    ax2 = axes[1]
    colors = ["#00C896" if r == "WIN" else "#FF4C4C" for r in df_t["Result"]]
    ax2.bar(range(len(df_t)), df_t["PnL"], color=colors, edgecolor="none")
    ax2.axhline(0, color="#555", linewidth=0.8)
    ax2.set_title(f"PnL Per Trade  (+${FIXED_SL_USDT*RISK_REWARD:.0f} win / -${FIXED_SL_USDT:.0f} loss)")
    ax2.set_ylabel("PnL (USDT)")
    wp = mpatches.Patch(color="#00C896", label=f"Win  {len(wins)}")
    lp = mpatches.Patch(color="#FF4C4C", label=f"Loss {len(losses)}")
    ax2.legend(handles=[wp, lp], facecolor="#161616", labelcolor="white")
    ax2.grid(alpha=0.08, color="white")

    # Cumulative PnL
    ax3 = axes[2]
    cum = df_t["PnL"].cumsum()
    ax3.plot(cum.values, color="#4CA3FF", linewidth=2)
    ax3.axhline(0, color="#555", linestyle="--", linewidth=0.8)
    ax3.fill_between(range(len(cum)), 0, cum.values,
                     where=[v >= 0 for v in cum], alpha=0.15, color="#00C896")
    ax3.fill_between(range(len(cum)), 0, cum.values,
                     where=[v < 0  for v in cum], alpha=0.15, color="#FF4C4C")
    ax3.set_title("Cumulative PnL")
    ax3.set_ylabel("USDT")
    ax3.set_xlabel("Trade #")
    ax3.grid(alpha=0.08, color="white")

    plt.tight_layout()
    plt.savefig("backtest_results_v4.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    if SHOW_PLOTS:
        try:
            plt.show(block=False)
            plt.pause(2)
        except Exception:
            pass
    plt.close(fig)
    print("Saved: backtest_results_v4.png")

    print("\nPrioritize statistical validity over trade count.")


def parse_args() -> argparse.Namespace:
    """CLI options for quick parameter experimentation."""
    parser = argparse.ArgumentParser(description="SMC Backtester")
    parser.add_argument("--symbol", default=SYMBOL, help="Trading symbol, e.g. BTC/USDT")
    parser.add_argument("--days", type=int, default=DAYS_BACK, help="Lookback days")
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=sorted(PARAM_PRESETS.keys()),
        help="Parameter preset profile",
    )
    parser.add_argument("--entry-tf", choices=["5m", "15m"], help="Entry timeframe override")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug logs")
    parser.add_argument("--show-plots", action="store_true", help="Show matplotlib windows")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Arbitrary param override. Repeatable. Example: --set msb_break_mult=0.02",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    SYMBOL = args.symbol
    DAYS_BACK = args.days
    SHOW_PLOTS = args.show_plots

    PARAMS = build_params(args.profile, args.set)
    if args.entry_tf:
        PARAMS["entry_timeframe"] = args.entry_tf
    if args.debug:
        PARAMS["debug_mode"] = True
    if args.no_debug:
        PARAMS["debug_mode"] = False

    run_backtest(profile_name=args.profile)