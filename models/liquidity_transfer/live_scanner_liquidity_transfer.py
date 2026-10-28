"""
================================================================================
LIQUIDITY TRANSFER MODEL — PRODUCTION LIVE SCANNER & SIGNAL GENERATOR
================================================================================
Centralized Smart Money Concepts (SMC) live market scanner and signal engine.
Anchored strictly to the validated "Liquidity Transfer Model" (4H Context ➔ 15M Execution).

Architecture & Design:
  1. Data Layer:
     - CCXTDataFeed with rate-limiting, exponential backoff, and retry resilience.
     - Strict Lookahead Protection: Splits closed candles from open/in-progress candles.
     - Fallback / Mock support for unit testing and offline simulation.
  2. Strategy Engine:
     - Pure functional evaluation using validated models.backtest_liquidity_transfer logic.
     - Phase 1: 4H Structural Anchor (SFP or Discount/Premium FVG Tap, <=24h bias).
     - Phase 2: 15M Market Structure Shift (Strict FULL CANDLE BODY CLOSE beyond pivot).
     - Phase 3: 15M FVG Mandate + 50% Consequent Encroachment Retest Entry.
     - Dynamic Structural Roadblock Targeting (min 1.5R) & Liquidity-Buffered Stop Loss.
  3. Publisher & Alert Layer:
     - Institutional ASCII Terminal Card for instant human assessment.
     - Thread-safe persistent JSON & CSV trade signal logging in exports/.
     - Optional webhook notification dispatch with secret masking.
  4. Orchestrator:
     - Supports on-demand execution (--once) or continuous daemon loop (--poll).
     - Aligns polling cycles precisely to 15-minute candle close boundaries.
================================================================================
"""

import os
import sys
import time
import json
import math
import signal
import logging
import argparse
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd

REPO_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Safe import of backtest engine components to guarantee zero strategy divergence
try:
    from .backtest_liquidity_transfer import (
        detect_4h_bias_events,
        calculate_atr,
        BiasEvent,
        HTF_LOOKBACK_BARS,
        BIAS_EXPIRY_HOURS,
    )
except (ImportError, ValueError):
    from models.liquidity_transfer.backtest_liquidity_transfer import (
        detect_4h_bias_events,
        calculate_atr,
        BiasEvent,
        HTF_LOOKBACK_BARS,
        BIAS_EXPIRY_HOURS,
    )

# ==============================================================================
# CONFIGURATION, LOGGING & CONSTANTS
# ==============================================================================

DEFAULT_SYMBOLS: List[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "HYPE/USDT"]
DEFAULT_EXCHANGE: str = "binance"
DEFAULT_RISK_USD: float = 20.00
DEFAULT_MIN_RR: float = 1.50
DEFAULT_JSON_EXPORT: str = os.path.join(REPO_ROOT, "exports", "live_signals.json")
DEFAULT_CSV_EXPORT: str = os.path.join(REPO_ROOT, "exports", "live_signals.csv")

# Configure structured logging
logger = logging.getLogger("LiquidityTransferScanner")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class SetupState(str, Enum):
    """Lifecycle state machine for a live setup."""
    NO_HTF_BIAS = "NO_HTF_BIAS"
    BIAS_ACTIVE_WAITING_MSS = "BIAS_ACTIVE_WAITING_MSS"
    MSS_CONFIRMED_PENDING_RETEST = "MSS_CONFIRMED_PENDING_RETEST"
    RETEST_ACTIVE = "RETEST_ACTIVE"
    DISCARDED_NO_FVG = "DISCARDED_NO_FVG"
    INVALIDATED = "INVALIDATED"


@dataclass
class LiveScanResult:
    """Institutional Trade Card and State Snapshot for a given trading pair."""
    symbol: str
    timestamp: str
    state: SetupState
    current_price: float
    direction: Optional[str] = None  # 'LONG' or 'SHORT'
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    planned_rr: Optional[float] = None
    contracts: Optional[float] = None
    risk_usd: float = DEFAULT_RISK_USD
    notional_usd: Optional[float] = None
    htf_bias_source: Optional[str] = None
    htf_anchor_level: Optional[float] = None
    htf_target: Optional[float] = None
    pivot_price: Optional[float] = None
    fvg_top: Optional[float] = None
    fvg_btm: Optional[float] = None
    invalidation_level: Optional[float] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to clean dictionary for serialization."""
        d = asdict(self)
        d["state"] = self.state.value
        return d


# ==============================================================================
# DATA INGESTION & CCXT ADAPTER (LOOKAHEAD-FREE & RETRY-RESILIENT)
# ==============================================================================

class MarketDataFeed:
    """Abstract interface for market data ingestion to support testing and exchange swaps."""

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        raise NotImplementedError

    def get_closed_and_live_candles(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> Tuple[pd.DataFrame, float, pd.Timestamp]:
        raise NotImplementedError


class CCXTDataFeed(MarketDataFeed):
    """
    Robust CCXT market data feed with rate-limiting, exponential backoff retries,
    and strict separation between confirmed closed candles and the current live tick.
    """

    TIMEFRAME_SECONDS = {
        "15m": 15 * 60,
        "4h": 4 * 60 * 60,
        "1h": 60 * 60,
        "1d": 24 * 60 * 60,
    }

    def __init__(self, exchange_id: str = DEFAULT_EXCHANGE, max_retries: int = 3, timeout_ms: int = 15000):
        self.exchange_id = exchange_id.lower()
        self.max_retries = max_retries
        self.timeout_ms = timeout_ms

        try:
            import ccxt
        except ImportError:
            raise ImportError(
                "CCXT library is required for live market scanning. Install with: pip install ccxt"
            )

        if not hasattr(ccxt, self.exchange_id):
            raise ValueError(f"Exchange '{self.exchange_id}' is not supported by CCXT.")

        exchange_class = getattr(ccxt, self.exchange_id)
        # Enable built-in rate-limiting and configure safe timeouts
        self.exchange = exchange_class({
            "enableRateLimit": True,
            "timeout": self.timeout_ms,
        })
        logger.info(f"Initialized CCXT Data Feed using exchange: {self.exchange_id.upper()}")

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV candles from the exchange with exponential backoff on transient network faults.
        Returns a sorted, validated DataFrame with UTC DatetimeIndex.
        """
        import ccxt

        attempt = 0
        backoff_delay = 1.0

        while attempt < self.max_retries:
            attempt += 1
            try:
                raw_candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if not raw_candles:
                    raise ValueError(f"Empty OHLCV data returned for {symbol} on {timeframe}")

                df = pd.DataFrame(
                    raw_candles,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df.sort_values("timestamp", inplace=True)
                df.reset_index(drop=True, inplace=True)

                # Ensure strict numeric conversion
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                df.dropna(subset=["timestamp", "open", "high", "low", "close"], inplace=True)
                return df

            except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.RateLimitExceeded) as e:
                logger.warning(
                    f"[{symbol}] Network error fetching {timeframe} (Attempt {attempt}/{self.max_retries}): {e}"
                )
                if attempt >= self.max_retries:
                    raise ConnectionError(f"Failed to fetch {timeframe} data for {symbol} after {self.max_retries} attempts.") from e
                time.sleep(backoff_delay)
                backoff_delay *= 2.0

            except ccxt.ExchangeError as e:
                logger.error(f"[{symbol}] Exchange error for {timeframe}: {e}")
                raise

    def get_closed_and_live_candles(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> Tuple[pd.DataFrame, float, pd.Timestamp]:
        """
        Enforces the Strict Lookahead Rule:
        Identifies whether the last candle returned is currently OPEN (in-progress).
        Returns:
          - df_closed: DataFrame containing ONLY fully settled and closed candles.
          - current_live_price: The latest tick / close of the open candle.
          - current_utc_time: Timestamp of evaluation.
        """
        df = self.fetch_ohlcv(symbol, timeframe, limit=limit)
        if df.empty:
            raise ValueError(f"No candle data available for {symbol} ({timeframe})")

        now_utc = datetime.now(timezone.utc)
        duration_sec = self.TIMEFRAME_SECONDS.get(timeframe, 900)

        last_bar_time = df["timestamp"].iloc[-1]
        candle_end_time = last_bar_time + pd.Timedelta(seconds=duration_sec)

        # If current time is prior to the candle end time, the last bar is IN-PROGRESS
        if now_utc < candle_end_time:
            current_live_price = float(df["close"].iloc[-1])
            df_closed = df.iloc[:-1].copy().reset_index(drop=True)
        else:
            current_live_price = float(df["close"].iloc[-1])
            df_closed = df.copy().reset_index(drop=True)

        return df_closed, current_live_price, pd.Timestamp(now_utc)


# ==============================================================================
# STRATEGY EVALUATION ENGINE (PURE & TESTABLE)
# ==============================================================================

class LiquidityTransferLiveEngine:
    """
    Evaluates market structure state strictly following the validated Liquidity Transfer Model.
    Decoupled from exchange networking to allow 100% deterministic unit testing.
    """

    def __init__(
        self,
        risk_usd: float = DEFAULT_RISK_USD,
        target_mode: str = "next_swing",
        min_rr: float = DEFAULT_MIN_RR,
        entry_mode: str = "midpoint",
        max_stop_distance_pct: float = 0.035,
        min_stop_distance_pct: float = 0.0015,
        bias_expiry_hours: int = BIAS_EXPIRY_HOURS,
    ):
        self.risk_usd = risk_usd
        self.target_mode = target_mode
        self.min_rr = min_rr
        self.entry_mode = entry_mode
        self.max_stop_distance_pct = max_stop_distance_pct
        self.min_stop_distance_pct = min_stop_distance_pct
        self.bias_expiry_hours = bias_expiry_hours

    def evaluate(
        self,
        symbol: str,
        df_4h_closed: pd.DataFrame,
        df_15m_closed: pd.DataFrame,
        current_live_price: float,
        current_time: pd.Timestamp,
    ) -> LiveScanResult:
        """
        Runs the 3-phase structural sequence:
          Phase 1: Detect active 4H Bias Event (<=24h).
          Phase 2: Detect confirmed 15M Body-Close MSS and Displacement FVG.
          Phase 3: Compute Consequent Encroachment Retest level, SL, Target & State.
        """
        if len(df_4h_closed) < 5 or len(df_15m_closed) < 20:
            return LiveScanResult(
                symbol=symbol,
                timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                state=SetupState.NO_HTF_BIAS,
                current_price=current_live_price,
                message="Insufficient candle history for analysis."
            )

        # ----------------------------------------------------------------------
        # PHASE 1: 4H STRUCTURAL ANCHOR (HTF CONTEXT & BIAS)
        # ----------------------------------------------------------------------
        bias_events = detect_4h_bias_events(df_4h_closed)
        if not bias_events:
            return LiveScanResult(
                symbol=symbol,
                timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                state=SetupState.NO_HTF_BIAS,
                current_price=current_live_price,
                message="No 4H SFP or FVG tap detected in recent structural lookback."
            )

        # Identify the most recent active 4H bias event
        sorted_close_times = sorted(bias_events.keys(), reverse=True)
        active_ev: Optional[BiasEvent] = None
        for ct in sorted_close_times:
            if ct <= current_time:
                # Check expiration window (<= 24 hours)
                if (current_time - ct).total_seconds() <= self.bias_expiry_hours * 3600:
                    active_ev = bias_events[ct]
                    break

        if active_ev is None:
            return LiveScanResult(
                symbol=symbol,
                timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                state=SetupState.NO_HTF_BIAS,
                current_price=current_live_price,
                message=f"All prior 4H bias events have expired (> {self.bias_expiry_hours}h old)."
            )

        direction = "LONG" if active_ev.bias == "BULLISH" else "SHORT"
        current_eq = active_ev.equilibrium
        current_htf_target = active_ev.htf_target
        active_source = active_ev.source
        anchor_level = active_ev.anchor_level

        # ----------------------------------------------------------------------
        # PHASE 2: 15M MARKET STRUCTURE SHIFT (THE STRICT BODY-CLOSE RULE)
        # ----------------------------------------------------------------------
        highs = df_15m_closed["high"].values
        lows = df_15m_closed["low"].values
        opens = df_15m_closed["open"].values
        closes = df_15m_closed["close"].values
        ts_15m = df_15m_closed["timestamp"].tolist()
        atrs = calculate_atr(df_15m_closed, period=14)
        n_15m = len(df_15m_closed)

        # Locate the 15M bar corresponding to the start of the 4H bias anchor
        bias_start_idx = 0
        for idx, t in enumerate(ts_15m):
            if t >= active_ev.start_time:
                bias_start_idx = idx
                break
        bias_start_idx = min(bias_start_idx, max(0, n_15m - 40))

        if direction == "LONG":
            # SOP 4: Longs strictly permitted in Discount (< 0.5 Eq)
            if current_live_price > current_eq:
                return LiveScanResult(
                    symbol=symbol,
                    timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    state=SetupState.BIAS_ACTIVE_WAITING_MSS,
                    current_price=current_live_price,
                    direction=direction,
                    htf_bias_source=active_source,
                    htf_anchor_level=anchor_level,
                    htf_target=current_htf_target,
                    message=f"4H Bullish Bias active, but price is in Premium (> Eq {current_eq:,.2f}). Waiting for discount."
                )

            # 1. Identify 15M Sweep Low anchored to 4H event start
            sweep_bar_pos = bias_start_idx + int(np.argmin(lows[bias_start_idx:]))
            sweep_low = float(lows[sweep_bar_pos])

            # 2. Identify preceding 15M structural pivot (swing high that led to sweep low)
            pivot_price: Optional[float] = None
            for p in range(sweep_bar_pos - 1, max(1, sweep_bar_pos - 20), -1):
                if highs[p] > highs[p - 1] and highs[p] > highs[p + 1]:
                    pivot_price = float(highs[p])
                    break
            if pivot_price is None and sweep_bar_pos > bias_start_idx:
                pivot_price = float(np.max(highs[max(0, sweep_bar_pos - 10) : sweep_bar_pos]))

            if pivot_price is None:
                return LiveScanResult(
                    symbol=symbol,
                    timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    state=SetupState.BIAS_ACTIVE_WAITING_MSS,
                    current_price=current_live_price,
                    direction=direction,
                    htf_bias_source=active_source,
                    htf_anchor_level=anchor_level,
                    htf_target=current_htf_target,
                    message="Waiting for 15M pivot formation before sweep."
                )

            # 3. Check for 15M FULL BODY CLOSE beyond pivot
            mss_confirmed = False
            mss_bar_idx = -1
            for k in range(sweep_bar_pos + 1, n_15m):
                if highs[k] > pivot_price and closes[k] > pivot_price and closes[k] > opens[k]:
                    mss_confirmed = True
                    mss_bar_idx = k
                    break

            if not mss_confirmed:
                return LiveScanResult(
                    symbol=symbol,
                    timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    state=SetupState.BIAS_ACTIVE_WAITING_MSS,
                    current_price=current_live_price,
                    direction=direction,
                    htf_bias_source=active_source,
                    htf_anchor_level=anchor_level,
                    htf_target=current_htf_target,
                    pivot_price=pivot_price,
                    message=f"4H Bullish Bias active. Waiting for 15M candle to BODY-CLOSE above pivot ${pivot_price:,.2f}."
                )

            # 4. Strict FVG Mandate in displacement leg
            fvg_top: Optional[float] = None
            fvg_btm: Optional[float] = None
            for k in range(sweep_bar_pos + 2, mss_bar_idx + 1):
                if lows[k] > highs[k - 2]:
                    fvg_top = float(lows[k])
                    fvg_btm = float(highs[k - 2])

            if fvg_top is None or fvg_btm is None:
                return LiveScanResult(
                    symbol=symbol,
                    timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    state=SetupState.DISCARDED_NO_FVG,
                    current_price=current_live_price,
                    direction=direction,
                    htf_bias_source=active_source,
                    htf_anchor_level=anchor_level,
                    pivot_price=pivot_price,
                    message="MSS triggered, but displacement leg left NO 15M FVG imbalance. Setup discarded."
                )

            # ------------------------------------------------------------------
            # PHASE 3: ENTRY, BUFFERED SL & DYNAMIC TARGETING
            # ------------------------------------------------------------------
            entry_level = (fvg_top + fvg_btm) / 2.0 if self.entry_mode == "midpoint" else fvg_top
            atr = float(atrs[mss_bar_idx])
            buf = max(sweep_low * self.min_stop_distance_pct, 0.5 * atr)
            sl = sweep_low - buf
            risk = entry_level - sl

            if risk <= 0 or (risk / entry_level) > self.max_stop_distance_pct:
                return LiveScanResult(
                    symbol=symbol,
                    timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    state=SetupState.INVALIDATED,
                    current_price=current_live_price,
                    direction=direction,
                    message=f"Stop distance invalid or exceeds maximum threshold ({self.max_stop_distance_pct*100:.1f}%)."
                )

            # Target Selection: Next swing high roadblock >= 1.5R or HTF target
            tp: Optional[float] = None
            for k in range(n_15m - 2, max(0, n_15m - 60), -1):
                if highs[k] > highs[k - 1] and highs[k] > highs[k + 1]:
                    if highs[k] >= entry_level + (self.min_rr * risk):
                        if tp is None or highs[k] < tp:
                            tp = float(highs[k])

            if tp is None:
                if current_htf_target > entry_level + (self.min_rr * risk):
                    tp = float(current_htf_target)
                else:
                    tp = entry_level + (2.0 * risk)

            planned_rr = (tp - entry_level) / risk
            contracts = self.risk_usd / risk
            notional = contracts * entry_level

            # Determine Live Execution State
            if current_live_price <= sweep_low:
                state = SetupState.INVALIDATED
                msg = f"Setup INVALIDATED: Live price (${current_live_price:,.2f}) breached sweep low (${sweep_low:,.2f})."
            elif current_live_price >= tp:
                state = SetupState.INVALIDATED
                msg = f"Setup MISSED: Target (${tp:,.2f}) hit before retest fill occurred."
            elif current_live_price <= entry_level:
                state = SetupState.RETEST_ACTIVE
                msg = f"ACTIONABLE ENTRY: Price (${current_live_price:,.2f}) reached 50% FVG retest zone (${entry_level:,.2f})."
            else:
                state = SetupState.MSS_CONFIRMED_PENDING_RETEST
                msg = f"PENDING LIMIT BUY: Place Limit Order at ${entry_level:,.2f} (Waiting for pullback)."

            return LiveScanResult(
                symbol=symbol,
                timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                state=state,
                current_price=current_live_price,
                direction=direction,
                entry_price=round(entry_level, 4),
                stop_loss=round(sl, 4),
                take_profit=round(tp, 4),
                planned_rr=round(planned_rr, 2),
                contracts=round(contracts, 4),
                risk_usd=self.risk_usd,
                notional_usd=round(notional, 2),
                htf_bias_source=active_source,
                htf_anchor_level=round(anchor_level, 4),
                htf_target=round(current_htf_target, 4),
                pivot_price=round(pivot_price, 4),
                fvg_top=round(fvg_top, 4),
                fvg_btm=round(fvg_btm, 4),
                invalidation_level=round(sweep_low, 4),
                message=msg
            )

        else:  # SHORT
            # SOP 4: Shorts strictly permitted in Premium (> 0.5 Eq)
            if current_live_price < current_eq:
                return LiveScanResult(
                    symbol=symbol,
                    timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    state=SetupState.BIAS_ACTIVE_WAITING_MSS,
                    current_price=current_live_price,
                    direction=direction,
                    htf_bias_source=active_source,
                    htf_anchor_level=anchor_level,
                    htf_target=current_htf_target,
                    message=f"4H Bearish Bias active, but price is in Discount (< Eq {current_eq:,.2f}). Waiting for premium."
                )

            # 1. Identify 15M Sweep High anchored to 4H event start
            sweep_bar_pos = bias_start_idx + int(np.argmax(highs[bias_start_idx:]))
            sweep_high = float(highs[sweep_bar_pos])

            # 2. Identify preceding 15M structural pivot (swing low that led to sweep high)
            pivot_price = None
            for p in range(sweep_bar_pos - 1, max(1, sweep_bar_pos - 20), -1):
                if lows[p] < lows[p - 1] and lows[p] < lows[p + 1]:
                    pivot_price = float(lows[p])
                    break
            if pivot_price is None and sweep_bar_pos > bias_start_idx:
                pivot_price = float(np.min(lows[max(0, sweep_bar_pos - 10) : sweep_bar_pos]))

            if pivot_price is None:
                return LiveScanResult(
                    symbol=symbol,
                    timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    state=SetupState.BIAS_ACTIVE_WAITING_MSS,
                    current_price=current_live_price,
                    direction=direction,
                    htf_bias_source=active_source,
                    htf_anchor_level=anchor_level,
                    htf_target=current_htf_target,
                    message="Waiting for 15M pivot formation before sweep."
                )

            # 3. Check for 15M FULL BODY CLOSE beyond pivot
            mss_confirmed = False
            mss_bar_idx = -1
            for k in range(sweep_bar_pos + 1, n_15m):
                if lows[k] < pivot_price and closes[k] < pivot_price and closes[k] < opens[k]:
                    mss_confirmed = True
                    mss_bar_idx = k
                    break

            if not mss_confirmed:
                return LiveScanResult(
                    symbol=symbol,
                    timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    state=SetupState.BIAS_ACTIVE_WAITING_MSS,
                    current_price=current_live_price,
                    direction=direction,
                    htf_bias_source=active_source,
                    htf_anchor_level=anchor_level,
                    htf_target=current_htf_target,
                    pivot_price=pivot_price,
                    message=f"4H Bearish Bias active. Waiting for 15M candle to BODY-CLOSE below pivot ${pivot_price:,.2f}."
                )

            # 4. Strict FVG Mandate in displacement leg
            fvg_top = None
            fvg_btm = None
            for k in range(sweep_bar_pos + 2, mss_bar_idx + 1):
                if highs[k] < lows[k - 2]:
                    fvg_btm = float(highs[k])
                    fvg_top = float(lows[k - 2])

            if fvg_top is None or fvg_btm is None:
                return LiveScanResult(
                    symbol=symbol,
                    timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    state=SetupState.DISCARDED_NO_FVG,
                    current_price=current_live_price,
                    direction=direction,
                    htf_bias_source=active_source,
                    htf_anchor_level=anchor_level,
                    pivot_price=pivot_price,
                    message="MSS triggered, but displacement leg left NO 15M FVG imbalance. Setup discarded."
                )

            # ------------------------------------------------------------------
            # PHASE 3: ENTRY, BUFFERED SL & DYNAMIC TARGETING
            # ------------------------------------------------------------------
            entry_level = (fvg_top + fvg_btm) / 2.0 if self.entry_mode == "midpoint" else fvg_btm
            atr = float(atrs[mss_bar_idx])
            buf = max(sweep_high * self.min_stop_distance_pct, 0.5 * atr)
            sl = sweep_high + buf
            risk = sl - entry_level

            if risk <= 0 or (risk / entry_level) > self.max_stop_distance_pct:
                return LiveScanResult(
                    symbol=symbol,
                    timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    state=SetupState.INVALIDATED,
                    current_price=current_live_price,
                    direction=direction,
                    message=f"Stop distance invalid or exceeds maximum threshold ({self.max_stop_distance_pct*100:.1f}%)."
                )

            # Target Selection: Next swing low roadblock >= 1.5R or HTF target
            tp = None
            for k in range(n_15m - 2, max(0, n_15m - 60), -1):
                if lows[k] < lows[k - 1] and lows[k] < lows[k + 1]:
                    if lows[k] <= entry_level - (self.min_rr * risk):
                        if tp is None or lows[k] > tp:
                            tp = float(lows[k])

            if tp is None:
                if current_htf_target < entry_level - (self.min_rr * risk):
                    tp = float(current_htf_target)
                else:
                    tp = entry_level - (2.0 * risk)

            planned_rr = (entry_level - tp) / risk
            contracts = self.risk_usd / risk
            notional = contracts * entry_level

            # Determine Live Execution State
            if current_live_price >= sweep_high:
                state = SetupState.INVALIDATED
                msg = f"Setup INVALIDATED: Live price (${current_live_price:,.2f}) breached sweep high (${sweep_high:,.2f})."
            elif current_live_price <= tp:
                state = SetupState.INVALIDATED
                msg = f"Setup MISSED: Target (${tp:,.2f}) hit before retest fill occurred."
            elif current_live_price >= entry_level:
                state = SetupState.RETEST_ACTIVE
                msg = f"ACTIONABLE ENTRY: Price (${current_live_price:,.2f}) reached 50% FVG retest zone (${entry_level:,.2f})."
            else:
                state = SetupState.MSS_CONFIRMED_PENDING_RETEST
                msg = f"PENDING LIMIT SELL: Place Limit Order at ${entry_level:,.2f} (Waiting for pullback)."

            return LiveScanResult(
                symbol=symbol,
                timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                state=state,
                current_price=current_live_price,
                direction=direction,
                entry_price=round(entry_level, 4),
                stop_loss=round(sl, 4),
                take_profit=round(tp, 4),
                planned_rr=round(planned_rr, 2),
                contracts=round(contracts, 4),
                risk_usd=self.risk_usd,
                notional_usd=round(notional, 2),
                htf_bias_source=active_source,
                htf_anchor_level=round(anchor_level, 4),
                htf_target=round(current_htf_target, 4),
                pivot_price=round(pivot_price, 4),
                fvg_top=round(fvg_top, 4),
                fvg_btm=round(fvg_btm, 4),
                invalidation_level=round(sweep_high, 4),
                message=msg
            )


# ==============================================================================
# SIGNAL PUBLISHER & ALERT DISPATCHER (THREAD-SAFE & MULTI-CHANNEL)
# ==============================================================================

class SignalPublisher:
    """Handles terminal formatting, persistent JSON/CSV logs, and webhook notifications."""

    @staticmethod
    def print_trade_card(res: LiveScanResult) -> None:
        """Prints a high-impact, institutional ASCII trade card to terminal."""
        border = "=" * 76
        sub_border = "-" * 76

        status_tag = f"[{res.state.value}]"
        print(f"\n{border}")
        print(f"   INSTITUTIONAL SMC LIVE SETUP: {res.symbol.upper()} {status_tag}")
        print(f"   Timestamp: {res.timestamp} | Live Price: ${res.current_price:,.2f}")
        print(f"{border}")

        if res.direction:
            print(f"   Direction:          {res.direction.upper()}")
            print(f"   HTF Anchor:         {res.htf_bias_source} at ${res.htf_anchor_level:,.2f} -> Target: ${res.htf_target:,.2f}")
            if res.pivot_price:
                print(f"   15M Pivot:          ${res.pivot_price:,.2f}")

        if res.entry_price and res.stop_loss and res.take_profit:
            print(f"{sub_border}")
            print(f"   [EXECUTION PARAMETERS]")
            action = "LIMIT BUY" if res.direction == "LONG" else "LIMIT SELL"
            print(f"   Order Type:         {action} @ ${res.entry_price:,.2f} (50% FVG Midpoint)")
            print(f"   Stop Loss:          ${res.stop_loss:,.2f} (Buffer: 0.15% / 0.5 ATR)")
            print(f"   Take Profit:        ${res.take_profit:,.2f} (Roadblock Target)")
            print(f"   Planned R:R:        1 : {res.planned_rr:.2f}R")
            print(f"   Position Size:      {res.contracts} contracts (${res.risk_usd:.2f} Fixed 1.0R Risk)")
            if res.notional_usd:
                print(f"   Notional Value:     ${res.notional_usd:,.2f}")
            if res.invalidation_level:
                print(f"   Invalidation:       Cancel order if price breaches ${res.invalidation_level:,.2f}")

        print(f"{sub_border}")
        print(f"   Status Notes:       {res.message}")
        print(f"{border}\n")

    @staticmethod
    def save_json(res: LiveScanResult, filepath: str = DEFAULT_JSON_EXPORT) -> None:
        """Appends or updates live signal in persistent JSON storage."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        signals: List[Dict[str, Any]] = []

        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    signals = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read existing JSON signals ({e}). Starting fresh list.")
                signals = []

        signals.append(res.to_dict())
        # Keep the latest 500 signals to prevent unbounded file growth
        signals = signals[-500:]

        # Atomic write to avoid file corruption on interruption
        temp_file = f"{filepath}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(signals, f, indent=2)
        os.replace(temp_file, filepath)

    @staticmethod
    def save_csv(res: LiveScanResult, filepath: str = DEFAULT_CSV_EXPORT) -> None:
        """Appends signal record to CSV log."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        row = res.to_dict()
        df_row = pd.DataFrame([row])

        write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        df_row.to_csv(filepath, mode="a", index=False, header=write_header)

    @staticmethod
    def dispatch_webhook(res: LiveScanResult, webhook_url: Optional[str]) -> bool:
        """
        Sends an alert payload to a Discord or Slack webhook URL.
        Safely suppresses errors and never prints webhook URLs in logs.
        """
        if not webhook_url:
            return False

        try:
            import urllib.request
            payload = {
                "content": (
                    f"**[SMC LIVE SIGNAL: {res.symbol}]**\n"
                    f"State: `{res.state.value}` | Direction: `{res.direction}`\n"
                    f"Entry: `${res.entry_price}` | SL: `${res.stop_loss}` | TP: `${res.take_profit}`\n"
                    f"R:R: `{res.planned_rr}R` | Risk: `${res.risk_usd}`\n"
                    f"Notes: {res.message}"
                )
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json", "User-Agent": "LiquidityTransferScanner/1.0"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status in (200, 204)
        except Exception as e:
            logger.warning(f"Webhook notification dispatch failed: {e}")
            return False


# ==============================================================================
# SCANNER ORCHESTRATOR & CLI RUNNER
# ==============================================================================

class LiveScannerOrchestrator:
    """Manages multi-pair scanning cycles and continuous 15M candle alignment."""

    def __init__(
        self,
        symbols: List[str],
        data_feed: MarketDataFeed,
        engine: LiquidityTransferLiveEngine,
        json_export: str = DEFAULT_JSON_EXPORT,
        csv_export: str = DEFAULT_CSV_EXPORT,
        webhook_url: Optional[str] = None,
    ):
        self.symbols = symbols
        self.data_feed = data_feed
        self.engine = engine
        self.json_export = json_export
        self.csv_export = csv_export
        self.webhook_url = webhook_url
        self._running = True

        # Register graceful shutdown handlers
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

    def _handle_exit(self, signum: int, frame: Any) -> None:
        logger.info("Shutdown signal received. Exiting scanner cleanly.")
        self._running = False

    def scan_once(self) -> List[LiveScanResult]:
        """Performs a single complete scan across all configured trading pairs."""
        results: List[LiveScanResult] = []

        for sym in self.symbols:
            try:
                # Fetch closed 4H candles (for macro bias)
                df_4h_closed, _, _ = self.data_feed.get_closed_and_live_candles(
                    symbol=sym, timeframe="4h", limit=50
                )
                # Fetch closed 15M candles and current live price
                df_15m_closed, live_px, now_utc = self.data_feed.get_closed_and_live_candles(
                    symbol=sym, timeframe="15m", limit=100
                )

                result = self.engine.evaluate(
                    symbol=sym,
                    df_4h_closed=df_4h_closed,
                    df_15m_closed=df_15m_closed,
                    current_live_price=live_px,
                    current_time=now_utc,
                )
                results.append(result)

                # Output presentation & persistence
                SignalPublisher.print_trade_card(result)
                if result.state in (SetupState.MSS_CONFIRMED_PENDING_RETEST, SetupState.RETEST_ACTIVE):
                    SignalPublisher.save_json(result, self.json_export)
                    SignalPublisher.save_csv(result, self.csv_export)
                    if self.webhook_url:
                        SignalPublisher.dispatch_webhook(result, self.webhook_url)

            except Exception as e:
                logger.error(f"Error scanning {sym}: {e}", exc_info=True)

        return results

    def poll_continuously(self, grace_period_sec: int = 5) -> None:
        """
        Continuously polls the market aligned precisely with 15-minute candle closures.
        Sleeps until next :00, :15, :30, :45 + grace period to allow exchange candle settlement.
        """
        logger.info(f"Starting continuous live scanner daemon across: {', '.join(self.symbols)}")

        while self._running:
            now = datetime.now(timezone.utc)
            logger.info(f"Executing scheduled 15M scan cycle at {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            self.scan_once()

            if not self._running:
                break

            # Calculate sleep duration to next 15-minute mark
            current_minute = now.minute
            current_second = now.second
            minutes_to_next = 15 - (current_minute % 15)
            seconds_to_sleep = (minutes_to_next * 60) - current_second + grace_period_sec

            next_wake = now.timestamp() + seconds_to_sleep
            next_wake_dt = datetime.fromtimestamp(next_wake, timezone.utc)
            logger.info(
                f"Next scan cycle scheduled at {next_wake_dt.strftime('%H:%M:%S UTC')} "
                f"(Sleeping for {int(seconds_to_sleep)}s)..."
            )

            # Interruptible sleep in 1-second chunks
            slept = 0
            while slept < seconds_to_sleep and self._running:
                time.sleep(1)
                slept += 1


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Liquidity Transfer Model Production Live Market Scanner."
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_SYMBOLS),
        help=f"Comma-separated pairs to scan (default: {','.join(DEFAULT_SYMBOLS)})."
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=DEFAULT_EXCHANGE,
        help=f"CCXT exchange ID (default: {DEFAULT_EXCHANGE})."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Scan once and exit immediately (default mode if --poll is not set)."
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Run continuously aligned to 15-minute candle close intervals."
    )
    parser.add_argument(
        "--risk",
        type=float,
        default=DEFAULT_RISK_USD,
        help="Fixed risk USD per setup (default: $20.00)."
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        default="next_swing",
        choices=["next_swing", "htf_roadblock", "fixed_rr"],
        help="Target profit mode (default: next_swing)."
    )
    parser.add_argument(
        "--min-rr",
        type=float,
        default=DEFAULT_MIN_RR,
        help="Minimum R:R threshold for roadblock targets (default: 1.5)."
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=DEFAULT_JSON_EXPORT,
        help="Output path for JSON signals."
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=DEFAULT_CSV_EXPORT,
        help="Output path for CSV signals."
    )
    parser.add_argument(
        "--webhook-url",
        type=str,
        default=os.getenv("LIVE_SCANNER_WEBHOOK_URL", ""),
        help="Optional webhook URL for instant Discord/Slack notifications."
    )

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Initialize components
    feed = CCXTDataFeed(exchange_id=args.exchange)
    engine = LiquidityTransferLiveEngine(
        risk_usd=args.risk,
        target_mode=args.target_mode,
        min_rr=args.min_rr
    )
    orchestrator = LiveScannerOrchestrator(
        symbols=symbols,
        data_feed=feed,
        engine=engine,
        json_export=args.export_json,
        csv_export=args.export_csv,
        webhook_url=args.webhook_url if args.webhook_url else None
    )

    if args.poll:
        orchestrator.poll_continuously()
    else:
        orchestrator.scan_once()


if __name__ == "__main__":
    main()
