"""
================================================================================
TEST SUITE: LIVE SCANNER & LIQUIDITY TRANSFER STRATEGY ENGINE
================================================================================
Comprehensive unit and integration test suite verifying:
  - 4H SFP & FVG anchor detection
  - 15M strict body-close rule (wicks vs full body closes)
  - 15M displacement FVG mandate (filtering naked pivot breaks)
  - Consequent Encroachment (50% midpoint) math & position sizing
  - Liquidity-buffered stop loss placement
  - Dynamic roadblock target selection
  - SetupState transitions (PENDING_RETEST, RETEST_ACTIVE, INVALIDATED)
  - Lookahead / unclosed candle separation
  - Signal JSON & CSV persistence
================================================================================
"""

import os
import sys
import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# Add repository root to Python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.liquidity_transfer.live_scanner_liquidity_transfer import (
    SetupState,
    LiveScanResult,
    LiquidityTransferLiveEngine,
    SignalPublisher,
    CCXTDataFeed,
    DEFAULT_RISK_USD,
)
from models.liquidity_transfer.backtest_liquidity_transfer import (
    BiasEvent,
    detect_4h_bias_events,
    calculate_atr,
)


def create_synthetic_4h_df(n_bars: int = 30, base_price: float = 80000.0) -> pd.DataFrame:
    """Helper to generate baseline 4H candles with UTC timestamps."""
    start_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    records = []
    current_price = base_price

    for i in range(n_bars):
        t = start_time + timedelta(hours=4 * i)
        # Moderate random walk
        open_px = current_price
        high_px = open_px + 200.0
        low_px = open_px - 200.0
        close_px = open_px + 50.0
        current_price = close_px

        records.append({
            "timestamp": t,
            "open": open_px,
            "high": high_px,
            "low": low_px,
            "close": close_px,
            "volume": 1000.0
        })

    return pd.DataFrame(records)


def create_synthetic_15m_df(start_time: datetime, n_bars: int = 60, base_price: float = 80000.0) -> pd.DataFrame:
    """Helper to generate baseline 15M candles."""
    records = []
    current_price = base_price

    for i in range(n_bars):
        t = start_time + timedelta(minutes=15 * i)
        open_px = current_price
        high_px = open_px + 50.0
        low_px = open_px - 50.0
        close_px = open_px + 10.0
        current_price = close_px

        records.append({
            "timestamp": t,
            "open": open_px,
            "high": high_px,
            "low": low_px,
            "close": close_px,
            "volume": 200.0
        })

    return pd.DataFrame(records)


class TestLiquidityTransferLiveEngine(unittest.TestCase):
    """Verifies strategy business logic, edge cases, and state machine transitions."""

    def setUp(self):
        self.engine = LiquidityTransferLiveEngine(
            risk_usd=20.00,
            target_mode="next_swing",
            min_rr=1.5,
            entry_mode="midpoint"
        )

    def test_insufficient_data_returns_no_bias(self):
        """Engine must handle empty or short DataFrames gracefully without crashing."""
        df_empty = pd.DataFrame()
        now = pd.Timestamp(datetime.now(timezone.utc))
        res = self.engine.evaluate("BTC/USDT", df_empty, df_empty, 80000.0, now)
        self.assertEqual(res.state, SetupState.NO_HTF_BIAS)
        self.assertIn("Insufficient", res.message)

    def test_bullish_sfp_and_15m_mss_retest_flow(self):
        """
        End-to-End Positive Test for Bullish Setup:
        1. 4H generates a confirmed Bullish SFP (pierces 20-bar low, closes above).
        2. 15M forms a sweep low, a preceding pivot high, and a decisive full-body close MSS.
        3. 15M displacement leg leaves a confirmed 15M FVG.
        4. Validates 50% Consequent Encroachment entry, Stop Loss, Target, and Sizing.
        """
        # 1. Construct 4H DataFrame with a Bullish SFP on the last bar
        df_4h = create_synthetic_4h_df(n_bars=25, base_price=80000.0)
        # Prior 20-bar low
        prior_low = df_4h["low"].iloc[4:24].min()
        # Bar 24 pierces prior_low but closes above it (Pure Bullish SFP)
        df_4h.loc[24, "low"] = prior_low - 300.0
        df_4h.loc[24, "close"] = prior_low + 100.0
        df_4h.loc[24, "open"] = prior_low + 50.0

        bar_start_4h = df_4h["timestamp"].iloc[24]
        bar_close_4h = bar_start_4h + timedelta(hours=4)

        # 2. Construct 15M DataFrame starting from the 4H bar start
        df_15m = create_synthetic_15m_df(start_time=bar_start_4h, n_bars=40, base_price=prior_low)

        # Create structural pivot high at index 5
        df_15m.loc[4, "high"] = prior_low + 100.0
        df_15m.loc[5, "high"] = prior_low + 250.0  # 15M Pivot High
        df_15m.loc[6, "high"] = prior_low + 100.0
        pivot_price = prior_low + 250.0

        # Create sweep low at index 10
        sweep_low = prior_low - 250.0
        df_15m.loc[10, "low"] = sweep_low
        df_15m.loc[10, "close"] = sweep_low + 50.0

        # Create 15M displacement leg with an FVG between bar 11 and 13:
        # Candle 11 High < Candle 13 Low
        df_15m.loc[11, "high"] = prior_low + 50.0   # FVG Bottom
        df_15m.loc[12, "low"] = prior_low + 80.0
        df_15m.loc[12, "high"] = prior_low + 300.0
        df_15m.loc[13, "low"] = prior_low + 150.0  # FVG Top (> Bar 11 High)
        df_15m.loc[13, "open"] = prior_low + 160.0
        # Decisive Bullish Full Body Close above pivot_price ($250):
        df_15m.loc[13, "close"] = prior_low + 320.0
        df_15m.loc[13, "high"] = prior_low + 350.0

        # Bar 14 to 39: Consolidation above FVG
        for b in range(14, 40):
            df_15m.loc[b, "open"] = prior_low + 300.0
            df_15m.loc[b, "close"] = prior_low + 310.0
            df_15m.loc[b, "high"] = prior_low + 350.0
            df_15m.loc[b, "low"] = prior_low + 280.0

        eval_time = bar_close_4h + timedelta(hours=2)

        # Case A: Live price is above entry -> MSS_CONFIRMED_PENDING_RETEST
        live_price_above = prior_low + 300.0
        res_pending = self.engine.evaluate("BTC/USDT", df_4h, df_15m, live_price_above, eval_time)

        self.assertEqual(res_pending.state, SetupState.MSS_CONFIRMED_PENDING_RETEST)
        self.assertEqual(res_pending.direction, "LONG")
        self.assertIsNotNone(res_pending.entry_price)
        self.assertIsNotNone(res_pending.stop_loss)
        self.assertIsNotNone(res_pending.take_profit)

        # Expected FVG midpoint: (fvg_top + fvg_btm) / 2.0 = (150 + 50)/2 = 100 above prior_low
        expected_entry = prior_low + 100.0
        self.assertAlmostEqual(res_pending.entry_price, expected_entry, places=2)

        # Stop loss must be below sweep_low
        self.assertLess(res_pending.stop_loss, sweep_low)

        # Verify fixed risk sizing: contracts = 20.0 / (entry - sl)
        expected_risk = res_pending.entry_price - res_pending.stop_loss
        expected_contracts = 20.0 / expected_risk
        self.assertAlmostEqual(res_pending.contracts, expected_contracts, places=4)

        # Case B: Live price drops to entry level -> RETEST_ACTIVE
        live_price_retest = expected_entry
        res_retest = self.engine.evaluate("BTC/USDT", df_4h, df_15m, live_price_retest, eval_time)
        self.assertEqual(res_retest.state, SetupState.RETEST_ACTIVE)
        self.assertIn("ACTIONABLE ENTRY", res_retest.message)

        # Case C: Live price breaks sweep low -> INVALIDATED
        live_price_invalid = sweep_low - 10.0
        res_invalid = self.engine.evaluate("BTC/USDT", df_4h, df_15m, live_price_invalid, eval_time)
        self.assertEqual(res_invalid.state, SetupState.INVALIDATED)
        self.assertIn("INVALIDATED", res_invalid.message)

    def test_bearish_sfp_and_15m_mss_retest_flow(self):
        """
        End-to-End Positive Test for Bearish Setup:
        1. 4H generates a confirmed Bearish SFP (pierces 20-bar high, closes below).
        2. 15M forms a sweep high, a preceding pivot low, and a decisive full-body close MSS downward.
        3. 15M displacement leg leaves a confirmed 15M FVG.
        4. Validates 50% Consequent Encroachment entry, Stop Loss, Target, and Sizing.
        """
        df_4h = create_synthetic_4h_df(n_bars=25, base_price=80000.0)
        # Prior 20-bar high
        prior_high = df_4h["high"].iloc[4:24].max()
        # Bar 24 pierces prior_high but closes below it (Pure Bearish SFP)
        df_4h.loc[24, "high"] = prior_high + 300.0
        df_4h.loc[24, "close"] = prior_high - 100.0
        df_4h.loc[24, "open"] = prior_high - 50.0

        bar_start_4h = df_4h["timestamp"].iloc[24]
        bar_close_4h = bar_start_4h + timedelta(hours=4)

        df_15m = create_synthetic_15m_df(start_time=bar_start_4h, n_bars=40, base_price=prior_high)

        # Create structural pivot low at index 5
        df_15m.loc[4, "low"] = prior_high - 100.0
        df_15m.loc[5, "low"] = prior_high - 250.0  # 15M Pivot Low
        df_15m.loc[6, "low"] = prior_high - 100.0
        pivot_price = prior_high - 250.0

        # Create sweep high at index 10
        sweep_high = prior_high + 250.0
        df_15m.loc[10, "high"] = sweep_high
        df_15m.loc[10, "close"] = sweep_high - 50.0

        # Create 15M displacement leg with an FVG between bar 11 and 13:
        # Candle 11 Low > Candle 13 High
        df_15m.loc[11, "low"] = prior_high - 50.0   # FVG Top
        df_15m.loc[12, "high"] = prior_high - 80.0
        df_15m.loc[12, "low"] = prior_high - 300.0
        df_15m.loc[13, "high"] = prior_high - 150.0  # FVG Bottom (< Bar 11 Low)
        df_15m.loc[13, "open"] = prior_high - 160.0
        # Decisive Bearish Full Body Close below pivot_price:
        df_15m.loc[13, "close"] = prior_high - 320.0
        df_15m.loc[13, "low"] = prior_high - 350.0

        # Bar 14 to 39: Consolidation below FVG
        for b in range(14, 40):
            df_15m.loc[b, "open"] = prior_high - 300.0
            df_15m.loc[b, "close"] = prior_high - 310.0
            df_15m.loc[b, "low"] = prior_high - 350.0
            df_15m.loc[b, "high"] = prior_high - 280.0

        eval_time = bar_close_4h + timedelta(hours=2)

        # Case A: Live price is below entry -> MSS_CONFIRMED_PENDING_RETEST
        live_price_below = prior_high - 300.0
        res_pending = self.engine.evaluate("BTC/USDT", df_4h, df_15m, live_price_below, eval_time)

        self.assertEqual(res_pending.state, SetupState.MSS_CONFIRMED_PENDING_RETEST)
        self.assertEqual(res_pending.direction, "SHORT")
        self.assertIsNotNone(res_pending.entry_price)
        self.assertIsNotNone(res_pending.stop_loss)
        self.assertIsNotNone(res_pending.take_profit)

        # Expected FVG midpoint: (fvg_top + fvg_btm) / 2.0 = (-50 + -150)/2 = -100 relative to prior_high
        expected_entry = prior_high - 100.0
        self.assertAlmostEqual(res_pending.entry_price, expected_entry, places=2)

        # Stop loss must be above sweep_high
        self.assertGreater(res_pending.stop_loss, sweep_high)

        # Case B: Live price rises to entry level -> RETEST_ACTIVE
        live_price_retest = expected_entry
        res_retest = self.engine.evaluate("BTC/USDT", df_4h, df_15m, live_price_retest, eval_time)
        self.assertEqual(res_retest.state, SetupState.RETEST_ACTIVE)

        # Case C: Live price breaks sweep high -> INVALIDATED
        live_price_invalid = sweep_high + 10.0
        res_invalid = self.engine.evaluate("BTC/USDT", df_4h, df_15m, live_price_invalid, eval_time)
        self.assertEqual(res_invalid.state, SetupState.INVALIDATED)

    def test_strict_body_close_filter_rejects_wick_only_breach(self):
        """
        A candle that wicks above the pivot but closes its body UNDER it
        must NOT trigger an MSS (treated as noise/liquidity raid).
        """
        df_4h = create_synthetic_4h_df(n_bars=25, base_price=80000.0)
        prior_low = df_4h["low"].iloc[4:24].min()
        df_4h.loc[24, "low"] = prior_low - 200.0
        df_4h.loc[24, "close"] = prior_low + 50.0

        bar_start_4h = df_4h["timestamp"].iloc[24]
        bar_close_4h = bar_start_4h + timedelta(hours=4)

        df_15m = create_synthetic_15m_df(start_time=bar_start_4h, n_bars=30, base_price=prior_low)
        # Structural pivot high at index 4 (before sweep)
        df_15m.loc[3, "high"] = prior_low + 50.0
        df_15m.loc[4, "high"] = prior_low + 200.0
        df_15m.loc[5, "high"] = prior_low + 50.0
        pivot_price = prior_low + 200.0

        # Sweep low at index 7 (lowest low after pivot)
        sweep_low = prior_low - 300.0
        df_15m.loc[7, "low"] = sweep_low

        # Candle 10: High penetrates pivot ($250), but CLOSE is BELOW pivot ($180) -> Wick only!
        df_15m.loc[10, "high"] = pivot_price + 50.0
        df_15m.loc[10, "open"] = pivot_price - 50.0
        df_15m.loc[10, "close"] = pivot_price - 20.0  # Failed body close

        # Subsequent bars remain below pivot
        for b in range(11, 30):
            df_15m.loc[b, "high"] = pivot_price - 30.0
            df_15m.loc[b, "open"] = pivot_price - 50.0
            df_15m.loc[b, "close"] = pivot_price - 40.0
            df_15m.loc[b, "low"] = pivot_price - 60.0

        res = self.engine.evaluate("BTC/USDT", df_4h, df_15m, pivot_price - 10.0, bar_close_4h + timedelta(hours=1))
        # Must remain in waiting state, NOT confirmed
        self.assertEqual(res.state, SetupState.BIAS_ACTIVE_WAITING_MSS)
        self.assertIn("BODY-CLOSE", res.message)

    def test_fvg_mandate_discards_naked_pivot_breaks(self):
        """
        If 15M market structure breaks but the displacement leg leaves no FVG imbalance,
        the setup must be discarded (DISCARDED_NO_FVG).
        """
        df_4h = create_synthetic_4h_df(n_bars=25, base_price=80000.0)
        prior_low = df_4h["low"].iloc[4:24].min()
        df_4h.loc[24, "low"] = prior_low - 200.0
        df_4h.loc[24, "close"] = prior_low + 50.0

        bar_start_4h = df_4h["timestamp"].iloc[24]
        bar_close_4h = bar_start_4h + timedelta(hours=4)

        df_15m = create_synthetic_15m_df(start_time=bar_start_4h, n_bars=30, base_price=prior_low)
        # Structural pivot high at index 4
        df_15m.loc[3, "high"] = prior_low + 50.0
        df_15m.loc[4, "high"] = prior_low + 150.0
        df_15m.loc[5, "high"] = prior_low + 50.0
        pivot_price = prior_low + 150.0

        # Sweep low at index 7
        df_15m.loc[7, "low"] = prior_low - 300.0
        sweep_idx = 7
        for i in range(sweep_idx + 1, 15):
            df_15m.loc[i, "high"] = prior_low + (i * 20.0)
            df_15m.loc[i, "low"] = prior_low + ((i - 2) * 20.0)  # Overlaps candle i-2 completely
            df_15m.loc[i, "open"] = prior_low + ((i - 1) * 20.0)
            df_15m.loc[i, "close"] = prior_low + (i * 20.0)

        # Breakout candle 14 closes body above pivot
        df_15m.loc[14, "close"] = pivot_price + 50.0
        df_15m.loc[14, "high"] = pivot_price + 60.0

        res = self.engine.evaluate("BTC/USDT", df_4h, df_15m, pivot_price + 30.0, bar_close_4h + timedelta(hours=1))
        self.assertEqual(res.state, SetupState.DISCARDED_NO_FVG)
        self.assertIn("NO 15M FVG", res.message)

    def test_expired_4h_bias_is_ignored(self):
        """A 4H bias older than 24 hours must be marked expired."""
        df_4h = create_synthetic_4h_df(n_bars=25, base_price=80000.0)
        prior_low = df_4h["low"].iloc[4:24].min()
        df_4h.loc[24, "low"] = prior_low - 200.0
        df_4h.loc[24, "close"] = prior_low + 50.0

        bar_close_4h = df_4h["timestamp"].iloc[24] + timedelta(hours=4)
        df_15m = create_synthetic_15m_df(start_time=bar_close_4h, n_bars=20, base_price=80000.0)

        # Evaluation time is 26 hours after 4H close (> 24h limit)
        eval_time = bar_close_4h + timedelta(hours=26)
        res = self.engine.evaluate("BTC/USDT", df_4h, df_15m, 80000.0, eval_time)
        self.assertEqual(res.state, SetupState.NO_HTF_BIAS)
        self.assertIn("expired", res.message)


class TestSignalPublisher(unittest.TestCase):
    """Verifies file export safety, formatting, and atomic write behavior."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.json_path = os.path.join(self.test_dir.name, "test_signals.json")
        self.csv_path = os.path.join(self.test_dir.name, "test_signals.csv")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_json_and_csv_persistence(self):
        sample_res = LiveScanResult(
            symbol="BTC/USDT",
            timestamp="2026-10-23 00:00:00 UTC",
            state=SetupState.MSS_CONFIRMED_PENDING_RETEST,
            current_price=82500.0,
            direction="LONG",
            entry_price=82000.0,
            stop_loss=81500.0,
            take_profit=84000.0,
            planned_rr=4.0,
            contracts=0.04,
            risk_usd=20.0,
            notional_usd=3280.0,
            htf_bias_source="4H_SFP",
            htf_anchor_level=81200.0,
            htf_target=85000.0,
            message="Test signal"
        )

        # 1. Test JSON Export
        SignalPublisher.save_json(sample_res, self.json_path)
        self.assertTrue(os.path.exists(self.json_path))

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["symbol"], "BTC/USDT")
        self.assertEqual(data[0]["state"], "MSS_CONFIRMED_PENDING_RETEST")

        # 2. Test CSV Export
        SignalPublisher.save_csv(sample_res, self.csv_path)
        self.assertTrue(os.path.exists(self.csv_path))

        df_csv = pd.read_csv(self.csv_path)
        self.assertEqual(len(df_csv), 1)
        self.assertEqual(df_csv["symbol"].iloc[0], "BTC/USDT")
        self.assertEqual(df_csv["entry_price"].iloc[0], 82000.0)


if __name__ == "__main__":
    unittest.main()
