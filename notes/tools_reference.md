# TTrades Tools & Scripts Reference Guide

This document provides a comprehensive overview of all the available tools and scripts in the repository. You can refer to this list when deciding how to analyze the market, debug a trade, or review performance.

> **Quick Market Check:** `python scratch/market_check.py BTCUSDT 15m 30`

## Core Trading Scripts (Root Directory)

### `challenge_trader.py`
- **Purpose:** The core Paper Trading Challenge Engine. It runs the automated simulated trading logic using Smart Money Concepts (SMC) against a predefined starting balance and ruleset.
- **Usage:**
  ```bash
  python challenge_trader.py
  python challenge_trader.py --interval 600
  ```

### `live_scanner.py`
- **Purpose:** The TTrades Omni-Scanner. It actively scans the live market across multiple timeframes (chains) for symbols in the watchlist and ranks found SMC setups by probability (Tier A++, A, B).
- **Usage:**
  ```bash
  python live_scanner.py
  python live_scanner.py --symbols BTC/USDT ETH/USDT
  ```

---

## Analytical & Debugging Tools (`/tools` directory)

### `tools/analyze_setup.py`
- **Purpose:** Performs a deep, multi-timeframe SMC analysis for a specific symbol. It evaluates the HTF Context (4H), MTF Bias (1H), and LTF Entry (5M). It identifies Swing Failure Patterns (SFPs), Order Blocks (OBs), Premium/Discount zones, and provides a final verdict with risk notes.
- **When to use:** When you spot a potential setup and want a detailed breakdown of whether it aligns with standard SMC rules before entering a trade.
- **Usage:**
  ```bash
  python tools/analyze_setup.py BTC/USDT
  ```

### `tools/check_candles.py`
- **Purpose:** A universal terminal-based candle viewer. It fetches and displays raw OHLCV data in a clean tabular format, highlighting bullish/bearish candles and providing period statistics (Max High, Min Low, Avg Volume).
- **When to use:** When you need a quick glance at raw price action, recent highs/lows, or want to verify data feeds without opening a chart.
- **Usage:**
  ```bash
  python tools/check_candles.py BTC/USDT -t 15m -l 20
  ```

### `tools/analyse_challenges.py`
- **Purpose:** Consolidates all paper trading CSV logs (from the `archive` directory) and prints a deep statistical breakdown. It analyzes Win Rate (WR), Profit Factor (PF), streaks, performance by chain/symbol, and flags issues like counter-trend trading.
- **When to use:** To review the overall performance of the trading system and identify what's working (e.g., With-Trend setups) and what's failing (e.g., Scalp chain).
- **Usage:**
  ```bash
  python tools/analyse_challenges.py
  ```

### `tools/dry_run_btc_sl.py`
- **Purpose:** A dry-run simulation specifically designed to test the structural Stop Loss (SL) placement logic (Nested SFPs vs FVG Boundaries) for BTC/USDT.
- **When to use:** When debugging or tuning the risk management and stop-loss placement algorithms to see if setups pass the Risk:Reward (RR) filters.
- **Usage:**
  ```bash
  python tools/dry_run_btc_sl.py
  ```

### `tools/view_candles.py`
- **Purpose:** A local HTTP server that acts as a backend for the SMC Challenge Dashboard (`view_candles.html` & `view_advanced.html`). It provides a TradingView UDF API feed and endpoints for active trades and live candle overlays (FVGs/OBs).
- **When to use:** Run this in the background when you want to visualize the active paper trades and live charts in your web browser.
- **Usage:**
  ```bash
  python tools/view_candles.py
  ```

---

## Scratch / Utility Scripts (`/scratch` directory)

### `scratch/market_check.py` ⭐ (Primary Manual Analysis Tool)
- **Purpose:** Centralized, universal market price fetcher and structure analyzer. Works for any Binance symbol on any timeframe. Replaces all per-coin scripts (`get_btc_data.py`, `get_eth_recent.py`, etc.).
- **What it outputs:**
  - Current price, period high/low, equilibrium (0.5 of range)
  - Price zone: PREMIUM / DISCOUNT / EQUILIBRIUM
  - Biggest displacement candle + strength vs average body
  - Automatic MSB detection (Higher High / Lower High / Higher Low / Lower Low)
  - Last 5 swing highs and swing lows with timestamps
  - All Bull/Bear FVGs detected in the window with exact zones
  - Last 10 OHLC candles in a clean table
- **When to use:** Any time you need a quick snapshot of a coin's current price action, structure, and key levels — before entering a trade, checking a live position, or doing a post-mortem.
- **Usage:**
  ```bash
  # Default: BTCUSDT, 15m, last 30 candles
  python scratch/market_check.py BTCUSDT

  # Custom symbol, interval, and candle count
  python scratch/market_check.py ETHUSDT 1h 50
  python scratch/market_check.py SOLUSDT 4h 20
  python scratch/market_check.py HYPEUSDT 15m 30
  python scratch/market_check.py BNBUSDT 5m 60
  ```

### `scratch/scan_now.py`
- **Purpose:** Instant one-shot scan that runs the full bot logic once across all watchlist symbols and prints any active setups with tier, entry, SL, and TP.
- **When to use:** Quick check for live setups without running the full `challenge_trader.py` loop.
- **Usage:**
  ```bash
  python scratch/scan_now.py
  ```
