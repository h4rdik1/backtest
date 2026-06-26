# TTrades Tools & Scripts Reference Guide

This document provides a comprehensive overview of all the available tools and scripts in the repository. You can refer to this list when deciding how to analyze the market, debug a trade, or review performance.

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
