# 🧠 TTrades SMC Trading Engine: Interview Study Guide

This guide is designed to help you understand your autonomous trading engine inside and out, making it easy to explain your architectural decisions and trading logic during technical interviews.

## 🚀 1. Project Elevator Pitch
"I built a professional-grade, autonomous algorithmic trading engine in Python based on Smart Money Concepts (SMC). It uses the CCXT library to pull multi-exchange market data (Binance, OKX) and leverages Pandas and NumPy to perform complex fractal and statistical analysis. The system strictly follows institutional trading logic, automatically identifying high-probability setups like Swing Failure Patterns (SFPs) across multiple timeframes, managing risk automatically, and executing trades through a tiered quality system."

## 🏗️ 2. High-Level Architecture
The project is structured modularly, separating the core logic, execution, and analysis:

*   **Execution Engines (`challenge_trader.py` & `live_scanner.py`)**
    *   **Live Scanner**: Runs continuous loops across multiple exchanges to monitor for A-Tier setups in real-time.
    *   **Challenge Trader**: The autonomous executor that manages a simulated $5,000 challenge capital, applying strict risk management ($50 risk per trade) and max daily trade caps.
*   **The Brain (`core/`)**
    *   **`config.py`**: The single source of truth for strategy parameters (stop loss distances, risk settings, timeframe alignments).
    *   **`lux_fvg.py`**: The mathematical engine that detects Fair Value Gaps (FVGs), Order Blocks (OBs), and Liquidity Sweeps (SFPs) using pandas and numpy array operations.
    *   **`analysis.py`**: Statistical engine for Monte Carlo simulations, market regime classification, and performance validation.
*   **Modeling & Validation (`models/` & `tools/`)**
    *   **`backtest_cisd.py`**: Contains the logic for structural Stop Loss (SL) placement and entering trades on FVG/OB touches.
    *   Visual dashboards (HTML/Pine) and CLI diagnostic tools for verifying AI signals manually.

## 📊 3. The Institutional Methodology (The Tier System)
The most unique feature of the engine is that it doesn't just trade every signal. It grades setups into **Tiers** based on the **Context → Bias → Entry (CBE)** philosophy (Higher Timeframe decides direction, Lower Timeframe decides timing).

*   🏆 **S-Tier (The Double Sweep)**: Highest conviction. Requires a Higher Timeframe (HTF) Sweep + Lower Timeframe (LTF) Sweep + Market Structure Break (MSB).
*   ⭐ **A++ Tier (The Confluence)**: HTF Sweep + LTF FVG/OB Overlap.
*   ✅ **A-Tier (Standard Setup)**: HTF Alignment + LTF Sweep + MSB.
*   ⚠️ **B-Tier (Basic FVG)**: Filtered out to avoid "volume kill" low-probability trades.

## 🔑 4. Key Terminology (Smart Money Concepts)
You should be comfortable defining these terms if asked:
*   **SFP (Swing Failure Pattern)**: A wick that breaks a swing high/low, but the candle *body* closes back inside the range. This indicates a liquidity sweep (institutions trapping retail traders) and is your primary signal.
*   **FVG (Fair Value Gap)**: A 3-candle imbalance zone where price moved too fast, leaving a gap. Price often returns here to fill the gap.
*   **OB (Order Block)**: The last bearish candle before a strong bullish push (or vice versa), representing institutional supply/demand.
*   **MSB (Market Structure Break) / BOS (Break of Structure)**: When a candle *body* closes beyond a previous swing high/low, signaling a trend change.
*   **HTF / LTF**: Higher Timeframe (e.g., 4H, 1D) vs Lower Timeframe (e.g., 15M, 5M).

## 💡 5. Common Interview Questions & How to Answer Them

**Q: How does your system manage risk?**
*A: Risk management is deeply integrated into `challenge_trader.py` and `config.py`. It uses a fixed fractional risk model (e.g., $50 or 1% per trade) and enforces minimum stop-loss distances (0.3% for BTC, 0.5% for Alts) based on structural points (recent swing highs/lows) rather than tight, arbitrary borders. It also has a hard daily max-trade cap to prevent overtrading during choppy market regimes.*

**Q: How do you handle real-time data and execution?**
*A: I use the `CCXT` library to interact asynchronously with exchange APIs like Binance and OKX. The system caches OHLCV data locally in `data/` to minimize redundant API calls, processes the incoming candles using Pandas DataFrames for performance, and executes limit/market orders while constantly logging state to a JSON file (`exports/challenge_state.json`) to persist state in case of a crash.*

**Q: What was the biggest challenge you faced building this?**
*A: (Great place to mention the April 2026 upgrade!). Initially, the system traded too many low-quality setups (B-Tier). I overhauled the engine to implement a strict Tiered system and refined the 3-candle fractal logic for SFP detection to ensure the system only flags sweeps on closed candles, vastly improving the win rate (e.g., BNB to 63.6%).*
