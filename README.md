# 💹 TTrades SMC Trading Engine (Phase 4 - April 2026 Overhaul)

A professional-grade Python backtesting and live-monitoring framework for **Smart Money Concepts (SMC)**. This engine implements the **TTrades Scalping Model** using displacement-based **LuxAlgo FVG** detection, HTF institutional bias alignment, and Order Block (OB) confluence.

> [!IMPORTANT]
> **April 2026 Update**: The engine has been completely overhauled to eliminate the "0 trade" bottleneck. It now uses a **Reaction-based entry** on the 15m/5m/1m timeframes with validated profitability across BTC, ETH, and SOL.

---

## 📁 Project Architecture

- **`challenge_trader.py`**: **Main Entry Point (Live/Paper)**. Runs a continuous scan for signals, manages a $5,000 paper trading challenge, and logs events.
- **`live_scanner.py`**: Real-time market monitor that prints SMC setups to the terminal without execution.
- **`core/`**: The brain of the project.
  - `config.py`: **Single source of truth** for all settings (Watchlist, R:R, Alignments).
  - `analysis.py`: Statistical validation (Monte Carlo, Regime Analysis, PnL math).
  - `lux_fvg.py`: Mathematical FVG and Liquidity Sweep detection based on LuxAlgo logic.
- **`models/`**: Trading strategies.
  - `backtest_cisd.py`: **The Core Model**. Enters on FVG/OB reaction (touch) + HTF bias alignment.
  - `backtest_limit.py`: **Limit Order Model**. Places limit orders at FVG midpoints.
- **`tools/`**: Utility scripts.
  - `view_candles.py`: Interactive plotter to visualize candles, FVGs, and OBs.
  - `fetch_candles.py`: Paginated data fetcher for CCXT/Binance.
- **`tradingview_smc_v1.pine`**: PineScript implementation for visual parity on TradingView.

---

## 🧠 SMC Methodology: CBE Alignment

The engine follows a strict **Context → Bias → Entry (CBE)** hierarchy. Trade signals only fire when all three timeframes align:

| Chain | Context (HTF) | Bias (MTF) | Entry (LTF) | Style |
| :--- | :--- | :--- | :--- | :--- |
| **Weekly** | 1 Month | 1 Day | 1 Hour | Macro Swing |
| **Daily** | 1 Day | 1 Hour | 5 Min | Day Trading |
| **Intraday** | 12 Hour | 1 Hour | 5 Min | High Frequency |
| **Scalp** | 4 Hour | 15 Min | 1 Min | Micro Scalping |

---

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.9+ installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Fetch Historical Data
You must build a local data cache (365 days recommended for alignment validity):
```bash
# Fetch 1 year of data for core assets
python fetch_candles.py --symbols BTC/USDT ETH/USDT SOL/USDT BNB/USDT --days 365
```

---

## 🧪 How to Run

### 🏆 Challenge Trader (Paper/Live)
Run the autonomous challenge engine. It manages a $5,000 account with strict risk rules:
```bash
python challenge_trader.py
```
*Features:* $300 Floor, $500 Target, Max 2 concurrent trades, 10-minute scan interval.

### 🔭 Live Scanner
Monitor the markets for signals without executing trades:
```bash
python live_scanner.py
```

### 📊 Master Backtest
Run the multi-asset runner to see combined performance of the CISD model:
```bash
python run_multi_asset.py --alignment daily
```

---

## 📊 Performance Benchmarks (1-Year Backtest)

The overhauled CISD strategy achieves a **~42% Win Rate** with a focused **2.0 Risk:Reward ratio**.

| Asset | Model | Trades | Profit Factor | PnL (1-Year) |
| :--- | :--- | :--- | :--- | :--- |
| **BTC/USDT** | CISD | 894 | **1.51** | +$6,450 |
| **ETH/USDT** | CISD | 882 | **1.43** | +$5,550 |
| **SOL/USDT** | CISD | 835 | **1.38** | +$4,700 |

---

## 🛠️ Configuration Tuning

All strategy rules are centralized in **`core/config.py`**. 

- `ACTIVE_CHAIN`: Switch between `daily`, `weekly`, or `scalp` for different trading intensities.
- `RISK_REWARD`: Set to `2.0` (validated optimal for SMC setups).
- `USE_LTF_OB_ENTRY`: Enable/Disable Order Block entries alongside FVGs.
- `LIVE_WATCHLIST`: Add or remove symbols for scanning (e.g., `LINK/USDT`, `BNB/USDT`).

---

**Disclaimer**: This project is for educational purposes. Trading carries significant risk. Refer to the [latest walkthrough](file:///C:/Users/hardi/.gemini/antigravity/brain/1dc5aac8-cb15-4735-a728-525d6b59f8b3/walkthrough.md) for full diagnostic and optimization details.
