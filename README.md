# 💹 TTrades SMC Trading Engine (Phase 4 - Overhauled)

A professional-grade Python backtesting and live-monitoring framework for **Smart Money Concepts (SMC)**. This engine implements the **TTrades Scalping Model** using displacement-based **LuxAlgo FVG** detection and institutional bias alignment.

> [!IMPORTANT]
> **April 2026 Update**: The engine has been completely overhauled to fix a "0 trade" bottleneck. It now uses a **Reaction-based entry** on the 15m timeframe with validated profitability across BTC, ETH, and SOL.

---

## 📁 Project Architecture

- **`core/`**: The brain of the project.
  - `config.py`: **Single source of truth** for R:R (2.0), 15m LTF settings, and filter toggles.
  - `analysis.py`: Statistical validation (Monte Carlo, Regime Analysis).
  - `lux_fvg.py`: Mathematical FVG and Liquidity Sweep detection.
- **`models/`**: Trading strategies.
  - `backtest_cisd.py`: **Overhauled Model**. Enters on FVG reaction (touch) + HTF bias.
  - `backtest_limit.py`: **Limit Order Model**. Places limit orders at FVG midpoints.
- **`tools/`**: Utility scripts.
  - `diagnose.py`: Deep diagnostic tool to identify strategy bottlenecks.
  - `fetch_candles.py`: Paginated data fetcher for CCXT/Binance.
- **`data/ohlcv/`**: Local cache where historical candle data is stored.

---

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.9+ installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Fetch Historical Data
You must build a local data cache (365 days recommended for 15m/1h alignment):
```bash
# Fetch 1 year of data for core assets
python fetch_candles.py --symbols BTC/USDT ETH/USDT SOL/USDT --days 365
```

---

## 🧪 How to Run

### Master Backtest
Run the multi-asset runner to see the combined performance of the CISD model:
```bash
python run_multi_asset.py --alignment daily
```

### Individual Model Testing
Test a specific asset with the overhauled CISD logic:
```bash
python models/backtest_cisd.py --symbol BTC/USDT
```

### Live Scanner
Monitor the markets in real-time for 15m FVG reaction setups:
```bash
python live_scanner.py
```

---

## 📊 Performance Benchmarks (1-Year Backtest)

The overhauled strategy achieves a **~42% Win Rate** with a **2.0 Risk:Reward ratio**.

| Asset | Model | Trades | Profit Factor | PnL |
| :--- | :--- | :--- | :--- | :--- |
| **BTC/USDT** | CISD | 894 | **1.51** | +$6,450 |
| **ETH/USDT** | CISD | 882 | **1.43** | +$5,550 |
| **SOL/USDT** | CISD | 835 | **1.38** | +$4,700 |

---

## 🛠️ Configuration Tuning
All strategy rules are centralized in **`core/config.py`**. 

- `DEFAULT_LTF`: Set to `15m` for the optimal balance of noise reduction and trade frequency.
- `RISK_REWARD`: Set to `2.0` (validated optimal).
- `USE_AUTO_BREAKEVEN`: Disabled by default to maximize win rate on 15m swings.

---
**Disclaimer**: This project is for educational purposes. Trading carries significant risk. Refer to the [latest walkthrough](file:///C:/Users/hardi/.gemini/antigravity/brain/1dc5aac8-cb15-4735-a728-525d6b59f8b3/walkthrough.md) for full diagnostic and optimization details.
