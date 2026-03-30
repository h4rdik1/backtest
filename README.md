# 💹 TTrades SMC Trading Engine (Phase 4)

A professional-grade Python backtesting and live-monitoring framework for **Smart Money Concepts (SMC)**. This engine implements the **TTrades Scalping Model** using displacement-based **LuxAlgo FVG** detection and institutional bias alignment.

---

## 📁 Project Architecture

The codebase follows a modular design for maximum scalability and maintainability:

- **`core/`**: The brain of the project.
  - `config.py`: Primary settings for R:R, risk, filter toggles, and data paths.
  - `analysis.py`: Statistical validation (Monte Carlo, Walk-Forward, Regime Analysis).
  - `lux_fvg.py`: Mathematical FVG and Liquidity Sweep detection.
- **`models/`**: Trading strategies.
  - `backtest_cisd.py`: Confirmation entry (Change in State of Delivery).
  - `backtest_limit.py`: Aggressive limit entry at FVG 50% CE.
- **`tools/`**: Utility scripts.
  - `check_signal.py`: On-demand "Go/No-Go" check for active signals.
  - `view_candles.py`: Interactive browser-based visualizer.
  - `watchlist_scanner.py`: Background automation for signal alerts.
- **`data/ohlcv/`**: Local cache where historical candle data is stored.
- **`exports/`**: Centralized output for Excel reports and CSV logs.

---

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.9+ installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Fetch Historical Data
You must build a local data cache (3 years recommended for statistical validity):
```bash
# Fetch 3 years of data for multiple assets
python fetch_candles.py --symbols BTC/USDT ETH/USDT SOL/USDT BNB/USDT XRP/USDT --days 1095
```

---

## 🧪 How to Run

### Master Backtest (Recommended)
The master runner executes both models across all configured symbols and performs full statistical validation.
```bash
python run_multi_asset.py --alignment daily --walk-forward
```
- **Results**: Check the `exports/` folder for `MultiAsset_Analysis_daily.xlsx`.

### Live Scanner
Monitor the markets in real-time for pending or active setups.
```bash
python live_scanner.py --alignment daily
```

### Interactive Visualization
Visualise FVGZones, HTF Bias, and trade entries in your browser:
```bash
python tools/view_candles.py
```

---

## 📊 Statistical Validation Framework

Every strategy performance is subjected to three tiers of validation:

1. **Monte Carlo (Tier 1)**: Shuffles trade order 10,000 times to test if the edge is "Real" or "Lucky".
2. **Walk-Forward (Tier 2)**: Compares performance on design data (Train) vs. unseen data (Test) to detect overfitting.
3. **Monthly Consistency (Tier 3)**: Requires a 60% profitable month ratio to be considered "Tradeable".

---

## 🛠️ Configuration
All strategy rules are centralized in **`core/config.py`**. You can toggle filters without touching the source code:
- `USE_FVG_QUALITY`: Only trade the first touch of an FVG.
- `USE_HTF_OB_CONFLUENCE`: Require price to be near a HTF Order Block.
- `USE_PREMIUM_DISCOUNT`: Only buy in Discount, Sell in Premium.

---
**Disclaimer**: This project is for educational and backtesting purposes. Trading carries significant risk. Always validate results on a demo account before live deployment.
