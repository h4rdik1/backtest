# Signal Engine (Phase 5 - April 2026 Institutional Upgrade)

A professional-grade Python autonomous trading and live-monitoring framework for **Smart Money Concepts (SMC)**. This engine implements the **TTrades Tiered Scalping Model** using displacement-based **LuxAlgo FVG** detection, HTF institutional bias alignment, and Order Block (OB) confluence.

> [!IMPORTANT]
> **April 2026 Overhaul**: The engine now features a **Multi-Exchange Live Scanner**, **Automated Position Management**, and a strict **Tiered Setup System** (S-Tier to B-Tier) to ensure only high-probability institutional footprints are traded.

---

## 📁 Project Architecture

- **`challenge_trader.py`**: **Main Execution Engine**. Runs continuous scans, manages $5,000 challenge capital with automated risk management. Now features **Institutional Stop Alignment** and **0.3%/0.5% Min-Risk protection**.
- **`live_scanner.py`**: **Multi-Exchange Real-Time Monitor**. Features an automated loop to scan the market across Binance, OKX, and others for A-Tier setups.
- **`core/`**: The brain of the project.
  - `config.py`: **Single source of truth**. Now with **Minimum SL Distances** (0.3% BTC, 0.5% Alts).
  - `analysis.py`: Statistical engine for Monte Carlo simulations, Market Regime classification, and PnL validation.
  - `lux_fvg.py`: Mathematical FVG and Liquidity Sweep (SFP) detection.
- **`models/`**:
  - `backtest_cisd.py`: Logic for entering on FVG/OB "touches". Now features an **Enhanced Structural SL Finder** (detects SFPs and prominent wicks).
- **`tools/`**:
  - `view_candles.py`: Graphical plotter for visualizing SMC structures.
  - `dry_run_btc_sl.py`: **Structural SL Tester**. Validate SL placement and HTF alignment on live data.
  - `analyse_challenges.py`: Deep-dive diagnostic for post-mortem analysis.
- **`exports/`**: Storage for `challenge_state.json`, trade logs, and **`trade_analysis_post_mortem.md`** reports.

---

## 🧠 Institutional Methodology: The Tier System (New)

The engine follows a strict **Context → Bias → Entry (CBE)** hierarchy. We no longer trade all valid signals; setups are ranked into Tiers defined in [TRADING_RULES.md](file:///e:/backtest/TRADING_RULES.md):

| Tier | Name | Requirement | Status |
| :--- | :--- | :--- | :--- |
| 🏆 **S-Tier** | The Double Sweep | HTF SFP + LTF Sweep + MSB | **Trade (Limit)** |
| ⭐ **A++ Tier** | The Confluence | HTF Sweep + LTF FVG/OB Overlap | **Trade** |
| ✅ **A-Tier** | Standard Setup | HTF Alignment + LTF Sweep + MSB | **Trade** |
| ⚠️ **B-Tier** | Basic FVG | Single Timeframe Reaction | **SKIP** |

### Timeframe Alignment Chains
Defined in `config.py` using standard institutional pairings:

| Chain | Context (HTF) | Bias (MTF) | Entry (LTF) | Strategy Class |
| :--- | :--- | :--- | :--- | :--- |
| **Weekly** | 1 Week | 4 Hour | 15 Min | Swing / High Conviction |
| **Daily** | 1 Day | 1 Hour | 5 Min | Day Trading (Standard) |

---

## 🚀 Getting Started

### 1. Installation
Install core dependencies (CCXT, Pandas, NumPy):
```bash
pip install -r requirements.txt
```

### 2. Live Scanner (Multi-Exchange)
Run the automated scanner loops. You can now specify symbols with exchange overrides:
```bash
# Default (Binance)
python live_scanner.py

# Custom symbols across different exchanges
python live_scanner.py --symbols BTC/USDT ETH/USDT:okx HYPE/USDT:okx
```

### 3. Challenge Trader
Autonomous engine targeting a $5,500 goal ($500 gain). Features automated "Daily Trade Cap" and state persistence.
```bash
python challenge_trader.py
```
*Current Fixed Risk:* $50 (1%) per trade. *Max Drawdown:* $300 (Floor at $4,700).

---

## 📊 Backtest Benchmarks (April 2026)

The Tiered A-Tier model achieves significantly higher quality outcomes by filtering out low-probability B-Tier "volume kill" trades.

| Asset | Model | Trades | Win Rate | Profit Factor |
| :--- | :--- | :--- | :--- | :--- |
| **BNB/USDT** | A-Tier | 120 | **63.6%** | **2.11** |
| **ETH/USDT** | A-Tier | 482 | **42.9%** | **1.43** |
| **BTC/USDT** | A-Tier | 494 | **38.2%** | **1.21** |

---

## 🛠️ Configuration Tuning

All strategy rules are centralized in **`core/config.py`**. 

- `MIN_SETUP_TIER`: Set to `"A"` to automatically block low-probability B-tier setups.
- `MAX_TRADES_PER_DAY`: Hard cap to prevent over-trading in ranging conditions.
- `PROTECTED_SWING_LOOKBACK`: Defines structural SL depth (replaces tight FVG boundary stops).
- `RETIRED_ALIGNMENTS`: Preserves management logic for legacy trades (Scalp, Intraday, Macro).

---

**Disclaimer**: This project is for educational purposes. Smart Money Concepts are statistical models and do NOT guarantee profit. Refer to [TRADING_RULES.md](file:///e:/backtest/TRADING_RULES.md) for full risk and execution protocol.
