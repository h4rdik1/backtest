# Product Requirement Document (PRD)

## 1. WHAT TO BUILD
**TTrades SMC Trading Engine** is a professional-grade Python autonomous trading engine built on Smart Money Concepts (SMC). The primary goal is to run a paper trading "challenge" that can autonomously scan, identify, and execute high-probability institutional setups (specifically Swing Failure Patterns, Liquidity Sweeps, and Fair Value Gaps) to achieve a target profit (e.g., $500 gain on a $5,000 account) using strict risk management.

## 2. TARGETED USER
- **Algorithmic Traders / Quantitative Developers** looking for a strict, rules-based engine that implements SMC theory without emotional interference.
- **Proprietary Trading Firm Applicants** (funding challenges) who need a system with hard drawdown limits, daily trade caps, and automated position sizing.

## 3. FEATURES
### Core Capabilities
- **Multi-Exchange Live Scanner:** Real-time monitoring across Binance, OKX, etc., to rank setups by Tier (S, A++, A, B).
- **Challenge Management:** Autonomous paper trading execution (`challenge_trader.py`) that tracks balance, risk per trade ($50 default), and halts trading if daily limits or drawdown thresholds are hit.
- **SMC Detection Math:** Algorithmic identification of Order Blocks (OBs), Fair Value Gaps (FVGs), and Market Structure Breaks (MSBs).
- **Tiered Setup System:** A strict filtering mechanism prioritizing "Context → Bias → Entry". Only high-tier setups (A-Tier and above) are executed; low-quality B-Tier setups are skipped.
- **Structural Risk Management:** Dynamic Stop-Loss placement based on swing extremes and prominent wicks rather than fixed percentages.

### Secondary Features
- **Visual Dashboards:** Local HTML/JS dashboards (`view_candles.html`) to visualize SMC structures and active trades.
- **Post-Mortem Analytics:** Automated markdown reporting (`MASTER_ANALYSIS.md`) to aggregate win rates, profit factors, and strategy performance across historical runs.
