# 🧠 Trading System Brain & Knowledge Hub

Welcome to the central command hub for the **Backtest & Live Trading System**.

---

## 📌 Core Documentation & Rules
- [[TRADING_RULES|Trading Rules & Execution Framework]] — Master trading rules, setup criteria, and risk parameters.
- [[S_TIER_TRADING_RULES|S-Tier Trading Rules]] — High-probability, strict setup criteria.
- [[AI_CONTEXT|AI Context & Guidelines]] — Instructions and context for AI analysis & subagents.
- [[project_study_guide|Project Study Guide]] — Complete system overview and conceptual walk-through.
- [[tools_reference|Tools Reference]] — Breakdown of scripts in `tools/` and how to run them.
- [[README|Project README]] — Setup, requirements, and general execution guide.

---

## ⚙️ Architecture & Code Map

### 1. 🤖 Telegram Bot (`telegramBot/`)
- [[telegramBot/telegram_bot|telegram_bot.py]] — Bot initialization & entry point.
- [[telegramBot/handlers|handlers.py]] — Message command handlers (`/scan`, `/status`, trade image analysis).
- [[telegramBot/agent|agent.py]] — AI tool dispatch & autonomous loop.
- [[telegramBot/ai|ai.py]] — Trade vision & setup evaluation API calls.

### 2. ⚡ Core Logic (`core/`)
- [[core/analysis|analysis.py]] — Technical analysis calculations, market regime detection.
- [[core/config|config.py]] — Strategy rules, risk parameters, and pair definitions.
- [[core/lux_fvg|lux_fvg.py]] — Fair Value Gap (FVG) detection module.

### 3. 📊 Backtest Engine (`models/`)
- [[models/backtest_cisd|backtest_cisd.py]] — Change in State of Delivery (CISD) backtest simulator.

### 4. 🛠️ Execution & Scanner Scripts
- [[challenge_trader|challenge_trader.py]] — Live challenge trading automation script.
- [[live_scanner|live_scanner.py]] — Continuous market scanner for active setups.
- [[tradingview_smc_v1.pine|tradingview_smc_v1.pine]] — Custom TradingView Pine Script indicator.

---

## 📂 Quick Links & Folders
- [[notes/Strategies|Strategies & Setup Notes]] — Detailed breakdown of SMC, CISD, and FVG mechanics.
- [[notes/Journal|Trading Journal & Post-Mortems]] — Daily trade logs and system audit notes.

---
*Created automatically to serve as the Obsidian Vault Brain.*
