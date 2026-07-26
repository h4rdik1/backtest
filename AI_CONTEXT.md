# 🧠 AI Context & Project Brain — TTrades SMC Trading Engine

> **For AI Assistants:** Read this file first. It tells you everything you need to know about the project's purpose, structure, and the rules you must follow before touching any code.

---

## 🎯 Project Overview

A professional-grade **Python autonomous trading engine** built on **Smart Money Concepts (SMC)**. It finds and executes high-probability institutional setups — specifically **Swing Failure Patterns (SFPs)**, **Liquidity Sweeps**, and **Fair Value Gaps (FVGs)** — by reading the market from the Highest Timeframe down to the Entry Timeframe (HTF → LTF).

- **Language/Stack:** Python 3 · CCXT · Pandas · NumPy
- **Exchanges:** Binance (default), OKX
- **Current Focus:** Refining 3-candle fractal-based SFP detection; ensuring sweeps are validated on closed candles only.

---

## 📜 Core Trading Rules (Read Before Changing Any Logic)

| Priority | Document | Purpose |
|---|---|---|
| 🔴 Must Read | [TRADING_RULES.md](file:///e:/backtest/TRADING_RULES.md) | Full tier system (S / A++ / A / B), conflict-of-bias rules, SOPs for range marking, execution checklist |
| 🔴 Must Read | [S_TIER_TRADING_RULES.md](file:///e:/backtest/S_TIER_TRADING_RULES.md) | Exact step-by-step sequence for the "Double Sweep" S-Tier setup |
| 🟡 Reference | [tools_reference.md](file:///e:/backtest/tools_reference.md) | How to run every debugging/analysis script |

**The one philosophy that overrides everything else:**  
> *Context → Bias → Entry. HTF determines direction. LTF determines timing. Never work backwards from LTF.*

---

## 🗂️ Full Project Hierarchy

```
e:/backtest/
│
├── 🔴 challenge_trader.py          ← MAIN ENGINE. Autonomous paper trading. Manages
│                                     challenge state, risk ($50/trade), daily trade cap.
│                                     DO NOT break its state-persistence logic.
│
├── 🟠 live_scanner.py              ← LIVE MONITOR. Scans multi-exchange markets in real
│                                     time, ranks setups by Tier (A++ → A → B).
│
├── 🟡 tradingview_smc_v1.pine      ← TradingView Pine Script companion (SMC overlays).
│
├── 📄 AI_CONTEXT.md                ← THIS FILE. The project brain for AI assistants.
├── 📄 TRADING_RULES.md             ← Institutional rules. Read before changing logic.
├── 📄 S_TIER_TRADING_RULES.md      ← Deep dive on S-Tier "Double Sweep" setups.
├── 📄 tools_reference.md           ← Index of all scripts and how to run them.
├── 📄 README.md                    ← General project intro and benchmarks.
│
├── 📄 requirements.txt             ← pip dependencies (ccxt, pandas, numpy, etc.)
├── 📄 .env                         ← API keys (NEVER commit, already in .gitignore)
├── 📄 start_trading_system.bat     ← Windows launcher for the trading system.
│
│
├── 🔴 core/                        ← THE BRAIN of the engine. Most logic lives here.
│   ├── config.py                   ← ⭐ SINGLE SOURCE OF TRUTH. All strategy constants:
│   │                                  MIN_SETUP_TIER, SL distances, TF alignment chains,
│   │                                  max trades/day, watchlist, retired alignments.
│   │                                  ALWAYS CHECK HERE before hardcoding any values.
│   ├── lux_fvg.py                  ← SMC Math Engine. FVG detection, Order Block (OB)
│   │                                  identification, Liquidity Sweep / SFP logic.
│   │                                  Primary file for fractal/SFP changes.
│   └── analysis.py                 ← Statistical engine. Monte Carlo simulations, Market
│                                      Regime classifier, PnL validation, risk analytics.
│
│
├── 🟠 models/                      ← Trade setup detection and entry logic.
│   └── backtest_cisd.py            ← Core setup model: FVG/OB touch entries, Enhanced
│                                      Structural SL Finder (detects SFPs + prominent wicks),
│                                      "visited_signals" deduplication to avoid re-trading zones.
│
│
├── 🟡 tools/                       ← Debugging, diagnostics, and visualization scripts.
│   ├── build_master_analysis.py    ← ⭐ CENTRALIZED ANALYSIS BUILDER. Reads ALL challenge CSVs
│   │                                  (archive + current) and writes exports/MASTER_ANALYSIS.md.
│   │                                  Run after every challenge ends or anytime to get fresh stats.
│   │                                  Run: python tools/build_master_analysis.py
│   ├── analyze_setup.py            ← Deep multi-TF SMC analysis for a single symbol.
│   │                                  Evaluates HTF (4H), MTF (1H), LTF (5M). Best tool
│   │                                  for pre-trade manual verification.
│   │                                  Run: python tools/analyze_setup.py BTC/USDT
│   ├── dry_run_btc_sl.py           ← Tests structural SL placement logic (SFP vs FVG).
│   │                                  Use when tuning stop-loss algorithms.
│   ├── analyse_challenges.py       ← Post-mortem analyzer. Reads all CSV logs from
│   │                                  /archive and outputs WR, PF, streaks, per-symbol stats.
│   ├── check_candles.py            ← Quick terminal candle viewer (OHLCV table).
│   │                                  Run: python tools/check_candles.py BTC/USDT -t 15m -l 20
│   ├── view_candles.py             ← Local HTTP server backend for the visual dashboard.
│   │                                  Powers view_candles.html and view_advanced.html.
│   ├── view_candles.html           ← Main browser dashboard for visualizing live SMC chart.
│   └── view_advanced.html          ← Advanced overlay dashboard (FVGs, OBs, active trades).
│
│
├── 🟢 exports/                     ← Live runtime outputs. Generated by challenge_trader.py.
│   ├── challenge_state.json        ← ⭐ LIVE STATE FILE. Tracks current balance, open trades,
│   │                                  daily trade count, visited signals. Critical — do not
│   │                                  manually edit unless you know what you're doing.
│   ├── MASTER_ANALYSIS.md          ← ⭐ CENTRALIZED STRATEGY KNOWLEDGE BASE. Cross-challenge
│   │                                  aggregated stats: WR by symbol/tier/regime/session/chain.
│   │                                  Refresh by running: python tools/build_master_analysis.py
│   │                                  READ THIS BEFORE making any strategy configuration changes.
│   ├── challenge_trades.csv        ← Log of all completed trades in the current challenge run.
│   ├── challenge_events.txt        ← Verbose event log (entries, exits, scan results).
│   ├── signal_log.csv              ← Full history of every signal detected (pass or skip).
│   ├── overnight_scan_log.txt      ← Log output from unattended overnight scan runs.
│   └── trade_analysis_post_mortem.md ← Auto-generated post-mortem report after a challenge ends.
│
│
├── 🗄️ archive/                     ← Completed challenge snapshots. Used for performance review.
│   ├── challenge_PASSED_*.json/csv ← Challenges that hit the $5,500 target. ✅
│   ├── challenge_STOPPED_*.json/csv← Challenges that were stopped early (drawdown/review). ⛔
│   ├── challenge_ACTIVE_*.json     ← Snapshots of challenges that were still active when archived.
│   └── backtest_limit.py           ← Standalone backtesting script (not part of live engine).
│                                     Replays historical data against the strategy rules.
│
│
├── 📦 data/                        ← Cached OHLCV price data (avoids repeated API calls).
│   └── ohlcv/
│       ├── BTC_USDT/               ← Cached candle data per timeframe for BTC/USDT.
│       ├── ETH_USDT/               ← Cached candle data per timeframe for ETH/USDT.
│       ├── SOL_USDT/               ← Cached candle data per timeframe for SOL/USDT.
│       └── HYPE_USDT/              ← Cached candle data per timeframe for HYPE/USDT.
│
│
├── 🔬 scratch/                     ← Utility and experimental scripts.
│   ├── market_check.py             ← ⭐ PRIMARY MANUAL TOOL. Universal market price fetcher
│   │                                  and SMC structure analyzer for any Binance symbol.
│   │                                  Outputs: price levels, displacement, MSB, swing
│   │                                  highs/lows, FVGs, and last 10 candles.
│   │                                  Run: python scratch/market_check.py BTCUSDT 15m 30
│   ├── scan_now.py                 ← Instant one-shot scan script for quick setup checks.
│   │                                  Run: python scratch/scan_now.py
│   ├── test_sfp.py                 ← Unit tests for the SFP detection logic.
│   ├── test_new_tasks.py           ← Experimental feature tests.
│   ├── analyze_sui.py              ← Ad-hoc SUI/USDT analysis script.
│   ├── check_all_4h_ranges.py      ← Quick scanner for 4H range structures.
│   ├── real_life_limits.py         ← Simulates limit order slippage in real conditions.
│   └── debug_regime.py             ← Debugs market regime classification output.
│
│
└── 📜 history/                     ← Strategy version history (read-only reference).
    ├── v1_base_smc.py              ← Original base SMC strategy (no tier system).
    ├── v2_refined_confirmation.py  ← Added multi-TF confirmation layer.
    └── v3_limit_expansion.py       ← Added limit order expansion logic.
```

---

## 🔑 Key Concepts Glossary

| Term | Meaning |
|---|---|
| **SFP** | Swing Failure Pattern — a wick that breaks a swing high/low but the *candle body closes back inside*. This is the primary institutional signal. |
| **FVG** | Fair Value Gap — a 3-candle imbalance zone where price moved too fast, leaving a gap to be revisited. |
| **OB** | Order Block — the last bearish candle before a bullish displacement (or vice versa). Institutional supply/demand zone. |
| **BOS** | Break of Structure — a candle *body* closes beyond a swing high/low, invalidating the previous range. |
| **MSB** | Market Structure Break — same as BOS; used interchangeably in this codebase. |
| **HTF** | Higher Timeframe (1D or 4H). Determines the macro bias direction. |
| **LTF** | Lower Timeframe (1H, 15M, 5M). Used for entry timing and confirmation. |
| **Tier** | Setup quality grade: S-Tier (best) → A++ → A → B-Tier (skip in challenge). |

---

## 🤖 AI Rules — Always Follow These

1. **Never violate the Tier System.** Any change to setup detection must preserve the S / A++ / A / B distinction. Check `TRADING_RULES.md` if unsure.
2. **Context → Bias → Entry.** Never invert this hierarchy. HTF bias first, LTF entry second.
3. **`core/config.py` is the single source of truth.** Never hardcode strategy constants anywhere else — always reference or update `config.py`.
4. **Use structural stops.** Prefer swing extremes + SFP wicks for SL over naked FVG boundaries.
5. **Preserve `challenge_state.json` integrity.** The state file tracks open trades, balance, and visited signals. A corrupt state file breaks the live engine.
6. **SFP requires a closed candle.** Never flag a wick as an SFP while the candle is still forming.
7. **Debugging workflow:** Run `python tools/analyze_setup.py <SYMBOL>` for a full pre-trade breakdown. Run `python tools/dry_run_btc_sl.py` to test SL logic in isolation.
