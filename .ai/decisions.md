# Architectural Decisions (MEMORY)

This document tracks *why* certain technical and architectural choices were made, preventing the AI from attempting to rewrite core systems.

## 1. File-Based State (JSON/CSV) over Databases
**Decision**: We use `challenge_state.json` and CSV files in `/exports` instead of SQLite or Postgres.
**Reason**: Transparency and easy archiving. At the end of a challenge, the entire `/exports` folder is simply zipped or moved to `/archive`. It allows the user to manually open the JSON and inspect/fix state instantly without database tools.

## 2. Pre-Caching Data in `/data/ohlcv/`
**Decision**: The engine aggressively caches OHLCV data locally.
**Reason**: To avoid CCXT rate limits during multi-timeframe analysis. The engine needs to constantly check 4H, 1H, and 15m/5m charts. Fetching this live every loop for multiple symbols would trigger API bans.

## 3. The Tier System (S, A++, A, B)
**Decision**: Implemented a strict grading system rather than taking every valid FVG/SFP touch.
**Reason**: Earlier versions suffered from "volume kills" where the bot took too many B-tier trades in choppy markets. The tier system enforces institutional logic (Context -> Bias -> Entry).

## 4. Separation of Live Scanner and Challenge Trader
**Decision**: `live_scanner.py` and `challenge_trader.py` are distinct scripts.
**Reason**: Allows for running the scanner simply as a notification/alert system without actually risking paper money in a challenge run.

## 5. Structural SLs over Fixed % SLs
**Decision**: Stop losses are placed behind swing extremes or SFP wicks (`PROTECTED_SWING_LOOKBACK`).
**Reason**: Fixed % SLs (e.g., 1%) get hunted by market makers. Using structural protection ensures the trade idea is completely invalidated before the stop is hit.
