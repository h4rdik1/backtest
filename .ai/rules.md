# Coding Rules & AI Boundaries

## 1. WHAT TO USE
- **Python 3.10+** features (Type hinting is highly encouraged).
- **Pandas and NumPy** for all array and timeseries operations (avoid slow python `for` loops on price data).
- **CCXT** for all exchange interactions.
- Explicit path handling using `os.path` or `pathlib` relative to the project root.

## 2. WHAT TO AVOID
- **No SQLite or Databases**: The project relies on transparent, human-readable JSON (`challenge_state.json`) and CSV files. Do not implement complex databases.
- **No Async/Await overrides**: Unless explicitly requested, stick to the synchronous loops currently implemented in the main runner scripts.
- **No Magic Numbers**: All trading parameters (SL %, max trades, tier thresholds) must be pulled from `core/config.py`.

## 3. LIBRARIES & DEPENDENCIES
- Allowed: `ccxt`, `pandas`, `numpy`, built-in python libraries (`json`, `time`, `logging`, `os`).
- UI dependencies: None (Use vanilla HTML/JS for dashboards).

## 4. ERROR HANDLING
- Graceful degradation on API rate limits: Ensure CCXT `RateLimitExceeded` exceptions trigger sleep/backoff routines rather than crashing the engine.
- Invalid Data: If an exchange returns incomplete candles, skip the cycle. Do not process corrupted data.

## 5. BOUNDARIES OF AI (Crucial)
- **DO NOT** modify `core/config.py` unless explicitly instructed by the user. This file is the central nervous system.
- **DO NOT** change the logic of `core/lux_fvg.py` unless the user asks for a math/logic change regarding SMC indicators.
- **NEVER** edit `exports/challenge_state.json` directly through code without ensuring data integrity (e.g., locking/safe writes), as this holds the live challenge progress.

## 6. GENERAL RULES
- **Code Style:** PEP 8, but prioritize readability over extreme strictness.
- **Comments:** Explain *why* a trading logic block exists, not just *what* it does. Reference SMC concepts (e.g., "Check if body closed below wick to confirm MSB").
- **State Integrity:** When restarting a script, it must seamlessly resume from where `challenge_state.json` left off.
