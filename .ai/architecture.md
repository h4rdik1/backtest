# Architecture

## 1. ARCHITECTURE
The system operates on a modular, event-driven loop primarily built around two main executors:
- **The Live Scanner (`live_scanner.py`)**: Continuously polls exchange APIs, caches OHLCV data, runs SMC logic, and outputs active setups.
- **The Challenge Trader (`challenge_trader.py`)**: Consumes the setups, validates them against risk constraints (balance, daily limits), and manages open paper trades through a state JSON file.

### Data Flow
`Exchange API` ➔ `CCXT` ➔ `Data Cache (data/ohlcv/)` ➔ `Math Engine (core/lux_fvg.py)` ➔ `Setup Logic (models/backtest_cisd.py)` ➔ `Execution/State Tracking (exports/challenge_state.json)`.

## 2. FOLDER & FILE STRUCTURE
- `core/`: Core business logic. `config.py` is the single source of truth for strategy parameters. `lux_fvg.py` contains the heavy math. `analysis.py` handles stats.
- `models/`: Trading models defining exact entry, SL, and TP rules based on core signals.
- `tools/`: Diagnostic scripts, manual backtesters, and HTML visualizers.
- `exports/`: Runtime state and analytics. Crucially houses `challenge_state.json` and `MASTER_ANALYSIS.md`.
- `data/`: Local cache of OHLCV data to prevent rate-limiting.
- `archive/`: Historical challenge data used for backtesting and performance reviews.

## 3. TECH STACK
- **Language**: Python 3
- **Data Manipulation**: Pandas, NumPy
- **Exchange Integration**: CCXT
- **Visualization**: HTML/JS (local web server)
- **State Management**: JSON (local file-based)
