# Testing Strategy & Quality Checks

## 1. PRE-TRADE VERIFICATION (Manual Testing)
Before trusting the engine with new logic, use the diagnostic scripts in the `tools/` directory:
- **`analyze_setup.py`**: Run `python tools/analyze_setup.py BTC/USDT` to get a deep, multi-timeframe text breakdown of a specific symbol. Use this to verify that the bot is reading HTF, MTF, and LTF alignment correctly.
- **`dry_run_btc_sl.py`**: Use this script to test changes to the Stop-Loss logic (e.g., structural placement, prominent wick detection) in isolation without risking fake capital in the challenge runner.

## 2. COMPONENT TESTING
- **SMC Math (`test_sfp.py`)**: Run the unit tests in `scratch/test_sfp.py` whenever modifying `core/lux_fvg.py` to ensure Swing Failure Pattern (SFP) detection remains mathematically accurate.
- **Data Integrity (`market_check.py`)**: Use `python scratch/market_check.py` to quickly fetch and print the raw structures and price levels to ensure CCXT is returning valid data.

## 3. POST-MORTEM & BACKTESTING
- **`analyse_challenges.py`**: After running a series of paper trades or a full challenge, run this to parse the CSV logs and ensure Win Rate and Profit Factor calculations match expectations.
- **`build_master_analysis.py`**: Generates `MASTER_ANALYSIS.md`. Review this file after changes to ensure strategy metrics aren't degrading.

## 4. AI QUALITY CHECKS
When writing new code:
1. Ensure new imports are added to `requirements.txt` if necessary.
2. Verify that new logic does not break the JSON schema of `challenge_state.json`.
3. Do not introduce infinite loops without a `time.sleep()` to respect exchange rate limits.
