# Strategy: The Liquidity Transfer Model (4H Context ➔ 15M Execution)

> **Folder:** `models/liquidity_transfer/`  
> **Type:** Institutional Smart Money Concepts (SMC) Execution Model  
> **Timeframe Hierarchy:** 4H Macro Bias (Context) ➔ 15M Structural Shift & Retest (Execution)  
> **Risk Engine:** Fixed \$20.00 USD per trade (1.0R) on \$5,000 account (\$4,700 floor)  

---

## 1. Executive Performance Scorecard

The strategy was evaluated across 2,190 4-Hour bars and 35,040 15-Minute bars of continuous historical market data on BTC/USDT.

### 1.1 Capital & Profit Metrics

| Metric | Score / Value | Status / Evaluation |
|---|---|---|
| **Starting Capital** | **\$5,000.00** | Initial balance |
| **Final Balance** | **\$5,701.45** | **+\$701.45 (+14.03% Account Growth)** |
| **Net Profit in R-Multiples** | **+35.07R** | **+\$701.47 USD** at \$20/R |
| **Profit Factor** | **1.42** | Robust institutional profitability |
| **Max Drawdown (USD)** | **\$229.99** | Well within the \$300.00 challenge drawdown floor |
| **Max Drawdown (%)** | **4.03%** | Low volatility drawdown profile |
| **Drawdown Floor Check** | **PASSED** | Floor (\$4,700.00) never breached |

### 1.2 Win Rate & Trade Distribution

| Metric | Score / Value | Interpretation |
|---|---|---|
| **Total Closed Trades** | **156** | High statistical sample size |
| **Outcome Split** | **45 TP \| 84 SL \| 27 BE** | 27 trades protected by Breakeven engine |
| **Decisive Win Rate (Excl. BE)** | **34.88%** | Ratio of TP to (TP + SL) |
| **Gross Win Rate (Incl. BE)** | **28.85%** | Raw percentage of trades hitting TP |
| **Average Realized R:R** | **2.65R** | Wins produce \$53.00+ vs \$20.00 standard loss |
| **Long Trade Win Rate** | **29.76% (25W / 84T)** | Longs in discount environments |
| **Short Trade Win Rate** | **27.78% (20W / 72T)** | Shorts in premium environments |
| **Max Consecutive Wins / Losses** | **4 Wins \| 11 Losses** | \$20 fixed sizing absorbs 11 SLs easily (-\$220) |

### 1.3 Execution Funnel Efficiency

```text
34,990 Total 15M Bars Scanned
  └── 10,047 Bars in Active 4H Bias (28.7% Market Exposure)
        └── 356 15M Full Body-Close MSS Confirmed (High Filter Strictness)
              └── 157 Retest Touch Entries Filled (44.1% Conversion Rate)
                    └── 156 Closed Trades (45 TP, 84 SL, 27 BE)
```

---

## 2. Quick-Start Commands (How to Run Everything)

All scripts can be run directly from the repository root:

### A. Run Strategy Backtest
```powershell
# Standard BTC/USDT Backtest
python models/liquidity_transfer/backtest_liquidity_transfer.py

# Portfolio Backtest across BTC/USDT, ETH/USDT, SOL/USDT
python models/liquidity_transfer/backtest_liquidity_transfer.py --symbol ALL

# Test alternative boundary entry mode with 2.5R target
python models/liquidity_transfer/backtest_liquidity_transfer.py --entry-mode boundary --rr 2.5
```

### B. Run Strategy Live Scanner
```powershell
# Instant one-time live scan across Binance spot
python models/liquidity_transfer/live_scanner_liquidity_transfer.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --once

# Continuous 24/7 daemon aligned to 15M candle closes
python models/liquidity_transfer/live_scanner_liquidity_transfer.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --poll

# Scanner with Discord/Slack webhook alert integration
python models/liquidity_transfer/live_scanner_liquidity_transfer.py --webhook YOUR_WEBHOOK_URL --poll
```

### C. Run Strategy Test Suite
```powershell
# Run the 7 isolated strategy verification tests (0.10s execution)
python -m unittest models/liquidity_transfer/test_liquidity_transfer.py
```

---

## 3. How to Fix & Tune Everything Accordingly

To adjust strategy aggression, win rate, or risk, modify the parameters either via CLI flags or directly in [`backtest_liquidity_transfer.py`](file:///e:/backtest/models/liquidity_transfer/backtest_liquidity_transfer.py) and [`live_scanner_liquidity_transfer.py`](file:///e:/backtest/models/liquidity_transfer/live_scanner_liquidity_transfer.py).

### Parameter Tuning Matrix

| Parameter | Location | Default | What Happens if Increased | What Happens if Decreased | Recommended Values |
|---|---|---|---|---|---|
| **`--entry-mode`** | `backtest_liquidity_transfer.py:1007`<br>`live_scanner_liquidity_transfer.py:270` | `midpoint` (50% FVG C.E.) | `boundary`: Fills more trades (~30% more entries), but widens SL distance and decreases average R:R. | `midpoint`: Tighter SL, higher R:R (avg 2.65R), but misses setups that only lightly touch FVG edge. | Use `midpoint` for prop firms; `boundary` for higher frequency. |
| **`--be-trigger`** | `backtest_liquidity_transfer.py:1001`<br>`backtest_liquidity_transfer.py:40` | `1.5` ($1.5\text{R}$) | Moves SL to breakeven later; allows trades more room to breathe, reducing premature BE scratch-outs. | Moves SL to BE earlier (e.g. $1.0\text{R}$); protects capital faster, but increases scratch rate. | `1.5` to `1.8` |
| **`--min-rr`** | `backtest_liquidity_transfer.py:989`<br>`live_scanner_liquidity_transfer.py:73` | `1.5` ($1.5\text{R}$) | Requires wider opposing roadblocks for targets; increases average profit per win, but discards closer setups. | Allows closer roadblock targets (e.g. $1.2\text{R}$); increases win rate %, but lowers profit factor. | `1.5` to `2.0` |
| **`--target-mode`** | `backtest_liquidity_transfer.py:982` | `next_swing` | `fixed_rr`: Ignores structure, targets static multiple (e.g. 2R/3R).<br>`htf_roadblock`: Targets full 4H range high/low. | `next_swing`: Dynamically targets the immediate opposing 15M liquidity block ($\ge 1.5\text{R}$). | `next_swing` is safest. |
| **`RISK_PER_TRADE_USD`** | `backtest_liquidity_transfer.py:38`<br>`live_scanner_liquidity_transfer.py:72` | `20.00` (\$20/trade) | Increases USD dollar PnL and drawdown proportionally. Contract sizing adjusts automatically. | Decreases dollar risk; safe for conservative accounts. | \$20.00 (0.4% risk on \$5k account). |
| **`BIAS_EXPIRY_HOURS`** | `backtest_liquidity_transfer.py:41`<br>`backtest_liquidity_transfer.py:128` | `24` (6 $\times$ 4H bars) | Allows 4H SFP bias to remain valid longer (e.g. 36h); triggers more trades, but risks trading into stale liquidity. | Stricter freshness (e.g. 16h); only trades immediate sweeps. | `24` hours is optimal. |

---

## 4. File Structure of This Strategy Folder

```text
models/liquidity_transfer/
├── README.md                            <-- Executive scorecard, percentages, tuning guide, and CLI instructions
├── decisions.md                         <-- GRAVITAS decision record, ADRs, invariant checks, and architecture proof
├── backtest_liquidity_transfer.py       <-- Pure backtest simulation engine (loads cached OHLCV, produces metrics & CSV)
├── live_scanner_liquidity_transfer.py   <-- Real-time CCXT market scanner & signal generator (supports --once and --poll)
├── test_liquidity_transfer.py           <-- 7 unit & integration tests validating state transitions, wick filters & exports
└── __init__.py                          <-- Package exports for clean imports across the codebase
```

---

## 5. Architectural Integrity & Lookahead Safety

1. **Strict 15M Body-Close Filter:** Wicks piercing swing highs/lows are explicitly rejected as fakeouts. A valid Market Structure Shift requires the 15-minute candle body (`close > open` for longs, `close < open` for shorts) to close beyond the pivot.
2. **Displacement FVG Requirement:** A market shift without a 3-candle Fair Value Gap is classified as `DISCARDED_NO_FVG` and never traded.
3. **Anti-Lookahead Separation:** In [`live_scanner_liquidity_transfer.py`](file:///e:/backtest/models/liquidity_transfer/live_scanner_liquidity_transfer.py), completed closed candles (`df.iloc[:-1]`) are separated from the live forming candle (`df.iloc[-1]`), guaranteeing signals never repaint.
