# Trade Post-Mortem Analysis: April 10-13

This report analyzes the trades that went wrong during the current challenge and identifies systemic issues in the execution logic.

## 📊 Performance Overview
The challenge recently suffered a string of losses, dropping the balance from a peak of approx. **$5,250** back down to **$5,000**.

### Recent Failed Trades
| Trade ID | Symbol | Direction | Result | Regime | Bias Reason | Issue Identified |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T0012** | BTC/USDT | LONG | LOSS | trending_up | Bull Sweep | **Fatal SL Logic** (0.01% SL) |
| **T0011** | BNB/USDT | SHORT | LOSS | trending_down | Bear FVG | Tight SL (0.2%) |
| **T0010** | SOL/USDT | LONG | LOSS | ranging | Bull Sweep | Indecision in Ranging market |
| **T0008** | BNB/USDT | SHORT | LOSS | trending_down | Bear FVG | Tight SL / Noise |
| **T0002** | ETH/USDT | SHORT | LOSS | trending_up | Bear Sweep | **Counter-Trend** |

---

## 🔍 Critical Flaws Identified

### 1. The "Slippage-into-Stop" Execution Bug (Fatal)
In `challenge_trader.py`, the `check_pending_trades` function has a major logic flaw that explains the "instant losses" like BTC T0012.

**The Workflow:**
1. A signal is found. A Limit Entry is set at the FVG midpoint (e.g., $71,034) and a "Protected Swing" SL is set (e.g., $70,885).
2. Price drops fast. The 5m candle closes at **$70,895**.
3. The engine sees `low_price <= entry_limit` and triggers the trade.
4. **The Bug:** It sets the entry price to the *current close* ($70,895) but keeps the *original SL* ($70,885).
5. **The Result:** You enter a BTC trade with a **$10 stop loss** (0.014%). This is effectively a market order at the edge of your invalidation zone. It is guaranteed to fail due to spread and noise.

### 2. Regime Mismatch (Counter-Trend)
Trade **T0002** (ETH SHORT) was taken in a `trending_up` regime. 
- While the MTF Bias was "Bearish" due to a sweep, the HTF Regime was still clearly "Trending Up".
- Institutional SMC rules suggest avoiding counter-trend shorts unless a clear HTF SFP *and* MSB have occurred. The current scanner is perhaps too eager to react to a single MTF wick.

### 3. "Tight" SL Fallbacks
When a "Protected Swing" isn't found within 20 bars, the engine falls back to `0.1%` beyond the FVG boundary.
- For high-volatility assets like SOL or during news events, 0.1% is too tight.
- Price often "over-wick" the FVG to tap the Order Block or grab liquidity just below it before reversing. This is the "Engineered Retrace" mentioned in your S-Tier rules.

### 4. B-Tier Fatigue
The `MIN_SETUP_TIER` is currently set to **"B"**.
- Analysis shows B-Tier setups (simple FVGs without HTF OB confluence) are being chopped up, especially in `ranging` regimes.
- Wins are mostly coming from A/A++ setups where multiple timeframes align.

---

## 🛠️ Proposed Fixes

### 1. Execution Refinement
- **Fix Pending Fill:** Use the `limit_price` as the entry, or if `cur_price` is better, recalibrate the SL to maintain a minimum percentage/structure distance.
- **Minimum Risk %:** Enforce a hard-coded minimum SL distance (e.g., 0.3% for BTC, 0.5% for Alts) to prevent noise-outs.

### 2. Tier Filtering
- Set `MIN_SETUP_TIER` back to **"A"**.
- Force **HTF Confluence** for all alt-coin trades.

### 3. Regime Filter
- Block trades that are directly counter to the `trending_up` or `trending_down` labels in `analysis.py`. 
- Only allow "Counter-Bias" trades in a `ranging` regime.

---

## ⏭️ Next Steps
1. **Fix the SL Logic bug** in `challenge_trader.py`.
2. **Review T0012 chart manually** to see if it was a deep liquidity grab or a trend reversal.
3. **Upgrade to S-Tier rules** where an LTF MSB (Market Structure Break) is *mandatory* for entry, not just an FVG touch.

> [!IMPORTANT]
> The BTC T0012 loss of -$50 was effectively a "garbage" trade caused by code logic, not a failed market thesis. Fixing this execution bug alone will significantly improve the win rate.
