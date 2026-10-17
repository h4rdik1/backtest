# 📓 Backtest Sprint Journal: BTC/USD (November - December)
**Tag:** `#backtest`  
**Account:** $5,000 Funded Account ($300 Max Drawdown Limit, $20 Fixed 1R Risk)  
**Sprint Period:** Nov 27 – Dec 11 Market Replay  
**Auditor:** Elite SMC Performance Coach & Strict Trade Auditor  

---

### 📊 Sprint Performance Summary

| Metric | Value |
| :--- | :--- |
| **Total Setups Taken** | 7 Closed |
| **Wins / Losses** | 3 Wins / 4 Losses (42.8% Win Rate) |
| **Gross Profit** | +9.78R (+$195.60) |
| **Gross Loss** | -4.00R (-$80.00) |
| **Net Performance** | **+5.78R (+$115.60 Net Profit)** |
| **Starting Balance** | $5,000.00 |
| **Final Account Balance** | **$5,115.60** (+2.31% Gain) |
| **Max Drawdown Experienced** | -$130.00 / $300.00 (43.3% buffer used; floor safe) |
| **Sprint Status** | **PASSED WITH HIGH ASYMMETRY** |

---

## 📉 Trade Log #1: BTC/USD Short (Nov 27 – Dec 1)
* **Tag:** `#backtest`
* **Strategy:** **4H OB Retest Trend Continuation (Re-Entry Playbook)**
* **Asset / Direction:** BTC/USD Short
* **Setup Tier:** **A-Tier**
* **Context (4H):** Master Bearish displacement leg from $93.1k to $81.0k. Price retraced into Premium (>0.5) tapping the 4H Bearish Order Block ($91.2k–$92.8k).
* **Execution (15M):** 15M sweep at $92,100 followed by structure break down to $91,200.
* **Trade Parameters:**
  * **Entry Price:** $91,679.02
  * **Stop Loss:** $93,146.55 (Anchored to Master 4H Range High)
  * **Take Profit:** $87,353.00 (First 4H Demand Roadblock)
  * **Risk:** $1,467.53 (1.60% / 1R = $20.00)
  * **Reward:** $4,326.02
  * **R:R:** **2.95R**
* **Outcome:** **WIN (+2.95R / +$59.00)**
* **Key Lessons & Diagnostic Post-Mortem:**
  1. **HTF SL Anchoring Saved the Trade:** Placing the stop loss above the Master Range High ($93,146) protected the trade from the internal sweep to $92,750 on Nov 28 (Sweep Exhaustion).
  2. **Trend Continuation Rules:** No need to wait for a Range High sweep when trading trend continuation in Premium.
* **Charts:**  
  ![Trade 1 4H Overview](../../images/backtest_sprint/trade_1_4h_overview.png)  
  ![Trade 1 15M Execution](../../images/backtest_sprint/trade_1_15m_execution.png)

---

## 📉 Trade Log #2: BTC/USD Short (Dec 2)
* **Tag:** `#backtest`
* **Strategy:** **Premature Breakdown Short in Discount (Invalid Trap)**
* **Asset / Direction:** BTC/USD Short
* **Setup Tier:** **Invalid (Trap)**
* **Context (4H):** Aggressive displacement down to $83.5k. Price in a shallow pullback.
* **Execution (15M):** Entered on a minor 15M dip after hovering above the $87,226 line.
* **Trade Parameters:**
  * **Entry Price:** $86,779.59
  * **Stop Loss:** $87,480.27 (Tight LTF Stop)
  * **Take Profit:** $83,787.28
* **Outcome:** **LOSS (-1.00R / -$20.00)**
* **Key Lessons & Diagnostic Post-Mortem:**
  1. **SOP 4 Violation (Shorting in Discount):** The displacement leg was $92,750 to $83,500 with 0.5 EQ at $88,125. Entry at $86,779 was deep in Discount (<0.5).
  2. **Market Acceptance vs SFP:** Price spent 4 hours closing candle bodies above $87,226. That is acceptance, not rejection.
  3. **Stop-Loss Timeframe Trap:** A $700 stop inside a $3,500 4H FVG gets liquidated by normal pullback momentum.
* **Chart:**  
  ![Trade 2 15M Stopout](../../images/backtest_sprint/trade_2_15m_stopout.png)

---

## 📉 Trade Log #3: BTC/USD Short (Dec 2)
* **Tag:** `#backtest`
* **Strategy:** **5-Minute Micro-Stare Scalp (Invalid Revenge Trade)**
* **Asset / Direction:** BTC/USD Short
* **Setup Tier:** **Invalid (Trap)**
* **Context (4H/1H):** 1H chart printing consecutive Higher Highs & Higher Lows.
* **Execution (5M):** Zoomed into the 5-Minute chart, saw a red tick, and entered immediately after Trade #2 loss.
* **Trade Parameters:**
  * **Entry Price:** $87,250.21
  * **Stop Loss:** $87,639.69 (Ultra-tight $389 stop)
  * **Take Profit:** $86,207.11
* **Outcome:** **LOSS (-1.00R / -$20.00)**
* **Key Lessons & Diagnostic Post-Mortem:**
  1. **5M Micro-Staring Trap:** Zooming into the 5M chart created phantom setups. 
  2. **Zero 1H/15M MSS:** Not a single 1H candle closed below a swing low.
  3. **Emotional Re-Entry:** Re-entering the exact same invalidated zone without a new HTF catalyst.
* **Chart:**  
  ![Trade 3 5M Stopout](../../images/backtest_sprint/trade_3_5m_stopout.png)

---

## 📉 Trade Log #4: BTC/USD Short (Dec 3)
* **Tag:** `#backtest`
* **Strategy:** **Internal 4H OB Sweep Reversal (Sweep Exhaustion Trap)**
* **Asset / Direction:** BTC/USD Short
* **Setup Tier:** **A-Tier (Setup Valid, SL Execution Flawed)**
* **Context (4H):** Price reached the 4H Bearish Order Block ($91.2k–$92.5k) in Master Premium.
* **Execution (15M):** 15M SFP of $92,100 to $92,413.43 followed by 15M MSS body close below $91,550.
* **Trade Parameters:**
  * **Entry Price:** $91,392.70
  * **Stop Loss:** $92,413.43 (Naked SL at local wick tip)
  * **Take Profit:** $86,800.00
* **Outcome:** **LOSS (-1.00R / -$20.00)**
* **Key Lessons & Diagnostic Post-Mortem:**
  1. **Sweep Exhaustion (Playbook Part 2):** Price spiked to **$92,753.96** to sweep the Nov 28 high ($92,750) by $3.96 before reversing.
  2. **The Magnet Above:** The Nov 28 swing high was unmitigated and sitting $300 above the stop. The algorithm sought that external liquidity before dropping.
  3. **Buffer Rule:** Placing naked stops at exact wick tips guarantees stop-outs from liquidity sweeps.
* **Charts:**  
  ![Trade 4 15M Entry](../../images/backtest_sprint/trade_4_15m_entry.png)  
  ![Trade 4 15M Stopout](../../images/backtest_sprint/trade_4_15m_stopout.png)

---

## 📉 Trade Log #5: BTC/USD Short (Dec 3)
* **Tag:** `#backtest`
* **Strategy:** **Breakout Fade against 4H BOS (Invalid Expansion Fight)**
* **Asset / Direction:** BTC/USD Short
* **Setup Tier:** **Invalid (Trap)**
* **Context (4H):** Price expanding through $93.1k Range High.
* **Execution (15M):** Shorted at $92,309 on a pullback while 15M was printing higher lows.
* **Trade Parameters:**
  * **Entry Price:** $92,309.91
  * **Stop Loss:** $94,068.38
* **Outcome:** **LOSS (-1.00R / -$20.00)**
* **Key Lessons & Diagnostic Post-Mortem:**
  1. **Fading a HTF BOS:** 4H candle body closed above the old Range High ($93,146). The range was temporarily invalidated into expansion.
  2. **Max 2 Attempts Rule:** Taking a 3rd short in a zone that already failed twice violates risk rules.
* **Chart:**  
  ![Trade 5 15M Stopout](../../images/backtest_sprint/trade_5_15m_stopout.png)

---

## 📉 Trade Log #6: BTC/USD Short (Dec 4 – Dec 7)
* **Tag:** `#backtest`
* **Strategy:** **Macro Range High 4H SFP + 15M MSS (External-to-Internal Rotation)**
* **Setup Tier:** **S-Tier**
* **Context (4H):** Price swept the Master Range High ($93,146) with a massive deviation to **$94,306.38**, then the 4H candle failed and closed back inside the range.
* **Execution (15M):** 15M displacement broke through $93,250 swing low. Lower high formed at $93.5k.
* **Trade Parameters:**
  * **Entry Price:** $93,004.36
  * **Stop Loss:** $94,306.38 (Absolute Macro Sweep Wick)
  * **Take Profit:** $88,322.65 (4H Gap Fill / 0.5 Equilibrium)
  * **Risk:** $1,302.02 (1.40%)
  * **Reward:** $4,681.71
  * **R:R:** **3.60R**
* **Outcome:** **WIN (+3.60R / +$72.00)**
* **Key Lessons & Diagnostic Post-Mortem:**
  1. **Patience for Macro Deviations:** Waiting for the true liquidity purge above $94k provided complete directional conviction with zero drawdown.
  2. **External to Internal Rotation:** Cleared buy-stops above the range, then rode the full delivery to the internal imbalance and 50% equilibrium.
* **Charts:**  
  ![Trade 6 15M Entry](../../images/backtest_sprint/trade_6_15m_entry.png)  
  ![Trade 6 4H TP Hit](../../images/backtest_sprint/trade_6_4h_tphit.png)

---

## 📉 Trade Log #7: BTC/USD Short (Dec 9 – Dec 11)
* **Tag:** `#backtest`
* **Strategy:** **Updated Range High 4H SFP + 15M FVG Retest (Double Sweep S-Tier)**
* **Setup Tier:** **S-Tier**
* **Context (4H):** Price tested the updated Range High ($94,306), wicked to **$94,588.80**, and closed back below (4H SFP).
* **Execution (15M):** Clean 15M MSS body close below $93,700. Limit entry placed inside 15M FVG at $93,542.51.
* **Trade Parameters:**
  * **Entry Price:** $93,542.51 (Limit filled on retest spike to $94,450)
  * **Stop Loss:** $94,624.83 (Wick + 0.2% Structural Buffer)
  * **Take Profit:** $90,048.35 (First Roadblock Hit)
  * **Risk:** $1,082.32 (1.16% / 1R = $20.00)
  * **Reward:** $3,494.16
  * **R:R:** **3.23R**
* **Outcome:** **WIN (+3.23R / +$64.60)**
* **Key Lessons & Diagnostic Post-Mortem:**
  1. **SL Buffer Saved the Trade:** The retest wicked to ~$94,450. A naked stop would have died; the 0.2% buffer held effortlessly and allowed the 3.23R drop to pay out in full.
* **Charts:**  
  ![Trade 7 15M Setup](../../images/backtest_sprint/trade_7_15m_setup.png)  
  ![Trade 7 15M TP Hit](../../images/backtest_sprint/trade_7_15m_tphit.png)
