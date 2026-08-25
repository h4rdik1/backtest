# 📓 Manual Trade Logs

This file records manual trades taken outside of the autonomous engine, tracking setup quality, execution, and key structural lessons.

---

## 📈 Trade Log: TAO/USDT Long (August 8, 2026)

*   **Asset:** TAO/USDT Perpetual
*   **Direction:** Long
*   **Setup Tier:** A-Tier (HTF POI Tap + LTF Sweep + LTF MSB)
*   **Context (4H):** Price tapped the 4H demand zone (green order block) between $189.50 - $192.30.
*   **Execution (15M):** 
    *   LTF Sweep of local lows down to $190.50.
    *   LTF Bullish MSB (BOS of local high).
*   **Trade Parameters:**
    *   **Entry Price:** $192.12
    *   **Stop Loss:** $190.14 (Risk: $1.98)
    *   **First Target (2R):** $196.08 (Hit - Closed 50%)
    *   **Second Target (Range High):** $200.02 (Closed at $198.50 - 3.22R)
    *   **Net Outcome:** Successful Win (Average P&L: **~2.6R**)

### 💡 Key Lessons & Post-Mortem
1.  **Timeframe Mismatch Caution (The Stop-Loss Trap):** The initial entry was placed at $192.03 with a stop at $190.05. This stop was placed inside the range boundary, making it highly vulnerable to being run. The lower entry at $192.12 with SL at $190.14 survived the local volatility because it was anchored properly.
2.  **Mitigation / Stale OB Risk:** The trade was taken on the 6th mitigation of the 4H order block. While it succeeded, trading a stale block is generally lower probability.
3.  **Opposing Supply Target:** Because the higher timeframe trend was bearish (Strong Bear on 4H) and had already printed a Bearish MSB at $195.00, this long was counter-trend. Opposing supply was hit at $194.50 - $195.00, causing a heavy stall before the final squeeze to $198.50. Taking partial profits at 2.0R was the mathematically correct decision.

---

## 📉 Trade Log: BTC/USDT Long (August 11, 2026)

*   **Asset:** BTC/USDT Perpetual
*   **Direction:** Long (Re-Entry)
*   **Setup Tier:** B-Tier (Counter-Trend, Premium Entry)
*   **Context (4H):** Macro trend flipped to **STRONG BEAR**. Price was in the Master Premium zone.
*   **Execution (15M):** 
    *   Missed the initial 2R bounce from the $63,800 SFP.
    *   Entered on the pullback expecting a continuation back to the Range High.
*   **Trade Parameters:**
    *   **Entry Price:** $64,045.20
    *   **Stop Loss:** $63,803.80 (Risk: $241.40)
    *   **Target:** $64,861.40
    *   **Outcome:** Stopped Out (Loss: **-1.0R**)

### 💡 Key Lessons & Post-Mortem
1.  **Premium Zone Long Violation (SOP 4):** The local range was $62,392.20 – $65,415.40, with a 0.5 Equilibrium at $63,903.80. The entry at $64,045.20 was in the **Premium Zone** of the local range. Longs in premium are strictly forbidden because you are buying right before the market seeks discount.
2.  **Trend Incoherence (Strong Bear Context):** The 4H macro trend had flipped bearish. In a strong bear market, upward bounces are weak, and local support zones are easily broken (BOS) to continue the downward expansion. 
3.  **FOMO Chasing Trap:** Entering a pullback after missing a clean 2R setup often leads to forced entries. If you miss the primary sweep entry, let the trade go. Never chase a re-entry in premium.

---

## 📈 Trade Log: TAO/USDT Swing Long (August 12, 2026)

*   **Asset:** TAO/USDT Perpetual
*   **Direction:** Long (Swing)
*   **Setup Tier:** A-Tier (Range Breakout + 1H SFP + 1H Bullish MSB)
*   **Context (4H):** Breakout of the old daily range ($187.52 – $199.92). Price is now establishing a new range expansion phase.
*   **Execution (5M):** 
    *   Price broke above the 1H Range High ($199.92).
    *   Pulled back and printed a 5M Bullish SFP of local lows (swept to $196.50 and closed back inside).
    *   Printed a 5M candle body close confirming a Bullish CHoCH above $199.50.
*   **Trade Parameters:**
    *   **Entry Price:** $200.68 (Half position entered at market)
    *   **Stop Loss:** $195.53 (Risk: $5.15)
    *   **Target:** $219.20
    *   **Outcome:** Active (Pending)

### 💡 Execution Strategy & Management
1.  **Half Position Risk Management:** Entering a half position (5 TAO, ~$1,003 with 2x leverage) is a very smart way to manage swing risk, resulting in a maximum risk of only **$25.75 (~0.53% of challenge balance)**.
2.  **Adding to Position:** The other half of the position will be added if price pulls back to retest the 5M/1H demand block / order block at **$198.50**, keeping the average entry tight.
3.  **Target R:R:** The target of $219.20 represents a **3.59R swing trade**, which is fully compliant with the 2.0R minimum requirement.

---

## 📉 Trade Log: BTC/USDT Short (August 12, 2026)

*   **Asset:** BTC/USDT Perpetual
*   **Direction:** Short (Intraday)
*   **Setup Tier:** A-Tier (BOS + 1H FVG Pullback Retest + 5M Bearish reversal)
*   **Context (4H):** Macro trend is **STRONG BEAR** after the break of the local $63,800 low.
*   **Execution (5M):** 
    *   Price pulled back to mitigate the 1H FVG at $63,800 – $64,000.
    *   5M structure printed a bearish reversal (lower low and lower high).
    *   Entered short at market.
*   **Trade Parameters:**
    *   **Entry Price:** $63,666.90
    *   **Stop Loss:** $63,900.00 (Risk: $233.10)
    *   **Target:** $62,800.00
    *   **Outcome:** Stopped Out (Loss: **-1.0R** / -$25.00)

### 💡 Execution Strategy & Management
1.  **Strict Risk Control:** Entering with a 3x leverage size (~$3,820 value) anchors your total risk to exactly **$25.00 (~0.52% of challenge balance)**. This is extremely safe and protects your drawdown cushion.
2.  **R:R Expectancy:** The target at $62,800.00 yields a **3.72R trade**, which has positive mathematical expectancy and keeps you fully compliant with the 2.0R rule.




