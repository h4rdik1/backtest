# 📚 SMC Core Concepts: Study & Practice Guide

Use this guide to study the three fundamental concepts required to master market structure, timeframe alignment, and candle closes.

---

## 1. Market Regimes (Order Flow Delivery)
You must learn how to distinguish between a **Trending Market (Expansion)** and a **Ranging Market (Rotation)**. 
*   **The Golden Rule:** 
    *   In a **Range**, the market respects wicks (sweeps/SFPs) and runs through body closes (order blocks get run).
    *   In a **Trend**, the market respects body closes (order blocks/FVGs hold) and runs through wicks (sweeps get run over).
*   **What to study:** 
    *   *Wyckoff Stage Analysis* (Accumulation ➔ Markup ➔ Distribution ➔ Markdown).
    *   *Premium vs. Discount Zones* (understanding that you only buy in discount and only short in premium relative to the macro displacement leg).

---

## 2. Timeframe Separation & Structural Anchoring
You need to study how to align your trade parameters (Stop Loss and Take Profit) with the timeframe of your trading thesis.
*   **The Golden Rule:** 
    *   If your **Target** is based on the **4H Range** (macro rotation), your **Stop Loss** *must* be anchored to the **4H structure**. 
    *   If you want to use a tight **15M Stop Loss**, your **Target** *must* be a **15M/1H support level** (short-term scalp). 
    *   You cannot mix a 15M risk model with a 4H target.
*   **What to study:**
    *   *Fractal Market Structure* (understanding how a 15M trend forms a single 4H pullback leg).
    *   *HTF Context ➔ MTF Bias ➔ LTF Entry* alignment frameworks.

---

## 3. Structural Acceptance vs. Rejection (BOS vs. SFP)
You need to master how to read candle closes to determine whether a key price level is being accepted or rejected by the market.
*   **The Golden Rule:** 
    *   **Candle bodies** represent value and acceptance. A body close outside a level is a **Break of Structure (BOS)**—the old range is dead, do not short it.
    *   **Candle wicks** represent liquidity grabs and rejection. A wick outside a level with a body close back inside is a **Swing Failure Pattern (SFP)**—the range is active, look for the reversal.
*   **What to study:**
    *   *Swing Failure Pattern (SFP) Mechanics* (specifically, how resting liquidity/stop orders are triggered and swept).
    *   *Failed Breakouts / Deviations* (learning how to identify and trade fakeouts only after the price returns and closes back inside the range).

---

## 4. MSS Confirmation: The Body Close Rule (NEW — Sep 10, 2026)
The exact pivot point for an MSS is the **swing low/high that directly created the most recent structural extreme** (the sweep wick).
*   **The Golden Rule:**
    *   A **wick** below/above that pivot level = **another Sweep** (liquidity grab). Do NOT enter yet.
    *   A **candle body close** beyond that pivot level = **True MSS confirmed**. This is your entry trigger.
*   **Why this matters:** If you enter on a wick, you are entering before the algorithm has proven it wants to reverse. Waiting for the body close filters out fake moves and aligns you with the real institutional shift.
*   **Practical Application:** Draw the red line at the swing low/high that created the high. Watch for a big displacement candle whose **body** closes clearly below/above it. That candle's FVG is your entry zone.

---

## 5. Nested Ranges & Internal Liquidity Layers (NEW — Sep 10, 2026)
Markets operate in layers. Inside every Macro Range, there are smaller local ranges and internal liquidity pools.

*   **When Price Sweeps the Macro Range High and Returns Inside:**
    *   The **New Range High** = the extreme wick of the breakout (the SFP sweep wick).
    *   The **Old Range High** = becomes **Internal Resistance / intermediate liquidity** (just a speed bump).
    *   The **Old Range Low** = remains the **Ultimate Target** (macro external liquidity).

*   **Internal Liquidity Pools** (the "speed bumps" between your entry and final target):
    *   Every local equal low, swing low, or prior range low is a pool of resting stop-loss orders.
    *   The algorithm will target each pool in sequence on its way to the Macro Range Low.
    *   **These are your TP cascade levels**, not your final target.

*   **The Playbook for Cascading TPs:**
    1.  **TP 1 (Internal):** Close 40-50% at the first major internal liquidity cluster.
    2.  **TP 2 (Internal):** Close another 30% at the next significant swing low.
    3.  **TP 3 / Runner (External):** Leave 20% to target the absolute Macro Range Low, front-running it by ~0.5% (never set TP exactly on the line — retail buy limits sit there and the market can reverse just before).

---

## 6. Full Trade Execution Playbook (NEW — Sep 10, 2026)
This is the complete, step-by-step process from idea to exit, applied to the HYPE short on Sep 9-10, 2026:

```
STEP 1 — HTF NARRATIVE (4H/Daily)
   → Identify the Macro Range (RH at wick extreme, RL at wick extreme).
   → Determine Premium (above 0.5 EQ) or Discount (below 0.5 EQ).
   → Is price in a valid POI for reversal? (Shorts in Premium only. Longs in Discount only.)

STEP 2 — MTF CONFIRMATION (1H)
   → Did price sweep the Range High / Range Low? (SFP = wick out, body close back inside)
   → Is there a 1H/4H Bearish MSS (body close below the swing low that created the sweep)?

STEP 3 — LTF ENTRY TRIGGER (5M/15M)
   → Find the exact pivot: the swing low/high that created the SFP wick.
   → Wait for a BODY CLOSE past that level (not a wick).
   → The displacement candle leaves an FVG. That FVG is your entry zone.

STEP 4 — STOP LOSS PLACEMENT
   → SL goes safely ABOVE the sweep wick tip + small buffer (0.10-0.15%).
   → NEVER place SL at the exact wick tip or at an equal high/low. The MM will run it.

STEP 5 — TRADE MANAGEMENT
   → Once the first internal liquidity pool is swept in your direction, move SL to Breakeven.
   → This creates a "free roll" — zero risk, unlimited upside.
   → Scale out at internal liquidity levels (TP1), not just at the final target.

STEP 6 — TAKE PROFIT
   → TP1: First major internal liquidity cluster.
   → TP2: Next significant swing low (local range low).
   → Runner: Slightly above/below the absolute Macro Range Low/High (front-run by ~0.5%).
```

---

## 7. The Sniper Entry Sequence: HTF to LTF (NEW — Sep 16, 2026)
Never execute a trade based on LTF (15M/5M) sweeps until the HTF objective is completed. The HTF dictates the *location*, and the LTF dictates the *timing*.

*   **Step 1: The HTF Anchor:** The 4H or 1H chart *must* tap your macro Order Block (OB) or sweep a macro liquidity line. If this hasn't happened, LTF action is just noise.
*   **Step 2: The LTF Trap:** Once the HTF sweeps, zoom into the 15M/5M. Price will often bounce slightly, tricking early retail traders into buying. This creates a weak LTF support line.
*   **Step 3: The LTF Sweep (The Spring):** Price drops one final time on the 15M/5M to sweep the stop-losses of those early retail buyers. This happens *during or just after* the HTF sweep.
*   **Step 4: The LTF Trigger (MSB/CHoCH):** After the final LTF sweep, institutional volume steps in and breaks the 15M/5M lower high. **Wait for a full candle body close** above this line. That is your strict entry trigger for a Return to Origin (RTO) limit order.

---

## 8. Liquidity Resistance & Sponsored Order Blocks (NEW — Sep 16, 2026)
This concept dictates exactly where your Stop Loss and Take Profit should go, and which Order Blocks are valid.

*   **HRL (High Resistance Liquidity) = Strong Pivots:**
    *   *What it is:* A swing high or low that swept liquidity (created an SFP) and then caused a major Market Structure Break (MSB).
    *   *How to use it:* Institutions defend these levels. Place your **Stop Loss** safely behind HRL points.
*   **LRL (Low Resistance Liquidity) = Weak Pivots:**
    *   *What it is:* Swing points that failed to sweep liquidity or failed to cause a new MSB. Examples include Equal Highs/Lows (EQH/EQL) and clean trendlines.
    *   *How to use it:* There is no institutional money defending these levels, making them magnetic. Use LRL as your **Take Profit / Draw on Liquidity** targets.
*   **Sponsored Order Blocks:**
    *   Not all Order Blocks are valid. A high-probability OB must form *immediately after* sweeping a major liquidity pool (stopping out retail) and must aggressively reverse to cause an MSB.
    *   *Logic:* Institutions use the liquidity from the sweep to fill massive orders. That resulting Order Block is their "home base," and they will violently defend it.

---

## 🛠️ Practical Backtesting Exercise
To build your intuition, perform this exercise weekly on your charts:
1.  Open any chart (e.g., BTC, ETH, SOL) on the 4H timeframe.
2.  Find a major displacement leg, mark the Range High and Range Low, and draw the 0.5 Equilibrium line.
3.  Look at every time the price touched the boundary: **Did it close outside as a body (BOS), or wick out and close inside (SFP)?** 
4.  Trace what happened next. You will quickly see how the market respects wicks in ranges and runs wicks in trends.
