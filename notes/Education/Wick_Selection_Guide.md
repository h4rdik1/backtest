# 🎯 Wick Selection & Range Boundary Blueprint

This guide defines the exact rules for selecting range highs and lows when there are multiple wicks nearby, helping you avoid drawing arbitrary lines that get run over by stop hunts.

---

## Rule 1: The Principle of the Absolute Extreme
*   **The Rule:** A range boundary must represent the **absolute highest wick** and **absolute lowest wick** of the displacement leg.
*   **Why:** If you draw a Range High line at a lower swing high (e.g., $65,391) while a higher swing high wick sits just above it (e.g., $65,746):
    *   The real pool of buy stops is resting above the absolute highest point ($65,746).
    *   If you short the sweep of the lower level, you will get stopped out when price wicks up to run the stops at the absolute highest point.
*   **The Decision:** Always choose the **larger, outer range** that covers the absolute extremes over a tighter, internal range.

---

## Rule 2: How to Handle Wick Clustering (Multiple Wicks Nearby)

When you see a cluster of multiple wicks near a range extreme, do not draw multiple horizontal lines. Instead, use the **Wick Zone** method:

```
[Absolute Highest Wick Tip]  <-- Range High Line (Stop Loss Invalidation)
      ▲
      │  (The Wick Zone / Liquidity Pool)
      ▼
[Highest Candle Body Close]  <-- Start of Supply Zone (Alert Trigger)
```

### The 3-Step Execution:
1.  **Mark the Range High (1):** Place your Range High line at the **absolute tip** of the highest wick in the cluster. This is your structural invalidation level (if a body closes above this, it is a BOS).
2.  **Draw the Wick Zone:** Draw a shaded box from the **highest candle body close** to the **highest wick tip**. This shaded box is your **Liquidity Pool / Supply Zone**.
3.  **Set the Alert:** Set your alert at the bottom of the Wick Zone (the highest body close). 
    *   When the alert triggers, look for your entry trigger (15M SFP + CHoCH) inside this zone.
    *   Place your stop loss safely above the top of the Wick Zone (the absolute highest wick tip).

---

## Rule 3: The 4H/12H Timeframe Alignment Check
If you are confused between two overlapping ranges:
*   Always check the **HTF Context (12H/1D)**.
*   If the 12H chart shows a clear swing high/low that aligns with the larger range, use the larger range.
*   A larger range is always more robust and less vulnerable to stop hunts than a tight internal range.
