# 🏛️ SMC Core Concepts & Centralized Strategy Guide
### The Master Manual for Market Structure, Bias, and Execution

> *"Patience is the edge. We do not chase price — we let the market manipulate into our zones, confirm intent with candle body closes, and enter on the retest."*

---

## 1. The Core Market Laws

Every trade must tell a coherent top-to-bottom institutional story. Markets are governed by three mechanical laws:

### Law 1: Order Flow Delivery (Expansion vs. Rotation)
* In a **Ranging Market (Rotation)**: The market respects **wicks** (sweeps/SFPs) and runs through body closes. Order blocks get violated; range boundaries hold.
* In a **Trending Market (Expansion)**: The market respects **body closes** (FVGs and Order Blocks hold) and violently runs through wicks (sweeps get overrun).
* **The Golden Rule:** Only buy in **Discount (< 0.5 EQ)** and only sell in **Premium (> 0.5 EQ)** relative to the active displacement leg.

### Law 2: Acceptance vs. Rejection (The Candle Close Law)
* **Candle Body Close = Acceptance (BOS / MSS):** If a candle body closes beyond a structural pivot, the level is broken. Trend continuation is confirmed.
* **Candle Wick Only = Rejection (Sweep / SFP):** If price pokes beyond a pivot but the body closes back inside, liquidity has been absorbed. A reversal is imminent.
* **The Body Close Rule:** Never call an MSS or structural shift on a wick. A wick beyond a pivot is just another liquidity sweep. Entry requires a full candle body close.

### Law 3: The TTrades Law ($ERL \rightleftarrows IRL$)
Markets strictly rotate between two types of liquidity:
* **External Range Liquidity (ERL):** Old Swing Highs, Old Swing Lows, Equal Highs, Equal Lows (Stop pools).
* **Internal Range Liquidity (IRL):** Fair Value Gaps (FVGs) and Order Blocks resting inside the range.
$$\mathbf{ERL \longrightarrow Targets \longrightarrow IRL}$$
$$\mathbf{IRL \longrightarrow Targets \longrightarrow ERL}$$
* When price sweeps an External Low $\longrightarrow$ Expect an expansion toward the nearest Internal FVG (Long Bias).
* When price fills an Internal FVG $\longrightarrow$ Expect a rejection seeking the External Lows (Short Bias).

### Law 4: Range Hierarchy & Subordination (Master vs. Internal)
* **Master HTF Range (4H/Daily):** Defines directional permission.
  * **Master Premium (> 0.5 EQ):** Only look for **Shorts**.
  * **Master Discount (< 0.5 EQ):** Only look for **Longs**.
* **Internal Range Subordination:** 
  * You **never** buy an internal discount low if it sits in the **Master Premium** zone.
  * You **never** short an internal premium high if it sits in the **Master Discount** zone.
  * When in doubt, trade only in the direction of the Master Range.

---

## 2. The 3-Step Daily Bias Engine (Zero Guesswork)

Determine directional bias every morning in under 2 minutes:

```
[Step 1: Last Liquidity Taken]
        ├── Swept External Low (Range Low / Prior Day Low)? ──► Bias is BULLISH (seeking Internal Highs).
        └── Swept External High (Range High / Prior Day High)? ──► Bias is BEARISH (seeking Internal Lows).
                        │
                        ▼
[Step 2: The Draw on Liquidity (DOL)]
        ├── Bullish Bias: Nearest unmitigated 4H Bearish FVG or Clean Buy-Side Liquidity.
        └── Bearish Bias: Nearest unmitigated 4H Bullish FVG or Clean Sell-Side Liquidity.
                        │
                        ▼
[Step 3: Daily Candle Confirmation (5:30 AM IST Close)]
        ├── Previous Daily closed with a rejection wick (SFP)? ──► Reversal bias today.
        └── Previous Daily closed with a full displacement body? ──► Continuation bias today.
```

---

## 3. The Centralized Strategy: "The Liquidity Transfer Model" (4H ➔ 15M)

This is the single core setup model for all manual execution. If a setup does not match this sequence, **no trade is taken**.

```
[Phase 1: 4H Structural Anchor]
        Price sweeps a key 4H/Daily High or Low (Liquidity Raid / SFP)
        OR taps into a pristine 4H FVG / Order Block in Discount/Premium.
                                │
                                ▼
[Phase 2: 15M Market Structure Shift (MSS)]
        Price aggressively breaks the opposing 15M structural pivot.
        MUST close with a FULL CANDLE BODY (Wicks do not count).
        Leaves behind a 15M Bullish/Bearish Fair Value Gap (FVG).
                                │
                                ▼
[Phase 3: The Retest Entry]
        Price retraces into the 15M FVG or Broken Structural Pivot.
        Enter on limit or touch. NEVER chase the breakout candle.
                                │
                                ▼
[Phase 4: Structural SL Placement (With Buffer)]
        SL placed beyond the sweep low/high with a 0.15% (or ATR) buffer.
        NEVER place SL directly on the obvious swing wick.
        Size position strictly: Contracts = $20.00 / (Entry - SL).
                                │
                                ▼
[Phase 5: Trade Management & Exits]
        Move SL to Break-Even at +1R / 15M Bearish Breaker ($0 risk).
        TP 1: 50% at +2.0R (nearest 15M liquidity pool).
        TP 2 (Runner): Opposing 4H unmitigated FVG / Swing High (3R - 5R).
```

---

## 4. Liquidity Architecture & Stop Loss Protection

### HRL vs. LRL (Where Stops and Targets Belong)
* **HRL (High Resistance Liquidity = Defended Pivots):**
  * Swing points that swept liquidity and caused a validated MSS.
  * Institutions defend these levels. **Place your Stop Loss safely behind HRL points.**
* **LRL (Low Resistance Liquidity = Magnetic Pools):**
  * Equal Highs/Lows (EQH/EQL), flat shelves of wicks, and clean trendlines.
  * No institutional volume defends these; they are stop-loss clusters. **Use LRL as your Take Profit / DOL targets.**

### The Stop-Loss Liquidity Buffer Law
* Obvious swing lows are the algorithm's targets (Sell-Side Liquidity).
* Placing your stop directly on a swing low means you *become the liquidity*.
* **Rule:** Always add a structural buffer:
  $$\text{SL (Long)} = \text{Swing Low} - (0.15\% \text{ to } 0.25\% \text{ buffer or } 0.5 \times \text{ATR})$$
* On BTC, this equals a **150–200 point buffer** below the key wick.

### The Wick Zone Method (Handling Multiple Clustered Wicks)
When a cluster of multiple wicks forms near a range boundary, do not draw arbitrary multiple lines:
1. **The Invalidation Line:** Place your Range High/Low line at the **absolute tip** of the highest/lowest wick in the cluster (the absolute extreme).
2. **The Wick Zone (Liquidity Pool):** Draw a shaded box from the **highest candle body close** to the **highest wick tip**. 
3. **Execution:** Set an alert at the edge of the Wick Zone (highest close). Only enter when 15M SFP + MSS confirms inside the zone.

### The Max 2 Attempts Rule (Sweep Exhaustion)
At major supply/demand zones, price often wicks 2 or 3 times to exhaust liquidity before reversing:
* If a valid SFP prints, you enter, and a higher wick stops you out:
* You are allowed **exactly ONE re-entry** if a new, higher SFP prints with a fresh 15M body-close MSS.
* **If stopped out twice in the same zone, STOP TRADING THAT ZONE.** Two failures indicate institutional trend expansion, not rotation.

---

## 5. The 5-Point Pre-Trade Checklist

Before submitting any order on the exchange, verify all 5 conditions. If any condition fails, **skip the trade**:

```
[ ] 1. HTF Anchor: Has a 4H/Daily sweep occurred, or is price tapping a 4H POI in Discount/Premium?
[ ] 2. 15M MSS Body Close: Did a 15M candle close its solid body beyond the structural pivot?
[ ] 3. Retest Entry: Am I buying a pullback into the 15M FVG/pivot, rather than chasing green candles?
[ ] 4. Protected SL: Is my stop anchored behind the structural sweep with a proper liquidity buffer?
[ ] 5. Fixed $20 Risk Math: Is position size calculated strictly as Contracts = $20.00 / (Entry - SL)?
```

---

## 6. Operational Discipline & Mental Capital

1. **Watchlist Restriction (2 Assets Only):**
   * **BTCUSDT** (Primary) and **HYPEUSDT** (Secondary).
   * Delete all other noise. Mastery of two assets generates more consistency than chasing 50 coins.
2. **Session Operating Windows:**
   * **London Open:** 12:30 PM – 4:30 PM IST (Prime expansion volume).
   * **New York Open:** 6:30 PM – 10:30 PM IST (Maximum liquidity and clean reversals).
   * **Asian Dead Zone (1:00 AM – 8:00 AM IST):** **NO NEW ENTRIES.** Active trades run on automated TP/SL.
3. **The "Hands-Off" Law:**
   * Once an order is live with hard TP and SL resting on the exchange, **close TradingView**.
   * Micro-managing 1M/5M candles inside normal pullbacks leads to panic closes and death by paper cuts.
   * You either take **+3R** or accept a controlled, professional **-$20.00** stop. Everything else is noise.
