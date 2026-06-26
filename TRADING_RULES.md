# TTrades Institutional Trading Rules
### Smart Money Concepts (SMC) — Complete Tier Framework

> *"Patience is the edge. We don't chase price — we let the market manipulate into our zones."*

---

## The Core Philosophy

Every trade must tell a **top-to-bottom story**. If you cannot explain the logic from the highest timeframe down to your entry candle in under 60 seconds, you are not ready to take the trade.

The framework has **4 tiers** of setup quality. You must correctly identify the tier of your setup BEFORE entering. The tier determines your position sizing and how strictly you follow the rules.

---

## Timeframe Alignment Chains

The system uses three specific "chains." Each chain scans from the macro context down to the entry trigger.

| Chain | Context (Structural Bias) | Bias (Direction) | Entry (Trigger) |
|-------|--------------------------|-------------------|-----------------|
| **Weekly** | 1W | 4H | 15M |
| **Daily** | 1D | 1H | 5M |

> **Rule:** Always read the chart from LEFT to RIGHT and TOP to BOTTOM (HTF → LTF). Never work your way up from the LTF to justify a bias.

---

## The Tier System

### 🏆 S-Tier — "The Double Sweep" (Highest Conviction)

**What it is:** Both the HTF and the LTF have independent, sequential liquidity sweeps before the entry. This is the **rarest and highest-probability** setup in the framework.

**The Full Sequence (Long Example):**
1. **HTF BOS:** Price breaks a major structural high on the 4H/1H, flipping bias to Bullish.
2. **HTF Pullback to Discount:** Price retraces to the 4H FVG or 0.5 Equilibrium of the move that created the BOS.
3. **HTF Sweep (SFP):** Price wicks *below* the key support level (or BOS retest point) and **closes back above it.** This is the institutional buy trigger.
4. **LTF MSB:** Following the HTF sweep, the 15M/5M market structure breaks to the upside (a Lower High is taken out), confirming the reversal direction.
5. **LTF Sweep (The S-Tier Stamp):** On the retrace back into the LTF FVG/OB, price forms local swing highs, then a candle **wicks above them and closes back below.** This sweep clears the liquidity that was resting above those highs.
6. **LTF MSB #2 (Confirmation of Confirmation):** Price breaks structure again after the LTF sweep, confirming entry.

**Entry:** Limit order inside the LTF FVG/OB zone (at or below the midpoint of the zone).
**Stop Loss:** Below the extreme of the LTF Sweep (the absolute lowest wick).
**Target:** The next major HTF liquidity pool or Range High/Low.
**Risk Sizing:** Full 1R (1% of capital / $50 in the challenge).

---

### ⭐ A++ Tier — "HTF Liquidity Grab + LTF FVG + OB Confluence"

**What it is:** The HTF sweep is confirmed, and on the LTF, the entry zone has **both an FVG AND an Order Block** overlapping. Double confirmation at the micro level.

**The Sequence:**
1. **HTF Sweep (SFP):** A clear wick above/below a key level with a close back inside structure.
2. **LTF MSB:** Market structure breaks in the direction of the trade.
3. **LTF Confluence Zone:** The FVG and an OB overlap in the same price area. The "entry box" is where they intersect.
4. **LTF Sweep into Zone (Optional but ideal):** Price taps the confluence zone, ideally with a minor sweep of the swing lows/highs just in front of it.

**Entry:** Into the confluence zone (FVG + OB overlap).
**Stop Loss:** Below/above the OB extreme.
**Target:** HTF liquidity pool (minimum 2R).
**Risk Sizing:** Full 1R.

---

### ✅ A-Tier — "HTF Aligned + LTF Sweep + LTF MSB" (Standard Conviction)

**What it is:** The HTF bias and the LTF signal are pointing in the same direction. You have a LTF liquidity sweep and structure break, but the HTF hasn't had a clean SFP (just a general bias from FVG or Sweep signal).

**This session's ETH Long (Apr-10) is a textbook A-Tier:**
- 4H BOS at ~2170 → Bullish HTF Bias ✅
- Price retraced to 0.5 Eq / FVG Discount ✅
- 15M Sweep of local lows (wick to 2183) ✅
- 15M MSB (broke above the local lower high) ✅

**What separates A-Tier from S-Tier:** The HTF does NOT have a clean SFP into an OB. The HTF just has a directional bias signal (FVG or Sweep). The entry is still valid but the "institutional footprint" is less clean.

**Entry:** Into the LTF FVG/OB on the retest of the MSB.
**Stop Loss:** Below the LTF Sweep wick extreme.
**Target:** Next HTF structural level (aim for 2R minimum).
**Risk Sizing:** Full 1R.

---

### ⚠️ B-Tier — "Single Timeframe FVG Bounce" (Low Conviction — Avoid in Challenge)

**What it is:** You see an FVG on the LTF and price taps it. There is no HTF sweep or structural confirmation. You are essentially just betting on a FVG bounce.

**Why we avoid it in the challenge:** Our own backtesting showed B-tier setups had a **29.6% Win Rate** (below breakeven for 2R). The filter was added on Apr-09 specifically because of this.

**The Bot Setting:** `MIN_SETUP_TIER = "A"` — The script will never automatically take a B-tier trade.

**When it is acceptable manually:** A B-tier becomes acceptable ONLY when:
- You are at maximum drawdown and need a defensive trade (not applicable in the challenge).
- You have significant additional context (news, liquidity, correlated asset confluence) that the bot cannot detect.
- In all other cases: **SKIP IT.**

---

## The "Conflict of Bias" Rule (Critical)

> **If your HTF says "Down" and your LTF says "Up" — this is a counter-trend trade. Do not take it.**

**The ETH 4H Bearish SFP vs. ETH 15M Bullish MSB:** The 4H wick above 2273 was institutional distribution. The 15M sweep at 2183 is institutional accumulation. They conflict. When we saw the 4H SFP was actually at the **Range High** (a full range extension sweep, not just a local one), the BOS Retest at equilibrium gave enough confluence to make the Long an A-Tier. But this distinction is subtle and should not be made casually.

**Rule for Conflict of Bias:**
- HTF SFP at a **Range Extreme** (top or bottom of a macro range) → The LTF retest at Equilibrium (0.5) can be valid Long/Short with strict LTF confirmation.
- HTF SFP at a **Local Swing High** (not a range extreme) → Do NOT take the counter-trend LTF signal. Wait for HTF bias to flip.

---

## Execution Rules (Non-Negotiable)

### Entry
- [ ] **Wait for the candle close.** An SFP is only confirmed on a closed candle. A wick that is still forming could be a genuine breakout.
- [ ] **Use limit orders.** Do not market-buy into an FVG. Set the limit at the midpoint of the zone.
- [ ] **One signal per zone.** If you already traded an FVG zone, do not trade it again. (The script tracks `visited_signals` for exactly this reason.)

### Stop Loss
- [ ] **Always use a structural stop.** Prefer the `protected_swing` SL (the last confirmed swing low/high) over the `fvg_boundary` SL.
- [ ] **Minimum buffer:** 0.1% beyond the structural extreme. Never a "naked" SL right at the boundary.
- [ ] **SL must be invalidating.** If price hits your SL, your entire thesis must be wrong. If you could see price hitting your SL and still believe the trade works, your SL is too tight.

### Take Profit
- [ ] **TP1:** The nearest major swing high/low on the bias TF (typically 1.5R-2R).
- [ ] **TP2 (Optional):** The range extreme / next liquidity pool, if the setup had exceptional confluence.
- [ ] **Break-Even at 1.8R:** Move SL to entry once trade reaches 1.8R in profit. Let it run to full target.

### Risk Management
- [ ] **1% per trade** — $50 risk in the challenge (fixed).
- [ ] **Max 2 open trades simultanoeusly.**
- [ ] **Daily trade cap: 4 trades.** After 4 trades, the session's scanning stops (new entries only; open trades continue to be managed).
- [ ] **Challenge floor: $4,700.** If balance drops here, halt all trading and review.

---

## Trade Journal Checklist

Before submitting any trade entry, run through this checklist:

```
[ ] 1. What is the HTF bias? (BOS / FVG / Sweep)
[ ] 2. Is price in a Discount (for Longs) or Premium (for Shorts)?
[ ] 3. Did price SWEEP liquidity before reversing? (Sweep = high conviction)
[ ] 4. Is there a confirmed LTF MSB after the sweep? (Closed candle only)
[ ] 5. Is there a clear FVG / OB to enter from?
[ ] 6. What is the minimum Tier? (Must be A or above for challenge)
[ ] 7. Where is my SL? Is it structural?
[ ] 8. What is the R:R? (Minimum 2R required)
[ ] 9. Am I fighting the HTF bias? (If yes → SKIP)
[ ] 10. Is this coin already in a trade? (If yes → SKIP, no duplicate symbols)
```

---

## Range & Structure Marking SOP

To maintain consistency and avoid bias, always follow these rules when marking trading ranges:

### 1. The Leg of Displacement (SOP 1)
Do not draw ranges in choppy, sideways environments. A range is only established after a **displacement leg**—an aggressive move that breaks previous structure (BOS or CHoCH/MSB).
* **Range Low (0):** The absolute lowest wick point that initiated the upward displacement.
* **Range High (1):** The absolute highest wick point reached before a clear pullback (defined as at least two consecutive candles closing in the opposite direction).

### 2. The 3-Candle Swing Confirmation (SOP 2)
A swing high/low must meet structural validity to be used as a range boundary:
* **Swing Low:** A candle low flanked by a higher low on the left and a higher low on the right.
* **Swing High:** A candle high flanked by a lower high on the left and a lower high on the right.

### 3. Breaks vs. Sweeps (SOP 3)
* **BOS (Break of Structure):** A candle body closes *outside* the range boundary. The current range is now invalidated. Wait for a new displacement leg.
* **Sweep / SFP (Swing Failure Pattern):** A wick breaks the boundary but the candle closes *back inside* the range. The range remains active, and this is a high-probability reversal signal.

### 4. Premium vs. Discount (SOP 4)
* **Discount (< 0.5 Equilibrium):** The only zone where **Longs** are permitted.
* **Premium (> 0.5 Equilibrium):** The only zone where **Shorts** are permitted.

---

## Tier Quick-Reference

| Tier | HTF Sweep | HTF + LTF Aligned | LTF FVG+OB Confluence | LTF Sweep | Win Rate (Backtest) | Challenge Eligible |
|------|-----------|-------------------|----------------------|-----------|---------------------|--------------------|
| **S** | ✅ Clean SFP | ✅ Same direction | ✅ Both present | ✅ Required | ~55-65% | ✅ Yes |
| **A++** | ✅ Clean SFP | ✅ Same direction | ✅ Both present | ➖ Optional | ~50-55% | ✅ Yes |
| **A** | ➖ Bias only | ✅ Same direction | ➖ FVG or OB | ✅ Required | ~42-50% | ✅ Yes |
| **B** | ❌ No sweep | ➖ Any | ❌ FVG only | ❌ Not required | ~29-35% | ❌ No |

---

*Last Updated: 2026-04-10 — Based on live trading session analysis and challenge post-mortems.*
