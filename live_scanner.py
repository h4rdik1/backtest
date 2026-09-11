import os
import sys
import time
import argparse
from datetime import datetime
from core.config import LIVE_WATCHLIST, ACTIVE_CHAIN, ALIGNMENTS, chain_label, format_price, LIVE_SCAN_INTERVAL
import challenge_trader

def get_signal_for_symbol(symbol: str, chain_name: str, exchange_id: str = "binance"):
    """
    Scans a single symbol and returns the signal if found using the main challenge_trader engine.
    """
    sig = challenge_trader.scan_for_signal(symbol, chain_name, exchange_id)
    return sig


def print_signal(sig: dict):
    if not sig: return
    
    cur = sig["current_price"]
    ent = sig["entry_price"]
    dire = sig["direction"]
    sl = sig["sl_price"]
    tp = sig["tp_price"]

    # Determine if it's PENDING or ACTIVE
    is_active = False
    if dire == "SHORT":
        if cur >= ent and cur < sl:
            is_active = True
    else:  # LONG
        if cur <= ent and cur > sl:
            is_active = True

    print(f"\n--- ⭐ [{sig['setup_tier']}] {sig['symbol']} {dire} [{sig['chain'].upper()}] ---")
    if sig.get('is_counter_trend'):
        print(f"   [⚠️ WARNING] This is a COUNTER-TREND trade (Against {sig['regime'].replace('_', ' ').title()})")
    
    print(f"   [REASON] Top-to-Bottom Analysis:")
    print(f"      Chain:          {chain_label(sig['chain'])}")
    print(f"      Context Layer:  {sig['regime'].replace('_', ' ').title()} market")
    print(f"      Bias Layer:     {sig['bias_reason'] or 'Active ' + dire}")
    print(f"      Entry Anchor:   {sig['anchor_type']} (zone: {format_price(sig['fvg_zone_low'])} - {format_price(sig['fvg_zone_high'])})")

    meta = sig.get('metadata', {})
    disp = meta.get('displacement_strength', 1.0)
    if meta:
        print(f"      Quality Meter:  Displacement Strength: {disp}x (High > 1.5)")

    # SL anchor label
    sl_type = sig.get('sl_type', 'fvg_boundary')
    if sl_type == 'htf_sfp':
        sl_label = "🛡️  HTF SFP (Strong Structural Floor)"
    elif sl_type == 'ltf_sfp':
        sl_label = "🔥  LTF SFP (Tight/High-RR Protection)"
    elif sl_type == 'protected_swing':
        sl_label = "🛡️  Protected Swing"
    else:
        sl_label = "⚠️  FVG Boundary (fallback — no SFP/swing found)"

    print(f"\n   [FULL LOGIC BREAKDOWN]")
    
    # 1. Context Explanation
    regime = sig['regime']
    if regime == 'strong_bull':
        ctx = "The HTF is in a strong UP-TREND. We prioritize LONGs as institutional flow is bullish."
    elif regime == 'strong_bear':
        ctx = "The HTF is in a strong DOWN-TREND. We prioritize SHORTs as institutional flow is bearish."
    else:
        ctx = "The HTF is RANGING. We are looking for mean-reversion or liquidity sweeps from the edges."
    print(f"      1. Context: {ctx}")

    # 2. Bias Explanation
    bias_desc = sig['bias_reason'].lower()
    print(f"      2. Bias:    Our directional bias shifted to {dire} because of a {bias_desc}.")

    # 3. Entry Explanation
    anc = sig['anchor_type']
    if anc == 'CONFLUENCE':
        ent_expl = f"We have high-confluence between an FVG and an Order Block. This is an 'A++' high-probability zone."
    elif anc == 'FVG':
        ent_expl = f"Institutional displacement created a Fair Value Gap. We enter at the 'Equilibrium' (midpoint)."
    else:
        ent_expl = f"Entry is based on an Order Block where institutional buying/selling was previously concentrated."
    print(f"      3. Entry:   {ent_expl} (Strength: {disp}x)")

    # 4. Stop-Loss Explanation
    if 'sfp' in sl_type:
        stop_expl = f"SL is anchored to a Structural Sweep ({sl_type.upper()}). Price must break logic to hit our stop."
    else:
        stop_expl = "SL is at the zone boundary. This is a tighter risk setup with higher RR but lower structural protection."
    print(f"      4. Risk:    {stop_expl}")

    # 5. Setup Tier Explanation
    tier = sig['setup_tier']
    if tier == 'A++':
        tier_expl = "A++ (Ultimate Confluence). This setup aligns an FVG with a fresh LTF Order Block. Highest probability."
    elif tier == 'A':
        tier_expl = "A (High Quality). The LTF entry zone has direct confluence with an HTF Order Block."
    else:
        tier_expl = "B (Standard). A valid SMC setup, but lacking secondary structural confluence from HTF OBs."
    print(f"      5. Tier:    {tier_expl}")

    if not is_active:
        print(f"\n      [STATUS] PENDING - Waiting for price to tap entry @ {format_price(ent)}")

        print(f"      [LAYOUT] Target: {format_price(tp)} (2.0R) | Stop: {format_price(sl)}")
    else:
        print(f"\n      [STATUS] ACTIVE - Price is currently within the execution zone.")
        print(f"      [LAYOUT] Entry ~{format_price(ent)} | SL: {format_price(sl)} | TP: {format_price(tp)} (2R)")
    
    print(f"      [PRICE]  Current: {format_price(cur)}")



def run_scanner():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=LIVE_WATCHLIST)
    parser.add_argument("--interval", type=int, default=LIVE_SCAN_INTERVAL, help="Scan interval in seconds")
    args = parser.parse_args()

    # The chains we want to scan (from core config)
    chains_to_scan = list(ALIGNMENTS.keys())

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("=" * 62)
        print(f"  TTrades Omni-Scanner - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Scanning all chains: {', '.join(chains_to_scan).upper()}")
        print(f"  Next scan in {args.interval}s")
        print("=" * 62)
        print("  Scanning the market, please wait...", end="\r")

        signals_found = []

        for sym_raw in args.symbols:
            if ":" in sym_raw:
                sym, ex_id = sym_raw.split(":")
            else:
                sym, ex_id = sym_raw, "binance"
                
            for chain_name in chains_to_scan:
                try:
                    sig = get_signal_for_symbol(sym, chain_name, ex_id)
                    if sig:
                        signals_found.append(sig)
                except Exception as e:
                    pass
                    
        # Sort the signals to find the strictly best probability setups
        # Tier A++ > A > B. For ties, prefer daily over weekly over intraday etc depending on alignment priority.
        chain_priority = {"daily": 0, "weekly": 1, "intraday": 2}
        tier_priority = {"A++": 0, "A": 1, "B": 2}
        
        signals_found.sort(
            key=lambda s: (
                tier_priority.get(s["setup_tier"], 9),
                chain_priority.get(s["chain"], 9),
            )
        )

        # Clear the "Scanning..." line
        print("                                       \r", end="")

        if not signals_found:
            print("\n  [STATE] No valid A/B tier setups found across any timeframe right now.")
            print("          Waiting for the next scan...")
        else:
            print(f"\n  Found {len(signals_found)} valid setups. Ranking by Probability (Win-Rate):")
            
            # De-duplicate: If the same symbol has a signal on multiple chains, just show the highest tier one
            deduped = []
            seen_syms = set()
            for s in signals_found:
                if s["symbol"] not in seen_syms:
                    deduped.append(s)
                    seen_syms.add(s["symbol"])
                    
            for sig in deduped:
                print_signal(sig)

        print("\n" + "=" * 62)
        time.sleep(args.interval)

if __name__ == "__main__":
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            # Fallback for older python versions
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    run_scanner()

