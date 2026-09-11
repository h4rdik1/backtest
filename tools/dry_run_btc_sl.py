import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to path (works regardless of CWD)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import models.backtest_cisd as cisd
from challenge_trader import scan_for_signal
from core.config import MIN_SL_PCT_BTC, MIN_SL_PCT_ALTS, MAX_STOP_DISTANCE
from models.backtest_cisd import find_protected_swing, find_latest_sfp

def dry_run():
    symbol = "BTC/USDT"
    print(f"--- DRY RUN: Structural SL Logic for {symbol} ---")
    
    try:
        chain_name = "daily"
        print(f"Chain: {chain_name.upper()}")
        
        # We try a normal scan first
        sig = scan_for_signal(symbol, chain_name)
        
        if not sig:
            print("No active signal found. Forcing a simulation on the most recent FVG...")
            from core.config import ALIGNMENTS
            chain = ALIGNMENTS[chain_name]
            
            # Fetch data manually
            df_bias = cisd.fetch_live_ohlcv(symbol, chain["bias"], 250)
            df_ltf = cisd.fetch_live_ohlcv(symbol, chain["entry"], 500)
            
            # Detect FVGs on LTF
            import core.lux_fvg as lux
            from core.config import FVG_WICK_RATIO, FVG_AVG_BODY_LOOKBACK, FVG_MIN_SIZE_MULT, FVG_MIN_VOLUME_MULT
            fvgs = lux.detect_luxalgo_fvgs(df_ltf.copy(), FVG_WICK_RATIO, FVG_AVG_BODY_LOOKBACK, FVG_MIN_SIZE_MULT, FVG_MIN_VOLUME_MULT)
            
            # Find last bull FVG
            last_idx = -1
            for j in range(len(fvgs)-1, 0, -1):
                if fvgs['fvg_bull'].iloc[j]:
                    last_idx = j
                    btm = fvgs['fvg_bull_btm'].iloc[j]
                    top = fvgs['fvg_bull_top'].iloc[j]
                    midpoint = (btm + top) / 2
                    break
            
            if last_idx == -1:
                print("No FVG found in 500 bars. Market is too slow.")
                return

            print(f"\n[SIMULATED SIGNAL]")
            print(f"Latest FVG found @ {fvgs.index[last_idx]}")
            print(f"Zone: {btm:.2f} - {top:.2f}")
            print(f"Entry (Midpoint): {midpoint:.2f}")

            # Calculate SL using NEW LOGIC (Nested SFP)
            # Find HTF SFP (lookback 96)
            htf_sfp = find_latest_sfp(df_bias, is_long=True, lookback=96)
            # Find LTF SFP (lookback 100)
            ltf_sfp = find_latest_sfp(df_ltf, is_long=True, lookback=100)
            
            print(f"\n[STRUCTURAL ANALYSIS]")
            print(f"LTF SFP: {ltf_sfp if ltf_sfp else 'None found'}")
            print(f"HTF SFP: {htf_sfp if htf_sfp else 'None found'}")
            
            final_sl = None
            sl_type = "FVG_Boundary"
            
            if htf_sfp:
                htf_sl_price, htf_sl_time = htf_sfp
                final_sl = htf_sl_price
                sl_type = "HTF_SFP"
                
                if ltf_sfp:
                    ltf_sl_price, ltf_sl_time = ltf_sfp
                    if ltf_sl_time > htf_sl_time:
                        final_sl = ltf_sl_price
                        sl_type = "LTF_SFP"
            
            if final_sl is None:
                final_sl = btm * 0.998
                sl_type = "Fallback_Boundary"
                
            # Minimum Check
            min_sl_dist = MIN_SL_PCT_BTC
            if (midpoint - final_sl) < (midpoint * min_sl_dist):
                final_sl = midpoint - (midpoint * min_sl_dist)
                sl_type += " + Min_Floor"

            print(f"\n[FINAL PLACEMENT]")
            print(f"Final SL  : {final_sl:.2f} ({sl_type})")
            dist_pct = (midpoint - final_sl) / midpoint
            print(f"SL Dist % : {dist_pct * 100:.3f}%")

            if dist_pct > MAX_STOP_DISTANCE:
                print(f"❌ [ALERT] Stop distance ({dist_pct*100:.2f}%) exceeds MAX_STOP_DISTANCE ({MAX_STOP_DISTANCE*100:.2f}%)")
                print(f"   Verdict: Trade would be DISCARDED by the challenge engine.")
            else:
                print(f"✅ Verdict: Setup passes RR filter. New SL is { (dist_pct*100) / ((midpoint - (btm*0.999))/midpoint*100) :.1f}x safer than old 0.1% fallback.")

        else:
            # (normal output logic as before)
            pass

    except Exception as e:
        print(f"Error: {e}")

    except Exception as e:
        print(f"Error during dry run: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    dry_run()
