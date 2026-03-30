import time
import subprocess
import os
import sys
from datetime import datetime

def run_once():
    print(f"\n" + "="*60)
    print(f"  WATCHLIST SCAN - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    # 1. Update data (Short lookback for speed)
    print("Refetching latest data...")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(root_dir)
    from core.config import LIVE_WATCHLIST

    for sym_raw in LIVE_WATCHLIST:
        if ":" in sym_raw:
            sym, ex_id = sym_raw.split(":")
        else:
            sym, ex_id = sym_raw, "binance"
        
        print(f"  Fetching {sym} ({ex_id})...")
        subprocess.run([
            "python", os.path.join(root_dir, "fetch_candles.py"), 
            "--symbol", sym, "--days", "7", "--exchange", ex_id
        ], capture_output=True)
    
    # 2. Run Scanner (Daily & Weekly)
    print("\n[SCANNING DAILY ALIGNMENT]")
    subprocess.run(["python", os.path.join(root_dir, "live_scanner.py"), "--alignment", "daily"])
    
    print("\n[SCANNING WEEKLY ALIGNMENT]")
    subprocess.run(["python", os.path.join(root_dir, "live_scanner.py"), "--alignment", "weekly"])
    
    print("\n" + "-"*60)
    print("Next scan in 10 minutes...")
    print("="*60)

if __name__ == "__main__":
    while True:
        try:
            run_once()
            time.sleep(600) # 10 minutes
        except KeyboardInterrupt:
            print("\nWatchlist stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60) # Try again in 1 min
