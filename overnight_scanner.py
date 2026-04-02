#!/usr/bin/env python
# ==============================================================
# overnight_scanner.py — Full overnight multi-chain watchlist scanner
# Scans all 6 curated coins across ALL 3 alignment chains every N minutes
# Run: python overnight_scanner.py [--interval 300]
# ==============================================================

import os
import time
import argparse
from datetime import datetime

from core.config import LIVE_WATCHLIST, ALIGNMENTS, chain_label


SCAN_LOG = "exports/overnight_scan_log.txt"


def header(msg: str, char="=", width=70):
    print(char * width)
    print(f"  {msg}")
    print(char * width)


def log(msg: str):
    """Append to overnight log file."""
    os.makedirs("exports", exist_ok=True)
    with open(SCAN_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_scanner_for_chain(chain_name: str, symbols: list, exchange_id: str = "binance"):
    """
    Inline import of scan_symbol so we don't spawn subprocesses -
    keeps all output captured and timestamped here.
    """
    from live_scanner import scan_symbol
    label = chain_label(chain_name)
    print("\n" + "-"*70)
    print(f"  CHAIN: {chain_name.upper()}  ({label})")
    print("-"*70)
    log(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CHAIN: {chain_name.upper()} ({label})")

    for sym in symbols:
        scan_symbol(sym, chain_name=chain_name, exchange_id=exchange_id)


def run_one_full_scan(symbols: list):
    """Run a complete scan across all 3 chains on all symbols."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.system("cls" if os.name == "nt" else "clear")

    header(f"TTrades Overnight Scanner  --  {ts}")
    print(f"  Watchlist : {', '.join(symbols)}")
    print(f"  Chains    : {', '.join(ALIGNMENTS.keys())}")
    print(f"  Capital   : $5,000  |  Risk/trade: $50 (1%)")
    header("Scan starting", char="-")
    log("\n" + "="*70)
    log(f"OVERNIGHT SCAN -- {ts}")
    log(f"Watchlist: {', '.join(symbols)}")

    for chain_name in ALIGNMENTS.keys():
        run_scanner_for_chain(chain_name, symbols)

    print(f"\n  Scan complete at {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Log: {SCAN_LOG}")


def main():
    parser = argparse.ArgumentParser(description="Overnight Multi-Chain Scanner")
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Scan interval in seconds (default: 300 = 5 min)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single scan and exit instead of looping"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Override symbol list"
    )
    args = parser.parse_args()

    symbols = args.symbols or LIVE_WATCHLIST

    print(f"\n  Starting Overnight Scanner")
    print(f"  Coins     : {', '.join(symbols)}")
    print(f"  Interval  : every {args.interval}s ({args.interval//60}min)")
    print(f"  Chains    : {list(ALIGNMENTS.keys())}")
    print(f"  Capital   : $5,000  |  1% risk = $50/trade")
    print("  Press Ctrl+C to stop\n")

    if args.once:
        run_one_full_scan(symbols)
        return

    scan_number = 1
    while True:
        print(f"\n  >>> Scan #{scan_number} starting...")
        try:
            run_one_full_scan(symbols)
        except KeyboardInterrupt:
            print("\n\n  Scanner stopped by user.")
            break
        except Exception as e:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"  [{ts}] ERROR during scan: {e}")
            log(f"  [{ts}] ERROR: {e}")

        scan_number += 1
        next_scan = datetime.now()
        print(f"\n  Sleeping {args.interval}s... Next scan at {next_scan.strftime('%H:%M:%S')}")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\n  Scanner stopped by user.")
            break


if __name__ == "__main__":
    main()
