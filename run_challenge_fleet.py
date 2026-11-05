#!/usr/bin/env python
# ==============================================================
# run_challenge_fleet.py  --  Dual Challenge Engine Supervisor
#
# Runs both CISD and Liquidity Transfer instances simultaneously,
# each isolated with its own parameters and persistence state:
#   - Instance 1: CISD Multi-TF Model (exports/challenge_state_cisd.json)
#   - Instance 2: Liquidity Transfer Model (exports/challenge_state_liquidity_transfer.json)
#
# Usage:
#   python run_challenge_fleet.py
#   python run_challenge_fleet.py --reset    (resets both states to fresh $5,000)
#   python run_challenge_fleet.py --once     (runs single scan on both)
# ==============================================================

import sys
import time
import signal
import argparse
import subprocess
import threading
from typing import List

from core.config import STRATEGY_INSTANCES


def stream_logs(process: subprocess.Popen, prefix: str):
    """Read lines from subprocess stdout and prepend strategy tag."""
    try:
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            print(f"[{prefix}] {line.rstrip()}", flush=True)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Dual Challenge Engine Supervisor")
    parser.add_argument("--once", action="store_true", help="Run one scan on all instances then exit")
    parser.add_argument("--reset", action="store_true", help="Reset both challenge states to initial $5,000")
    parser.add_argument("--models", nargs="+", default=["cisd", "liquidity_transfer"], help="Models to launch")
    args = parser.parse_args()

    procs: List[subprocess.Popen] = []
    threads: List[threading.Thread] = []

    def shutdown(sig, frame):
        print("\n[FLEET] Shutting down challenge instances gracefully...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("=" * 66)
    print("  LAUNCHING DUAL CHALLENGE FLEET (PARALLEL EXECUTION)")
    print(f"  Models: {args.models}")
    print("=" * 66)

    for model in args.models:
        if model not in STRATEGY_INSTANCES:
            print(f"[ERROR] Unknown model '{model}'. Valid: {list(STRATEGY_INSTANCES.keys())}")
            continue

        cmd = [sys.executable, "-u", "challenge_trader.py", "--model", model, "--no-live-check"]
        if args.once:
            cmd.append("--once")
        if args.reset:
            cmd.append("--reset")

        prefix = "CISD" if model == "cisd" else "LIQUIDITY"
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs.append(p)

        t = threading.Thread(target=stream_logs, args=(p, prefix), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(1)

    print(f"[FLEET] Successfully running {len(procs)} parallel challenge instance(s).\n")

    try:
        while True:
            all_done = True
            for p in procs:
                if p.poll() is None:
                    all_done = False
                    break
            if all_done:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)

    print("\n[FLEET] All challenge instances completed.")


if __name__ == "__main__":
    main()
