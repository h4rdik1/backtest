import ccxt
import pandas as pd
import time
import os
from datetime import datetime, timedelta, timezone

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
from core.config import TIMEFRAMES, DAYS_BACK as LOOKBACK_DAYS, DATA_DIR
EX_ID = 'binance'

def fetch_history(symbol, timeframe, days, exchange_id='binance'):
    """Fetch full history for a given timeframe using pagination."""
    exchange_class = getattr(ccxt, exchange_id)
    if exchange_id == 'binance':
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
                'urls': {
                    'api': {
                        'public': 'https://api3.binance.com/api/v3',
                        'private': 'https://api3.binance.com/api/v3',
                    }
                }
            }
        })
    else:
        exchange = exchange_class({'enableRateLimit': True})
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    limit = 1000
    all_data = []
    
    print(f"   Fetching {timeframe} history for {symbol} on {exchange_id}...")
    
    while True:
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not raw:
                break
            all_data.extend(raw)
            if len(raw) < limit:
                break
            since = raw[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"      [WARN] Retry after error: {e}")
            time.sleep(5)
            continue
            
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df

import argparse


def fetch_single_symbol(symbol: str, days: int, exchange_id: str = "binance"):
    """Fetch all timeframes for a single symbol and save to data/ohlcv/."""
    print("=" * 62)
    print(f"  SMC DATA FETCHER - {symbol} ({exchange_id})")
    print(f"  Fetching {days} days for: {TIMEFRAMES}")
    print("=" * 62)

    safe_symbol = symbol.replace("/", "_")
    output_dir = os.path.join(DATA_DIR, safe_symbol)
    os.makedirs(output_dir, exist_ok=True)

    for tf in TIMEFRAMES:
        df = fetch_history(symbol, tf, days, exchange_id)
        filename = os.path.join(output_dir, f"{tf}.csv")
        df.to_csv(filename, index=False)
        print(f"   SUCCESS: Saved {len(df):,} candles to {filename}")

    print(f"\n  Done: {symbol}\n")


def run_fetcher():
    parser = argparse.ArgumentParser(description="SMC Data Fetcher")
    # CHANGED: Added --symbols for batch fetching multiple symbols at once
    # Build list of symbols to fetch
    parser.add_argument("--symbol", nargs="+", default=None, help="Symbol(s) e.g BTC/USDT ETH/USDT")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Multiple symbols e.g BTC/USDT ETH/USDT SOL/USDT")
    parser.add_argument("--days", type=int, default=LOOKBACK_DAYS, help="Lookback days")
    parser.add_argument("--exchange", default="binance", help="Exchange ID (binance, okx, etc.)")
    args = parser.parse_args()

    symbols = []
    if args.symbols:
        symbols.extend(args.symbols)
    if args.symbol:
        symbols.extend([s for s in args.symbol if s not in symbols])
    
    if not symbols:
        symbols = ["BTC/USDT"]  # Default fallback

    print(f"\n  Fetching {len(symbols)} symbol(s): {symbols}")
    print(f"  Lookback: {args.days} days | Exchange: {args.exchange}\n")

    for sym in symbols:
        try:
            fetch_single_symbol(sym, args.days, args.exchange)
        except Exception as e:
            print(f"  [ERROR] Failed to fetch {sym}: {e}")
            continue

    print("=" * 62)
    print(f"  All data refreshed. You can now run backtest scripts.")
    print("=" * 62)


if __name__ == "__main__":
    run_fetcher()
