import ccxt
import pandas as pd
import time
import os
from datetime import datetime, timedelta, timezone

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SYMBOL = 'SOL/USDT'
EXCHANGE_ID = 'binance'
TIMEFRAMES = ["1d", "1h", "15m", "5m"]
LOOKBACK_DAYS = 365

def fetch_history(symbol, timeframe, days):
    """Fetch full history for a given timeframe using pagination."""
    exchange = getattr(ccxt, EXCHANGE_ID)({'enableRateLimit': True})
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    limit = 1000
    all_data = []
    
    print(f"   Fetching {timeframe} history for {symbol}...")
    
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

def run_fetcher():
    print("=" * 62)
    print(f"  SMC DATA FETCHER - {SYMBOL}")
    print(f"  Fetching {LOOKBACK_DAYS} days for: {TIMEFRAMES}")
    print("=" * 62)
    
    safe_symbol = SYMBOL.replace("/", "_")
    output_dir = os.path.join("prev_candles", safe_symbol)
    os.makedirs(output_dir, exist_ok=True)
    
    for tf in TIMEFRAMES:
        df = fetch_history(SYMBOL, tf, LOOKBACK_DAYS)
        filename = os.path.join(output_dir, f"{tf}.csv")
        df.to_csv(filename, index=False)
        print(f"   SUCCESS: Saved {len(df):,} candles to {filename}")
        
    print("\nAll data refreshed. You can now run backtest.py.")

if __name__ == "__main__":
    run_fetcher()
