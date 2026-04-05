import http.server
import socketserver
import os
import json
import webbrowser
import sys
import pandas as pd
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import time

# Add root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import models.backtest_cisd as cisd
import core.lux_fvg as lux

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(ROOT_DIR, "exports", "challenge_state.json")
TRADES_FILE = os.path.join(ROOT_DIR, "exports", "challenge_trades.csv")

# Simple in-memory cache to prevent redundant Binance calls
CACHE = {}
CACHE_TTL = 60 # 60 seconds

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def get_cached(self, key):
        if key in CACHE:
            entry = CACHE[key]
            if (datetime.now() - entry['time']).total_seconds() < CACHE_TTL:
                return entry['data']
        return None

    def set_cache(self, key, data):
        CACHE[key] = {'time': datetime.now(), 'data': data}

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def do_GET(self):
        # --- TRADINGVIEW UDF API (For Advanced Charts) ---
        
        # 5. UDF: Config
        if self.path.startswith('/config'):
            config = {
                "supports_search": True,
                "supports_group_request": False,
                "supports_marks": False,
                "supports_timescale_marks": False,
                "supports_time": True,
                "exchanges": [{"value": "Binance", "name": "Binance", "desc": "Binance Exchange"}],
                "symbols_types": [{"name": "Crypto", "value": "crypto"}],
                "supported_resolutions": ["1", "5", "15", "60", "240", "1D", "1W"]
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(config).encode())
            return

        # 6. UDF: Symbols
        if self.path.startswith('/symbols'):
            query = urlparse(self.path).query
            params = parse_qs(query)
            sym = params.get('symbol', ['BTC/USDT'])[0].upper().replace('_', '/')
            
            # Metadata for the chart
            symbol_info = {
                "name": sym,
                "short_name": sym,
                "full_name": f"Binance:{sym}",
                "description": sym,
                "type": "crypto",
                "session": "24x7",
                "exchange": "Binance",
                "listed_exchange": "Binance",
                "timezone": "Etc/UTC",
                "format": "price",
                "pricescale": 100 if "USDT" in sym else 10000,
                "minmov": 1,
                "has_intraday": True,
                "supported_resolutions": ["1", "5", "15", "60", "240", "1D", "1W"],
                "data_status": "streaming"
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(symbol_info).encode())
            return

        # 7. UDF: History (The core candle data feed)
        if self.path.startswith('/history'):
            query = urlparse(self.path).query
            params = parse_qs(query)
            symbol = params.get('symbol', ['BTC/USDT'])[0].replace('_', '/')
            resolution = params.get('resolution', ['15'])[0]
            
            # Map TV resolution to Binance TF
            tf_map = {"1": "1m", "5": "5m", "15": "15m", "60": "1h", "240": "4h", "1D": "1d", "1W": "1w"}
            tf = tf_map.get(resolution, "15m")
            
            try:
                df = cisd.fetch_live_ohlcv(symbol, tf, limit=300)
                if df.empty:
                    res = {"s": "no_data"}
                else:
                    res = {
                        "s": "ok",
                        "t": [int(idx.timestamp()) for idx in df.index],
                        "o": [float(v) for v in df['open']],
                        "h": [float(v) for v in df['high']],
                        "l": [float(v) for v in df['low']],
                        "c": [float(v) for v in df['close']],
                        "v": [float(v) for v in df['volume']]
                    }
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode())
            except Exception as e:
                print(f" [UDF History] Error: {e}")
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"s": "error", "errmsg": str(e)}).encode())
            return

        # 8. UDF: Time
        if self.path.startswith('/time'):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(str(int(time.time())).encode())
            return

        # --- EXISTING DASHBOARD APIs ---
        
        # 1. API: Get Challenge Status
        if self.path == '/api/status':
            try:
                data = {"balance": 5000, "status": "INACTIVE"}
                if os.path.exists(STATE_FILE):
                    with open(STATE_FILE, 'r') as f:
                        data = json.load(f)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            except Exception as e:
                print(f" [/api/status] Error: {e}")
                self.send_error(500, str(e))
            return

        # 2. API: Get Trade History
        if self.path == '/api/trades':
            try:
                trades = []
                if os.path.exists(TRADES_FILE):
                    df = pd.read_csv(TRADES_FILE)
                    df = df.fillna("")
                    trades = df.to_dict('records')
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(trades).encode())
            except Exception as e:
                 print(f" [/api/trades] Error: {e}")
                 self.send_error(500, str(e))
            return

        # 3. API: Get Live Candles from Binance
        if self.path.startswith('/api/live_candles'):
            query = urlparse(self.path).query
            params = parse_qs(query)
            symbol = params.get('symbol', ['BTC/USDT'])[0].replace('_', '/')
            tf = params.get('tf', ['15m'])[0]
            cache_key = f"candles_{symbol}_{tf}"
            
            cached_data = self.get_cached(cache_key)
            if cached_data:
                print(f" [CACHE] HIT: {symbol} {tf}")
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(cached_data).encode())
                return

            print(f" [API] Fetching candles: {symbol} {tf}...", end="", flush=True)
            
            try:
                df = cisd.fetch_live_ohlcv(symbol, tf, limit=150)
                candles = []
                if not df.empty:
                    for idx, row in df.iterrows():
                        candles.append({
                            "time": int(idx.timestamp()),
                            "open": float(row['open']), "high": float(row['high']),
                            "low": float(row['low']), "close": float(row['close']),
                            "volume": float(row['volume'])
                        })
                
                self.set_cache(cache_key, candles)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(candles).encode())
                print(" OK")
            except Exception as e:
                print(f" FAIL: {e}")
                self.send_response(200) 
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps([]).encode())
            return

        # 4. API: Get Zones Overlay (FVGs/OBs)
        if self.path.startswith('/api/overlays'):
            query = urlparse(self.path).query
            params = parse_qs(query)
            symbol = params.get('symbol', ['BTC/USDT'])[0].replace('_', '/')
            tf = params.get('tf', ['15m'])[0]
            cache_key = f"overlays_{symbol}_{tf}"
            
            cached_data = self.get_cached(cache_key)
            if cached_data:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(cached_data).encode())
                return

            print(f" [API] Fetching overlays: {symbol} {tf}...", end="", flush=True)
            
            try:
                overlays = {"fvgs": [], "obs": []}
                df = cisd.fetch_live_ohlcv(symbol, tf, limit=100)
                if not df.empty:
                    fvgs = cisd.calc_fvgs(df)
                    if not fvgs.empty:
                        for idx, row in fvgs.iterrows():
                            t = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(idx[0].timestamp())
                            overlays['fvgs'].append({
                                "start": t, "top": float(row['top']), "btm": float(row['btm']), "type": str(row['type'])
                            })
                    obs = cisd.calc_obs(df)
                    if not obs.empty:
                        for idx, row in obs.iterrows():
                            t = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(idx[0].timestamp())
                            overlays['obs'].append({
                                "start": t, "top": float(row['top']), "btm": float(row['btm']), "type": str(row['type'])
                            })
                
                self.set_cache(cache_key, overlays)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(overlays).encode())
                print(" OK")
            except Exception as e:
                print(f" FAIL: {e}")
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"fvgs":[], "obs":[]}).encode())
            return

        return super().do_GET()

def run_viewer():
    os.chdir(DIRECTORY)
    try:
        with ThreadedHTTPServer(("", PORT), ViewerHandler) as httpd:
            url = f"http://localhost:{PORT}/view_candles.html"
            print(f"\n  [SUCCESS] SMC Challenge Dashboard started at {url}")
            print("  [EXIT] Press Ctrl+C to stop.")
            # webbrowser.open(url) # Uncomment if you want auto-open
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  [EXIT] Dashboard server stopped.")

if __name__ == "__main__":
    run_viewer()
