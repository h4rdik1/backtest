import http.server
import socketserver
import os
import json
import webbrowser
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import core.lux_fvg as lux
import models.backtest_cisd as cisd

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def do_GET(self):
        # Custom API to list trades
        if self.path == '/api/list_trades':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Use absolute path for trade log
            log_file = os.path.abspath(os.path.join(DIRECTORY, "..", "exports", "trade_log_v5.csv"))
            trades = []
            if os.path.exists(log_file):
                import pandas as pd
                try:
                    df = pd.read_csv(log_file).fillna("")
                    trades = df.to_dict(orient='records')
                except Exception as e:
                    print(f"Error reading trades: {e}")
            
            self.wfile.write(json.dumps(trades).encode())
            return

        # Custom API to list symbols and their timeframes
        if self.path == '/api/list_data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            data = {}
            # Use absolute path to avoid Windows pathing issues
            base_path = os.path.abspath(os.path.join(DIRECTORY, "..", "data", "ohlcv"))
            if not os.path.exists(base_path):
                # Fallback for relative paths in case of different execution context
                base_path = os.path.abspath(os.path.join(os.getcwd(), "data", "ohlcv"))
            
            if os.path.exists(base_path):
                symbols = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                for symbol in symbols:
                    tf_path = os.path.join(base_path, symbol)
                    tfs = [f.replace(".csv", "") for f in os.listdir(tf_path) if f.endswith(".csv")]
                    data[symbol] = tfs
            
            self.wfile.write(json.dumps(data).encode())
            return

        # NEW: API to get FVG and OB zones for visualization
        if self.path.startswith('/api/get_overlays'):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            
            symbol = params.get('symbol', [None])[0]
            tf = params.get('tf', [None])[0]
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            overlays = {"fvgs": [], "obs": []}
            
            if symbol and tf:
                try:
                    sym_load = symbol.replace('_', '/')
                    df = cisd.fetch_ohlcv_full(sym_load, tf, 30)
                    
                    # 1. Calc FVGs
                    fvgs_df = lux.detect_luxalgo_fvgs(df.copy())
                    bulls = fvgs_df[fvgs_df['fvg_bull']]
                    bears = fvgs_df[fvgs_df['fvg_bear']]
                    
                    for idx, row in bulls.iterrows():
                        overlays["fvgs"].append({
                            "type": "bull",
                            "start": idx.timestamp(),
                            "top": row['fvg_bull_top'],
                            "btm": row['fvg_bull_btm']
                        })
                    for idx, row in bears.iterrows():
                        overlays["fvgs"].append({
                            "type": "bear",
                            "start": idx.timestamp(),
                            "top": row['fvg_bear_top'],
                            "btm": row['fvg_bear_btm']
                        })
                    
                    # 2. Calc OBs
                    obs_list = cisd.find_order_blocks(df)
                    for ob in obs_list:
                        overlays["obs"].append({
                            "start": ob['time'].timestamp(),
                            "top": ob['high'],
                            "btm": ob['low']
                        })
                except Exception as e:
                    print(f"Overlay error: {e}")
            
            self.wfile.write(json.dumps(overlays).encode())
            return

        # NEW: Custom API to serve CSV data directly
        if self.path.startswith('/prev_candles/'):
            parts = self.path.split('/')
            if len(parts) >= 4:
                symbol = parts[2]
                tf_file = parts[3]
                csv_path = os.path.abspath(os.path.join(DIRECTORY, "..", "data", "ohlcv", symbol, tf_file))
                
                if os.path.exists(csv_path):
                    self.send_response(200)
                    self.send_header('Content-type', 'text/csv')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    with open(csv_path, 'rb') as f:
                        self.wfile.write(f.read())
                    return

        return super().do_GET()

def run_viewer():
    print("=" * 62)
    print("  SMC VISION - INTERACTIVE VIEWER")
    print("=" * 62)
    
    Handler = ViewerHandler
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/view_candles.html"
        print(f"\n  [SUCCESS] Server started at {url}")
        print("  [ACTION] Opening browser now...")
        print("  [EXIT] Press Ctrl+C to stop the server.")
        
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  [INFO] Server stopped.")

if __name__ == "__main__":
    run_viewer()
