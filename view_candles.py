import http.server
import socketserver
import os
import json
import webbrowser
from urllib.parse import urlparse

PORT = 8000
DIRECTORY = os.getcwd()

class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Custom API to list trades
        if self.path == '/api/list_trades':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            log_file = "trade_log_v4.csv"
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
            base_path = "prev_candles"
            if os.path.exists(base_path):
                symbols = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                for symbol in symbols:
                    tf_path = os.path.join(base_path, symbol)
                    tfs = [f.replace(".csv", "") for f in os.listdir(tf_path) if f.endswith(".csv")]
                    data[symbol] = tfs
            
            self.wfile.write(json.dumps(data).encode())
            return
            
        return super().do_GET()

def run_viewer():
    print("=" * 62)
    print("  SMC VISION - INTERACTIVE VIEWER")
    print("=" * 62)
    print(f"  Scanning: {os.path.join(DIRECTORY, 'prev_candles')}")
    
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
