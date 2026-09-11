import http.server
import socketserver
import os
import json
import webbrowser
import sys
import threading
import pandas as pd
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import time

# Add root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import models.backtest_cisd as cisd
import core.lux_fvg as lux

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(ROOT_DIR, "exports", "challenge_state.json")
TRADES_FILE = os.path.join(ROOT_DIR, "exports", "challenge_trades.csv")

# Simple in-memory cache to prevent redundant Binance calls
CACHE = {}
CACHE_TTL = 60  # 60 seconds

# Thread lock for state file writes
STATE_LOCK = threading.Lock()


def read_state():
    """Read the challenge state JSON file safely."""
    if not os.path.exists(STATE_FILE):
        return {"balance": 5000, "status": "INACTIVE", "open_trades": [], "pending_trades": [], "closed_trades": []}
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def write_state(data):
    """Write the challenge state JSON file safely."""
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def get_cached(self, key):
        if key in CACHE:
            entry = CACHE[key]
            if (datetime.now() - entry["time"]).total_seconds() < CACHE_TTL:
                return entry["data"]
        return None

    def set_cache(self, key, data):
        CACHE[key] = {"time": datetime.now(), "data": data}

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b'{}'
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        def send_json(data, status=200):
            resp = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(resp)

        # --- POST /api/trade/toggle --- enable/disable an open/pending trade
        if self.path == "/api/trade/toggle":
            trade_id = payload.get("trade_id")
            if not trade_id:
                send_json({"ok": False, "error": "Missing trade_id"}, 400)
                return
            with STATE_LOCK:
                state = read_state()
                found = False
                for group in ("open_trades", "pending_trades"):
                    for t in state.get(group, []):
                        if t.get("trade_id") == trade_id:
                            t["disabled"] = not t.get("disabled", False)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    send_json({"ok": False, "error": f"Trade {trade_id} not found"}, 404)
                    return
                write_state(state)
            send_json({"ok": True})
            return

        # --- POST /api/trade/close --- manually close an open trade
        if self.path == "/api/trade/close":
            trade_id = payload.get("trade_id")
            exit_price = payload.get("exit_price")
            if not trade_id or exit_price is None:
                send_json({"ok": False, "error": "Missing trade_id or exit_price"}, 400)
                return
            with STATE_LOCK:
                state = read_state()
                trade = None
                new_open = []
                for t in state.get("open_trades", []):
                    if t.get("trade_id") == trade_id:
                        trade = t
                    else:
                        new_open.append(t)
                if trade is None:
                    send_json({"ok": False, "error": f"Open trade {trade_id} not found"}, 404)
                    return
                try:
                    exit_price = float(exit_price)
                except Exception:
                    send_json({"ok": False, "error": "Invalid exit_price"}, 400)
                    return
                trade["exit_price"] = exit_price
                trade["closed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Get or reconstruct initial SL price
                entry = trade.get("entry_price", 0)
                tp = trade.get("tp_price", 0)
                initial_sl = trade.get("initial_sl_price")
                if initial_sl is None and tp and entry:
                    risk_dist = abs(tp - entry) / 2.0
                    initial_sl = entry - risk_dist if trade.get("direction") == "LONG" else entry + risk_dist
                
                if initial_sl is not None:
                    sl_dist = abs(entry - initial_sl)
                else:
                    sl_dist = abs(entry - trade.get("sl_price", 0))
                    
                if not sl_dist:
                    sl_dist = entry * 0.005
                
                if trade.get("direction") == "LONG":
                    pnl_r = (exit_price - entry) / sl_dist
                else:
                    pnl_r = (entry - exit_price) / sl_dist
                    
                risk = trade.get("trade_risk", 35.0)
                reward = trade.get("trade_reward", 70.0)
                partial_exit_done = trade.get("partial_exit_done", False)
                
                partial_pnl = (reward * 0.50) if partial_exit_done else 0.0
                position_factor = 0.50 if partial_exit_done else 1.0
                
                remaining_pnl = round(pnl_r * risk * position_factor, 2)
                total_pnl = partial_pnl + remaining_pnl
                
                trade["pnl_usd"] = round(total_pnl, 2)
                if initial_sl is not None:
                    trade["initial_sl_price"] = round(initial_sl, 8)
                
                if total_pnl > 0.01:
                    trade["result"] = "WIN"
                    state["wins"] = state.get("wins", 0) + 1
                elif total_pnl < -0.01:
                    trade["result"] = "LOSS"
                    state["losses"] = state.get("losses", 0) + 1
                else:
                    trade["result"] = "BE"
                    
                trade["balance_after"] = round(state.get("balance", 5000) + remaining_pnl, 2)
                state["balance"] = trade["balance_after"]
                state["open_trades"] = new_open
                if "closed_trades" not in state:
                    state["closed_trades"] = []
                state["closed_trades"].append(trade)
                state["total_trades"] = state.get("total_trades", 0) + 1
                write_state(state)
            send_json({"ok": True, "pnl_usd": round(total_pnl, 2)})
            return

        # --- POST /api/trade/add --- add a manual trade
        if self.path == "/api/trade/add":
            required = ["symbol", "direction", "entry_price", "sl_price", "tp_price"]
            for r in required:
                if r not in payload:
                    send_json({"ok": False, "error": f"Missing field: {r}"}, 400)
                    return
            with STATE_LOCK:
                state = read_state()
                next_id = state.get("next_trade_id", 1)
                trade_id = f"T{next_id:04d}"
                sl_dist = abs(float(payload["entry_price"]) - float(payload["sl_price"]))
                risk = float(payload.get("trade_risk", 35.0))
                reward = risk * 2.0
                new_trade = {
                    "trade_id": trade_id,
                    "opened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "closed_at": None,
                    "symbol": payload["symbol"].upper().replace("_", "/"),
                    "chain": payload.get("chain", "manual"),
                    "direction": payload["direction"].upper(),
                    "entry_price": float(payload["entry_price"]),
                    "sl_price": float(payload["sl_price"]),
                    "initial_sl_price": float(payload["sl_price"]),
                    "tp_price": float(payload["tp_price"]),
                    "exit_price": None,
                    "result": "OPEN",
                    "pnl_usd": 0.0,
                    "balance_after": state.get("balance", 5000),
                    "regime": payload.get("regime", "manual"),
                    "bias_reason": payload.get("notes", "Manual trade"),
                    "anchor_type": payload.get("anchor_type", "MANUAL"),
                    "setup_tier": payload.get("setup_tier", "B"),
                    "setup_score": int(payload.get("setup_score", 5)),
                    "trade_risk": risk,
                    "trade_reward": reward,
                    "session": payload.get("session", "manual"),
                    "notes": payload.get("notes", ""),
                    "be_triggered": False,
                    "partial_exit_done": False,
                    "trailing_active": False,
                    "trailing_sl": None,
                    "max_r_reached": 0.0,
                    "entry_bar_idx": 0,
                    "exchange_id": "binance",
                    "scan_number": 0,
                    "disabled": False,
                    "manual": True,
                }
                if "open_trades" not in state:
                    state["open_trades"] = []
                state["open_trades"].append(new_trade)
                state["next_trade_id"] = next_id + 1
                write_state(state)
            send_json({"ok": True, "trade_id": trade_id})
            return

        send_json({"ok": False, "error": "Unknown POST endpoint"}, 404)

    def do_GET(self):
        # --- TRADINGVIEW UDF API (For Advanced Charts) ---

        # 5. UDF: Config
        if self.path.startswith("/config"):
            config = {
                "supports_search": True,
                "supports_group_request": False,
                "supports_marks": False,
                "supports_timescale_marks": False,
                "supports_time": True,
                "exchanges": [
                    {"value": "Binance", "name": "Binance", "desc": "Binance Exchange"}
                ],
                "symbols_types": [{"name": "Crypto", "value": "crypto"}],
                "supported_resolutions": ["1", "5", "15", "60", "240", "1D", "1W"],
            }
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(config).encode())
            return

        # 6. UDF: Symbols
        if self.path.startswith("/symbols"):
            query = urlparse(self.path).query
            params = parse_qs(query)
            sym = params.get("symbol", ["BTC/USDT"])[0].upper().replace("_", "/")

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
                "data_status": "streaming",
            }
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(symbol_info).encode())
            return

        # 7. UDF: History (The core candle data feed)
        if self.path.startswith("/history"):
            query = urlparse(self.path).query
            params = parse_qs(query)
            symbol = params.get("symbol", ["BTC/USDT"])[0].replace("_", "/")
            resolution = params.get("resolution", ["15"])[0]

            # Map TV resolution to Binance TF
            tf_map = {
                "1": "1m",
                "5": "5m",
                "15": "15m",
                "60": "1h",
                "240": "4h",
                "1D": "1d",
                "1W": "1w",
            }
            tf = tf_map.get(resolution, "15m")

            try:
                df = cisd.fetch_live_ohlcv(symbol, tf, limit=300)
                if df.empty:
                    res = {"s": "no_data"}
                else:
                    res = {
                        "s": "ok",
                        "t": [int(idx.timestamp()) for idx in df.index],
                        "o": [float(v) for v in df["open"]],
                        "h": [float(v) for v in df["high"]],
                        "l": [float(v) for v in df["low"]],
                        "c": [float(v) for v in df["close"]],
                        "v": [float(v) for v in df["volume"]],
                    }
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode())
            except Exception as e:
                print(f" [UDF History] Error: {e}")
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"s": "error", "errmsg": str(e)}).encode())
            return

        # 8. UDF: Time
        if self.path.startswith("/time"):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(str(int(time.time())).encode())
            return

        # --- EXISTING DASHBOARD APIs ---

        # 1. API: Get Challenge Status
        if self.path == "/api/status":
            try:
                data = {"balance": 5000, "status": "INACTIVE"}
                if os.path.exists(STATE_FILE):
                    with open(STATE_FILE, "r") as f:
                        data = json.load(f)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            except Exception as e:
                print(f" [/api/status] Error: {e}")
                self.send_error(500, str(e))
            return

        # 2. API: Get Trade History
        if self.path == "/api/trades":
            try:
                trades = []
                if os.path.exists(TRADES_FILE):
                    df = pd.read_csv(TRADES_FILE)
                    df = df.fillna("")
                    trades = df.to_dict("records")

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(trades).encode())
            except Exception as e:
                print(f" [/api/trades] Error: {e}")
                self.send_error(500, str(e))
            return

        # 3. API: Get Live Candles from Binance
        if self.path.startswith("/api/live_candles"):
            query = urlparse(self.path).query
            params = parse_qs(query)
            symbol = params.get("symbol", ["BTC/USDT"])[0].replace("_", "/")
            tf = params.get("tf", ["15m"])[0]
            cache_key = f"candles_{symbol}_{tf}"

            cached_data = self.get_cached(cache_key)
            if cached_data:
                print(f" [CACHE] HIT: {symbol} {tf}")
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(cached_data).encode())
                return

            print(f" [API] Fetching candles: {symbol} {tf}...", end="", flush=True)

            try:
                df = cisd.fetch_live_ohlcv(symbol, tf, limit=150)
                candles = []
                if not df.empty:
                    for idx, row in df.iterrows():
                        candles.append(
                            {
                                "time": int(idx.timestamp()),
                                "open": float(row["open"]),
                                "high": float(row["high"]),
                                "low": float(row["low"]),
                                "close": float(row["close"]),
                                "volume": float(row["volume"]),
                            }
                        )

                self.set_cache(cache_key, candles)
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(candles).encode())
                print(" OK")
            except Exception as e:
                print(f" FAIL: {e}")
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps([]).encode())
            return

        # 4. API: Get Zones Overlay (FVGs/OBs)
        if self.path.startswith("/api/overlays"):
            query = urlparse(self.path).query
            params = parse_qs(query)
            symbol = params.get("symbol", ["BTC/USDT"])[0].replace("_", "/")
            tf = params.get("tf", ["15m"])[0]
            cache_key = f"overlays_{symbol}_{tf}"

            cached_data = self.get_cached(cache_key)
            if cached_data:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
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
                            t = (
                                int(idx.timestamp())
                                if hasattr(idx, "timestamp")
                                else int(idx[0].timestamp())
                            )
                            overlays["fvgs"].append(
                                {
                                    "start": t,
                                    "top": float(row["top"]),
                                    "btm": float(row["btm"]),
                                    "type": str(row["type"]),
                                }
                            )
                    obs = cisd.calc_obs(df)
                    if not obs.empty:
                        for idx, row in obs.iterrows():
                            t = (
                                int(idx.timestamp())
                                if hasattr(idx, "timestamp")
                                else int(idx[0].timestamp())
                            )
                            overlays["obs"].append(
                                {
                                    "start": t,
                                    "top": float(row["top"]),
                                    "btm": float(row["btm"]),
                                    "type": str(row["type"]),
                                }
                            )

                self.set_cache(cache_key, overlays)
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(overlays).encode())
                print(" OK")
            except Exception as e:
                print(f" FAIL: {e}")
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"fvgs": [], "obs": []}).encode())
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
