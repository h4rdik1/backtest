#!/usr/bin/env python
# ==============================================================
# api_server.py  --  Lightweight Multi-Challenge REST API Server
#
# Zero third-party dependencies (built-in standard library http.server).
# Serves live states and closed trade journals for both instances:
#   GET /api/challenges
#   GET /api/challenges/cisd
#   GET /api/challenges/liquidity_transfer
#   GET /health
#
# CORS enabled for React frontend running on any port (e.g. 5173).
# ==============================================================

import os
import time
import json
import csv
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Tuple

from core.config import STRATEGY_INSTANCES

PORT = 8000

# In-memory short-lived candle cache to prevent hitting exchange rate limits
CANDLE_CACHE: Dict[str, Tuple[float, list]] = {}


def fetch_candles_json(symbol: str, tf: str) -> list:
    """Fetch live OHLCV candles from Binance via ccxt and format for Lightweight Charts."""
    clean_sym = symbol.replace("_", "/")
    cache_key = f"{clean_sym}_{tf}"
    now = time.time()

    if cache_key in CANDLE_CACHE:
        ts, cached = CANDLE_CACHE[cache_key]
        if now - ts < 10.0:  # 10 second TTL
            return cached

    try:
        import models.backtest_cisd as cisd
        df = cisd.fetch_live_ohlcv(clean_sym, tf, limit=120)
        if df.empty:
            return []

        candles = [
            {
                "time": int(idx.timestamp()),
                "open": round(float(row["open"]), 4),
                "high": round(float(row["high"]), 4),
                "low": round(float(row["low"]), 4),
                "close": round(float(row["close"]), 4),
                "volume": round(float(row.get("volume", 0)), 2),
            }
            for idx, row in df.iterrows()
        ]
        CANDLE_CACHE[cache_key] = (now, candles)
        return candles
    except Exception as e:
        print(f"[API] Error fetching live candles for {symbol} {tf}: {e}")
        return []


def read_json_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def read_trades_csv(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception:
        return []


def get_instance_data(model_name: str) -> Dict[str, Any]:
    cfg = STRATEGY_INSTANCES.get(model_name, {})
    state_file = cfg.get("state_file", f"exports/challenge_state_{model_name}.json")
    trade_file = cfg.get("trade_log", f"exports/challenge_trades_{model_name}.csv")

    state = read_json_file(state_file)
    trades = read_trades_csv(trade_file)

    balance = state.get("balance", cfg.get("start_balance", 5000.0))
    start_bal = cfg.get("start_balance", 5000.0)
    pnl = balance - start_bal
    target = cfg.get("target", 5500.0)
    floor = cfg.get("floor", 4700.0)
    progress_pct = max(0.0, min(100.0, ((balance - start_bal) / (target - start_bal)) * 100)) if target > start_bal else 0.0

    return {
        "model": model_name,
        "name": cfg.get("name", model_name.upper()),
        "description": cfg.get("description", ""),
        "status": state.get("status", "ACTIVE" if state else "INITIALIZING"),
        "balance": balance,
        "start_balance": start_bal,
        "pnl_usd": round(pnl, 2),
        "peak_balance": state.get("peak_balance", balance),
        "floor": floor,
        "target": target,
        "progress_pct": round(progress_pct, 1),
        "drawdown_usd": round(max(0.0, start_bal - balance), 2),
        "scan_count": state.get("scan_count", 0),
        "total_trades": state.get("total_trades", len(trades)),
        "wins": state.get("wins", 0),
        "losses": state.get("losses", 0),
        "open_trades": state.get("open_trades", []),
        "pending_trades": state.get("pending_trades", []),
        "closed_trades": trades[-50:],  # last 50 closed trades
    }


class ChallengeAPIHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/health" or path == "":
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "service": "Challenge Trading Fleet API"}).encode("utf-8"))
            return

        if path == "/api/challenges":
            data = {
                model_name: get_instance_data(model_name)
                for model_name in STRATEGY_INSTANCES.keys()
            }
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))
            return

        if path == "/api/live_candles":
            qs = parse_qs(parsed.query)
            symbol = qs.get("symbol", ["BTC/USDT"])[0].replace("_", "/")
            tf = qs.get("tf", ["15m"])[0]
            candles = fetch_candles_json(symbol, tf)
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(candles).encode("utf-8"))
            return

        if path == "/api/status":
            data = get_instance_data("cisd")
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        if path == "/api/trades":
            data = get_instance_data("cisd")["closed_trades"]
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        for model_name in STRATEGY_INSTANCES.keys():
            if path == f"/api/challenges/{model_name}":
                data = get_instance_data(model_name)
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))
                return

        self.send_response(404)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": "Not Found"}).encode("utf-8"))

    def log_message(self, format, *args):
        # Silence console log spam for heartbeat polling
        pass


def run_server(port: int = PORT):
    server_address = ("", port)
    httpd = HTTPServer(server_address, ChallengeAPIHandler)
    print(f"[API] Challenge Fleet REST API running at http://localhost:{port}/api/challenges")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
