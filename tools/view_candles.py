"""SMC Challenge Dashboard – lightweight HTTP server."""
import http.server
import json
import os
import socketserver
import sys
import threading
import time
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import pandas as pd

# ── path bootstrap ────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import models.backtest_cisd as cisd  # noqa: E402

# ── config ────────────────────────────────────────────────────────────────────
PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(ROOT_DIR, "exports", "challenge_state.json")
TRADES_FILE = os.path.join(ROOT_DIR, "exports", "challenge_trades.csv")

CACHE: dict = {}
CACHE_TTL = 60  # seconds

STATE_LOCK = threading.Lock()

_EMPTY_STATE = {
    "balance": 5000,
    "status": "INACTIVE",
    "open_trades": [],
    "pending_trades": [],
    "closed_trades": [],
}


# ── state helpers ─────────────────────────────────────────────────────────────
def read_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return dict(_EMPTY_STATE)
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def write_state(data: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── data helpers ──────────────────────────────────────────────────────────────
def _df_to_candles(df) -> list:
    """Convert an OHLCV DataFrame to a list of chart-ready dicts."""
    return [
        {
            "time": int(idx.timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        for idx, row in df.iterrows()
    ]


def _df_to_overlays(df) -> dict:
    """Compute FVG and OB overlays from an OHLCV DataFrame."""
    overlays: dict = {"fvgs": [], "obs": []}
    if df.empty:
        return overlays

    def _ts(idx):
        return int(idx.timestamp()) if hasattr(idx, "timestamp") else int(idx[0].timestamp())

    fvgs = cisd.calc_fvgs(df)
    if not fvgs.empty:
        for idx, row in fvgs.iterrows():
            overlays["fvgs"].append(
                {"start": _ts(idx), "top": float(row["top"]), "btm": float(row["btm"]), "type": str(row["type"])}
            )

    obs = cisd.calc_obs(df)
    if not obs.empty:
        for idx, row in obs.iterrows():
            overlays["obs"].append(
                {"start": _ts(idx), "top": float(row["top"]), "btm": float(row["btm"]), "type": str(row["type"])}
            )

    return overlays


def _calc_trade_pnl(trade: dict, exit_price: float) -> tuple[float, float]:
    """Return (remaining_pnl, total_pnl) for a closing trade."""
    entry = trade.get("entry_price", 0)
    tp = trade.get("tp_price", 0)
    initial_sl = trade.get("initial_sl_price")

    if initial_sl is None and tp and entry:
        risk_dist = abs(tp - entry) / 2.0
        initial_sl = entry - risk_dist if trade.get("direction") == "LONG" else entry + risk_dist

    sl_dist = abs(entry - initial_sl) if initial_sl is not None else abs(entry - trade.get("sl_price", 0))
    if not sl_dist:
        sl_dist = entry * 0.005

    pnl_r = (exit_price - entry) / sl_dist if trade.get("direction") == "LONG" else (entry - exit_price) / sl_dist

    risk = trade.get("trade_risk", 35.0)
    reward = trade.get("trade_reward", 70.0)
    partial_exit_done = trade.get("partial_exit_done", False)

    partial_pnl = (reward * 0.50) if partial_exit_done else 0.0
    position_factor = 0.50 if partial_exit_done else 1.0
    remaining_pnl = round(pnl_r * risk * position_factor, 2)
    return remaining_pnl, partial_pnl + remaining_pnl


# ── server ────────────────────────────────────────────────────────────────────
_DEAD_SOCKET = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        exc_type, _, _ = sys.exc_info()
        if exc_type in _DEAD_SOCKET:
            return  # suppress WinError 10053 / ECONNRESET tracebacks
        super().handle_error(request, client_address)


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    # ── utils ─────────────────────────────────────────────────────────────────
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()

    def send_json(self, data, status: int = 200):
        try:
            resp = json.dumps(data, default=str).encode()
            self.send_response(status)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(resp)
        except _DEAD_SOCKET:
            pass

    def _parse_qs(self) -> dict:
        return parse_qs(urlparse(self.path).query)

    def _cached(self, key: str):
        entry = CACHE.get(key)
        if entry and (datetime.now() - entry["time"]).total_seconds() < CACHE_TTL:
            return entry["data"]
        return None

    def _cache(self, key: str, data):
        CACHE[key] = {"time": datetime.now(), "data": data}

    def _read_payload(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(body)
        except Exception:
            return {}

    # ── OPTIONS ───────────────────────────────────────────────────────────────
    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    # ── POST dispatch ─────────────────────────────────────────────────────────
    def do_POST(self):
        payload = self._read_payload()
        routes = {
            "/api/trade/toggle": self._post_trade_toggle,
            "/api/trade/close": self._post_trade_close,
            "/api/trade/add": self._post_trade_add,
        }
        handler = routes.get(self.path)
        if handler:
            handler(payload)
        else:
            self.send_json({"ok": False, "error": "Unknown endpoint"}, 404)

    def _post_trade_toggle(self, payload: dict):
        trade_id = payload.get("trade_id")
        if not trade_id:
            self.send_json({"ok": False, "error": "Missing trade_id"}, 400)
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
                self.send_json({"ok": False, "error": f"Trade {trade_id} not found"}, 404)
                return
            write_state(state)
        self.send_json({"ok": True})

    def _post_trade_close(self, payload: dict):
        trade_id = payload.get("trade_id")
        exit_price_raw = payload.get("exit_price")
        if not trade_id or exit_price_raw is None:
            self.send_json({"ok": False, "error": "Missing trade_id or exit_price"}, 400)
            return
        try:
            exit_price = float(exit_price_raw)
        except Exception:
            self.send_json({"ok": False, "error": "Invalid exit_price"}, 400)
            return

        with STATE_LOCK:
            state = read_state()
            trade = next((t for t in state.get("open_trades", []) if t.get("trade_id") == trade_id), None)
            if trade is None:
                self.send_json({"ok": False, "error": f"Open trade {trade_id} not found"}, 404)
                return

            state["open_trades"] = [t for t in state["open_trades"] if t.get("trade_id") != trade_id]
            remaining_pnl, total_pnl = _calc_trade_pnl(trade, exit_price)

            trade.update(
                exit_price=exit_price,
                closed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                pnl_usd=round(total_pnl, 2),
                result="WIN" if total_pnl > 0.01 else ("LOSS" if total_pnl < -0.01 else "BE"),
                balance_after=round(state.get("balance", 5000) + remaining_pnl, 2),
            )
            state["balance"] = trade["balance_after"]
            state.setdefault("wins" if total_pnl > 0.01 else "losses", 0)
            if total_pnl > 0.01:
                state["wins"] = state.get("wins", 0) + 1
            elif total_pnl < -0.01:
                state["losses"] = state.get("losses", 0) + 1
            state.setdefault("closed_trades", []).append(trade)
            state["total_trades"] = state.get("total_trades", 0) + 1
            write_state(state)

        self.send_json({"ok": True, "pnl_usd": round(total_pnl, 2)})

    def _post_trade_add(self, payload: dict):
        required = ["symbol", "direction", "entry_price", "sl_price", "tp_price"]
        for field in required:
            if field not in payload:
                self.send_json({"ok": False, "error": f"Missing field: {field}"}, 400)
                return

        risk = float(payload.get("trade_risk", 35.0))
        with STATE_LOCK:
            state = read_state()
            next_id = state.get("next_trade_id", 1)
            trade_id = f"T{next_id:04d}"
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
                "trade_reward": risk * 2.0,
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
            state.setdefault("open_trades", []).append(new_trade)
            state["next_trade_id"] = next_id + 1
            write_state(state)

        self.send_json({"ok": True, "trade_id": trade_id})

    # ── GET dispatch ──────────────────────────────────────────────────────────
    def do_GET(self):
        if self.path == "/api/status":
            return self._get_status()
        if self.path == "/api/trades":
            return self._get_trades()
        if self.path.startswith("/api/live_candles"):
            return self._get_live_candles()
        if self.path.startswith("/api/overlays"):
            return self._get_overlays()
        return super().do_GET()

    def _get_status(self):
        data = dict(_EMPTY_STATE)
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f" [/api/status] {e}")
        self.send_json(data)

    def _get_trades(self):
        trades = []
        if os.path.exists(TRADES_FILE):
            try:
                df = pd.read_csv(TRADES_FILE).fillna("")
                trades = df.to_dict("records")
            except Exception as e:
                print(f" [/api/trades] {e}")
        self.send_json(trades)

    def _get_live_candles(self):
        p = self._parse_qs()
        symbol = p.get("symbol", ["BTC/USDT"])[0].replace("_", "/")
        tf = p.get("tf", ["15m"])[0]
        key = f"candles_{symbol}_{tf}"

        cached = self._cached(key)
        if cached is not None:
            return self.send_json(cached)

        print(f" [API] Fetching candles: {symbol} {tf}...", end="", flush=True)
        try:
            df = cisd.fetch_live_ohlcv(symbol, tf, limit=150)
            candles = _df_to_candles(df) if not df.empty else []
            self._cache(key, candles)
            self.send_json(candles)
            print(" OK")
        except _DEAD_SOCKET:
            print(" DISCONNECTED")
        except Exception as e:
            print(f" FAIL: {e}")
            self.send_json([])

    def _get_overlays(self):
        p = self._parse_qs()
        symbol = p.get("symbol", ["BTC/USDT"])[0].replace("_", "/")
        tf = p.get("tf", ["15m"])[0]
        key = f"overlays_{symbol}_{tf}"

        cached = self._cached(key)
        if cached is not None:
            return self.send_json(cached)

        print(f" [API] Fetching overlays: {symbol} {tf}...", end="", flush=True)
        try:
            df = cisd.fetch_live_ohlcv(symbol, tf, limit=100)
            overlays = _df_to_overlays(df)
            self._cache(key, overlays)
            self.send_json(overlays)
            print(" OK")
        except _DEAD_SOCKET:
            print(" DISCONNECTED")
        except Exception as e:
            print(f" FAIL: {e}")
            self.send_json({"fvgs": [], "obs": []})


# ── entry point ───────────────────────────────────────────────────────────────
def run_viewer():
    os.chdir(DIRECTORY)
    url = f"http://localhost:{PORT}/view_candles.html"
    print(f"\n  [SUCCESS] SMC Challenge Dashboard started at {url}")
    print("  [EXIT] Press Ctrl+C to stop.")
    try:
        with ThreadedHTTPServer(("", PORT), ViewerHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  [EXIT] Dashboard server stopped.")


if __name__ == "__main__":
    run_viewer()
