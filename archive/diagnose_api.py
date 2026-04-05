import requests
import time

def check():
    base_url = "http://localhost:8000"
    endpoints = ["/", "/api/status", "/api/trades", "/api/live_candles?symbol=BTC_USDT&tf=1h"]
    
    for ep in endpoints:
        print(f"Checking {ep}...", end=" ", flush=True)
        try:
            r = requests.get(base_url + ep, timeout=35)
            print(f"Status: {r.status_code} | Size: {len(r.content)} bytes")
            if r.status_code == 200:
                print("   [O] Response: ", r.text[:100], "...")
        except Exception as e:
            print(f"FAIL: {e}")

if __name__ == "__main__":
    check()
