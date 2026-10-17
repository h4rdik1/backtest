# Visual Chart Snapshot Pipeline (SMC / FVG Edition)

## 1. Overview
This pipeline automatically generates high-resolution, annotated candlestick chart snapshots whenever a valid Smart Money Concepts (SMC) setup is detected by `live_scanner.py` or executed by `challenge_trader.py`.

The system is decoupled from the trading loop: setup detection and order routing complete in sub-millisecond time, while an isolated background thread handles the chart rendering using `mplfinance` and Matplotlib's headless `"Agg"` engine.

---

## 2. Key Architecture & Design Choices

### Non-Blocking Producer / Consumer Queue
* **Producer:** `live_scanner.py` or `challenge_trader.py` calls `snapshot_trade_setup(df_ltf, sig)`. This executes a non-blocking `queue.put_nowait()` and returns in `< 0.05ms`.
* **Consumer:** `core/chart_worker.py` runs a single dedicated background daemon worker thread (`ChartSnapshotWorker`) that pulls tasks from the queue and saves PNGs sequentially.
* **Backpressure Protection:** The queue has a finite capacity (`maxsize=50`). If charts back up during rapid market volatility, excess requests are safely dropped with a warning rather than stalling trading execution.

### Headless & Memory-Safe Rendering
* **`matplotlib.use("Agg")`:** Explicitly set before any `pyplot` imports to prevent GUI-window initialization errors on headless VPS servers or Windows task environments.
* **Guaranteed Cleanup:** Matplotlib does not garbage-collect figures automatically. Every render cycle closes figures inside a `finally: plt.close(fig)` block to prevent memory leaks over weeks of uptime.
* **X-Axis Coordinate Mapping:** Because `mplfinance` plots candles on integer categorical coordinates ($0, 1, 2, \dots, N-1$), the worker maps the setup's `anchor_time` (when the FVG formed) to the integer index of the sliced DataFrame.
* **Right-Edge Margin Padding:** `ax.set_xlim(right=n_bars + 11)` creates dedicated margin space on the right side of the chart so Entry, SL, and TP badges are never clipped or occluded by candles.

---

## 3. Visual Layout

Each generated chart includes:
1. **Dark Theme:** Binance/TradingView dark style (`#131722` background, clean `#26a69a` / `#ef5350` candles, subtle gridlines).
2. **Shaded FVG Box (Rectangle):**
   - **Bullish FVG:** Cyan/Teal translucent box (`alpha=0.22`) with dashed outline.
   - **Bearish FVG:** Purple translucent box (`alpha=0.22`) with dashed outline.
   - Spans from the candle that created the FVG to the current bar + margin.
   - Header tag displaying `FVG [Low - High]` price zone.
3. **Trade Levels (Dashed Lines & Badges):**
   - **Entry:** Light blue dash-dot line (`#29b6f6`) with an `"ENTRY <price>"` tag.
   - **Stop Loss:** Crimson red dotted line (`#ef5350`) with an `"SL <price> (<sl_type>)"` tag.
   - **Take Profit:** Emerald green dotted line (`#26a69a`) with a `"TP (2R) <price>"` tag.
4. **Header Banner:**
   - Displays Symbol, Direction (LONG/SHORT), Timeframe Chain (e.g. `INTRADAY`), Setup Tier (`[A++]`, `[A]`, or `[B]`), Context Regime (e.g. `Strong Bull`), and MTF Bias Reason.

---

## 4. Configuration & Usage

### Global Config Toggle (`core/config.py`)
```python
ENABLE_CHART_SNAPSHOTS = True       # Set to False to disable all chart generation
CHART_OUTPUT_DIR       = "images/signals" # Destination directory for PNG files
CHART_LOOKBACK_BARS    = 120        # Number of recent candlesticks plotted on the chart
```

### Live Scanner CLI Flags (`live_scanner.py`)
You can override the snapshot setting on startup:
```bash
# Run scanner with chart generation (default based on config):
python live_scanner.py

# Force chart generation on:
python live_scanner.py --charts

# Disable chart generation:
python live_scanner.py --no-charts
```

### De-duplication
`live_scanner.py` tracks `seen_chart_keys` using the unique `signal_key` (`{chain}|{symbol}|{low}|{high}`). If a signal remains active across multiple 15-minute scan cycles, it only renders the snapshot once upon initial detection rather than flooding your disk every scan.

---

## 5. Files Modified / Created

| File | Status | Description |
| :--- | :--- | :--- |
| `core/config.py` | Modified | Added `ENABLE_CHART_SNAPSHOTS`, `CHART_OUTPUT_DIR`, and `CHART_LOOKBACK_BARS`. |
| `core/chart_worker.py` | **Created** | Background daemon thread, queue worker, and `mplfinance` FVG/Trade level renderer. |
| `challenge_trader.py` | Modified | Added `anchor_time` and `df_ltf` to `scan_for_signal()`; triggered snapshot in `open_trade()`. |
| `live_scanner.py` | Modified | Added `--charts` / `--no-charts` CLI args and automatic snapshot dispatch for detected setups. |
| `notes/CHART_SNAPSHOT_PIPELINE.md` | **Created** | Comprehensive documentation for the charting pipeline. |
