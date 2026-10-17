"""
core/chart_worker.py — Background Non-Blocking Chart Snapshot Generator
Renders SMC trade setups (FVGs, SFPs, trade levels) using mplfinance.

Changes (Oct-06):
  - Per-trade output folder:  images/signals/{trade_id}_{symbol}/
  - Multi-TF charts:          htf_context.png, mtf_bias.png, ltf_entry.png
  - SFP annotation:           minimal dashed horizontal line from liquidity level
                              formation (C2 bar) to the sweep bar only — NOT
                              a full-page axhline. Sweep bar gets a triangle marker.
  - Trade levels:             short horizontal segments (±15 bars from current bar)
                              with a small badge on the right edge — not full-width.
  - Memory-safe:              plt.close(fig) guaranteed in a finally block.
"""

import os
import queue
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd

# CRITICAL: Set headless backend BEFORE importing pyplot to avoid GUI crashes/hangs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle, FancyArrowPatch
import mplfinance as mpf

from core.config import (
    ENABLE_CHART_SNAPSHOTS,
    CHART_OUTPUT_DIR,
    CHART_LOOKBACK_BARS,
    format_price,
)


# ─────────────────────────────────────────────────────────────────────────────
# DARK STYLE (shared across all chart renders)
# ─────────────────────────────────────────────────────────────────────────────

def _make_style():
    mc = mpf.make_marketcolors(
        up="#26a69a", down="#ef5350",
        edge="inherit", wick="inherit", volume="in",
    )
    return mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        facecolor="#131722",
        edgecolor="#2a2e39",
        figcolor="#131722",
        gridcolor="#1e222d",
        gridstyle="--",
        y_on_right=True,
        rc={
            "text.color": "#d1d4dc",
            "axes.labelcolor": "#d1d4dc",
            "xtick.color": "#787b86",
            "ytick.color": "#787b86",
            "font.size": 9,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# SFP ANNOTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _draw_sfp(ax, plot_df: pd.DataFrame, sfp_info: Dict, is_long: bool, color: str):
    """
    Draw a minimal SFP annotation:
      1. A short dashed horizontal line at sfp_level, from the C2 bar to the
         sweep bar (not the full chart width).
      2. A triangle marker on the sweep bar pointing in the direction of the
         rejection (up-triangle for bullish, down-triangle for bearish).
      3. A small text label 'SFP' near the sweep wick.

    sfp_info keys:
        c2_idx, sweep_idx, sfp_level  — all relative to the FULL df passed
        to find_latest_sfp.  We map them to the sliced plot_df coordinates.
    """
    if sfp_info is None:
        return

    sfp_level  = sfp_info.get("sfp_level")
    c2_abs     = sfp_info.get("c2_idx")
    sweep_abs  = sfp_info.get("sweep_idx")
    if sfp_level is None or c2_abs is None or sweep_abs is None:
        return

    n = len(plot_df)
    # The plot_df is a tail() slice. The absolute indices in sfp_info are
    # relative to the FULL df; we need to remap to the sliced indices.
    # c2_plot = c2_abs - (full_len - n)  but we only receive plot_df here.
    # We attach the offset via sfp_info["plot_offset"] set by the caller.
    offset    = sfp_info.get("plot_offset", 0)   # = full_len - n
    c2_plot   = c2_abs - offset
    sweep_plot = sweep_abs - offset

    # Clamp to visible range
    if c2_plot < 0 and sweep_plot < 0:
        return  # entirely outside the visible window
    c2_plot    = max(0, c2_plot)
    sweep_plot = min(n - 1, sweep_plot)

    if c2_plot >= sweep_plot:
        return  # degenerate or out of order

    # 1. Horizontal dashed line from C2 to sweep bar
    ax.hlines(
        y=sfp_level,
        xmin=c2_plot,
        xmax=sweep_plot + 0.5,
        colors=color,
        linestyles="--",
        linewidths=1.2,
        alpha=0.80,
    )

    # 2. Triangle marker on the sweep bar
    marker = "^" if is_long else "v"
    y_marker = (
        plot_df["low"].iloc[sweep_plot] * 0.9995
        if is_long
        else plot_df["high"].iloc[sweep_plot] * 1.0005
    )
    ax.plot(
        sweep_plot, y_marker,
        marker=marker, color=color,
        markersize=9, zorder=5, alpha=0.95,
    )

    # 3. Label
    label_y = sfp_level * (0.9998 if is_long else 1.0002)
    ax.text(
        (c2_plot + sweep_plot) / 2,
        label_y,
        "SFP",
        color=color, fontsize=7.5, fontweight="bold",
        va="top" if is_long else "bottom",
        ha="center", alpha=0.90,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TRADE-LEVEL DRAWING (SHORT SEGMENTS, NOT FULL-WIDTH)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_level(ax, n_bars: int, price: float, color: str, label: str,
                bg_color: str, linestyle: str = "--", linewidth: float = 1.3,
                segment_left_bars: int = 30):
    """
    Draw a short horizontal price-level segment (not a full-page axhline).
    The segment spans from  (n_bars - segment_left_bars)  to  (n_bars + 1),
    i.e. it's anchored near the right edge of the visible candles.
    A small badge label is placed just to the right.
    """
    x_start = max(0, n_bars - segment_left_bars)
    x_end   = n_bars + 0.5

    ax.hlines(
        y=price,
        xmin=x_start,
        xmax=x_end,
        colors=color,
        linestyles=linestyle,
        linewidths=linewidth,
        alpha=0.85,
    )

    badge_x = n_bars + 0.8
    ax.text(
        badge_x, price, f" {label} ",
        color="#ffffff", fontsize=8, fontweight="bold",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_color,
                  edgecolor="none", alpha=0.90),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORE CHART RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def _render_chart(
    df: pd.DataFrame,
    sig: Dict,
    save_path: str,
    tf_label: str,
    show_trade_levels: bool = True,
):
    """
    Render a single candlestick chart for one timeframe and save as PNG.

    Args:
        df              : OHLCV DataFrame (already sliced to desired length).
        sig             : Signal dict — carries FVG, SFP, trade levels, etc.
        save_path       : Absolute path to save the PNG.
        tf_label        : Human-readable label, e.g. "LTF Entry (5m)".
        show_trade_levels: Only draw Entry/SL/TP badges on the entry-TF chart.
    """
    fig = None
    try:
        plot_df = df.tail(CHART_LOOKBACK_BARS).copy()
        if not isinstance(plot_df.index, pd.DatetimeIndex):
            plot_df.index = pd.to_datetime(plot_df.index, utc=True)

        if plot_df.empty or len(plot_df) < 3:
            return

        symbol    = sig.get("symbol", "ASSET")
        direction = sig.get("direction", "LONG").upper()
        setup_tier = sig.get("setup_tier", "B")
        regime    = sig.get("regime", "").replace("_", " ").title()
        chain     = sig.get("chain", "").upper()
        is_long   = direction == "LONG"
        dir_color = "#26a69a" if is_long else "#ef5350"

        style = _make_style()

        fig, axlist = mpf.plot(
            plot_df,
            type="candle",
            style=style,
            returnfig=True,
            figsize=(13, 7.5),
            tight_layout=False,
        )
        ax = axlist[0]
        n_bars = len(plot_df)

        # Extend X-axis right (+12 bars) for badge room
        ax.set_xlim(left=-1, right=n_bars + 13)

        # ── FVG Zone (only if price data is close to the FVG) ──────────
        fvg_low  = sig.get("fvg_zone_low")
        fvg_high = sig.get("fvg_zone_high")
        anchor_type = sig.get("anchor_type", "FVG")

        if fvg_low and fvg_high and fvg_high > fvg_low:
            anchor_time = sig.get("anchor_time")
            start_idx = max(0, n_bars - 20)
            if anchor_time:
                try:
                    ts = pd.to_datetime(anchor_time, utc=True)
                    matched = plot_df.index.get_indexer([ts], method="nearest")[0]
                    if matched >= 0:
                        start_idx = matched
                except Exception:
                    pass

            zone_color = "#00bcd4" if is_long else "#e040fb"
            rect_x     = start_idx - 0.4
            rect_width = (n_bars - start_idx) + 1.5
            rect_h     = fvg_high - fvg_low

            fvg_rect = Rectangle(
                (rect_x, fvg_low), rect_width, rect_h,
                linewidth=1.2, edgecolor=zone_color, facecolor=zone_color,
                alpha=0.22, linestyle="--",
            )
            ax.add_patch(fvg_rect)

            ax.text(
                rect_x + 0.5, fvg_high,
                f" {anchor_type} [{format_price(fvg_low)} – {format_price(fvg_high)}]",
                color=zone_color, fontsize=8, fontweight="bold",
                va="bottom", ha="left",
            )

        # ── SFP Annotations ─────────────────────────────────────────────
        # We look for SFP info stored in sig under "htf_sfp_detail" /
        # "ltf_sfp_detail" depending on which chart we're rendering.
        full_df_len = sig.get("_full_df_len", {})
        tf_key_map  = sig.get("_tf_key", "ltf")   # "htf", "mtf", or "ltf"

        sfp_detail_key = f"{tf_key_map}_sfp_detail"
        sfp_info = sig.get(sfp_detail_key)

        if sfp_info:
            # Attach the offset so _draw_sfp can map absolute → plot indices
            full_len_for_tf = full_df_len.get(tf_key_map, n_bars)
            sfp_info = dict(sfp_info)  # shallow copy so we don't mutate
            sfp_info["plot_offset"] = full_len_for_tf - n_bars
            sfp_color = "#80cbc4" if is_long else "#ce93d8"
            _draw_sfp(ax, plot_df, sfp_info, is_long, sfp_color)

        # ── Trade Levels (Entry / SL / TP) — entry TF only ──────────────
        if show_trade_levels:
            entry = sig.get("entry_price")
            sl    = sig.get("sl_price")
            tp    = sig.get("tp_price")
            sl_type = sig.get("sl_type", "structural")

            if entry is not None:
                _draw_level(ax, n_bars, entry,
                            "#29b6f6", f"ENTRY  {format_price(entry)}",
                            "#0288d1", linestyle="-.", linewidth=1.4)

            if tp is not None:
                _draw_level(ax, n_bars, tp,
                            "#26a69a", f"TP (2R)  {format_price(tp)}",
                            "#1b5e20", linestyle=":", linewidth=1.3)

            if sl is not None:
                _draw_level(ax, n_bars, sl,
                            "#ef5350", f"SL  {format_price(sl)}  ({sl_type})",
                            "#b71c1c", linestyle=":", linewidth=1.3)

        # ── Header ───────────────────────────────────────────────────────
        tier_badge = f"[{setup_tier}]"
        title_line = (
            f"{symbol}  {direction}  {chain}  {tier_badge}  "
            f"|  Regime: {regime}  |  {tf_label}"
        )
        bias_line = f"Bias: {sig.get('bias_reason', 'N/A')}"

        fig.text(0.08, 0.945, title_line, fontsize=11.5, fontweight="bold", color=dir_color)
        fig.text(0.08, 0.910, bias_line, fontsize=8.5, color="#90a4ae")

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=130,
                    facecolor=fig.get_facecolor())

    except Exception as e:
        print(f"   [CHART ERROR] {save_path}: {e}")
    finally:
        if fig is not None:
            plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TF ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _render_trade_folder(
    df_htf: pd.DataFrame,
    df_mtf: pd.DataFrame,
    df_ltf: pd.DataFrame,
    sig: Dict,
    folder: str,
    tf_names: Tuple[str, str, str],
):
    """
    Render three charts (HTF / MTF / LTF) into a per-trade folder.

    Each chart:
      - htf_context.png  — big picture structure (weekly / daily)
      - mtf_bias.png     — bias timeframe  (4H / 1H)   + SFP if HTF SFP was used
      - ltf_entry.png    — entry timeframe (15m / 5m)  + SFP + trade levels
    """
    tf_htf, tf_mtf, tf_ltf = tf_names

    # Store df lengths so SFP offset mapping works correctly
    sig = dict(sig)  # shallow copy — don't mutate caller's dict
    sig["_full_df_len"] = {
        "htf": len(df_htf),
        "mtf": len(df_mtf),
        "ltf": len(df_ltf),
    }

    # HTF context chart — show structure only, no trade levels, no SFP overlay
    sig["_tf_key"] = "htf"
    _render_chart(
        df_htf, sig,
        save_path=os.path.join(folder, "htf_context.png"),
        tf_label=f"HTF Context ({tf_htf})",
        show_trade_levels=False,
    )

    # MTF bias chart — show HTF SFP if that's what drove the SL anchor
    sig["_tf_key"] = "mtf"
    _render_chart(
        df_mtf, sig,
        save_path=os.path.join(folder, "mtf_bias.png"),
        tf_label=f"MTF Bias ({tf_mtf})",
        show_trade_levels=False,
    )

    # LTF entry chart — FVG zone + LTF SFP + Entry/SL/TP levels
    sig["_tf_key"] = "ltf"
    _render_chart(
        df_ltf, sig,
        save_path=os.path.join(folder, "ltf_entry.png"),
        tf_label=f"LTF Entry ({tf_ltf})",
        show_trade_levels=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND WORKER
# ─────────────────────────────────────────────────────────────────────────────

class ChartSnapshotWorker:
    """
    Background worker that receives chart rendering requests via a Queue
    and processes them sequentially on a single dedicated daemon thread.

    Task payload:  (df_htf, df_mtf, df_ltf, sig, folder, tf_names)
    """
    def __init__(self, maxsize: int = 50):
        self._queue  = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="ChartSnapshotWorker"
        )
        self._thread.start()

    def _worker_loop(self):
        while True:
            try:
                task = self._queue.get()
                if task is None:
                    break
                df_htf, df_mtf, df_ltf, sig, folder, tf_names = task
                _render_trade_folder(df_htf, df_mtf, df_ltf, sig, folder, tf_names)
            except Exception as e:
                print(f"   [CHART WORKER ERROR] {e}")
            finally:
                self._queue.task_done()

    def submit(
        self,
        df_htf: pd.DataFrame,
        df_mtf: pd.DataFrame,
        df_ltf: pd.DataFrame,
        sig: Dict,
        folder: Optional[str] = None,
        tf_names: Tuple[str, str, str] = ("HTF", "MTF", "LTF"),
    ):
        """
        Submit a multi-TF chart-snapshot job.  Non-blocking — returns in <0.1ms.

        Args:
            df_htf / df_mtf / df_ltf : The three DataFrames for the chain.
            sig                      : Signal dict (will be shallow-copied).
            folder                   : Output folder path. Auto-built if None.
            tf_names                 : (htf_tf, mtf_tf, ltf_tf) label strings.
        """
        if not ENABLE_CHART_SNAPSHOTS:
            return
        if df_ltf is None or df_ltf.empty or not sig:
            return

        if not folder:
            sym_clean  = sig.get("symbol", "ASSET").replace("/", "_")
            chain      = sig.get("chain", "chain")
            direction  = sig.get("direction", "SIGNAL").lower()
            ts_str     = datetime.now().strftime("%Y%m%d_%H%M%S")
            trade_id   = sig.get("trade_id", "live")
            subfolder  = f"{trade_id}_{sym_clean}_{chain}_{direction}_{ts_str}"
            folder     = os.path.join(CHART_OUTPUT_DIR, subfolder)

        # Use empty DataFrame placeholders if HTF/MTF weren't provided
        if df_htf is None or df_htf.empty:
            df_htf = df_ltf
        if df_mtf is None or df_mtf.empty:
            df_mtf = df_ltf

        try:
            self._queue.put_nowait((
                df_htf.copy(), df_mtf.copy(), df_ltf.copy(),
                sig.copy(), folder, tf_names,
            ))
        except queue.Full:
            print("   [CHART WORKER] Queue full — dropping snapshot to protect throughput.")


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SINGLETON + CONVENIENCE API
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_WORKER: Optional[ChartSnapshotWorker] = None


def get_chart_worker() -> ChartSnapshotWorker:
    """Returns (or initialises) the global singleton ChartSnapshotWorker."""
    global _GLOBAL_WORKER
    if _GLOBAL_WORKER is None:
        _GLOBAL_WORKER = ChartSnapshotWorker()
    return _GLOBAL_WORKER


def snapshot_trade_setup(
    df_ltf: pd.DataFrame,
    sig: Dict,
    df_htf: Optional[pd.DataFrame] = None,
    df_mtf: Optional[pd.DataFrame] = None,
    folder: Optional[str] = None,
    tf_names: Tuple[str, str, str] = ("HTF", "MTF", "LTF"),
):
    """
    Fire-and-forget multi-TF chart snapshot.
    Safe to call directly from live_scanner or trader loops.

    Backwards-compatible: if only df_ltf + sig are given it still works —
    the HTF and MTF charts will just mirror the LTF data.
    """
    if ENABLE_CHART_SNAPSHOTS:
        worker = get_chart_worker()
        worker.submit(df_htf, df_mtf, df_ltf, sig, folder, tf_names)
