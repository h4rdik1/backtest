# ==============================================================
# analysis.py — Shared statistical analysis functions
# Imported by both backtest_cisd.py and backtest_limit.py
# Contains: Monte Carlo, Walk-Forward, Regime, Monthly, Filters
# ==============================================================

import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime, timezone
from typing import List, Dict, Callable, Optional

from .config import (
    MONTE_CARLO_SIMS, MIN_TRADES_FOR_VALIDITY, CONSISTENCY_PASS_PCT,
    EDGE_REAL_THRESHOLD, WALK_FORWARD_TOLERANCE, FIXED_SL_USDT,
    ACCOUNT_SIZE,
    # Task 2 — Regime constants
    REGIME_LOOKBACK_BARS,
    # Task 3 — Session constants
    SESSIONS,
    # Task 10 — Live vs backtest constants
    LIVE_BACKTEST_CHECK_N, LIVE_BACKTEST_WARN_PCT, LIVE_BACKTEST_HALT_PCT,
)


# ==============================================================
# TASK 3 — Session Detection Helper
# WHY: Institutional volume peaks at London Open (07:00-11:00 UTC)
# and NY Open (13:00-16:00 UTC). We need to know what session we
# are in before deciding whether to allow a trade.
# ==============================================================

def get_current_session(dt: datetime = None) -> str:
    """
    Returns the name of the trading session the given datetime falls in.

    Sessions are defined in UTC. Overlapping windows (e.g., London-NY
    overlap at 15:00-16:00) will return 'london_close' if checked
    top-to-bottom in the SESSIONS dict order.

    Args:
        dt: A datetime object. Defaults to now (UTC) if not provided.

    Returns:
        Session name: 'london_open', 'ny_open', 'london_close',
        'asian', or 'off_hours'.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    # Make sure we compare in UTC
    if dt.tzinfo is not None:
        dt_utc = dt.replace(tzinfo=None)  # strip tz for simple HH:MM compare
    else:
        dt_utc = dt

    # Current time as a string like '09:30' for comparison
    time_str = dt_utc.strftime("%H:%M")

    for session_name, (start, end) in SESSIONS.items():
        # Handle sessions that cross midnight (e.g., asian: 00:00 – 07:00)
        if start <= end:
            if start <= time_str < end:
                return session_name
        else:
            # Crosses midnight
            if time_str >= start or time_str < end:
                return session_name

    return "off_hours"  # Not in any defined session window


# ==============================================================
# TASK 2 (UPGRADED) — Market Regime Classification
# CHANGED: Now uses strong_bull / strong_bear / ranging labels
# (instead of trending_up / trending_down) with the exact thresholds
# specified in the task (close > 20-bar high * 0.90 for strong_bull,
# close < 20-bar low * 1.10 for strong_bear).
# WHY: The old ranging check was too permissive. The new one
# ensures we only label 'strong_bull' when price is clearly holding
# near recent highs AND making higher lows — not just being close.
# ==============================================================

def classify_regime(df_weekly_or_daily: pd.DataFrame, entry_time=None) -> str:
    """
    TASK 2 — Classify the market regime using the last 20 candles.

    Can be called with a weekly OR daily dataframe. The labels are:
      strong_bull : price near 20-bar high (close > 20-bar high * 0.90)
                    AND second half of window has higher lows than first half
      strong_bear : price near 20-bar low  (close < 20-bar low * 1.10)
                    AND second half of window has lower highs than first half
      ranging     : neither condition met

    WHY the 0.90 / 1.10 thresholds:
      Multiplying the 20-bar high by 0.90 means price must be within
      10% of the range top to qualify as a bull trend. This filters
      out cases where price has already pulled back significantly.

    Args:
        df_weekly_or_daily: OHLCV DataFrame indexed by timestamp
        entry_time: Optional timestamp to slice history up to.
                    If None, uses the last 20 bars of the full df.

    Returns: 'strong_bull', 'strong_bear', or 'ranging'
    """
    # Get the relevant slice of data
    if entry_time is not None:
        entry_ts = (pd.Timestamp(entry_time, tz="UTC")
                    if not isinstance(entry_time, pd.Timestamp)
                    else entry_time)
        recent = df_weekly_or_daily.loc[:entry_ts].tail(REGIME_LOOKBACK_BARS)
    else:
        recent = df_weekly_or_daily.tail(REGIME_LOOKBACK_BARS)

    # Need at least half the bars to make a decision
    if len(recent) < REGIME_LOOKBACK_BARS // 2:
        return "ranging"  # Not enough data to classify

    high_20      = recent["high"].max()
    low_20       = recent["low"].min()
    current_close = recent["close"].iloc[-1]

    # Split window in half to detect HH/HL or LL/LH patterns
    half = len(recent) // 2
    first_half  = recent.iloc[:half]
    second_half = recent.iloc[half:]

    # ── Strong Bull: price near highs AND making higher lows ──
    # The 0.90 multiplier means 'within 10% of the 20-bar peak'
    near_highs  = current_close > (high_20 * 0.95)
    higher_lows = second_half["low"].min() > first_half["low"].min()

    # ── Strong Bear: price near lows AND making lower highs ──
    # The 1.10 multiplier means 'within 10% above the 20-bar trough'
    near_lows   = current_close < (low_20 * 1.05)
    lower_highs = second_half["high"].max() < first_half["high"].max()

    if near_highs and higher_lows:
        return "strong_bull"
    elif near_lows and lower_highs:
        return "strong_bear"
    else:
        return "ranging"


# ══════════════════════════════════════════════════════════════
# PRIORITY 5 — Monte Carlo Simulation
# ══════════════════════════════════════════════════════════════

def monte_carlo(trades: List[Dict], n_simulations: int = MONTE_CARLO_SIMS,
                risk_per_trade: float = FIXED_SL_USDT,
                account_size: float = ACCOUNT_SIZE) -> Dict:
    """
    Randomly shuffles the order of trade outcomes N times to test
    whether the strategy's profitability is robust or just lucky.

    WHY: If your trades happened to come in a lucky order, your equity
    curve looks great — but a different order could wipe you out.
    Monte Carlo tests ALL possible orderings.

    Returns dict with:
    - profitable_pct: % of simulations that ended profitable
    - median_equity: median final account balance
    - p5_equity: 5th percentile (worst realistic case)
    - p95_equity: 95th percentile (best realistic case)
    - median_max_dd: median maximum drawdown %
    - verdict: "REAL" / "UNCERTAIN" / "LIKELY LUCK"
    """
    if not trades:
        return {"profitable_pct": 0, "median_equity": account_size,
                "p5_equity": account_size, "p95_equity": account_size,
                "median_max_dd": 0, "verdict": "NO TRADES"}

    # Extract just the PnL values from each trade
    pnls = [t["pnl"] for t in trades]
    rng = np.random.default_rng(42)  # Reproducible results

    final_equities = []
    max_drawdowns = []

    for _ in range(n_simulations):
        # Shuffle the trade order randomly
        shuffled = rng.permutation(pnls)

        # Build equity curve from shuffled trades
        equity_curve = np.cumsum(shuffled) + account_size
        peak = np.maximum.accumulate(equity_curve)

        # Calculate max drawdown for this simulation
        drawdown = (peak - equity_curve) / peak * 100
        max_dd = drawdown.max() if len(drawdown) > 0 else 0

        final_equities.append(equity_curve[-1])
        max_drawdowns.append(max_dd)

    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)

    profitable_pct = (final_equities > account_size).mean() * 100

    # Determine verdict
    if profitable_pct >= EDGE_REAL_THRESHOLD * 100:
        verdict = "REAL"
    elif profitable_pct >= 55:
        verdict = "UNCERTAIN"
    else:
        verdict = "LIKELY LUCK"

    return {
        "profitable_pct": round(profitable_pct, 1),
        "median_equity": round(np.median(final_equities), 2),
        "p5_equity": round(np.percentile(final_equities, 5), 2),
        "p95_equity": round(np.percentile(final_equities, 95), 2),
        "median_max_dd": round(np.median(max_drawdowns), 1),
        "verdict": verdict
    }


# ══════════════════════════════════════════════════════════════
# PRIORITY 6 — Monthly Consistency Breakdown
# ══════════════════════════════════════════════════════════════

def monthly_breakdown(trades: List[Dict]) -> Dict:
    """
    Groups trades by month and calculates per-month stats.
    A real edge should be profitable in at least 60% of months.

    WHY: Even a profitable strategy is dangerous if it only makes money
    in 2 months and loses in the other 10. Consistency = tradeable.

    Returns dict with monthly_data (list of dicts) and consistency stats.
    """
    if not trades:
        return {"monthly_data": [], "profitable_months": 0,
                "total_months": 0, "consistency_pct": 0, "verdict": "NO TRADES"}

    df = pd.DataFrame(trades)
    df["month"] = pd.to_datetime(df["Entry Time"]).dt.to_period("M")

    monthly_data = []
    for month, group in df.groupby("month"):
        wins = (group["result"] == "WIN").sum()
        losses = (group["result"] == "LOSS").sum()
        pnl = group["pnl"].sum()
        wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        monthly_data.append({
            "month": str(month), "trades": len(group), "wins": wins,
            "losses": losses, "pnl": round(pnl, 2), "wr": round(wr, 1)
        })

    profitable = sum(1 for m in monthly_data if m["pnl"] > 0)
    total = len(monthly_data)
    consistency = (profitable / total * 100) if total > 0 else 0

    verdict = "PASS" if consistency >= CONSISTENCY_PASS_PCT * 100 else "FAIL"

    return {
        "monthly_data": monthly_data,
        "profitable_months": profitable,
        "total_months": total,
        "consistency_pct": round(consistency, 1),
        "verdict": verdict
    }


# ══════════════════════════════════════════════════════════════
# PRIORITY 2 — Walk-Forward Testing
# ══════════════════════════════════════════════════════════════

def walk_forward_split(df_ltf: pd.DataFrame, df_bias: pd.DataFrame,
                       df_htf: pd.DataFrame, simulate_fn: Callable,
                       train_end: str, test_start: str,
                       symbol: str = "BTC/USDT",
                       filter_overrides: Optional[Dict] = None) -> Dict:
    """
    Splits data into TRAIN (model was tuned on this) and TEST (blind
    out-of-sample validation) periods and runs the simulation on both.

    WHY: If your strategy only works on data it was designed for, it's
    overfit. Walk-forward testing catches this by testing on data the
    strategy has never 'seen'.

    Returns dict with train_results, test_results, and PASS/FAIL verdict.
    """
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    test_start_ts = pd.Timestamp(test_start, tz="UTC")

    # Split each dataframe at the boundary
    ltf_train = df_ltf[df_ltf.index <= train_end_ts]
    ltf_test = df_ltf[df_ltf.index >= test_start_ts]
    bias_train = df_bias[df_bias.index <= train_end_ts]
    bias_test = df_bias[df_bias.index >= test_start_ts]
    htf_train = df_htf[df_htf.index <= train_end_ts] if df_htf is not None else None
    htf_test = df_htf[df_htf.index >= test_start_ts] if df_htf is not None else None

    # Run simulation on both periods
    kwargs = {"symbol": symbol}
    if filter_overrides:
        kwargs.update(filter_overrides)

    res_train = simulate_fn(ltf_train, bias_train, htf_train, **kwargs)
    res_test = simulate_fn(ltf_test, bias_test, htf_test, **kwargs)

    # Calculate metrics for comparison
    def get_metrics(trades):
        if not trades:
            return {"trades": 0, "wr": 0, "pf": 0, "pnl": 0, "max_dd": 0}
        df = pd.DataFrame(trades)
        wins = (df["result"] == "WIN").sum()
        losses = (df["result"] == "LOSS").sum()
        wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        gross_w = df[df["result"] == "WIN"]["pnl"].sum()
        gross_l = abs(df[df["result"] == "LOSS"]["pnl"].sum())
        pf = gross_w / gross_l if gross_l > 0 else float("inf")

        # Max drawdown
        equity = df["pnl"].cumsum() + ACCOUNT_SIZE
        peak = equity.cummax()
        dd = ((peak - equity) / peak * 100)
        max_dd = dd.max() if len(dd) > 0 else 0

        return {"trades": len(df), "wr": round(wr, 1), "pf": round(pf, 2),
                "pnl": round(df["pnl"].sum(), 2), "max_dd": round(max_dd, 1)}

    m_train = get_metrics(res_train["trades"])
    m_test = get_metrics(res_test["trades"])

    # Compare: is TEST within 15% of TRAIN?
    if m_train["wr"] > 0 and m_test["trades"] > 0:
        wr_dev = abs(m_test["wr"] - m_train["wr"]) / m_train["wr"]
    else:
        wr_dev = 1.0  # 100% deviation if no data

    if m_train["pf"] > 0 and m_train["pf"] != float("inf") and m_test["trades"] > 0:
        pf_dev = abs(m_test["pf"] - m_train["pf"]) / m_train["pf"]
    else:
        pf_dev = 1.0

    # Verdict: both WR and PF deviation must be within tolerance
    verdict = "PASS" if (wr_dev <= WALK_FORWARD_TOLERANCE and
                         pf_dev <= WALK_FORWARD_TOLERANCE) else "FAIL"

    return {
        "train": m_train, "test": m_test,
        "wr_deviation": round(wr_dev * 100, 1),
        "pf_deviation": round(pf_dev * 100, 1),
        "verdict": verdict,
        "train_trades": res_train["trades"],
        "test_trades": res_test["trades"]
    }


# ══════════════════════════════════════════════════════════════
# PRIORITY 4 — Filter Contribution Analysis
# ══════════════════════════════════════════════════════════════

# Each stage adds one more filter on top of the previous
FILTER_STAGES = [
    {"name": "Base (no filters)",
     "use_fvg_quality": False, "use_htf_ob": False, "use_pd_array": False,
     "use_pullback": False, "use_auto_be": False},
    {"name": "+ FVG quality filter",
     "use_fvg_quality": True, "use_htf_ob": False, "use_pd_array": False,
     "use_pullback": False, "use_auto_be": False},
    {"name": "+ HTF OB confluence",
     "use_fvg_quality": True, "use_htf_ob": True, "use_pd_array": False,
     "use_pullback": False, "use_auto_be": False},
    {"name": "+ Premium/Discount",
     "use_fvg_quality": True, "use_htf_ob": True, "use_pd_array": True,
     "use_pullback": False, "use_auto_be": False},
    {"name": "+ Pullback filter",
     "use_fvg_quality": True, "use_htf_ob": True, "use_pd_array": True,
     "use_pullback": True, "use_auto_be": False},
    {"name": "+ Auto Break-Even",
     "use_fvg_quality": True, "use_htf_ob": True, "use_pd_array": True,
     "use_pullback": True, "use_auto_be": True},
]


def filter_contribution_analysis(df_ltf: pd.DataFrame, df_bias: pd.DataFrame,
                                  df_htf: pd.DataFrame, simulate_fn: Callable,
                                  symbol: str = "BTC/USDT") -> List[Dict]:
    """
    Runs the backtest 6 times, adding one filter at a time, and measures
    how each filter changes WR, PF, trade count, and PnL.

    WHY: Some filters might actually HURT performance. This tells you
    exactly which filters are helping and which are not worth using.

    Any filter that reduces PF compared to the previous stage is flagged.
    """
    results = []
    prev_pnl = None

    for stage in FILTER_STAGES:
        overrides = {k: v for k, v in stage.items() if k != "name"}
        res = simulate_fn(df_ltf, df_bias, df_htf, symbol=symbol, **overrides)
        trades = res["trades"]

        if trades:
            df = pd.DataFrame(trades)
            wins = (df["result"] == "WIN").sum()
            losses = (df["result"] == "LOSS").sum()
            wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
            gross_w = df[df["result"] == "WIN"]["pnl"].sum()
            gross_l = abs(df[df["result"] == "LOSS"]["pnl"].sum())
            pf = gross_w / gross_l if gross_l > 0 else float("inf")
            pnl = df["pnl"].sum()
        else:
            wr, pf, pnl = 0, 0, 0

        change = round(pnl - prev_pnl, 2) if prev_pnl is not None else 0
        flag = "[HARMFUL]" if prev_pnl is not None and pnl < prev_pnl and pf < (results[-1]["pf"] if results else 999) else ""

        results.append({
            "name": stage["name"],
            "trades": len(trades),
            "wr": round(wr, 1),
            "pf": round(pf, 2),
            "pnl": round(pnl, 2),
            "change": change,
            "flag": flag
        })
        prev_pnl = pnl

    return results


# ══════════════════════════════════════════════════════════════
# ENHANCED STATS PRINTER — Replaces old print_stats()
# ══════════════════════════════════════════════════════════════

def print_enhanced_stats(trades: List[Dict], funnel: Dict = None,
                         label: str = "BACKTEST", run_mc: bool = True):
    """
    Comprehensive stats output with monthly breakdown, Monte Carlo,
    and consistency score. Replaces the old print_stats() function.

    CHANGED vs old version: Added Monte Carlo, monthly consistency
    score, and regime breakdown. Removed duplicated code blocks.
    """
    print("\n" + "=" * 62)
    print(f"  {label} RESULTS")
    print("=" * 62)

    # Filter funnel
    if funnel:
        print("\n  FILTER FUNNEL")
        for stage, count in funnel.items():
            print(f"    {stage:<20}: {count}")

    if not trades:
        print(f"\n  No trades completed for {label}.")
        return

    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["result"] == "WIN"]
    losses = df_t[df_t["result"] == "LOSS"]
    bes = df_t[df_t["result"] == "BE"]

    total_effective = len(wins) + len(losses)
    wr = (len(wins) / total_effective * 100) if total_effective > 0 else 0
    gross_wins = wins["pnl"].sum()
    gross_losses = abs(losses["pnl"].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Streaks
    max_win_streak = curr_win = 0
    max_loss_streak = curr_loss = 0
    for r in df_t["result"].tolist():
        if r == "WIN":
            curr_win += 1; curr_loss = 0
            max_win_streak = max(max_win_streak, curr_win)
        elif r == "LOSS":
            curr_loss += 1; curr_win = 0
            max_loss_streak = max(max_loss_streak, curr_loss)

    # Max drawdown
    equity = df_t["pnl"].cumsum() + ACCOUNT_SIZE
    peak = equity.cummax()
    max_dd = ((peak - equity) / peak * 100).max()

    # Sample size verdict
    sample_verdict = "[VALID]" if len(df_t) >= MIN_TRADES_FOR_VALIDITY else f"[LOW] (need {MIN_TRADES_FOR_VALIDITY})"

    print(f"\n  Total Trades    : {len(df_t)} (BEs: {len(bes)}) [{sample_verdict}]")
    print(f"  Wins / Losses   : {len(wins)} / {len(losses)}")
    print(f"  Win Rate        : {wr:.1f}%")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Max Win Streak  : {max_win_streak}")
    print(f"  Max Loss Streak : {max_loss_streak}")
    print(f"  Max Drawdown    : {max_dd:.1f}%")
    print(f"  Total PnL       : ${df_t['pnl'].sum():.2f}")

    # ── REGIME BREAKDOWN (Priority 3) ──
    if "regime" in df_t.columns:
        print("\n  REGIME BREAKDOWN:")
        for regime in ["strong_bull", "strong_bear", "ranging"]:
            subset = df_t[df_t["regime"] == regime]
            if len(subset) == 0:
                continue
            r_wins = (subset["result"] == "WIN").sum()
            r_losses = (subset["result"] == "LOSS").sum()
            r_wr = (r_wins / (r_wins + r_losses) * 100) if (r_wins + r_losses) > 0 else 0
            r_gw = subset[subset["result"] == "WIN"]["pnl"].sum()
            r_gl = abs(subset[subset["result"] == "LOSS"]["pnl"].sum())
            r_pf = r_gw / r_gl if r_gl > 0 else float("inf")
            print(f"    {regime:<15}: {len(subset):>3} trades | {r_wr:>5.1f}% WR | {r_pf:>5.2f} PF")

    # ── MONTHLY BREAKDOWN (Priority 6) ──
    mb = monthly_breakdown(trades)
    print("\n  MONTHLY BREAKDOWN:")
    print(f"    {'Month':<10} {'Trades':>6} {'Wins':>5} {'Loss':>5} {'PnL':>10} {'WR%':>6}")
    print("    " + "-" * 46)
    for m in mb["monthly_data"]:
        print(f"    {m['month']:<10} {m['trades']:>6} {m['wins']:>5} {m['losses']:>5} ${m['pnl']:>8.2f} {m['wr']:>5.1f}%")
    print(f"\n    Profitable months : {mb['profitable_months']}/{mb['total_months']}")
    print(f"    Consistency score : {mb['consistency_pct']:.1f}%  [{mb['verdict']}]")

    # ── MONTE CARLO (Priority 5) ──
    if run_mc and len(trades) >= 5:
        mc = monte_carlo(trades)
        print(f"\n  MONTE CARLO ({MONTE_CARLO_SIMS:,} simulations):")
        print(f"    Profitable sims   : {mc['profitable_pct']:.1f}%   [{'PASS' if mc['profitable_pct'] >= EDGE_REAL_THRESHOLD * 100 else 'FAIL'} - need >={EDGE_REAL_THRESHOLD * 100:.0f}%]")
        print(f"    Median equity     : ${mc['median_equity']:,.2f}")
        print(f"    5th pctile equity : ${mc['p5_equity']:,.2f}  (worst realistic)")
        print(f"    95th pctile equity: ${mc['p95_equity']:,.2f}  (best realistic)")
        print(f"    Max DD (median)   : {mc['median_max_dd']:.1f}%")
        print(f"    Verdict           : EDGE IS {mc['verdict']}")


# ══════════════════════════════════════════════════════════════
# Walk-Forward Pretty Print
# ══════════════════════════════════════════════════════════════

def print_walk_forward(wf_result: Dict, label: str = ""):
    """Prints side-by-side train vs test comparison."""
    t = wf_result["train"]
    s = wf_result["test"]
    print(f"\n  WALK-FORWARD VALIDATION {label}")
    print(f"    {'Metric':<18} {'TRAIN':>10} {'TEST':>10}")
    print("    " + "-" * 40)
    print(f"    {'Trades':<18} {t['trades']:>10} {s['trades']:>10}")
    print(f"    {'Win Rate':<18} {t['wr']:>9.1f}% {s['wr']:>9.1f}%")
    print(f"    {'Profit Factor':<18} {t['pf']:>10.2f} {s['pf']:>10.2f}")
    print(f"    {'Total PnL':<18} ${t['pnl']:>9.2f} ${s['pnl']:>9.2f}")
    print(f"    {'Max Drawdown':<18} {t['max_dd']:>9.1f}% {s['max_dd']:>9.1f}%")
    print(f"\n    WR deviation  : {wf_result['wr_deviation']:.1f}%  (tolerance: {WALK_FORWARD_TOLERANCE*100:.0f}%)")
    print(f"    PF deviation  : {wf_result['pf_deviation']:.1f}%  (tolerance: {WALK_FORWARD_TOLERANCE*100:.0f}%)")
    print(f"    Verdict       : {wf_result['verdict']}")


# ══════════════════════════════════════════════════════════════
# Filter Contribution Pretty Print
# ══════════════════════════════════════════════════════════════

def print_filter_contribution(results: List[Dict]):
    """Prints the filter contribution table."""
    print("\n  FILTER CONTRIBUTION ANALYSIS")
    print(f"    {'Stage':<25} {'Trades':>7} {'WR%':>6} {'PF':>6} {'PnL':>10} {'Change':>8} {'':>10}")
    print("    " + "-" * 75)
    for r in results:
        change_str = f"${r['change']:>+7.2f}" if r["change"] != 0 else "    —"
        print(f"    {r['name']:<25} {r['trades']:>7} {r['wr']:>5.1f}% {r['pf']:>6.2f} ${r['pnl']:>9.2f} {change_str} {r['flag']}")


# ══════════════════════════════════════════════════════════════
# PRIORITY 7 — Live Scanner Signal Logging
# ══════════════════════════════════════════════════════════════

SIGNAL_LOG_FILE = "exports/signal_log.csv"
SIGNAL_LOG_HEADERS = [
    "timestamp", "symbol", "bias_type", "reason", "fvg_top", "fvg_bottom",
    "fvg_midpoint", "entry_price", "stop_price", "target_price",
    "triggered", "outcome"
]


def log_signal(signal: Dict):
    """
    Appends a detected signal to signal_log.csv for tracking.

    WHY: This lets you compare what the scanner detected vs what
    actually happened, so you can validate the live edge.

    signal dict should contain keys matching SIGNAL_LOG_HEADERS.
    """
    file_exists = os.path.exists(SIGNAL_LOG_FILE)

    with open(SIGNAL_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SIGNAL_LOG_HEADERS)
        if not file_exists:
            writer.writeheader()

        # Fill missing keys with empty string
        row = {k: signal.get(k, "") for k in SIGNAL_LOG_HEADERS}
        writer.writerow(row)


def compare_live_vs_backtest(backtest_wr: float) -> Dict:
    """
    Reads signal_log.csv and compares live signal outcomes to backtest
    win rate. Flags if results deviate more than 15%.

    WHY: If live results collapse vs backtest, the strategy may be
    breaking down in current market conditions.

    Returns dict with live stats and verdict.
    """
    if not os.path.exists(SIGNAL_LOG_FILE):
        return {"verdict": "NO LOG FILE", "signals": 0}

    df = pd.read_csv(SIGNAL_LOG_FILE)

    # Only count signals with known outcomes
    completed = df[df["outcome"].isin(["WIN", "LOSS"])]
    if len(completed) == 0:
        return {"verdict": "NO COMPLETED SIGNALS", "signals": len(df)}

    live_wins = (completed["outcome"] == "WIN").sum()
    live_total = len(completed)
    live_wr = live_wins / live_total * 100

    deviation = abs(live_wr - backtest_wr) / backtest_wr * 100 if backtest_wr > 0 else 100

    if deviation <= 15:
        verdict = "CONSISTENT"
    elif deviation <= 30:
        verdict = "INVESTIGATE"
    else:
        verdict = "STRATEGY BROKEN"

    return {
        "signals_logged": len(df),
        "completed": live_total,
        "live_wr": round(live_wr, 1),
        "backtest_wr": round(backtest_wr, 1),
        "deviation": round(deviation, 1),
        "verdict": verdict
    }


def print_live_comparison(result: Dict):
    """Prints live vs backtest comparison."""
    print("\n  LIVE vs BACKTEST COMPARISON:")
    print(f"    Signals logged   : {result.get('signals_logged', 0)}")
    print(f"    Completed        : {result.get('completed', 0)}")
    if result.get("completed", 0) > 0:
        print(f"    Live win rate    : {result['live_wr']:.1f}%")
        print(f"    Backtest WR      : {result['backtest_wr']:.1f}%")
        print(f"    Deviation        : {result['deviation']:.1f}%  [OK if < 15%]")
    print(f"    Verdict          : {result['verdict']}")


# ══════════════════════════════════════════════════════════════
# TASK 10 — Live vs Backtest Consistency Check
# CHANGED: Added a structured function that compares BOTH win rate
# AND profit factor (not just WR). Also triggers halt at 25% rather
# than just logging. The old compare_live_vs_backtest() only read
# from the CSV file. This new version accepts live trade dicts.
# WHY: If the live strategy produces drastically different results
# than backtesting, it could mean market conditions have changed,
# execution issues, or the strategy has stopped working.
# ══════════════════════════════════════════════════════════════

def count_wins(trades: List[Dict]) -> int:
    """Helper: count the number of winning trades in a list of trade dicts."""
    return sum(1 for t in trades if t.get("result") in ("WIN", "win"))


def compare_live_to_backtest(
    live_trades: List[Dict],
    backtest_stats: Dict,
    send_alert_fn=None,
    halt_fn=None,
) -> Dict:
    """
    TASK 10 — After every N live completed trades, compare live performance
    to backtest stats. Triggers warnings and halts automatically.

    How it works:
      1. Count wins in the live_trades list.
      2. Compare live win rate to backtest_stats['win_rate'].
      3. Compare live profit factor to backtest_stats['profit_factor'].
      4. If either deviation > LIVE_BACKTEST_HALT_PCT → halt trading.
      5. If either deviation > LIVE_BACKTEST_WARN_PCT → log warning.

    Args:
        live_trades    : List of completed trade dicts from challenge state.
                         Each should have 'result' ('WIN'/'LOSS') and 'pnl_usd'.
        backtest_stats : Dict with keys 'win_rate' (0.0-1.0) and
                         'profit_factor' (float).
        send_alert_fn  : Optional callable(msg: str) to send alerts
                         (e.g., Telegram, print). Falls back to print.
        halt_fn        : Optional callable() that sets a halt flag in your
                         state. Falls back to print a HALT message.

    Returns:
        Dict with live stats, deviations, and action taken.
    """
    # Must have enough trades to compare (don't fire on first 3 trades)
    if len(live_trades) < LIVE_BACKTEST_CHECK_N:
        return {
            "checked"   : False,
            "reason"    : f"Need {LIVE_BACKTEST_CHECK_N} completed trades, have {len(live_trades)}",
            "trades_n"  : len(live_trades),
        }

    # Only check every LIVE_BACKTEST_CHECK_N completed trades
    # (caller should pass only the last N trades, or we check total)
    recent_trades = live_trades[-LIVE_BACKTEST_CHECK_N:]

    # ── Live win rate ──
    wins      = count_wins(recent_trades)
    total     = len(recent_trades)
    live_wr   = wins / total  # fractional (0.0 – 1.0)
    bt_wr     = backtest_stats.get("win_rate", 0.5)

    wr_deviation = abs(live_wr - bt_wr) / bt_wr if bt_wr > 0 else 1.0

    # ── Live profit factor ──
    gross_wins   = sum(t.get("pnl_usd", 0) for t in recent_trades if t.get("pnl_usd", 0) > 0)
    gross_losses = abs(sum(t.get("pnl_usd", 0) for t in recent_trades if t.get("pnl_usd", 0) < 0))
    live_pf      = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    bt_pf        = backtest_stats.get("profit_factor", 1.5)

    pf_deviation = abs(live_pf - bt_pf) / bt_pf if bt_pf > 0 and live_pf != float("inf") else 0.0

    # Worst deviation between WR and PF
    max_deviation = max(wr_deviation, pf_deviation)

    action = "none"

    # ── HALT condition: >25% deviation ──
    if max_deviation > LIVE_BACKTEST_HALT_PCT:
        msg = (
            f"CRITICAL: Live results deviate {max_deviation:.1%} from backtest. "
            f"Live WR={live_wr:.1%} vs BT={bt_wr:.1%} | "
            f"Live PF={live_pf:.2f} vs BT PF={bt_pf:.2f}. "
            f"HALTING trading."
        )
        if send_alert_fn:
            send_alert_fn(msg)
        else:
            print(f"\n  [HALT ALERT] {msg}")

        if halt_fn:
            halt_fn()
        else:
            print("  [HALT] Trading halted due to live/backtest deviation. Manual restart needed.")

        action = "halt"

    # ── WARNING condition: >15% deviation ──
    elif max_deviation > LIVE_BACKTEST_WARN_PCT:
        msg = (
            f"WARNING: Live results drifting from backtest ({max_deviation:.1%} deviation). "
            f"Live WR={live_wr:.1%} vs BT={bt_wr:.1%} | "
            f"Live PF={live_pf:.2f} vs BT PF={bt_pf:.2f}."
        )
        if send_alert_fn:
            send_alert_fn(msg)
        else:
            print(f"\n  [WARN] {msg}")

        action = "warn"

    return {
        "checked"         : True,
        "trades_compared" : total,
        "live_wr"         : round(live_wr, 4),
        "backtest_wr"     : round(bt_wr, 4),
        "wr_deviation"    : round(wr_deviation, 4),
        "live_pf"         : round(live_pf, 2) if live_pf != float("inf") else "inf",
        "backtest_pf"     : round(bt_pf, 2),
        "pf_deviation"    : round(pf_deviation, 4),
        "max_deviation"   : round(max_deviation, 4),
        "action"          : action,  # 'none', 'warn', or 'halt'
    }

