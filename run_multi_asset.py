# ==============================================================
# run_multi_asset.py — Multi-symbol backtester with full analysis
# ==============================================================

import pandas as pd
import os
import argparse
from typing import List, Dict

import models.backtest_cisd as cisd
import models.backtest_limit as limit
from core.config import (
    SYMBOLS, DAYS_BACK, TRAIN_END, TEST_START,
    MIN_TRADES_FOR_VALIDITY
)
from core.analysis import (
    print_enhanced_stats, walk_forward_split, print_walk_forward,
    filter_contribution_analysis, print_filter_contribution
)


def get_metrics(trades: List[Dict], label: str) -> Dict:
    """Calculate summary metrics for a list of trades."""
    if not trades:
        return {"Model": label, "Trades": 0, "WR%": 0, "PF": 0, "PnL": 0}
    df = pd.DataFrame(trades)
    wins = (df["result"] == "WIN").sum()
    losses = (df["result"] == "LOSS").sum()
    wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    gross_w = df[df["result"] == "WIN"]["pnl"].sum()
    gross_l = abs(df[df["result"] == "LOSS"]["pnl"].sum())
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    return {
        "Model": label, "Trades": len(df), "WR%": round(wr, 1),
        "PF": round(pf, 2), "PnL": round(df["pnl"].sum(), 2)
    }


def run_multi_asset():
    parser = argparse.ArgumentParser(description="Multi-Asset Backtest Runner")
    parser.add_argument("--alignment", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override symbol list (default: config.py SYMBOLS)")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward train/test validation")
    parser.add_argument("--filter-analysis", action="store_true",
                        help="Run filter contribution analysis")
    args = parser.parse_args()

    symbols = args.symbols or SYMBOLS

    # Determine timeframes from alignment
    if args.alignment == "daily":
        tf_context, tf_bias, tf_ltf = "1d", "1h", "5m"
    else:
        tf_context, tf_bias, tf_ltf = "1w", "4h", "15m"

    print("=" * 70)
    print(f"  MULTI-ASSET BACKTEST — {len(symbols)} symbols")
    print(f"  Alignment: {args.alignment.upper()}")
    print(f"  Symbols: {', '.join(symbols)}")
    print("=" * 70)

    summary_rows = []            # for the combined table
    all_cisd_trades = []         # combined across all symbols
    all_limit_trades = []
    wf_results = []              # walk-forward results per symbol

    for sym in symbols:
        print("\n" + "-" * 70)
        print(f"  Loading {sym}...")
        print("-" * 70)

        try:
            df_htf = cisd.fetch_ohlcv_full(sym, tf_context, DAYS_BACK)
            df_bias = cisd.fetch_ohlcv_full(sym, tf_bias, DAYS_BACK)
            df_ltf = cisd.fetch_ohlcv_full(sym, tf_ltf, DAYS_BACK)
        except FileNotFoundError as e:
            print(f"  [SKIP] {sym}: {e}")
            continue

        # Load daily for regime classification
        try:
            df_daily = cisd.fetch_ohlcv_full(sym, "1d", DAYS_BACK)
        except FileNotFoundError:
            df_daily = None

        # ── Run CISD Model ──
        res_cisd = cisd.simulate_trades(df_ltf, df_bias, df_htf,
                                         symbol=sym, df_daily=df_daily)
        print_enhanced_stats(res_cisd["trades"], res_cisd["funnel"],
                            f"CISD — {sym}", run_mc=False)
        all_cisd_trades.extend(res_cisd["trades"])
        summary_rows.append({"Symbol": sym, **get_metrics(res_cisd["trades"], "CISD")})

        # ── Run Limit Model ──
        res_limit = limit.simulate_trades(df_ltf, df_bias, df_htf,
                                           symbol=sym, df_daily=df_daily)
        print_enhanced_stats(res_limit["trades"], res_limit["funnel"],
                            f"LIMIT — {sym}", run_mc=False)
        all_limit_trades.extend(res_limit["trades"])
        summary_rows.append({"Symbol": sym, **get_metrics(res_limit["trades"], "Limit")})

        # ── Walk-Forward (per symbol) ──
        if args.walk_forward:
            wf_cisd = walk_forward_split(df_ltf, df_bias, df_htf,
                                          cisd.simulate_trades,
                                          TRAIN_END, TEST_START, symbol=sym)
            print_walk_forward(wf_cisd, f"({sym} CISD)")

            wf_limit = walk_forward_split(df_ltf, df_bias, df_htf,
                                           limit.simulate_trades,
                                           TRAIN_END, TEST_START, symbol=sym)
            print_walk_forward(wf_limit, f"({sym} Limit)")
            wf_results.append({"symbol": sym, "cisd": wf_cisd, "limit": wf_limit})

    # ══════════════════════════════════════════════════════════
    # COMBINED SUMMARY TABLE
    # ══════════════════════════════════════════════════════════
    summary_rows.append({"Symbol": "COMBINED", **get_metrics(all_cisd_trades, "CISD")})
    summary_rows.append({"Symbol": "COMBINED", **get_metrics(all_limit_trades, "Limit")})

    print("\n" + "=" * 70)
    print("  COMBINED SUMMARY TABLE")
    print("=" * 70)
    print(f"  {'Symbol':<12} {'Model':<8} {'Trades':>7} {'WR%':>7} {'PF':>7} {'PnL':>10}")
    print("  " + "-" * 55)
    for r in summary_rows:
        pnl_str = f"${r['PnL']:>8.2f}"
        print(f"  {r['Symbol']:<12} {r['Model']:<8} {r['Trades']:>7} {r['WR%']:>6.1f}% {r['PF']:>7.2f} {pnl_str}")

    # Sample size verdict
    cisd_count = get_metrics(all_cisd_trades, "CISD")["Trades"]
    limit_count = get_metrics(all_limit_trades, "Limit")["Trades"]
    print(f"\n  Sample size: CISD={cisd_count} Limit={limit_count}  (target: {MIN_TRADES_FOR_VALIDITY})")
    if cisd_count >= MIN_TRADES_FOR_VALIDITY:
        print("  CISD sample size: [VALID]")
    else:
        print(f"  CISD sample size: [LOW] (need {MIN_TRADES_FOR_VALIDITY})")
    if limit_count >= MIN_TRADES_FOR_VALIDITY:
        print("  Limit sample size: [VALID]")
    else:
        print(f"  Limit sample size: [LOW] (need {MIN_TRADES_FOR_VALIDITY})")

    # ── Combined Monte Carlo ──
    if all_cisd_trades:
        print("\n" + "=" * 70)
        print("  COMBINED MONTE CARLO")
        print("=" * 70)
        print_enhanced_stats(all_cisd_trades, label="COMBINED CISD", run_mc=True)

    if all_limit_trades:
        print_enhanced_stats(all_limit_trades, label="COMBINED LIMIT", run_mc=True)

    # ── Filter Contribution Analysis ──
    if args.filter_analysis:
        print("\n" + "=" * 70)
        print("  FILTER CONTRIBUTION ANALYSIS (using first available symbol)")
        print("=" * 70)

        # Use first symbol that has data
        for sym in symbols:
            try:
                df_htf = cisd.fetch_ohlcv_full(sym, tf_context, DAYS_BACK)
                df_bias = cisd.fetch_ohlcv_full(sym, tf_bias, DAYS_BACK)
                df_ltf = cisd.fetch_ohlcv_full(sym, tf_ltf, DAYS_BACK)
                print(f"\n  [CISD filter analysis on {sym}]")
                fc_cisd = filter_contribution_analysis(df_ltf, df_bias, df_htf,
                                                        cisd.simulate_trades,
                                                        symbol=sym)
                print_filter_contribution(fc_cisd)

                print(f"\n  [LIMIT filter analysis on {sym}]")
                fc_limit = filter_contribution_analysis(df_ltf, df_bias, df_htf,
                                                         limit.simulate_trades,
                                                         symbol=sym)
                print_filter_contribution(fc_limit)
                break
            except FileNotFoundError:
                continue

    # ── Excel Export ──
    os.makedirs("exports", exist_ok=True)
    output_path = f"exports/MultiAsset_Analysis_{args.alignment}.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Summary sheet
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        # CISD trades sheet
        if all_cisd_trades:
            df_cisd = pd.DataFrame(all_cisd_trades)
            export_cols = ["symbol", "model", "Entry Time", "exit_time",
                          "Entry Type", "result", "pnl", "Entry $", "Stop $",
                          "Target $", "regime", "bias_type", "setup_tier", "fvg_size",
                          "duration_bars", "be_triggered", "r_multiple", "train_test"]
            available = [c for c in export_cols if c in df_cisd.columns]
            df_c = df_cisd[available].copy()
            for col in ["Entry Time", "exit_time"]:
                if col in df_c.columns:
                    df_c[col] = pd.to_datetime(df_c[col]).dt.tz_localize(None)
            df_c.to_excel(writer, sheet_name="CISD Trades", index=False)

        # Limit trades sheet
        if all_limit_trades:
            df_lim = pd.DataFrame(all_limit_trades)
            available = [c for c in export_cols if c in df_lim.columns]
            df_l = df_lim[available].copy()
            for col in ["Entry Time", "exit_time"]:
                if col in df_l.columns:
                    df_l[col] = pd.to_datetime(df_l[col]).dt.tz_localize(None)
            df_l.to_excel(writer, sheet_name="Limit Trades", index=False)

    print(f"\n  Excel report saved: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_multi_asset()
