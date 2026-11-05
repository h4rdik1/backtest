// ─────────────────────────────────────────────────────────────
// src/lib/math.ts — Pure utility functions.
// NO side-effects, NO imports from react or the DOM.
// 100% unit-testable in isolation.
// ─────────────────────────────────────────────────────────────

import type { Trade, Trend, EquityPoint } from '../types';

// Cache formatter at module level (server-safe, never recreated)
const USD_FMT = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export function formatUSD(amount: number): string {
  if (!isFinite(amount)) return '$0.00';
  return USD_FMT.format(amount);
}

export function formatPnL(amount: number): string {
  const str = USD_FMT.format(Math.abs(amount));
  return amount >= 0 ? `+${str}` : `-${str}`;
}

export function formatPercent(value: number, decimals = 1): string {
  if (!isFinite(value)) return '0.0%';
  return `${value.toFixed(decimals)}%`;
}

/** Returns Trend for a numeric PnL vs starting baseline */
export function getBalanceTrend(current: number, start: number): Trend {
  if (current > start) return 'bullish';
  if (current < start) return 'bearish';
  return 'neutral';          // strict equality → neutral (not bullish)
}

/** Tailwind class name for a Trend value */
export function trendColor(trend?: Trend): string {
  if (trend === 'bullish') return 'text-emerald-400';
  if (trend === 'bearish') return 'text-rose-400';
  return 'text-zinc-400';
}

/**
 * Alias for `trendColor` — kept for backwards compatibility with older code
 * and test files that still reference `getTrendClass`.
 */
export const getTrendClass = trendColor;

/** Win rate as a percentage, safe against zero-trade division */
export function calcWinRate(wins: number, totalClosed: number): number {
  if (totalClosed <= 0) return 0;
  return (wins / totalClosed) * 100;
}

/** Build cumulative equity curve from closed trades */
export function buildEquityCurve(startBalance: number, trades: Trade[]): EquityPoint[] {
  let running = startBalance;
  const points: EquityPoint[] = [{
    time: 'Start',
    balance: startBalance,
    pnlDelta: 0,
    tradeId: 'INIT',
  }];

  for (const t of trades) {
    if (t.result === 'OPEN') continue;
    running = Number((running + t.pnlUsd).toFixed(2));
    points.push({
      time: t.closedAt ?? t.openedAt,
      balance: running,
      pnlDelta: t.pnlUsd,
      tradeId: t.id,
    });
  }

  return points;
}
