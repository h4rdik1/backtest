import { describe, it, expect } from 'vitest';
import {
  formatUSD,
  formatPercent,
  getTrendClass,
  getBalanceTrend,
  calcWinRate,
  buildEquityCurve,
} from '../math';
import type { Trade } from '../../types';

describe('lib/math utilities', () => {
  it('formatUSD formats standard, negative, and edge amounts', () => {
    expect(formatUSD(5000)).toBe('$5,000.00');
    expect(formatUSD(-25.5)).toBe('-$25.50');
    expect(formatUSD(0)).toBe('$0.00');
    expect(formatUSD(NaN)).toBe('$0.00');
  });

  it('formatPercent formats values safely', () => {
    expect(formatPercent(64.285)).toBe('64.3%');
    expect(formatPercent(0)).toBe('0.0%');
    expect(formatPercent(NaN)).toBe('0.0%');
  });

  it('getTrendClass returns expected CSS classes', () => {
    expect(getTrendClass('bullish')).toContain('emerald');
    expect(getTrendClass('bearish')).toContain('rose');
    expect(getTrendClass('neutral')).toContain('zinc');
    expect(getTrendClass(undefined)).toContain('zinc');
  });

  it('getBalanceTrend correctly detects profit, loss, and strict equality neutral', () => {
    expect(getBalanceTrend(5100, 5000)).toBe('bullish');
    expect(getBalanceTrend(4900, 5000)).toBe('bearish');
    expect(getBalanceTrend(5000, 5000)).toBe('neutral');
  });

  it('calcWinRate prevents division by zero', () => {
    expect(calcWinRate(0, 0)).toBe(0);
    expect(calcWinRate(5, 10)).toBe(50);
    expect(calcWinRate(3, 4)).toBe(75);
  });

  it('buildEquityCurve creates a cumulative balance progression', () => {
    const mockTrades: Trade[] = [
      {
        id: 'T1',
        openedAt: '2026-11-05 10:00:00',
        closedAt: '2026-11-05 11:00:00',
        symbol: 'BTC/USDT',
        direction: 'LONG',
        entryPrice: 90000,
        slPrice: 89000,
        tpPrice: 92000,
        exitPrice: 92000,
        result: 'WIN',
        pnlUsd: 100,
      },
      {
        id: 'T2',
        openedAt: '2026-11-05 12:00:00',
        closedAt: '2026-11-05 13:00:00',
        symbol: 'SOL/USDT',
        direction: 'SHORT',
        entryPrice: 100,
        slPrice: 105,
        tpPrice: 90,
        exitPrice: 105,
        result: 'LOSS',
        pnlUsd: -50,
      },
    ];

    const curve = buildEquityCurve(5000, mockTrades);
    expect(curve).toHaveLength(3);
    expect(curve[0].balance).toBe(5000);
    expect(curve[1].balance).toBe(5100);
    expect(curve[2].balance).toBe(5050);
  });
});
