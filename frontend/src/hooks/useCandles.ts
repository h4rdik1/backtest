// ─────────────────────────────────────────────────────────────
// src/hooks/useCandles.ts
// React Query hook for fetching live Binance OHLCV candles.
// Fallback to deterministic mock candles if backend or exchange fails.
// ─────────────────────────────────────────────────────────────

import { useQuery } from '@tanstack/react-query';
import { fetchJson } from '../api/client';
import { generateMockCandles } from '../lib/mockCandles';
import type { Candle } from '../types';

export interface UseCandlesResult {
  candles: Candle[];
  isLive: boolean;
  isLoading: boolean;
  error: Error | null;
}

export function useCandles(symbol: string, timeframe: string, basePrice: number): UseCandlesResult {
  const cleanSymbol = symbol.replace('/', '_');

  const { data, isLoading, error } = useQuery<Candle[], Error>({
    queryKey: ['candles', cleanSymbol, timeframe],
    queryFn: () => fetchJson<Candle[]>(`/api/live_candles?symbol=${cleanSymbol}&tf=${timeframe}`),
    refetchInterval: 15_000, // Refresh every 15s for live market
    staleTime: 10_000,
    retry: 1,
  });

  const isLive = Boolean(data && data.length > 0 && !error);
  const candles = isLive && data ? data : generateMockCandles(basePrice, 120, timeframe);

  return {
    candles,
    isLive,
    isLoading,
    error: isLive ? null : error,
  };
}
