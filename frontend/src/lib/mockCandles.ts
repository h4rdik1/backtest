// ─────────────────────────────────────────────────────────────
// src/lib/mockCandles.ts
// Deterministic seeded mock candles shown while real candle API
// is not wired into api_server.py yet.
// Labelled with a "DEMO DATA" flag in CandleChart wrapper.
// ─────────────────────────────────────────────────────────────

import type { Candle } from '../types';

const TF_SECONDS: Record<string, number> = {
  '1m': 60,
  '5m': 300,
  '15m': 900,
  '1h': 3600,
  '4h': 14400,
};

export function generateMockCandles(basePrice: number, count = 120, timeframe = '15m'): Candle[] {
  const candles: Candle[] = [];
  let price = basePrice;
  const now = Math.floor(Date.now() / 1000);
  const step = TF_SECONDS[timeframe] ?? 900;

  for (let i = count; i >= 0; i--) {
    const time = now - i * step;
    const drift  = (Math.random() - 0.495) * basePrice * 0.007;
    const open   = price;
    const close  = open + drift;
    const wick   = Math.random() * basePrice * 0.003;
    const high   = Math.max(open, close) + wick;
    const low    = Math.min(open, close) - wick;
    price = close;

    candles.push({
      time,
      open:  +open.toFixed(2),
      high:  +high.toFixed(2),
      low:   +low.toFixed(2),
      close: +close.toFixed(2),
    });
  }
  return candles;
}
