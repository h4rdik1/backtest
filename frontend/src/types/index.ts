// ─────────────────────────────────────────────────────────────
// src/types/index.ts — single source of truth for all domain types.
// Every union, interface, and enum is defined ONCE and imported everywhere.
// ─────────────────────────────────────────────────────────────

export type Trend = 'bullish' | 'bearish' | 'neutral';

export type TradeDirection = 'LONG' | 'SHORT';

export type TradeResult = 'WIN' | 'LOSS' | 'BE' | 'OPEN';

export type ModelStatus = 'ACTIVE' | 'INITIALIZING' | 'PAUSED' | 'HALTED' | 'INACTIVE';

/** Normalised trade record — both model CSVs map to this via adapters */
export interface Trade {
  id: string;
  openedAt: string;
  closedAt: string | null;
  symbol: string;
  direction: TradeDirection;
  entryPrice: number;
  slPrice: number;
  tpPrice: number;
  exitPrice: number | null;
  result: TradeResult;
  pnlUsd: number;
  rMultiple?: number;
  tier?: string;
  regime?: string;
  anchorType?: string;
  notes?: string;
  disabled?: boolean;
  beTriggered?: boolean;
  trailingActive?: boolean;
  trailingSlPrice?: number | null;
}

/** Normalised challenge snapshot for one model */
export interface ChallengeStatus {
  modelId: string;
  name: string;
  description: string;
  status: ModelStatus;
  balance: number;
  startBalance: number;
  pnlUsd: number;
  peakBalance: number;
  floor: number;
  target: number;
  progressPct: number;
  drawdownUsd: number;
  scanCount: number;
  totalTrades: number;
  wins: number;
  losses: number;
  openTrades: Trade[];
  pendingTrades: Trade[];
  closedTrades: Trade[];
  lastUpdated: string;
}

/** OHLCV bar for Lightweight Charts v4 */
export interface Candle {
  time: number;   // Unix seconds (UTC)
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

/** One point on the cumulative equity curve */
export interface EquityPoint {
  time: string;
  balance: number;
  pnlDelta: number;
  tradeId: string;
}
