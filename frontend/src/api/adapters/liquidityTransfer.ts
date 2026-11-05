// ─────────────────────────────────────────────────────────────
// src/api/adapters/liquidityTransfer.ts
// Maps raw /api/challenges/liquidity_transfer JSON to ChallengeStatus.
// The LT CSV uses PascalCase + different field names from CISD.
// The adapter bridges both historical CSVs and live state JSON.
// ─────────────────────────────────────────────────────────────

import type { ChallengeStatus, Trade, TradeDirection, TradeResult, ModelStatus } from '../../types';

export interface RawLtTrade {
  /* Live state JSON (snake_case from challenge_trader.py) */
  trade_id?: string;
  opened_at?: string;
  closed_at?: string | null;
  symbol?: string;
  direction?: string;
  entry_price?: number | string;
  sl_price?: number | string;
  tp_price?: number | string;
  exit_price?: number | string | null;
  result?: string;
  pnl_usd?: number | string;
  setup_tier?: string;
  regime?: string;
  anchor_type?: string;
  notes?: string;
  disabled?: boolean;
  be_triggered?: boolean;
  trailing_active?: boolean;
  trailing_sl?: number | null;
  /* Historical CSV columns (PascalCase) */
  Timestamp?: string;
  Symbol?: string;
  Direction?: string;
  Entry?: number | string;
  SL?: number | string;
  TP?: number | string;
  'Exit Price'?: number | string;
  'Exit Reason'?: string;
  PnL?: number | string;
  R_Multiple?: number | string;
  Risk_USD?: number | string;
  HTF_Source?: string;
  Entry_Type?: string;
}

export interface RawLtResponse {
  model?: string;
  name?: string;
  description?: string;
  status?: string;
  balance?: number;
  start_balance?: number;
  pnl_usd?: number;
  peak_balance?: number;
  floor?: number;
  target?: number;
  progress_pct?: number;
  drawdown_usd?: number;
  scan_count?: number;
  total_trades?: number;
  wins?: number;
  losses?: number;
  open_trades?: RawLtTrade[];
  pending_trades?: RawLtTrade[];
  closed_trades?: RawLtTrade[];
}

function adaptTrade(raw: RawLtTrade, idx: number): Trade {
  /* Resolve fields from either live JSON or historical CSV */
  const symbol    = (raw.symbol ?? raw.Symbol ?? 'UNKNOWN').replace('_', '/');
  const direction = ((raw.direction ?? raw.Direction ?? 'LONG').toUpperCase() === 'SHORT'
    ? 'SHORT' : 'LONG') as TradeDirection;
  const entry  = Number(raw.entry_price ?? raw.Entry ?? 0);
  const sl     = Number(raw.sl_price    ?? raw.SL    ?? 0);
  const tp     = Number(raw.tp_price    ?? raw.TP    ?? 0);
  const exit   = raw.exit_price != null ? Number(raw.exit_price)
               : raw['Exit Price'] != null ? Number(raw['Exit Price'])
               : null;
  const pnl = Number(raw.pnl_usd ?? raw.PnL ?? 0);
  const rMult = raw.R_Multiple != null ? Number(raw.R_Multiple) : undefined;

  /* Infer result */
  const reason = (raw.result ?? raw['Exit Reason'] ?? '').toUpperCase();
  let result: TradeResult = 'OPEN';
  if (reason.includes('WIN') || pnl > 0.01)                            result = 'WIN';
  else if (reason.includes('LOSS') || reason.includes('SL') || pnl < -0.01) result = 'LOSS';
  else if (reason.includes('BE') || (exit !== null && Math.abs(pnl) <= 0.01)) result = 'BE';

  const openedAt = raw.opened_at ?? raw.Timestamp ?? new Date().toISOString();

  return {
    id:             raw.trade_id ?? `LT-${String(idx + 1).padStart(4, '0')}`,
    openedAt,
    closedAt:       raw.closed_at ?? (exit !== null ? openedAt : null),
    symbol,
    direction,
    entryPrice:     entry,
    slPrice:        sl,
    tpPrice:        tp,
    exitPrice:      exit,
    result,
    pnlUsd:         pnl,
    rMultiple:      rMult,
    tier:           raw.setup_tier ?? raw.HTF_Source ?? raw.Entry_Type,
    anchorType:     raw.anchor_type,
    notes:          raw.notes ?? (raw['Exit Reason'] ? `Exit: ${raw['Exit Reason']}` : undefined),
    disabled:       raw.disabled ?? false,
    beTriggered:    raw.be_triggered ?? false,
    trailingActive: raw.trailing_active ?? false,
    trailingSlPrice: raw.trailing_sl ?? null,
  };
}

export function adaptLtStatus(raw: RawLtResponse): ChallengeStatus {
  const balance      = Number(raw.balance ?? 5000);
  const startBalance = Number(raw.start_balance ?? 5000);

  return {
    modelId:      'liquidity_transfer',
    name:         raw.name ?? 'Liquidity Transfer Strategy',
    description:  raw.description ?? '',
    status:       (raw.status as ModelStatus) ?? 'ACTIVE',
    balance,
    startBalance,
    pnlUsd:       Number(raw.pnl_usd ?? balance - startBalance),
    peakBalance:  Number(raw.peak_balance ?? balance),
    floor:        Number(raw.floor ?? 4700),
    target:       Number(raw.target ?? 5500),
    progressPct:  Number(raw.progress_pct ?? 0),
    drawdownUsd:  Number(raw.drawdown_usd ?? 0),
    scanCount:    Number(raw.scan_count ?? 0),
    totalTrades:  Number(raw.total_trades ?? 0),
    wins:         Number(raw.wins ?? 0),
    losses:       Number(raw.losses ?? 0),
    openTrades:   (raw.open_trades ?? []).map(adaptTrade),
    pendingTrades: (raw.pending_trades ?? []).map(adaptTrade),
    closedTrades: (raw.closed_trades ?? []).map(adaptTrade),
    lastUpdated:  new Date().toISOString(),
  };
}

export const adaptLiquidityTransferStatus = adaptLtStatus;
