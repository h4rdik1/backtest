// ─────────────────────────────────────────────────────────────
// src/api/adapters/cisd.ts
// Maps raw /api/challenges/cisd JSON (snake_case) → ChallengeStatus.
// Never throws — missing fields fall back to safe defaults.
// ─────────────────────────────────────────────────────────────

import type { ChallengeStatus, Trade, TradeDirection, TradeResult, ModelStatus } from '../../types';

/* Raw shapes coming from api_server.py */
export interface RawCisdTrade {
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
}

export interface RawCisdResponse {
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
  open_trades?: RawCisdTrade[];
  pending_trades?: RawCisdTrade[];
  closed_trades?: RawCisdTrade[];
}

function adaptTrade(raw: RawCisdTrade, idx: number): Trade {
  const dir = (raw.direction ?? 'LONG').toUpperCase();
  const direction: TradeDirection = dir === 'SHORT' ? 'SHORT' : 'LONG';

  const rawResult = (raw.result ?? '').toUpperCase();
  const result: TradeResult = (['WIN', 'LOSS', 'BE', 'OPEN'] as string[]).includes(rawResult)
    ? (rawResult as TradeResult)
    : 'OPEN';

  return {
    id:             raw.trade_id ?? `CISD-${String(idx).padStart(4, '0')}`,
    openedAt:       raw.opened_at ?? new Date().toISOString(),
    closedAt:       raw.closed_at ?? null,
    symbol:         (raw.symbol ?? 'UNKNOWN').replace('_', '/'),
    direction,
    entryPrice:     Number(raw.entry_price ?? 0),
    slPrice:        Number(raw.sl_price ?? 0),
    tpPrice:        Number(raw.tp_price ?? 0),
    exitPrice:      raw.exit_price != null ? Number(raw.exit_price) : null,
    result,
    pnlUsd:         Number(raw.pnl_usd ?? 0),
    tier:           raw.setup_tier,
    regime:         raw.regime,
    anchorType:     raw.anchor_type,
    notes:          raw.notes,
    disabled:       raw.disabled ?? false,
    beTriggered:    raw.be_triggered ?? false,
    trailingActive: raw.trailing_active ?? false,
    trailingSlPrice: raw.trailing_sl ?? null,
  };
}

export function adaptCisdStatus(raw: RawCisdResponse): ChallengeStatus {
  const balance      = Number(raw.balance ?? 5000);
  const startBalance = Number(raw.start_balance ?? 5000);

  return {
    modelId:      'cisd',
    name:         raw.name ?? 'CISD + FVG Multi-TF Strategy',
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
