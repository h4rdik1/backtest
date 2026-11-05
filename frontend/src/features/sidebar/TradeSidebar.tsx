// ─────────────────────────────────────────────────────────────
// src/features/sidebar/TradeSidebar.tsx
// Three-tab sidebar: Open / Pending / History.
// Trade cards show directional badge, tier chip, price levels, and PnL.
// Pure display — no fetch calls. Data comes from props.
// ─────────────────────────────────────────────────────────────

import { useState } from 'react';
import { clsx } from 'clsx';
import { ArrowUpRight, ArrowDownRight, Clock } from 'lucide-react';
import { formatPnL } from '../../lib/math';
import type { Trade } from '../../types';

interface TradeSidebarProps {
  openTrades: Trade[];
  pendingTrades: Trade[];
  closedTrades: Trade[];
  selectedId?: string;
  onSelect?: (trade: Trade) => void;
  accentColor: string;
}

type Tab = 'open' | 'pending' | 'history';

function TabBtn({ label, count, active, onClick, accent }: {
  label: string; count: number; active: boolean; onClick: () => void; accent?: string;
}) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'flex-1 py-2 text-xs font-semibold flex items-center justify-center gap-1.5 rounded-md transition-all duration-150',
        active ? 'bg-white/10 text-white border border-white/10' : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
      )}
    >
      {label}
      <span
        className="text-[10px] font-mono px-1.5 py-0.5 rounded-full"
        style={active ? { backgroundColor: `${accent}22`, color: accent } : { backgroundColor: 'rgba(255,255,255,0.05)', color: '#64748b' }}
      >
        {count}
      </span>
    </button>
  );
}

function TradeCard({ trade, selected, onSelect }: { trade: Trade; selected: boolean; onSelect?: () => void }) {
  const isLong = trade.direction === 'LONG';
  const isWin = trade.result === 'WIN';
  const isLoss = trade.result === 'LOSS';
  const isClosed = trade.result !== 'OPEN';

  return (
    <div
      onClick={onSelect}
      className={clsx(
        'rounded-xl border p-3.5 cursor-pointer transition-all duration-150 group',
        selected
          ? 'bg-white/[0.06] border-blue-500/40'
          : 'bg-[var(--bg-card)] border-[var(--border)] hover:bg-[var(--bg-hover)] hover:border-[var(--border-hover)]'
      )}
    >
      {/* Row 1: direction badge + symbol + PnL */}
      <div className="flex items-center justify-between gap-2 mb-2.5">
        <div className="flex items-center gap-2">
          <span
            className={clsx(
              'flex items-center gap-0.5 text-[10px] font-bold px-2 py-0.5 rounded font-mono border',
              isLong
                ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
                : 'text-rose-400 bg-rose-500/10 border-rose-500/20'
            )}
          >
            {isLong ? <ArrowUpRight size={10} /> : <ArrowDownRight size={10} />}
            {trade.direction}
          </span>
          <span className="text-sm font-bold text-slate-100 tracking-tight">{trade.symbol}</span>
          {trade.tier && (
            <span className="text-[9px] font-bold font-mono text-amber-400 bg-amber-500/10 border border-amber-500/20 px-1.5 py-0.5 rounded">
              {trade.tier}
            </span>
          )}
        </div>
        {isClosed && (
          <span className={clsx('text-xs font-mono font-bold tabular-nums', isWin ? 'text-emerald-400' : isLoss ? 'text-rose-400' : 'text-slate-400')}>
            {formatPnL(trade.pnlUsd)}
          </span>
        )}
      </div>

      {/* Row 2: price grid */}
      <div className="grid grid-cols-3 gap-2 py-2 border-y border-[var(--border)]">
        {[
          { label: 'Entry', value: trade.entryPrice, cls: 'text-slate-200' },
          { label: 'SL', value: trade.slPrice, cls: 'text-rose-400' },
          { label: 'TP', value: trade.tpPrice, cls: 'text-emerald-400' },
        ].map(({ label, value, cls }) => (
          <div key={label}>
            <div className="text-[9px] text-slate-600 uppercase font-semibold">{label}</div>
            <div className={clsx('text-[11px] font-mono tabular-nums font-semibold mt-0.5', cls)}>
              {Number(value).toLocaleString('en-US', { maximumFractionDigits: 5, minimumFractionDigits: 2 })}
            </div>
          </div>
        ))}
      </div>

      {/* Row 3: timestamp + regime */}
      <div className="flex items-center justify-between mt-2 text-[10px] text-slate-600">
        <span className="truncate max-w-[140px]">{trade.regime ?? trade.anchorType ?? '—'}</span>
        <span className="font-mono shrink-0">{trade.openedAt.split(' ')[1] ?? trade.openedAt.slice(11, 16)}</span>
      </div>
    </div>
  );
}

export function TradeSidebar({
  openTrades, pendingTrades, closedTrades, selectedId, onSelect, accentColor
}: TradeSidebarProps) {
  const [tab, setTab] = useState<Tab>('open');

  const list: Trade[] = tab === 'open' ? openTrades : tab === 'pending' ? pendingTrades : closedTrades;

  return (
    <aside className="w-80 xl:w-96 shrink-0 flex flex-col h-full border-r border-[var(--border)] bg-[var(--bg-base)] overflow-hidden">
      {/* Tab strip */}
      <div className="flex gap-1 p-2 border-b border-[var(--border)] bg-black/20">
        <TabBtn label="Open"    count={openTrades.length}    active={tab === 'open'}    onClick={() => setTab('open')}    accent={accentColor} />
        <TabBtn label="Pending" count={pendingTrades.length} active={tab === 'pending'} onClick={() => setTab('pending')} accent={accentColor} />
        <TabBtn label="History" count={closedTrades.length}  active={tab === 'history'} onClick={() => setTab('history')} accent={accentColor} />
      </div>

      {/* Scrollable list */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2.5">
        {list.length === 0 ? (
          <div className="h-48 flex flex-col items-center justify-center text-center gap-2">
            <Clock size={28} className="text-slate-700 stroke-[1.5]" />
            <p className="text-xs text-slate-500 font-medium">No {tab} trades</p>
            <p className="text-[11px] text-slate-700 max-w-[180px]">
              {tab === 'open'
                ? 'Live institutional setups will appear here when executed.'
                : tab === 'pending'
                ? 'Limit orders waiting for price to enter zone.'
                : 'Closed trade journal is empty.'}
            </p>
          </div>
        ) : (
          list.map((t) => (
            <TradeCard
              key={t.id}
              trade={t}
              selected={t.id === selectedId}
              onSelect={() => onSelect?.(t)}
            />
          ))
        )}
      </div>
    </aside>
  );
}
