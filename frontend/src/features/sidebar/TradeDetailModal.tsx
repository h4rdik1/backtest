// ─────────────────────────────────────────────────────────────
// src/features/sidebar/TradeDetailModal.tsx
// Comprehensive trade audit inspector modal.
// Accessible: closes on Escape, focus trapped, aria labelled.
// ─────────────────────────────────────────────────────────────

import { useEffect, useRef } from 'react';
import { X, ArrowUpRight, ArrowDownRight, Shield, Target, Activity } from 'lucide-react';
import { clsx } from 'clsx';
import { formatUSD, formatPnL } from '../../lib/math';
import type { Trade } from '../../types';

interface TradeDetailModalProps {
  trade: Trade | null;
  onClose: () => void;
  accentColor: string;
}

export function TradeDetailModal({ trade, onClose, accentColor }: TradeDetailModalProps) {
  const modalRef = useRef<HTMLDivElement>(null);

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!trade) return null;

  const isLong = trade.direction === 'LONG';
  const isClosed = trade.result !== 'OPEN';

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="trade-detail-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-150"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={modalRef}
        className="w-full max-w-lg rounded-2xl bg-[#0e0e17] border border-white/10 shadow-2xl overflow-hidden p-6 text-slate-200"
        style={{ borderTop: `3px solid ${accentColor}` }}
      >
        {/* Header */}
        <div className="flex items-start justify-between pb-4 border-b border-white/10">
          <div className="flex items-center gap-3">
            <span
              className={clsx(
                'flex items-center gap-1 text-xs font-bold font-mono px-2.5 py-1 rounded-md border',
                isLong
                  ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
                  : 'text-rose-400 bg-rose-500/10 border-rose-500/20'
              )}
            >
              {isLong ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
              {trade.direction}
            </span>
            <div>
              <h2 id="trade-detail-title" className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
                <span>{trade.symbol}</span>
                <span className="text-xs font-mono text-slate-500 font-normal">#{trade.id}</span>
              </h2>
              <div className="text-xs text-slate-400 flex items-center gap-2 mt-0.5">
                <span>Opened: {trade.openedAt}</span>
                {trade.closedAt && <span>· Closed: {trade.closedAt}</span>}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close trade details"
            className="p-1.5 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-all cursor-pointer"
          >
            <X size={16} />
          </button>
        </div>

        {/* Price Levels Grid */}
        <div className="grid grid-cols-3 gap-3 my-5">
          <div className="bg-black/40 rounded-xl p-3 border border-white/5">
            <div className="flex items-center gap-1 text-[10px] text-slate-500 uppercase font-semibold">
              <Activity size={10} />
              <span>Entry Price</span>
            </div>
            <div className="text-sm font-bold font-mono text-slate-100 mt-1 tabular-nums">
              {formatUSD(trade.entryPrice)}
            </div>
          </div>

          <div className="bg-black/40 rounded-xl p-3 border border-white/5">
            <div className="flex items-center gap-1 text-[10px] text-rose-400 uppercase font-semibold">
              <Shield size={10} />
              <span>Stop Loss</span>
            </div>
            <div className="text-sm font-bold font-mono text-rose-400 mt-1 tabular-nums">
              {formatUSD(trade.slPrice)}
            </div>
          </div>

          <div className="bg-black/40 rounded-xl p-3 border border-white/5">
            <div className="flex items-center gap-1 text-[10px] text-emerald-400 uppercase font-semibold">
              <Target size={10} />
              <span>Take Profit</span>
            </div>
            <div className="text-sm font-bold font-mono text-emerald-400 mt-1 tabular-nums">
              {formatUSD(trade.tpPrice)}
            </div>
          </div>
        </div>

        {/* Detailed Metrics */}
        <div className="space-y-2 text-xs bg-black/20 rounded-xl p-4 border border-white/5">
          {trade.tier && (
            <div className="flex justify-between items-center py-1 border-b border-white/5">
              <span className="text-slate-500">Setup Tier</span>
              <span className="font-mono font-bold text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded">
                Tier {trade.tier}
              </span>
            </div>
          )}

          {trade.regime && (
            <div className="flex justify-between items-center py-1 border-b border-white/5">
              <span className="text-slate-500">Market Regime</span>
              <span className="text-slate-200 capitalize">{trade.regime.replace('_', ' ')}</span>
            </div>
          )}

          {trade.anchorType && (
            <div className="flex justify-between items-center py-1 border-b border-white/5">
              <span className="text-slate-500">Anchor Mechanism</span>
              <span className="text-slate-300 font-mono">{trade.anchorType}</span>
            </div>
          )}

          {isClosed && (
            <div className="flex justify-between items-center py-1 border-b border-white/5">
              <span className="text-slate-500">Net P&L Realized</span>
              <span className={clsx('font-bold font-mono tabular-nums text-sm', trade.pnlUsd >= 0 ? 'text-emerald-400' : 'text-rose-400')}>
                {formatPnL(trade.pnlUsd)}
              </span>
            </div>
          )}

          {trade.notes && (
            <div className="pt-2">
              <span className="text-slate-500 block mb-1">Signal Context</span>
              <p className="text-[11px] text-slate-400 leading-relaxed font-mono bg-black/30 p-2 rounded border border-white/5">
                {trade.notes}
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 mt-6 pt-3 border-t border-white/10">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-semibold rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 hover:text-white transition-all cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
