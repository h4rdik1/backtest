// ─────────────────────────────────────────────────────────────
// src/features/compare/CompareView.tsx
// Side-by-side comparison of both challenge models.
// Called when activeTab === 'compare'.
// ─────────────────────────────────────────────────────────────

import { clsx } from 'clsx';
import { MODEL_REGISTRY } from '../../config/models';
import { formatUSD, formatPnL, formatPercent, calcWinRate } from '../../lib/math';
import { EquityCurveChart } from './EquityCurveChart';
import type { ChallengeStatus } from '../../types';

interface CompareViewProps {
  challenges: Record<string, ChallengeStatus>;
  onNavigate: (modelId: string) => void;
}

interface MetricRowProps {
  label: string;
  values: Array<{ text: string; cls?: string }>;
}

function MetricRow({ label, values }: MetricRowProps) {
  return (
    <div className="grid grid-cols-[180px_1fr_1fr] items-center py-3 border-b border-[var(--border)] last:border-0">
      <span className="text-xs text-slate-500 font-medium">{label}</span>
      {values.map((v, i) => (
        <span key={i} className={clsx('text-sm font-mono font-bold tabular-nums text-right pr-6', v.cls ?? 'text-slate-200')}>
          {v.text}
        </span>
      ))}
    </div>
  );
}

export function CompareView({ challenges, onNavigate }: CompareViewProps) {
  const models = Object.values(MODEL_REGISTRY);
  const statuses = models.map((m) => challenges[m.id]);

  return (
    <div className="flex-1 overflow-y-auto bg-[var(--bg-void)] p-8">
      <div className="max-w-5xl mx-auto">
        {/* Page title */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-white tracking-tight">Fleet Performance Matrix</h1>
          <p className="text-sm text-slate-500 mt-1">
            Head-to-head comparison of both institutional models under identical prop firm risk constraints.
          </p>
        </div>

        {/* Model header cards */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          {models.map((m, i) => {
            const s = statuses[i];
            if (!s) return (
              <div key={m.id} className="glass rounded-2xl p-6 border border-[var(--border)] opacity-50">
                <div className="text-sm font-bold text-slate-400">{m.name}</div>
                <div className="text-xs text-slate-600 mt-1">Not connected</div>
              </div>
            );
            const isProfit = s.pnlUsd >= 0;
            return (
              <div
                key={m.id}
                className="glass rounded-2xl p-6 border border-[var(--border)] hover:border-[var(--border-hover)] transition-all"
                style={{ borderTopColor: m.accentColor }}
              >
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: m.accentColor }} />
                      <span className="text-[10px] font-mono uppercase tracking-widest text-slate-500">{m.id}</span>
                    </div>
                    <h3 className="text-lg font-bold text-white">{m.name}</h3>
                    <p className="text-xs text-slate-500 mt-0.5 line-clamp-1">{m.description}</p>
                  </div>
                  <button
                    onClick={() => onNavigate(m.id)}
                    className="text-xs font-semibold text-slate-400 hover:text-white bg-white/5 hover:bg-white/10 px-3 py-1.5 rounded-lg border border-[var(--border)] transition-all shrink-0"
                  >
                    View Chart →
                  </button>
                </div>

                {/* Key numbers */}
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { label: 'Balance', value: formatUSD(s.balance), cls: isProfit ? 'text-emerald-400' : 'text-rose-400' },
                    { label: 'P/L', value: formatPnL(s.pnlUsd), cls: isProfit ? 'text-emerald-400' : 'text-rose-400' },
                    { label: 'Progress', value: formatPercent(s.progressPct), cls: 'text-blue-400' },
                  ].map(({ label, value, cls }) => (
                    <div key={label} className="bg-black/30 rounded-xl p-3 border border-[var(--border)]">
                      <div className="text-[9px] text-slate-500 uppercase font-semibold mb-1">{label}</div>
                      <div className={clsx('text-base font-bold font-mono tabular-nums', cls)}>{value}</div>
                    </div>
                  ))}
                </div>

                {/* Progress bar */}
                <div className="mt-4">
                  <div className="h-1.5 w-full rounded-full bg-black/50 overflow-hidden border border-white/5 p-px">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{ width: `${Math.min(100, Math.max(0, s.progressPct))}%`, backgroundColor: m.accentColor }}
                    />
                  </div>
                  <div className="flex justify-between text-[9px] font-mono text-slate-700 mt-1">
                    <span>{formatUSD(s.startBalance)}</span>
                    <span>{formatUSD(s.target)}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Comparative Equity Curve */}
        <EquityCurveChart challenges={challenges} />

        {/* Detailed comparison table */}
        <div className="glass rounded-2xl border border-[var(--border)] p-6">
          <h2 className="text-sm font-bold text-slate-300 mb-4">Detailed Metrics</h2>

          {/* Column headers */}
          <div className="grid grid-cols-[180px_1fr_1fr] pb-3 border-b border-[var(--border)] mb-1">
            <span />
            {models.map((m) => (
              <span key={m.id} className="text-xs font-bold text-right pr-6" style={{ color: m.accentColor }}>
                {m.shortName}
              </span>
            ))}
          </div>

          {statuses[0] && statuses[1] && (() => {
            const a = statuses[0]!;
            const b = statuses[1]!;
            const wrA = calcWinRate(a.wins, a.wins + a.losses);
            const wrB = calcWinRate(b.wins, b.wins + b.losses);

            const rows: MetricRowProps[] = [
              { label: 'Starting Balance', values: [{ text: formatUSD(a.startBalance) }, { text: formatUSD(b.startBalance) }] },
              { label: 'Current Balance',  values: [{ text: formatUSD(a.balance), cls: a.pnlUsd >= 0 ? 'text-emerald-400' : 'text-rose-400' }, { text: formatUSD(b.balance), cls: b.pnlUsd >= 0 ? 'text-emerald-400' : 'text-rose-400' }] },
              { label: 'Peak Balance',     values: [{ text: formatUSD(a.peakBalance) }, { text: formatUSD(b.peakBalance) }] },
              { label: 'Total P/L',        values: [{ text: formatPnL(a.pnlUsd), cls: a.pnlUsd >= 0 ? 'text-emerald-400' : 'text-rose-400' }, { text: formatPnL(b.pnlUsd), cls: b.pnlUsd >= 0 ? 'text-emerald-400' : 'text-rose-400' }] },
              { label: 'Target',           values: [{ text: formatUSD(a.target) }, { text: formatUSD(b.target) }] },
              { label: 'Progress',         values: [{ text: formatPercent(a.progressPct), cls: 'text-blue-400' }, { text: formatPercent(b.progressPct), cls: 'text-blue-400' }] },
              { label: 'Win Rate',         values: [{ text: formatPercent(wrA) }, { text: formatPercent(wrB) }] },
              { label: 'Total Trades',     values: [{ text: String(a.totalTrades) }, { text: String(b.totalTrades) }] },
              { label: 'Wins / Losses',    values: [{ text: `${a.wins}W / ${a.losses}L` }, { text: `${b.wins}W / ${b.losses}L` }] },
              { label: 'Max Drawdown Floor', values: [{ text: formatUSD(a.floor), cls: 'text-amber-400' }, { text: formatUSD(b.floor), cls: 'text-amber-400' }] },
              { label: 'Open Positions',   values: [{ text: String(a.openTrades.length) }, { text: String(b.openTrades.length) }] },
              { label: 'Status',           values: [{ text: a.status, cls: 'text-emerald-400' }, { text: b.status, cls: 'text-emerald-400' }] },
            ];

            return rows.map((r) => <MetricRow key={r.label} {...r} />);
          })()}
        </div>
      </div>
    </div>
  );
}
