// ─────────────────────────────────────────────────────────────
// src/features/stats/StatsBar.tsx
// Horizontal stat strip below the header: Balance, P/L, Win Rate,
// Drawdown floor, challenge progress. Reuses model accent colour.
// ─────────────────────────────────────────────────────────────

import { clsx } from 'clsx';
import { formatUSD, formatPnL, formatPercent, calcWinRate, getBalanceTrend, trendColor } from '../../lib/math';
import type { ChallengeStatus } from '../../types';

interface StatsBarProps {
  status: ChallengeStatus;
  accentColor: string;
}

interface StatItemProps {
  label: string;
  value: string;
  sub?: string;
  valueClass?: string;
}

function StatItem({ label, value, sub, valueClass }: StatItemProps) {
  return (
    <div className="flex flex-col gap-0.5 px-5 py-3 border-r border-[var(--border)] last:border-r-0">
      <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">{label}</span>
      <span className={clsx('text-base font-bold mono tabular-nums leading-tight', valueClass ?? 'text-slate-100')}>
        {value}
      </span>
      {sub && <span className="text-[10px] font-mono text-slate-600">{sub}</span>}
    </div>
  );
}

export function StatsBar({ status, accentColor }: StatsBarProps) {
  const balanceTrend = getBalanceTrend(status.balance, status.startBalance);
  const pnlTrend = status.pnlUsd > 0.01 ? 'bullish' : status.pnlUsd < -0.01 ? 'bearish' : 'neutral';
  const winRate = calcWinRate(status.wins, status.wins + status.losses);
  const closedCount = status.wins + status.losses;

  return (
    <div
      className="flex items-stretch shrink-0 border-b border-[var(--border)] bg-[var(--bg-base)]"
      style={{ borderTop: `1px solid ${accentColor}22` }}
    >
      <StatItem
        label="Balance"
        value={formatUSD(status.balance)}
        sub={`Start: ${formatUSD(status.startBalance)}`}
        valueClass={trendColor(balanceTrend)}
      />
      <StatItem
        label="Total P/L"
        value={formatPnL(status.pnlUsd)}
        sub={`Peak: ${formatUSD(status.peakBalance)}`}
        valueClass={trendColor(pnlTrend)}
      />
      <StatItem
        label="Win Rate"
        value={formatPercent(winRate)}
        sub={closedCount > 0 ? `${status.wins}W · ${status.losses}L (${closedCount})` : 'No closed trades'}
        valueClass="text-slate-200"
      />
      <StatItem
        label="Drawdown Floor"
        value={formatUSD(status.floor)}
        sub={`Risk: ${formatUSD(Math.max(0, status.balance - status.floor))}`}
        valueClass="text-amber-400"
      />
      <StatItem
        label="Open Positions"
        value={String(status.openTrades.length)}
        sub={`${status.pendingTrades.length} pending`}
        valueClass="text-slate-200"
      />
      <div className="flex flex-col gap-0.5 px-5 py-3 ml-auto justify-center min-w-[180px]">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-500 mb-1">
          Challenge Progress
        </span>
        <div className="h-1.5 w-full rounded-full bg-black/50 overflow-hidden border border-white/5 p-px">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{
              width: `${Math.min(100, Math.max(0, status.progressPct))}%`,
              background: `linear-gradient(90deg, ${accentColor}, #10b981)`,
            }}
          />
        </div>
        <span className="text-[10px] font-mono text-slate-500 mt-0.5">
          {formatPercent(status.progressPct)} of {formatUSD(status.target)}
        </span>
      </div>
    </div>
  );
}
