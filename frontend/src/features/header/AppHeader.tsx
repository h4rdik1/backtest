// ─────────────────────────────────────────────────────────────
// src/features/header/AppHeader.tsx
// Top navigation bar: brand, model tabs with live P&L badges,
// challenge progress meter, and status pill.
// ─────────────────────────────────────────────────────────────

import { TrendingUp, Circle } from 'lucide-react';
import { clsx } from 'clsx';
import { ALL_MODELS } from '../../config/models';
import { formatPnL, formatPercent, formatUSD } from '../../lib/math';
import type { ChallengeStatus } from '../../types';

interface AppHeaderProps {
  activeTab: string;
  onSelectTab: (tab: string) => void;
  current?: ChallengeStatus;
  allStatus: Record<string, { pnlUsd: number; status: string }>;
}

export function AppHeader({ activeTab, onSelectTab, current, allStatus }: AppHeaderProps) {
  return (
    <header className="flex items-center px-5 h-14 shrink-0 border-b border-[var(--border)] bg-[var(--bg-base)] z-30">
      {/* Brand */}
      <div className="flex items-center gap-2.5 mr-6 shrink-0">
        <div className="w-7 h-7 rounded-lg bg-[var(--accent-cisd)] flex items-center justify-center shadow-lg shadow-blue-500/30">
          <TrendingUp size={14} className="text-white" />
        </div>
        <span className="text-sm font-bold tracking-tight text-white whitespace-nowrap">SMC FLEET</span>
        <span className="text-[9px] font-mono text-slate-500 tracking-widest bg-white/5 px-1.5 py-0.5 rounded border border-white/5">
          v2.0
        </span>
      </div>

      {/* Model Switcher Tabs */}
      <nav className="flex items-center gap-1 p-1 rounded-lg bg-black/30 border border-[var(--border)]">
        {ALL_MODELS.map((m) => {
          const info = allStatus[m.id];
          const isActive = activeTab === m.id;
          const pnl = info?.pnlUsd ?? 0;
          const isProfit = pnl >= 0;

          return (
            <button
              key={m.id}
              onClick={() => onSelectTab(m.id)}
              className={clsx(
                'flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-semibold transition-all duration-150',
                isActive
                  ? 'bg-white/10 text-white border border-white/10 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
              )}
            >
              <span
                className="w-1.5 h-1.5 rounded-full shrink-0"
                style={{
                  backgroundColor: m.accentColor,
                  boxShadow: isActive ? `0 0 8px ${m.accentColor}` : 'none',
                }}
              />
              <span>{m.shortName}</span>
              {info && (
                <span
                  className={clsx(
                    'text-[10px] font-mono px-1.5 py-0.5 rounded tabular-nums',
                    isProfit ? 'text-emerald-400 bg-emerald-500/10' : 'text-rose-400 bg-rose-500/10'
                  )}
                >
                  {formatPnL(pnl)}
                </span>
              )}
            </button>
          );
        })}

        {/* Divider */}
        <div className="w-px h-5 bg-white/10 mx-0.5" />

        {/* Compare tab */}
        <button
          onClick={() => onSelectTab('compare')}
          className={clsx(
            'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all duration-150',
            activeTab === 'compare'
              ? 'bg-blue-500/15 text-blue-400 border border-blue-500/25'
              : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
          )}
        >
          <span>⊞</span>
          <span>Compare</span>
        </button>
      </nav>

      {/* Challenge Progress — only shown in model view */}
      {current && activeTab !== 'compare' && (
        <div className="hidden lg:flex flex-col gap-1 mx-6 w-48">
          <div className="flex justify-between text-[10px] font-mono">
            <span className="text-slate-400">Target Progress</span>
            <span className="text-white">{formatPercent(current.progressPct)}</span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-black/50 overflow-hidden border border-white/5 p-px">
            <div
              className="h-full rounded-full bg-gradient-to-r from-blue-500 to-emerald-400 transition-all duration-700"
              style={{ width: `${Math.min(100, Math.max(0, current.progressPct))}%` }}
            />
          </div>
          <div className="flex justify-between text-[9px] font-mono text-slate-600">
            <span>{formatUSD(current.startBalance)}</span>
            <span className="text-emerald-500">{formatUSD(current.target)}</span>
          </div>
        </div>
      )}

      {/* Status pill — right-aligned */}
      {current && (
        <div className="ml-auto flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/8 border border-emerald-500/20">
          <Circle size={6} className="text-emerald-400 fill-emerald-400 animate-pulse" />
          <span className="text-[10px] font-mono font-semibold text-emerald-400 tracking-widest uppercase">
            {current.status}
          </span>
        </div>
      )}
    </header>
  );
}
