// ─────────────────────────────────────────────────────────────
// src/features/debug/DebugPanel.tsx
// Dev-only floating diagnostic HUD.
// Completely stripped from production builds via import.meta.env.DEV.
// ─────────────────────────────────────────────────────────────

import { useState, useEffect } from 'react';
import { Terminal, X, ChevronRight, RefreshCw, CheckCircle2, AlertTriangle } from 'lucide-react';
import type { ChallengeStatus } from '../../types';

interface DebugPanelProps {
  activeModelId: string;
  apiPath: string;
  status?: ChallengeStatus;
  isLoading: boolean;
  error: Error | null;
  lastUpdated?: string;
  onRefresh?: () => void;
}

export function DebugPanel({
  activeModelId,
  apiPath,
  status,
  isLoading,
  error,
  lastUpdated,
  onRefresh,
}: DebugPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [showRaw, setShowRaw] = useState(false);

  // Keyboard shortcut Ctrl + Shift + D to toggle
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'd') {
        setIsOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Never render in production (stripped by bundler dead-code elimination)
  if (!import.meta.env.DEV) return null;

  return (
    <div className="fixed bottom-4 left-4 z-50 font-mono select-none">
      {!isOpen ? (
        <button
          onClick={() => setIsOpen(true)}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-black/80 hover:bg-black text-slate-400 hover:text-white border border-white/10 shadow-2xl backdrop-blur-md text-[11px] transition-all cursor-pointer group"
          title="Toggle Debug HUD (Ctrl+Shift+D)"
        >
          <Terminal size={12} className="text-blue-400 group-hover:rotate-12 transition-transform" />
          <span>DEBUG</span>
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        </button>
      ) : (
        <div className="w-96 rounded-2xl bg-[#09090f]/95 border border-white/15 shadow-2xl backdrop-blur-xl p-4 text-xs text-slate-300">
          {/* Header */}
          <div className="flex items-center justify-between pb-3 border-b border-white/10">
            <div className="flex items-center gap-2">
              <Terminal size={14} className="text-blue-400" />
              <span className="font-bold text-white tracking-wider">DEV DIAGNOSTICS</span>
            </div>
            <div className="flex items-center gap-1.5">
              {onRefresh && (
                <button
                  onClick={onRefresh}
                  className="p-1 rounded hover:bg-white/10 text-slate-400 hover:text-white transition-all cursor-pointer"
                  title="Force re-fetch"
                >
                  <RefreshCw size={12} className={isLoading ? 'animate-spin text-blue-400' : ''} />
                </button>
              )}
              <button
                onClick={() => setIsOpen(false)}
                className="p-1 rounded hover:bg-white/10 text-slate-400 hover:text-white transition-all cursor-pointer"
              >
                <X size={13} />
              </button>
            </div>
          </div>

          {/* Grid rows */}
          <div className="py-3 space-y-2 text-[11px]">
            <div className="flex justify-between items-center">
              <span className="text-slate-500">Active Model</span>
              <span className="text-blue-400 font-bold">{activeModelId}</span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-slate-500">Endpoint</span>
              <code className="text-slate-300 bg-white/5 px-1.5 py-0.5 rounded text-[10px]">
                {apiPath}
              </code>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-slate-500">Query Status</span>
              <span className="flex items-center gap-1">
                {isLoading ? (
                  <span className="text-amber-400 flex items-center gap-1">
                    <RefreshCw size={10} className="animate-spin" /> Fetching
                  </span>
                ) : error ? (
                  <span className="text-rose-400 flex items-center gap-1">
                    <AlertTriangle size={10} /> Error
                  </span>
                ) : (
                  <span className="text-emerald-400 flex items-center gap-1">
                    <CheckCircle2 size={10} /> Connected
                  </span>
                )}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-slate-500">Last Updated</span>
              <span className="text-slate-400">{lastUpdated ?? status?.lastUpdated?.slice(11, 19) ?? 'Never'}</span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-slate-500">Open / Pending</span>
              <span className="text-slate-300">
                {status?.openTrades.length ?? 0} open · {status?.pendingTrades.length ?? 0} pending
              </span>
            </div>

            {error && (
              <div className="p-2 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-[10px]">
                {error.message}
              </div>
            )}
          </div>

          {/* Raw payload expander */}
          <div className="pt-2 border-t border-white/10">
            <button
              onClick={() => setShowRaw((v) => !v)}
              className="flex items-center gap-1 text-[10px] text-slate-500 hover:text-slate-300 cursor-pointer"
            >
              <ChevronRight size={10} className={`transform transition-transform ${showRaw ? 'rotate-90' : ''}`} />
              <span>{showRaw ? 'Hide Raw Response' : 'Inspect Raw JSON'}</span>
            </button>

            {showRaw && (
              <pre className="mt-2 p-2 rounded bg-black/60 text-[9px] text-slate-400 max-h-40 overflow-auto whitespace-pre-wrap break-all">
                {JSON.stringify(status, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
