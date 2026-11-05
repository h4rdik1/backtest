// ─────────────────────────────────────────────────────────────
// src/features/empty/ModelNotConnected.tsx
// Graceful empty state when a model's backend endpoint is unreachable.
// Allows tab switching and retry without blocking the application.
// ─────────────────────────────────────────────────────────────

import { AlertCircle, RefreshCw, ArrowRight } from 'lucide-react';
import type { ModelConfig } from '../../config/models';

interface ModelNotConnectedProps {
  model: ModelConfig;
  error?: Error | null;
  onRetry: () => void;
  onSwitchModel: (modelId: string) => void;
}

export function ModelNotConnected({ model, error, onRetry, onSwitchModel }: ModelNotConnectedProps) {
  return (
    <div className="flex-1 flex items-center justify-center bg-[var(--bg-void)] p-8">
      <div className="max-w-lg w-full glass rounded-3xl p-8 border border-[var(--border)] text-center relative overflow-hidden">
        {/* Glow accent */}
        <div
          className="absolute -top-24 left-1/2 -translate-x-1/2 w-48 h-48 rounded-full blur-3xl opacity-20 pointer-events-none"
          style={{ backgroundColor: model.accentColor }}
        />

        <div className="w-12 h-12 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center mx-auto mb-4">
          <AlertCircle size={24} className="text-amber-400" />
        </div>

        <h3 className="text-xl font-bold text-white tracking-tight">
          {model.name} is not connected yet
        </h3>

        <p className="text-sm text-slate-400 mt-2 leading-relaxed">
          The dashboard is actively listening for live state updates, but the backend endpoint returned no response or is currently offline.
        </p>

        <div className="mt-5 p-3.5 rounded-xl bg-black/40 border border-white/5 text-left font-mono text-xs space-y-1.5">
          <div className="flex justify-between text-slate-500">
            <span>Expected URL:</span>
            <span className="text-slate-300">http://localhost:8000{model.apiPath}</span>
          </div>
          {error && (
            <div className="flex justify-between text-rose-400">
              <span>Status:</span>
              <span className="truncate max-w-[240px]">{error.message}</span>
            </div>
          )}
        </div>

        <div className="flex items-center justify-center gap-3 mt-6">
          <button
            onClick={onRetry}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 text-white text-xs font-semibold border border-white/10 transition-all cursor-pointer"
          >
            <RefreshCw size={13} />
            <span>Retry Connection</span>
          </button>

          {model.id !== 'cisd' && (
            <button
              onClick={() => onSwitchModel('cisd')}
              className="flex items-center gap-1.5 px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-lg shadow-blue-500/25 transition-all cursor-pointer"
            >
              <span>Switch to CISD</span>
              <ArrowRight size={13} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
