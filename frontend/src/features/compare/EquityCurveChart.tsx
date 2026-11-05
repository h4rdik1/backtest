// ─────────────────────────────────────────────────────────────
// src/features/compare/EquityCurveChart.tsx
// Overlaid equity curve chart comparing both challenge models.
// Renders cumulative balance progression with interactive tooltip.
// ─────────────────────────────────────────────────────────────

import { useState, useMemo } from 'react';
import { buildEquityCurve, formatUSD } from '../../lib/math';
import { MODEL_REGISTRY } from '../../config/models';
import type { ChallengeStatus } from '../../types';

interface EquityCurveChartProps {
  challenges: Record<string, ChallengeStatus>;
}

export function EquityCurveChart({ challenges }: EquityCurveChartProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const cisdStatus = challenges['cisd'];
  const ltStatus = challenges['liquidity_transfer'];

  const cisdPoints = useMemo(() => {
    return buildEquityCurve(cisdStatus?.startBalance ?? 5000, cisdStatus?.closedTrades ?? []);
  }, [cisdStatus]);

  const ltPoints = useMemo(() => {
    return buildEquityCurve(ltStatus?.startBalance ?? 5000, ltStatus?.closedTrades ?? []);
  }, [ltStatus]);

  // Compute overall min and max balance for Y-axis scaling
  const allBalances = [
    ...cisdPoints.map((p) => p.balance),
    ...ltPoints.map((p) => p.balance),
    4700, // include floor
    5500, // include target
  ];
  const minY = Math.min(...allBalances) - 50;
  const maxY = Math.max(...allBalances) + 50;
  const rangeY = maxY - minY || 1;

  // Chart dimensions
  const width = 800;
  const height = 260;
  const padLeft = 60;
  const padRight = 20;
  const padTop = 20;
  const padBottom = 30;
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;

  const getX = (idx: number, total: number) => {
    if (total <= 1) return padLeft;
    return padLeft + (idx / (total - 1)) * plotWidth;
  };

  const getY = (val: number) => {
    const norm = (val - minY) / rangeY;
    return padTop + (1 - norm) * plotHeight;
  };

  const cisdPath = cisdPoints
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${getX(i, cisdPoints.length)} ${getY(p.balance)}`)
    .join(' ');

  const ltPath = ltPoints
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${getX(i, ltPoints.length)} ${getY(p.balance)}`)
    .join(' ');

  const baselineY = getY(5000);
  const targetY = getY(5500);
  const floorY = getY(4700);

  return (
    <div className="glass rounded-2xl border border-[var(--border)] p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-sm font-bold text-slate-200">Comparative Equity Progression</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Cumulative account balance from trade executions overlaid across both models.
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-0.5 rounded-full" style={{ backgroundColor: MODEL_REGISTRY.cisd?.accentColor ?? '#3b82f6' }} />
            <span className="text-slate-300">CISD</span>
            <span className="text-slate-500">({cisdPoints[cisdPoints.length - 1]?.balance ? formatUSD(cisdPoints[cisdPoints.length - 1]!.balance) : '$5,000.00'})</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-0.5 rounded-full" style={{ backgroundColor: MODEL_REGISTRY.liquidity_transfer?.accentColor ?? '#a855f7' }} />
            <span className="text-slate-300">LT Model</span>
            <span className="text-slate-500">({ltPoints[ltPoints.length - 1]?.balance ? formatUSD(ltPoints[ltPoints.length - 1]!.balance) : '$5,000.00'})</span>
          </div>
        </div>
      </div>

      <div className="relative w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto max-h-[300px] select-none"
          style={{ minWidth: '600px' }}
        >
          <defs>
            <linearGradient id="cisdGlow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.0" />
            </linearGradient>
            <linearGradient id="ltGlow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#a855f7" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#a855f7" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          <line x1={padLeft} y1={targetY} x2={width - padRight} y2={targetY} stroke="rgba(16, 185, 129, 0.25)" strokeDasharray="3 3" />
          <text x={padLeft - 8} y={targetY + 4} textAnchor="end" fill="#10b981" fontSize="10" fontFamily="monospace">
            Target ($5,500)
          </text>

          <line x1={padLeft} y1={baselineY} x2={width - padRight} y2={baselineY} stroke="rgba(255, 255, 255, 0.12)" strokeDasharray="2 2" />
          <text x={padLeft - 8} y={baselineY + 4} textAnchor="end" fill="#94a3b8" fontSize="10" fontFamily="monospace">
            $5,000
          </text>

          <line x1={padLeft} y1={floorY} x2={width - padRight} y2={floorY} stroke="rgba(244, 63, 94, 0.25)" strokeDasharray="3 3" />
          <text x={padLeft - 8} y={floorY + 4} textAnchor="end" fill="#f43f5e" fontSize="10" fontFamily="monospace">
            Floor ($4,700)
          </text>

          {/* CISD Series */}
          {cisdPoints.length > 1 && (
            <path d={cisdPath} fill="none" stroke="#3b82f6" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
          )}

          {/* LT Series */}
          {ltPoints.length > 1 && (
            <path d={ltPath} fill="none" stroke="#a855f7" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
          )}

          {/* Point markers for CISD */}
          {cisdPoints.map((p, i) => (
            <circle
              key={`c-${i}`}
              cx={getX(i, cisdPoints.length)}
              cy={getY(p.balance)}
              r={hoverIndex === i ? 5 : 3}
              fill="#3b82f6"
              stroke="#07070b"
              strokeWidth="1.5"
              className="cursor-pointer transition-all"
              onMouseEnter={() => setHoverIndex(i)}
              onMouseLeave={() => setHoverIndex(null)}
            />
          ))}

          {/* Point markers for LT */}
          {ltPoints.map((p, i) => (
            <circle
              key={`lt-${i}`}
              cx={getX(i, ltPoints.length)}
              cy={getY(p.balance)}
              r={hoverIndex === i ? 5 : 3}
              fill="#a855f7"
              stroke="#07070b"
              strokeWidth="1.5"
              className="cursor-pointer transition-all"
              onMouseEnter={() => setHoverIndex(i)}
              onMouseLeave={() => setHoverIndex(null)}
            />
          ))}
        </svg>
      </div>
    </div>
  );
}
