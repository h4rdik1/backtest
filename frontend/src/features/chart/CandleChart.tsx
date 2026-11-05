// ─────────────────────────────────────────────────────────────
// src/features/chart/CandleChart.tsx
// Lightweight Charts v4 canvas mount.
//
// TWO effects — this is intentional (Vercel rerender-split-combined-hooks):
//   1. Mount effect  → createChart once, ResizeObserver, cleanup chart.remove()
//   2. Data effect   → series.setData() imperatively when candles change
//
// NEVER create the chart inside render or inside the data effect.
// ─────────────────────────────────────────────────────────────

import { useEffect, useRef } from 'react';
import { createChart, type IChartApi, type ISeriesApi, type UTCTimestamp, type CandlestickData } from 'lightweight-charts';
import { clsx } from 'clsx';
import type { Candle } from '../../types';

const TF_OPTIONS = ['1m', '5m', '15m', '1h', '4h'] as const;

interface CandleChartProps {
  candles: Candle[];
  symbol: string;
  timeframe: string;
  onTimeframeChange: (tf: string) => void;
  accentColor?: string;
}

export function CandleChart({ candles, symbol, timeframe, onTimeframeChange }: CandleChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef     = useRef<IChartApi | null>(null);
  const seriesRef    = useRef<ISeriesApi<'Candlestick'> | null>(null);

  /* ── Effect 1: Mount chart once ──────────────────────────── */
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { color: 'transparent' },
        textColor:  '#475569',
        fontFamily: "'Outfit', sans-serif",
        fontSize:   11,
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.03)' },
        horzLines: { color: 'rgba(255,255,255,0.03)' },
      },
      crosshair: {
        mode: 1,
        vertLine: { color: 'rgba(255,255,255,0.15)', width: 1, style: 2, labelBackgroundColor: '#1e293b' },
        horzLine: { color: 'rgba(255,255,255,0.15)', width: 1, style: 2, labelBackgroundColor: '#1e293b' },
      },
      timeScale: {
        borderColor:     'rgba(255,255,255,0.06)',
        timeVisible:     true,
        secondsVisible:  false,
        rightOffset:     8,
        barSpacing:      8,
      },
      rightPriceScale: {
        borderColor: 'rgba(255,255,255,0.06)',
        textColor:   '#475569',
        scaleMargins: { top: 0.08, bottom: 0.08 },
      },
      handleScroll: { mouseWheel: true, pressedMouseMove: true },
      handleScale:  { mouseWheel: true, pinch: true },
    });

    // v4 API — pinned: addCandlestickSeries
    const series = chart.addCandlestickSeries({
      upColor:     '#10b981',
      downColor:   '#f43f5e',
      borderVisible: false,
      wickUpColor:   '#10b981',
      wickDownColor: '#f43f5e',
    });

    chartRef.current  = chart;
    seriesRef.current = series;

    /* Responsive resize */
    const ro = new ResizeObserver(() => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width:  containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current  = null;
      seriesRef.current = null;
    };
  }, []);

  /* ── Effect 2: Push data imperatively (no chart recreation) ─ */
  useEffect(() => {
    if (!seriesRef.current || candles.length === 0) return;

    const sorted: CandlestickData<UTCTimestamp>[] = candles
      .slice()
      .sort((a, b) => a.time - b.time)
      .filter((c, i, arr) => i === 0 || c.time > arr[i - 1]!.time)
      .map((c) => ({
        time: (typeof c.time === 'number' ? c.time : Math.floor(new Date(c.time).getTime() / 1000)) as UTCTimestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }));

    seriesRef.current.setData(sorted);
    chartRef.current?.timeScale().fitContent();
  }, [candles]);

  return (
    <div className="flex-1 flex flex-col h-full bg-[var(--bg-void)] relative overflow-hidden">
      {/* Floating chart controls */}
      <div className="absolute top-4 left-4 z-20 flex items-center gap-3 glass rounded-xl px-3.5 py-2 shadow-xl">
        <span className="text-sm font-bold tracking-tight text-slate-200 font-mono">{symbol}</span>
        <div className="w-px h-4 bg-white/10" />
        <div className="flex items-center gap-0.5">
          {TF_OPTIONS.map((tf) => (
            <button
              key={tf}
              onClick={() => onTimeframeChange(tf)}
              className={clsx(
                'px-2.5 py-1 text-[11px] font-mono font-semibold rounded-md transition-all',
                timeframe === tf
                  ? 'bg-blue-600 text-white shadow shadow-blue-500/30'
                  : 'text-slate-500 hover:text-slate-200 hover:bg-white/5'
              )}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>

      {/* Canvas mount point — flex-1 so it fills remaining height */}
      <div ref={containerRef} className="w-full flex-1" />
    </div>
  );
}
