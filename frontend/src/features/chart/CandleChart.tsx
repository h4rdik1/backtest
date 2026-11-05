// ─────────────────────────────────────────────────────────────
// src/features/chart/CandleChart.tsx
// Lightweight Charts v4 canvas mount.
//
// THREE effects — this is intentional (Vercel rerender-split-combined-hooks):
//   1. Mount effect      → createChart once, ResizeObserver, cleanup chart.remove()
//   2. Data effect       → series.setData() imperatively when candles change
//   3. Trade line effect → series.createPriceLine() for Entry, SL, TP of selected trade
//
// NEVER create the chart inside render or inside data effects.
// ─────────────────────────────────────────────────────────────

import { useEffect, useRef, useState } from 'react';
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type IPriceLine,
  type UTCTimestamp,
  type CandlestickData,
  LineStyle,
} from 'lightweight-charts';
import { clsx } from 'clsx';
import { ChevronDown, ExternalLink, Activity } from 'lucide-react';
import type { Candle, Trade } from '../../types';

const TF_OPTIONS = ['1m', '5m', '15m', '1h', '4h'] as const;
const DEFAULT_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'];

interface CandleChartProps {
  candles: Candle[];
  symbol: string;
  timeframe: string;
  onTimeframeChange: (tf: string) => void;
  onSymbolChange?: (symbol: string) => void;
  selectedTrade?: Trade;
  onInspectTrade?: (trade: Trade) => void;
  accentColor?: string;
  availableSymbols?: string[];
}

export function CandleChart({
  candles,
  symbol,
  timeframe,
  onTimeframeChange,
  onSymbolChange,
  selectedTrade,
  onInspectTrade,
  availableSymbols,
}: CandleChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef     = useRef<IChartApi | null>(null);
  const seriesRef    = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const priceLinesRef = useRef<IPriceLine[]>([]);

  const [symbolMenuOpen, setSymbolMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const symbolsList = availableSymbols && availableSymbols.length > 0 ? availableSymbols : DEFAULT_SYMBOLS;

  // Close dropdown on outside click
  useEffect(() => {
    if (!symbolMenuOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setSymbolMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [symbolMenuOpen]);

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
      priceLinesRef.current = [];
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

  /* ── Effect 3: Overlay Entry, SL, TP price lines ─────────── */
  useEffect(() => {
    if (!seriesRef.current) return;

    // Remove previous price lines
    priceLinesRef.current.forEach((line) => {
      try {
        seriesRef.current?.removePriceLine(line);
      } catch {
        // ignore if already cleared
      }
    });
    priceLinesRef.current = [];

    // Only render price lines if the selected trade is on the active symbol
    if (!selectedTrade || selectedTrade.symbol !== symbol) return;

    if (selectedTrade.entryPrice > 0) {
      const entryLine = seriesRef.current.createPriceLine({
        price: selectedTrade.entryPrice,
        color: '#38bdf8', // sky-400
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        axisLabelVisible: true,
        title: `ENTRY (${selectedTrade.direction})`,
      });
      priceLinesRef.current.push(entryLine);
    }

    if (selectedTrade.slPrice > 0) {
      const slLine = seriesRef.current.createPriceLine({
        price: selectedTrade.slPrice,
        color: '#f43f5e', // rose-500
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: 'SL',
      });
      priceLinesRef.current.push(slLine);
    }

    if (selectedTrade.tpPrice > 0) {
      const tpLine = seriesRef.current.createPriceLine({
        price: selectedTrade.tpPrice,
        color: '#10b981', // emerald-500
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: 'TP',
      });
      priceLinesRef.current.push(tpLine);
    }
  }, [selectedTrade, symbol, candles]);

  return (
    <div className="flex-1 flex flex-col h-full bg-[var(--bg-void)] relative overflow-hidden">
      {/* Floating chart controls */}
      <div className="absolute top-4 left-4 z-20 flex items-center gap-3 glass rounded-xl px-3.5 py-2 shadow-xl">
        {/* Symbol Selector Dropdown */}
        <div ref={menuRef} className="relative">
          <button
            type="button"
            onClick={() => setSymbolMenuOpen((v) => !v)}
            className="flex items-center gap-1.5 text-sm font-bold tracking-tight text-slate-100 font-mono hover:text-white px-2 py-1 rounded-md hover:bg-white/10 transition-all cursor-pointer"
            aria-label="Select symbol"
          >
            <span>{symbol}</span>
            <ChevronDown size={13} className={clsx('text-slate-400 transition-transform duration-150', symbolMenuOpen && 'rotate-180')} />
          </button>

          {symbolMenuOpen && (
            <div className="absolute left-0 mt-2 w-36 rounded-xl bg-[#0e0e17] border border-white/10 shadow-2xl py-1.5 z-50 backdrop-blur-md animate-in fade-in zoom-in-95 duration-100">
              <div className="px-3 py-1 text-[10px] font-mono font-bold text-slate-500 uppercase tracking-wider">
                Markets
              </div>
              {symbolsList.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => {
                    onSymbolChange?.(s);
                    setSymbolMenuOpen(false);
                  }}
                  className={clsx(
                    'w-full text-left px-3 py-1.5 text-xs font-mono transition-all flex items-center justify-between cursor-pointer',
                    s === symbol ? 'text-blue-400 bg-blue-500/10 font-bold' : 'text-slate-300 hover:bg-white/5 hover:text-white'
                  )}
                >
                  <span>{s}</span>
                  {s === symbol && <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="w-px h-4 bg-white/10" />

        {/* Timeframe Selector */}
        <div className="flex items-center gap-0.5">
          {TF_OPTIONS.map((tf) => (
            <button
              key={tf}
              onClick={() => onTimeframeChange(tf)}
              className={clsx(
                'px-2.5 py-1 text-[11px] font-mono font-semibold rounded-md transition-all cursor-pointer',
                timeframe === tf
                  ? 'bg-blue-600 text-white shadow shadow-blue-500/30'
                  : 'text-slate-500 hover:text-slate-200 hover:bg-white/5'
              )}
            >
              {tf}
            </button>
          ))}
        </div>

        {/* Selected Trade Indicator */}
        {selectedTrade && selectedTrade.symbol === symbol && (
          <div className="hidden sm:flex items-center gap-2 pl-2 border-l border-white/10">
            <span className="flex items-center gap-1 text-[10px] font-mono font-bold text-sky-400 bg-sky-500/10 border border-sky-500/20 px-2 py-0.5 rounded-full">
              <Activity size={10} />
              <span>#{selectedTrade.id}</span>
            </span>
            {onInspectTrade && (
              <button
                type="button"
                onClick={() => onInspectTrade(selectedTrade)}
                className="text-[10px] text-slate-400 hover:text-slate-200 flex items-center gap-1 underline underline-offset-2 transition-colors cursor-pointer"
              >
                <span>Audit</span>
                <ExternalLink size={10} />
              </button>
            )}
          </div>
        )}
      </div>

      {/* Canvas mount point — flex-1 so it fills remaining height */}
      <div ref={containerRef} className="w-full flex-1" />
    </div>
  );
}
