// ─────────────────────────────────────────────────────────────
// src/App.tsx — Root shell. Composes the full layout.
//
// URL-synced model state: ?model=cisd | ?model=liquidity_transfer | ?model=compare
// All data fetching lives in hooks — App.tsx is a pure orchestrator.
// ─────────────────────────────────────────────────────────────

import { useState, useMemo } from 'react';
import { QueryClient, QueryClientProvider, useQueryClient } from '@tanstack/react-query';
import { AppHeader }        from './features/header/AppHeader';
import { StatsBar }         from './features/stats/StatsBar';
import { TradeSidebar }     from './features/sidebar/TradeSidebar';
import { CandleChart }      from './features/chart/CandleChart';
import { CompareView }      from './features/compare/CompareView';
import { ModelNotConnected } from './features/empty/ModelNotConnected';
import { TradeDetailModal } from './features/sidebar/TradeDetailModal';
import { DebugPanel }       from './features/debug/DebugPanel';
import { useModelStatus, useAllChallenges } from './hooks/useModelStatus';
import { useCandles }         from './hooks/useCandles';
import { MODEL_REGISTRY, DEFAULT_MODEL_ID, getModel } from './config/models';
import type { Trade } from './types';
import './App.css';

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 10_000 } },
});

// ─── Main Dashboard ───────────────────────────────────────────
function Dashboard() {
  const qc = useQueryClient();

  // Persist active tab in URL ?model= parameter
  const [activeTab, setActiveTab] = useState<string>(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('model') ?? DEFAULT_MODEL_ID;
  });
  const [timeframe, setTimeframe] = useState('15m');
  const [selectedTradeId, setSelectedTradeId] = useState<string | undefined>();
  const [selectedSymbol, setSelectedSymbol] = useState<string | undefined>();
  const [inspectingTrade, setInspectingTrade] = useState<Trade | null>(null);

  const handleSelectTab = (tab: string) => {
    setActiveTab(tab);
    setSelectedTradeId(undefined);
    setSelectedSymbol(undefined);
    const url = new URL(window.location.href);
    url.searchParams.set('model', tab);
    window.history.replaceState({}, '', url.toString());
  };

  const modelId = activeTab === 'compare' ? DEFAULT_MODEL_ID : activeTab;
  const modelCfg = getModel(modelId) ?? MODEL_REGISTRY[DEFAULT_MODEL_ID]!;

  const { data: current, isLoading, error } = useModelStatus(modelId, { retry: 2 });
  const { data: allChallenges } = useAllChallenges();

  // Build quick summary map for switcher P&L badges
  const allStatus: Record<string, { pnlUsd: number; status: string }> = useMemo(() => {
    if (!allChallenges) return {};
    return Object.fromEntries(
      Object.entries(allChallenges).map(([id, s]) => [id, { pnlUsd: s.pnlUsd, status: s.status }])
    );
  }, [allChallenges]);

  // Aggregate all trades for the active challenge
  const allTrades = useMemo(() => {
    if (!current) return [];
    return [...current.openTrades, ...current.pendingTrades, ...current.closedTrades];
  }, [current]);

  // Selected trade entity
  const selectedTrade = useMemo(() => {
    if (!selectedTradeId) return undefined;
    return allTrades.find((t) => t.id === selectedTradeId);
  }, [allTrades, selectedTradeId]);

  const openTradeSymbol = current?.openTrades[0]?.symbol;
  const defaultSymbol = modelCfg.defaultSymbol;

  // Active symbol: explicit selection > selected trade's symbol > first open trade > default symbol
  const symbol = useMemo(() => {
    if (selectedSymbol) return selectedSymbol;
    if (selectedTrade) return selectedTrade.symbol;
    return openTradeSymbol ?? defaultSymbol;
  }, [selectedSymbol, selectedTrade, openTradeSymbol, defaultSymbol]);

  // Base price for realistic mock fallback
  const basePrice = useMemo(() => {
    if (symbol.includes('SOL')) return 145;
    if (symbol.includes('ETH')) return 2650;
    if (symbol.includes('BNB')) return 590;
    return 89_500; // BTC default
  }, [symbol]);

  const { candles, isLive } = useCandles(symbol, timeframe, basePrice);

  const handleSelectTrade = (trade: Trade) => {
    setSelectedTradeId(trade.id);
    setSelectedSymbol(trade.symbol);
  };

  const handleSymbolChange = (newSymbol: string) => {
    setSelectedSymbol(newSymbol);
    if (selectedTrade && selectedTrade.symbol !== newSymbol) {
      setSelectedTradeId(undefined);
    }
  };

  const handleRetry = () => {
    qc.invalidateQueries({ queryKey: ['challenge', modelId] });
    qc.invalidateQueries({ queryKey: ['challenges'] });
  };

  return (
    <div className="flex flex-col h-screen w-full overflow-hidden bg-[var(--bg-void)]">
      {/* ── Top Navigation (Always visible) ─────────── */}
      <AppHeader
        activeTab={activeTab}
        onSelectTab={handleSelectTab}
        current={current}
        allStatus={allStatus}
      />

      {/* ── Stats Strip (hidden in compare mode or error) ── */}
      {activeTab !== 'compare' && current && (
        <StatsBar status={current} accentColor={modelCfg.accentColor} />
      )}

      {/* ── Main Workspace ─────────────────────────── */}
      <div className="flex-1 flex overflow-hidden">
        {activeTab === 'compare' ? (
          <CompareView
            challenges={allChallenges ?? (current ? { [current.modelId]: current } : {})}
            onNavigate={handleSelectTab}
          />
        ) : error && !current ? (
          <ModelNotConnected
            model={modelCfg}
            error={error}
            onRetry={handleRetry}
            onSwitchModel={handleSelectTab}
          />
        ) : isLoading && !current ? (
          <div className="flex-1 flex items-center justify-center bg-[var(--bg-void)]">
            <div className="flex flex-col items-center gap-4">
              <div
                className="w-8 h-8 border-2 border-t-transparent rounded-full animate-spin"
                style={{ borderColor: `${modelCfg.accentColor} transparent ${modelCfg.accentColor} ${modelCfg.accentColor}` }}
              />
              <p className="text-xs text-slate-500 font-mono tracking-wider">
                Connecting to {modelCfg.name} on port 8000...
              </p>
            </div>
          </div>
        ) : (
          <>
            <TradeSidebar
              openTrades={current?.openTrades ?? []}
              pendingTrades={current?.pendingTrades ?? []}
              closedTrades={current?.closedTrades ?? []}
              selectedId={selectedTradeId}
              onSelect={handleSelectTrade}
              onInspect={setInspectingTrade}
              accentColor={modelCfg.accentColor}
            />

            <div className="flex-1 flex flex-col overflow-hidden relative">
              {/* Live Feed vs Demo Data badge */}
              <div className="absolute top-4 right-4 z-30 flex items-center gap-1.5 px-2.5 py-1 rounded-full border shadow-lg backdrop-blur-md transition-all">
                {isLive ? (
                  <>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse shadow-sm shadow-emerald-400" />
                    <span className="text-[9px] font-mono font-bold text-emerald-400 uppercase tracking-widest">
                      LIVE BINANCE
                    </span>
                  </>
                ) : (
                  <>
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                    <span className="text-[9px] font-mono font-bold text-amber-400 uppercase tracking-widest">
                      DEMO DATA
                    </span>
                  </>
                )}
              </div>

              <CandleChart
                candles={candles}
                symbol={symbol}
                timeframe={timeframe}
                onTimeframeChange={setTimeframe}
                onSymbolChange={handleSymbolChange}
                selectedTrade={selectedTrade}
                onInspectTrade={setInspectingTrade}
                accentColor={modelCfg.accentColor}
              />
            </div>
          </>
        )}
      </div>

      {/* ── Modal & Dev Inspector Overlay ─────────── */}
      <TradeDetailModal
        trade={inspectingTrade}
        onClose={() => setInspectingTrade(null)}
        accentColor={modelCfg.accentColor}
      />

      <DebugPanel
        activeModelId={modelId}
        apiPath={modelCfg.apiPath}
        status={current}
        isLoading={isLoading}
        error={error}
        onRefresh={handleRetry}
      />
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Dashboard />
    </QueryClientProvider>
  );
}
