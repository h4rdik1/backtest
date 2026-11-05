import { describe, it, expect } from 'vitest';
import { adaptCisdStatus } from '../adapters/cisd';
import { adaptLiquidityTransferStatus } from '../adapters/liquidityTransfer';

describe('API Adapters', () => {
  it('adaptCisdStatus correctly normalizes CISD JSON schema', () => {
    const mockRaw = {
      model: 'cisd',
      name: 'CISD Test',
      balance: 5120,
      start_balance: 5000,
      pnl_usd: 120,
      open_trades: [
        {
          trade_id: 'T0001',
          symbol: 'SOL_USDT',
          direction: 'SHORT',
          entry_price: 96.24,
          sl_price: 96.67,
          tp_price: 95.36,
          result: 'OPEN',
        },
      ],
      closed_trades: [],
    };

    const adapted = adaptCisdStatus(mockRaw);
    expect(adapted.modelId).toBe('cisd');
    expect(adapted.balance).toBe(5120);
    expect(adapted.openTrades).toHaveLength(1);
    expect(adapted.openTrades[0].symbol).toBe('SOL/USDT');
    expect(adapted.openTrades[0].direction).toBe('SHORT');
    expect(adapted.openTrades[0].entryPrice).toBe(96.24);
  });

  it('adaptLiquidityTransferStatus unifies PascalCase and CSV trade properties', () => {
    const mockRaw = {
      model: 'liquidity_transfer',
      name: 'Liquidity Transfer',
      balance: 4980,
      start_balance: 5000,
      closed_trades: [
        {
          Timestamp: '2026-11-05 10:00',
          Symbol: 'BTC_USDT',
          Direction: 'LONG',
          Entry: '83500.5',
          SL: '81700',
          TP: '88500',
          'Exit Price': '81700',
          'Exit Reason': 'SL',
          PnL: '-20.0',
          R_Multiple: '-1.0',
        },
      ],
    };

    const adapted = adaptLiquidityTransferStatus(mockRaw);
    expect(adapted.modelId).toBe('liquidity_transfer');
    expect(adapted.balance).toBe(4980);
    expect(adapted.closedTrades).toHaveLength(1);

    const trade = adapted.closedTrades[0];
    expect(trade.symbol).toBe('BTC/USDT');
    expect(trade.direction).toBe('LONG');
    expect(trade.entryPrice).toBe(83500.5);
    expect(trade.exitPrice).toBe(81700);
    expect(trade.result).toBe('LOSS');
    expect(trade.pnlUsd).toBe(-20);
    expect(trade.rMultiple).toBe(-1);
    expect(trade.id).toBe('LT-0001');
  });
});
