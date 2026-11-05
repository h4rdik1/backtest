import { http, HttpResponse } from 'msw';

export const handlers = [
  http.get('/api/challenges/cisd', () => {
    return HttpResponse.json({
      model: 'cisd',
      name: 'CISD + FVG Multi-TF Strategy',
      balance: 5120.0,
      start_balance: 5000.0,
      pnl_usd: 120.0,
      status: 'ACTIVE',
      total_trades: 3,
      wins: 2,
      losses: 1,
      open_trades: [],
      pending_trades: [],
      closed_trades: [],
    });
  }),

  http.get('/api/challenges/liquidity_transfer', () => {
    return HttpResponse.json({
      model: 'liquidity_transfer',
      name: 'Liquidity Transfer Strategy',
      balance: 4980.0,
      start_balance: 5000.0,
      pnl_usd: -20.0,
      status: 'ACTIVE',
      total_trades: 1,
      wins: 0,
      losses: 1,
      open_trades: [],
      pending_trades: [],
      closed_trades: [],
    });
  }),
];
