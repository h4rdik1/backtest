# API Contract Documentation

## Endpoints

### 1. `GET /health`
- **Confirmed**: Returns API operational health.
```json
{
  "status": "ok",
  "service": "Challenge Trading Fleet API"
}
```

### 2. `GET /api/challenges`
- **Confirmed**: Dictionary of all models keyed by model id (`cisd`, `liquidity_transfer`).
```json
{
  "cisd": {
    "model": "cisd",
    "name": "CISD + FVG Multi-TF Strategy",
    "description": "Multi-timeframe Order Blocks, FVG Displacement, SFP & Trend Alignment",
    "status": "ACTIVE",
    "balance": 5000.0,
    "start_balance": 5000.0,
    "pnl_usd": 0.0,
    "peak_balance": 5000.0,
    "floor": 4700.0,
    "target": 5500.0,
    "progress_pct": 0.0,
    "drawdown_usd": 0.0,
    "scan_count": 1,
    "total_trades": 1,
    "wins": 0,
    "losses": 0,
    "open_trades": [],
    "pending_trades": [],
    "closed_trades": []
  }
}
```

### 3. `GET /api/challenges/:modelId`
- **Confirmed**: Returns details for a single model identical to its record in `/api/challenges`.

### 4. Trade Log Schemas
- **CISD Log (`challenge_trades_cisd.csv`)**:
  `trade_id, opened_at, closed_at, symbol, chain, direction, entry_price, sl_price, tp_price, exit_price, result, pnl_usd, balance_after, regime, bias_reason, anchor_type, setup_tier`
- **Liquidity Transfer Log (`liquidity_transfer_trades.csv`)**:
  `Timestamp, Symbol, Direction, Entry, SL, TP, Exit Price, Exit Reason, PnL, R_Multiple, Contracts, Risk_USD, Duration_Bars, Entry_Type, HTF_Source`
- **Normalization (Confirmed Strategy)**:
  `src/api/adapters/` unifies both CSV records into domain `Trade` objects.
