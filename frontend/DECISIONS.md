# Architectural Decisions Log

| ID | Decision | Status | Date |
|---|---|---|---|
| D-001 | Single Port Unified Backend with Vite Proxy | Accepted | 2026-11-05 |
| D-002 | Adapter Pattern for Multi-Model Schemas | Accepted | 2026-11-05 |
| D-003 | Lightweight Charts v4 API Pinning | Accepted | 2026-11-05 |
| D-004 | URL Param as Single Source of Truth for Model | Accepted | 2026-11-05 |

---

### D-001: Single Port Unified Backend with Vite Proxy
- **Status**: Accepted
- **Context**: Need both models monitored side-by-side without multi-port connection hurdles or CORS friction.
- **Options**:
  1. Multiple ports (e.g. 8000 and 8001).
  2. Single port unified endpoint (`/api/challenges/:modelId`) with Vite `/api` proxy.
- **Decision**: Option 2.
- **Why**: `api_server.py` already exposes both `cisd` and `liquidity_transfer` simultaneously on port 8000.
- **Trade-offs**: Requires single server process to be running.
- **How to reverse**: Change `apiBase` in `models.ts` to absolute origins if separated later.

---

### D-002: Adapter Pattern for Multi-Model Schemas
- **Status**: Accepted
- **Context**: Backend stores trade records in distinct formats (CISD uses snake_case and `pnl_usd`; Liquidity Transfer uses PascalCase, `PnL`, `R_Multiple`).
- **Options**:
  1. Hardcode backend CSV schema migrations.
  2. Frontend adapter layer in `src/api/adapters/`.
- **Decision**: Option 2.
- **Why**: Decouples UI components from backend serialization quirks. Adding a 3rd model requires only 1 new adapter.
- **Trade-offs**: Small runtime normalization step in JS.

---

### D-003: Lightweight Charts v4 API Pinning
- **Status**: Accepted
- **Context**: TradingView Lightweight Charts v5 introduces breaking API changes (`addCandlestickSeries` removed in favor of `addSeries(CandlestickSeries)`).
- **Decision**: Pin strictly to v4 (`4.2.3`).

---

### D-004: URL Param as Single Source of Truth for Model
- **Status**: Accepted
- **Context**: Model view needs to be bookmarkable and sharable.
- **Decision**: Use `?model=<id>` or `?view=compare` in URL search params instead of transient React component state.
