# Project Milestones & Tasks

## Milestones

### [x] M0: Architecture, Skills Alignment, Dependency Setup & Phase 0 Report
- [x] Align on requirements and tech stack.
- [x] Install Vitest, MSW, Testing Library, Tailwind CSS v4.
- [x] Create project docs skeleton.

### [ ] M1: Scaffold + Model Registry + CSS Tokens + Lib Functions + API Adapters & Hooks
- [ ] Model registry (`src/config/models.ts`).
- [ ] Core types (`src/types/index.ts`).
- [ ] CSS tokens and layout styles (`src/index.css`, `src/App.css`).
- [ ] Pure library functions (`src/lib/*.ts`) with 100% unit test coverage.
- [ ] API client and adapters (`src/api/`).
- [ ] React Query hooks with MSW integration tests.

### [ ] M2: Per-Model View (Stats, Sidebar, Trades, Modals)
- [ ] Stats bar with dynamic progress.
- [ ] Open/Pending/Closed trades sidebar.
- [ ] Lightweight Charts v4 candlestick display with timeframe switcher and overlays.

### [ ] M3: Model Switcher + Compare View & Equity Curve
- [ ] URL-synced tab switcher (`?model=cisd`, `?model=liquidity_transfer`, `?view=compare`).
- [ ] Side-by-side comparison matrix.
- [ ] Multi-model overlaid equity curve.

### [ ] M4: Debug Panel, Accessibility & Final Documentation Review
- [ ] Dev-only debug drawer.
- [ ] Full keyboard navigation and Esc-modal handling.
- [ ] Final docs verification.
