# Agent Directives & Repository Rules

## 1. Stack & Pinned Decisions
- **Vite + React + TypeScript Strict**: Zero `any`, strict null handling.
- **Lightweight Charts v4**: Strictly pinned to `4.2.3`. Use `chart.addCandlestickSeries()`. DO NOT use v5 methods.
- **TanStack React Query v5**: All query keys must include `modelId` (e.g. `['modelStatus', modelId]`).
- **Styling**: Tailwind CSS v4 + standard glassmorphism theme variables.
- **Testing**: Vitest + MSW. Tests gate every milestone.

## 2. Directory Architecture
```
src/
├── config/models.ts      # Model Registry (id, label, accent, apiBase, enabled)
├── api/                  # Unified typed HTTP client (NO direct fetch outside)
│   ├── client.ts
│   └── adapters/         # Raw API -> normalized domain model per strategy
├── types/                # Domain models, unions, Trend, Trade, Challenge
├── hooks/                # React Query hooks (useModelStatus, useTrades, etc.)
├── features/             # Feature domains (stats, chart, sidebar, compare, switcher)
├── components/ui/        # Reusable primitive UI (StatCard, Badge, Modal, Toast)
├── lib/                  # Pure math and format functions
└── test/                 # Test setup, MSW handlers & fixtures
```

## 3. Definition of Done
Before completing any milestone or submitting:
1. `npm run typecheck` passes with zero errors.
2. `npm run lint` passes with zero warnings.
3. `npm run test` passes with all tests green.
4. `npm run build` succeeds cleanly.
