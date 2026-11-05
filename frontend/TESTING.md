# Testing Guide & Manual QA Checklist

## Automated Testing Suite
Run all tests via:
```bash
npm run test
```

### Coverage Scope
1. **Pure Units (`src/lib/__tests__/`)**:
   - `formatUSD`: Currency formatting edge cases ($0, negatives, large numbers).
   - `getTrendClass`: Maps bullish/bearish to css classes.
   - `getBalanceTrend`: Strict equality returns neutral.
   - `calcWinRate`: Zero trades handles division by zero safely.
   - `buildEquityCurve`: Calculates cumulative PnL series.
2. **Adapters (`src/api/adapters/__tests__/`)**:
   - CISD payload normalization.
   - Liquidity Transfer schema mapping (handling PascalCase, PnL to pnl_usd).
3. **Integration with MSW (`src/hooks/__tests__/`)**:
   - Fetching status, 500 error handling, network failure recovery.

## Manual QA Checklist
- [ ] **Backend Down**: Shut down Python server; ensure clear "Backend Offline" toast/banner without app crash.
- [ ] **Model Offline**: Unreachable model displays distinct Not-Connected empty state with expected URL.
- [ ] **URL Navigation**: Pasting `?model=liquidity_transfer` loads the second model directly on refresh.
- [ ] **Responsive**: Scales seamlessly down to tablet resolution.
