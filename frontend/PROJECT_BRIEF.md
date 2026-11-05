# Project Brief — SMC Multi-Model Trading Dashboard

## 🎯 Goal
Provide a production-ready, highly responsive dashboard to monitor and compare autonomous institutional trading models (`cisd` and `liquidity_transfer`) side-by-side.

## 👥 Users
- Quantitative traders and prop firm operators running autonomous challenge fleets.
- Maintainers scaling from 2 models to N models without frontend rewrites.

## 📌 Scope
- Dynamic model switching via URL parameter (`?model=cisd` / `?model=liquidity_transfer`).
- Real-time Challenge & Prop Firm risk stats (Balance, P/L, Win Rate, Drawdown Floor, Progress).
- Open, Pending, and Closed Trade logs with uniform schemas.
- Overview / Compare tab with comparative equity curve.
- Clear Not-Connected empty states with expected URLs.
- Dev Debug Panel.

## 🚫 Non-Goals (M5 Scope)
- Live database persistence (handled by backend state files).
- User authentication and multi-tenant authorization.
- Cloud hosting deployment pipeline.
