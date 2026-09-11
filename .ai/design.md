# UI & Design System

While this is primarily a backend CLI application, it includes local HTML visualization tools (`view_candles.html`, `view_advanced.html`). 

## 1. DESIGN PHILOSOPHY
- **Function over Form:** The dashboard is a diagnostic tool for traders, not a consumer web app. Focus on data density and clarity.
- **Dark Mode Default:** Given the nature of trading interfaces (like TradingView), all web dashboards must default to a dark theme (e.g., `#1e1e24` backgrounds, `#e0e0e0` text).

## 2. COMPONENT PATTERNS
- **Candlestick Charts:** Must be rendered clearly, distinguishing between bullish (Green/White) and bearish (Red/Black) candles.
- **SMC Overlays:** 
  - FVGs should be highlighted as semi-transparent boxes extending forward in time.
  - MSB/BOS lines should be clearly labeled horizontal lines.
  - Liquidity Sweeps (SFPs) should be marked with distinct icons (e.g., ❌ or 💧) at the wick extremes.

## 3. TECH STACK (UI)
- Vanilla HTML/CSS/JS.
- No heavy frameworks (React, Vue) unless absolutely necessary.
- Data is fed via lightweight local HTTP servers (e.g., `tools/view_candles.py`).
