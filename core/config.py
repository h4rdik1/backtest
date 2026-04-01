# ==============================================================
# config.py — Single source of truth for ALL backtester settings
# Both backtest_cisd.py and backtest_limit.py import from here.
# ==============================================================

# --------------------------------------------------
# SETUP
# --------------------------------------------------
SETUP                     = "daily"
SYMBOLS                   = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
EXCHANGE                  = "binance"
LIVE_WATCHLIST            = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "LINK/USDT", "HYPE/USDT:okx"]

# ---------------------------------------------------
# DATA
# ---------------------------------------------------
DATA_DIR                  = "data/ohlcv"
TIMEFRAMES                = ["5m", "15m", "1h", "4h", "1d", "1w"]
DEFAULT_LTF               = "15m"     # Centralized Low Timeframe for entries
DAYS_BACK                 = 365       # 365 days for statistical validity
TRAIN_END                 = "2025-12-31"
TEST_START                = "2026-01-01"

# ---------------------------------------------------
# RISK
# ---------------------------------------------------
RISK_REWARD               = 2.0       # 2R — at 40% WR this gives PF 1.33
FIXED_SL_USDT             = 25.0      # Dollar risk per trade
ACCOUNT_SIZE              = 1000.0    # Starting account balance
AUTO_BREAKEVEN_R          = 1.8       # Move SL to entry at 1.8R (close to TP, reduces BE noise)

# ── FVG ENGINE (LuxAlgo exact values) ──────────────
FVG_WICK_RATIO            = 0.5       # Relaxed: allow slightly wickier displacement candles
FVG_AVG_BODY_LOOKBACK     = 10        # Bars for average body calculation
FVG_MIN_SIZE_MULT         = 0.3       # Relaxed: smaller FVGs are still valid on 15m
FVG_MIN_VOLUME_MULT       = 1.0       # Relaxed: volume spike not always reliable on crypto

# ── BIAS ───────────────────────────────────────────
BIAS_EXPIRY_BARS          = 24        # 1H bias stays active for 24 bars (1 day)

# ── FILTERS (toggle individually for analysis) ─────
USE_FVG_QUALITY           = True      # FVG first-touch only
USE_HTF_OB_CONFLUENCE     = False     # DISABLED: kills too many trades
USE_PREMIUM_DISCOUNT      = False     # DISABLED: was cutting 1357->517, losing good setups
USE_PULLBACK_FILTER       = False     # DISABLED: too strict
USE_AUTO_BREAKEVEN        = False     # DISABLED: too many BEs destroying WR (150/233 in limit)
USE_LTF_OB_ENTRY          = True      # Enable LTF Order Block entries

# ── HTF OB ─────────────────────────────────────────
OB_MOVE_MULT              = 1.2       # Relaxed: impulse 1.2x avg range
OB_SWING_LOOKBACK         = 5         # Bars to look for swing point

# ── CISD ───────────────────────────────────────────
CISD_BODY_MULT            = 0.3       # Relaxed: 30% of avg range

# ── PREMIUM/DISCOUNT ───────────────────────────────
PD_ARRAY_BARS             = 20        # Bars for HTF range calculation

# ── VALIDATION ─────────────────────────────────────
MONTE_CARLO_SIMS          = 10000     # Number of random simulations
MIN_TRADES_FOR_VALIDITY   = 100       # Lowered for initial validation
CONSISTENCY_PASS_PCT      = 0.55      # 55% profitable months = pass
EDGE_REAL_THRESHOLD       = 0.65      # 65% MC sims profitable = real edge
WALK_FORWARD_TOLERANCE    = 0.20      # 20% deviation train vs test = pass
 
# ── UTILITIES ──────────────────────────────────────
def format_price(p):
    """Dynamic precision based on price magnitude."""
    if p == 0:
        return "0"
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 1:
        return f"{p:.4f}"
    if p >= 0.01:
        return f"{p:.6f}"
    return f"{p:.8f}"
