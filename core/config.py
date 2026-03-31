# ==============================================================
# config.py — Single source of truth for ALL backtester settings
# Both backtest_cisd.py and backtest_limit.py import from here.
# ==============================================================

# --------------------------------------------------
# SETUP
# --------------------------------------------------
SETUP                     = "daily"
SYMBOLS                   = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
EXCHANGE                  = "binance"
LIVE_WATCHLIST            = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "LINK/USDT", "HYPE/USDT:okx"]

# ---------------------------------------------------
# DATA
# ---------------------------------------------------
DATA_DIR                  = "data/ohlcv"
TIMEFRAMES                = ["5m", "15m", "1h", "4h", "1d", "1w"]
DAYS_BACK                 = 60      # 60 days of historical data
TRAIN_END                 = "2023-12-31"
TEST_START                = "2024-01-01"

# ---------------------------------------------------
# RISK
# ---------------------------------------------------
RISK_REWARD               = 3.0       # Take profit at 3x the risk
FIXED_SL_USDT             = 25.0      # Dollar risk per trade
ACCOUNT_SIZE              = 1000.0    # Starting account balance
AUTO_BREAKEVEN_R          = 2.0       # Increased from 1.5R to let trades breathe

# ── FVG ENGINE (LuxAlgo exact values) ──────────────
FVG_WICK_RATIO            = 0.36      # Max wick-to-body ratio for displacement
FVG_AVG_BODY_LOOKBACK     = 10        # Bars for average body calculation
FVG_MIN_SIZE_MULT         = 0.5       # Min FVG size as multiple of avg range
FVG_MIN_VOLUME_MULT       = 1.5       # Min volume multiplier for displacement

# ── BIAS ───────────────────────────────────────────
BIAS_EXPIRY_BARS          = 10        # Signal expires if no momentum in N bars

# ── FILTERS (toggle individually for analysis) ─────
USE_FVG_QUALITY           = True      # FVG first-touch only
USE_HTF_OB_CONFLUENCE     = True      # Require price near HTF Order Block
USE_PREMIUM_DISCOUNT      = True      # Only buy in discount, sell in premium
USE_PULLBACK_FILTER       = True      # Require counter-trend candle before FVG
USE_AUTO_BREAKEVEN        = True      # Move SL to entry after 1.5R
USE_LTF_OB_ENTRY          = True      # Enable LTF Order Block entries

# ── HTF OB ─────────────────────────────────────────
OB_MOVE_MULT              = 1.5       # Impulse must be 1.5x avg range
OB_SWING_LOOKBACK         = 5         # Bars to look for swing point

# ── CISD ───────────────────────────────────────────
CISD_BODY_MULT            = 0.5       # CISD candle body must be > 50% of avg range

# ── PREMIUM/DISCOUNT ───────────────────────────────
PD_ARRAY_BARS             = 20        # Bars for HTF range calculation

# ── VALIDATION ─────────────────────────────────────
MONTE_CARLO_SIMS          = 10000     # Number of random simulations
MIN_TRADES_FOR_VALIDITY   = 200       # Minimum trades to trust results
CONSISTENCY_PASS_PCT      = 0.60      # 60% profitable months = pass
EDGE_REAL_THRESHOLD       = 0.70      # 70% MC sims profitable = real edge
WALK_FORWARD_TOLERANCE    = 0.15      # 15% deviation train vs test = pass
 
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
