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
# TIMEFRAME ALIGNMENT CHAINS
# Each chain defines: Context (HTF) → Bias (MTF) → Entry (LTF)
# Based on standard SMC timeframe pairing rules:
#   Monthly → Daily, Weekly → H4, Daily → H1, H4 → M15, H1 → M5
# ---------------------------------------------------
ALIGNMENTS = {
    "daily":     {"context": "1d",  "bias": "1h",  "entry": "5m"},   # Day trading
    "weekly":    {"context": "1w",  "bias": "4h",  "entry": "15m"},  # Swing trading
    "intraday":  {"context": "12h", "bias": "1h",  "entry": "5m"},   # Intraday scalping
}

ACTIVE_CHAIN              = "weekly"   # <<< CHANGE THIS to switch your entire alignment

# Derived from ACTIVE_CHAIN — used everywhere
_chain = ALIGNMENTS[ACTIVE_CHAIN]
TF_CONTEXT                = _chain["context"]   # HTF context (OBs, structure)
TF_BIAS                   = _chain["bias"]       # MTF bias (FVG/sweep → bias direction)
TF_ENTRY                  = _chain["entry"]      # LTF entry (FVG reaction, triggers)

# Legacy alias (used by some imports)
DEFAULT_LTF               = TF_ENTRY

# ---------------------------------------------------
# DATA
# ---------------------------------------------------
DATA_DIR                  = "data/ohlcv"
TIMEFRAMES                = ["5m", "15m", "1h", "4h", "12h", "1d", "1w"]
DAYS_BACK                 = 365       # 365 days for statistical validity
TRAIN_END                 = "2025-12-31"
TEST_START                = "2026-01-01"

# ---------------------------------------------------
# RISK
# ---------------------------------------------------
RISK_REWARD               = 2.0       # 2R — at 40% WR this gives PF 1.33
FIXED_SL_USDT             = 25.0      # Dollar risk per trade
ACCOUNT_SIZE              = 1000.0    # Starting account balance
AUTO_BREAKEVEN_R          = 1.8       # Move SL to entry at 1.8R

# ── FVG ENGINE (LuxAlgo exact values) ──────────────
FVG_WICK_RATIO            = 0.5
FVG_AVG_BODY_LOOKBACK     = 10
FVG_MIN_SIZE_MULT         = 0.3
FVG_MIN_VOLUME_MULT       = 1.0

# ── BIAS ───────────────────────────────────────────
BIAS_EXPIRY_BARS          = 24        # Bias stays active for 24 bars on the bias TF

# ── FILTERS (toggle individually for analysis) ─────
USE_FVG_QUALITY           = True      # FVG first-touch only
USE_HTF_OB_CONFLUENCE     = False     # Require price near HTF OB
USE_PREMIUM_DISCOUNT      = False     # Only buy in discount, sell in premium
USE_PULLBACK_FILTER       = False     # Require pullback candle before entry
USE_AUTO_BREAKEVEN        = False     # Move SL to entry after N-R
USE_LTF_OB_ENTRY          = True      # Enable LTF Order Block entries

# ── HTF OB ─────────────────────────────────────────
OB_MOVE_MULT              = 1.2       # Impulse 1.2x avg range
OB_SWING_LOOKBACK         = 5         # Bars to look for swing point

# ── CISD ───────────────────────────────────────────
CISD_BODY_MULT            = 0.3       # 30% of avg range

# ── PREMIUM/DISCOUNT ───────────────────────────────
PD_ARRAY_BARS             = 20

# ── VALIDATION ─────────────────────────────────────
MONTE_CARLO_SIMS          = 10000
MIN_TRADES_FOR_VALIDITY   = 100
CONSISTENCY_PASS_PCT      = 0.55
EDGE_REAL_THRESHOLD       = 0.65
WALK_FORWARD_TOLERANCE    = 0.20

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

def get_chain(name: str = None) -> dict:
    """Get a specific alignment chain by name, or the active one."""
    chain_name = name or ACTIVE_CHAIN
    if chain_name not in ALIGNMENTS:
        raise ValueError(f"Unknown chain '{chain_name}'. Available: {list(ALIGNMENTS.keys())}")
    return ALIGNMENTS[chain_name]

def chain_label(name: str = None) -> str:
    """Human-readable label for a chain."""
    c = get_chain(name)
    return f"{c['context'].upper()} -> {c['bias'].upper()} -> {c['entry'].upper()}"
