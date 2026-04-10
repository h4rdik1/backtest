# ==============================================================
# config.py — Single source of truth for ALL backtester settings
# Both backtest_cisd.py and backtest_limit.py import from here.
# ==============================================================

# --------------------------------------------------
# SETUP
# --------------------------------------------------
SETUP                     = "daily"

# ── CURATED WATCHLIST (April 2026) ─────────────────
# UPDATED based on challenge analysis (Apr-09):
# BNB  - 63.6% WR, best performer across all challenges
# ETH  - 42.9% WR, consistent reactions, follows BTC with lag
# BTC  - kept for structure/context only, not primary trade vehicle
# SOL  - borderline, kept for now
# REMOVED: XRP (22% WR), LINK (28% WR) — statistically unprofitable
SYMBOLS                   = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
EXCHANGE                  = "binance"
LIVE_WATCHLIST            = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

# ---------------------------------------------------
# TIMEFRAME ALIGNMENT CHAINS
# Each chain defines: Context (HTF) → Bias (MTF) → Entry (LTF)
# Based on standard SMC timeframe pairing rules:
#   Monthly → Daily, Weekly → H4, Daily → H1, H4 → M15, H1 → M5
# ---------------------------------------------------
ALIGNMENTS = {
    "weekly":    {"context": "1w",  "bias": "4h",  "entry": "15m"},  # Swing trading    — keep (46% WR, 1.71 PF)
    "daily":     {"context": "1d",  "bias": "1h",  "entry": "5m"},   # Day trading      — keep (42% WR, 1.47 PF)
    # DISABLED based on challenge analysis (Apr-09):
    # "intraday":  33% WR, 1.00 PF — break-even at best
    # "scalp":     39% WR but 50% of all trades — volume killer, no edge
    # "macro":     0% WR, 3 trades all lost — remove
}

# ── RETIRED ALIGNMENTS (for position management of existing trades) ──
RETIRED_ALIGNMENTS = {
    "intraday":  {"context": "4h",  "bias": "15m", "entry": "5m"},
    "scalp":     {"context": "1h",  "bias": "5m",  "entry": "1m"},
    "macro":     {"context": "1M",  "bias": "1d",  "entry": "4h"},
}

ACTIVE_CHAIN              = "daily"   # <<< Default starting chain

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
FIXED_SL_USDT             = 50.0      # Dollar risk per trade (1% of $5000 account)
ACCOUNT_SIZE              = 5000.0    # Live capital: $5,000
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
USE_HTF_OB_CONFLUENCE     = True      # Require price near HTF OB  [ENABLED — analysis shows +quality]
USE_PREMIUM_DISCOUNT      = True      # Only buy in discount, sell in premium [ENABLED — kills counter-trend]
USE_PULLBACK_FILTER       = False     # Require pullback candle before entry
USE_AUTO_BREAKEVEN        = True      # Move SL to entry at 1.8R  [ENABLED — protect profits]
USE_LTF_OB_ENTRY          = True      # Enable LTF Order Block entries

# ── HTF OB ─────────────────────────────────────────
OB_MOVE_MULT              = 1.2       # Impulse 1.2x avg range
OB_SWING_LOOKBACK         = 5         # Bars to look for swing point

# ── PROTECTED SWING SL ──────────────────────────────
# Instead of placing SL just below the FVG boundary (too tight),
# the engine walks back up to N bars to find the last STRUCTURAL
# swing low (for longs) or swing high (for shorts) and places the
# SL just beyond it.  This gives the trade room to breathe without
# widening risk arbitrarily.
PROTECTED_SWING_LOOKBACK  = 20        # bars to look back for swing high/low
PROTECTED_SWING_BUFFER    = 0.001     # 0.1% clearance beyond the swing point

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

# ── CHALLENGE QUALITY GATES (added Apr-09 post-mortem) ──
# Based on analysis: B-tier signals had 29.6% WR, weak displacement
MIN_DISPLACEMENT_STRENGTH = 1.2      # Require explosive move (avg body 1.2x recent baseline)
MIN_SETUP_TIER            = "A"      # Only trade A or A++ setups — no more B-Tier
MAX_TRADES_PER_DAY        = 4        # Hard cap: max 4 new trades per calendar day

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
