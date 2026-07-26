# ==============================================================
# config.py — Single source of truth for ALL backtester settings
# Both backtest_cisd.py and backtest_limit.py import from here.
# UPDATED: Added Task 1-10 config constants (Tasks 1-10 additions)
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
LIVE_SCAN_INTERVAL        = 900       # Scan interval in seconds (15 minutes)


# ---------------------------------------------------
# TIMEFRAME ALIGNMENT CHAINS
# Each chain defines: Context (HTF) → Bias (MTF) → Entry (LTF)
# Based on standard SMC timeframe pairing rules:
#   Monthly → Daily, Weekly → H4, Daily → H1, H4 → M15, H1 → M5
# ---------------------------------------------------
ALIGNMENTS = {
    "swing":     {"context": "1w",  "bias": "4h",  "entry": "15m"},  # Weekly -> 4H -> 15M (W/m15)
    "intraday":  {"context": "1d",  "bias": "1h",  "entry": "5m"},   # Daily -> 1H -> 5M (D/m5)
}

# ── RETIRED ALIGNMENTS (for position management of existing trades) ──
RETIRED_ALIGNMENTS = {
    "intraday_legacy": {"context": "4h",  "bias": "1h",  "entry": "15m"},  # Legacy intraday
    "scalp_legacy":     {"context": "1h",  "bias": "15m", "entry": "5m"},   # Legacy scalp
    "macro":            {"context": "1M",  "bias": "1d",  "entry": "4h"},
}

ACTIVE_CHAIN              = "intraday"   # <<< Default starting chain

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
AUTO_BREAKEVEN_R          = 1.8       # Move SL to entry at 1.8R (legacy, see BE_TRIGGER_R)

# ── FVG ENGINE (LuxAlgo exact values) ──────────────
FVG_WICK_RATIO            = 0.36      # CHANGED from 0.5 to match user spec
FVG_AVG_BODY_LOOKBACK     = 10
FVG_MIN_SIZE_MULT         = 0.5       # CHANGED from 0.3 to filter weak FVGs
FVG_MIN_VOLUME_MULT       = 1.0

# ── TASK 7 — FVG Expiry by Timeframe ───────────────
# WHY: An FVG that formed 3 months ago on a 5-minute chart is no longer
# relevant. These limits keep the system trading fresh structure only.
FVG_EXPIRY_BARS           = {
    "1w" : 12,   # weekly FVG valid for 12 bars = ~3 months
    "4h" : 30,   # 4H FVG valid for 30 bars = 5 days
    "1h" : 48,   # 1H FVG valid for 48 bars = 2 days
    "15m": 96,   # 15M FVG valid for 96 bars = 1 day
    "5m" : 144,  # 5M FVG valid for 144 bars = 12 hours
}

# ── BIAS ───────────────────────────────────────────
BIAS_EXPIRY_BARS          = 12        # Reduced from 24 to prevent stale biases in fast markets

# ── FILTERS (toggle individually for analysis) ─────
USE_FVG_QUALITY           = True      # FVG first-touch only
USE_HTF_OB_CONFLUENCE     = True      # Require price near HTF OB
USE_PREMIUM_DISCOUNT      = False     # Only buy in discount, sell in premium
USE_PULLBACK_FILTER       = False     # Require pullback candle before entry
USE_AUTO_BREAKEVEN        = True      # Move SL to entry at 1.8R
USE_LTF_OB_ENTRY          = True      # Enable LTF Order Block entries
SKIP_RANGING_REGIME       = True      # Avoid trading lower timeframes when HTF context is ranging
SKIP_COUNTER_TREND        = True      # Block trades running counter to the HTF trend

# ── HTF OB ─────────────────────────────────────────
OB_MOVE_MULT              = 1.2       # Impulse 1.2x avg range
OB_SWING_LOOKBACK         = 5         # Bars to look for swing point

# ── PROTECTED SWING SL ──────────────────────────────
PROTECTED_SWING_LOOKBACK  = 20        # bars to look back for swing high/low
PROTECTED_SWING_BUFFER    = 0.001     # 0.1% clearance beyond the swing point
PREFER_HTF_STRUCTURAL_SL  = True      # If True, prioritize HTF SFP stops & use Bias TF for swing SL fallback (avoids LTF shakeouts)

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

# ── CHALLENGE QUALITY GATES (updated May-05) ──
MIN_DISPLACEMENT_STRENGTH = 1.2      # Raised from 1.2 for better momentum
MIN_SETUP_TIER            = "B"      # Allow B-tier setups
MAX_TRADES_PER_DAY        = 4        # CHANGED from 5 — match Task 3 daily limit (MAX_DAILY_TRADES)
MIN_SL_PCT_BTC            = 0.003
MIN_SL_PCT_ALTS           = 0.005    # CHANGED to match MIN_SL_DISTANCE config
MAX_STOP_DISTANCE         = 0.035

# ── UTILITIES ──────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# TASK 1 — SFP Detection Constants
# WHY: The new 3-candle SFP needs precise thresholds to filter
# micro-wicks (noise) and ensure the sweep is real (price goes
# beyond the level AND closes firmly back inside it).
# ══════════════════════════════════════════════════════════════
MIN_SWEEP_BUFFER_PCT      = 0.0005   # 0.05% — wick must go 0.05% beyond C2 level
MAX_SWEEP_WAIT_BARS       = 10       # pattern expires if no sweep in 10 bars after C3
SFP_CLOSE_BUFFER_PCT      = 0.0001   # 0.01% — close must be 0.01% back inside level


# ══════════════════════════════════════════════════════════════
# TASK 2 — Market Regime Constants
# WHY: We need to know if the market is trending or ranging to
# avoid fighting the dominant structure.
# ══════════════════════════════════════════════════════════════
REGIME_LOOKBACK_BARS      = 20       # 20 weekly bars for regime classification
COUNTER_TREND_MIN_TIER    = "S"      # Only S-tier setups allowed against the trend


# ══════════════════════════════════════════════════════════════
# TASK 3 — Session Constants
# WHY: Institutional players are most active at London Open and
# NY Open. These are the highest-quality signal windows.
# Asian session is low-liquidity and produces fake moves.
# ══════════════════════════════════════════════════════════════
SESSIONS = {
    "london_open" : ("07:00", "11:00"),   # UTC — highest quality
    "ny_open"     : ("13:00", "16:00"),   # UTC — high quality
    "london_close": ("15:00", "17:00"),   # UTC — medium quality
    "asian"       : ("00:00", "07:00"),   # UTC — low quality, avoid unless S-tier
}
# Minimum tier required per session type
SESSION_MIN_TIER          = {"asian": "S", "other": "A"}


# ══════════════════════════════════════════════════════════════
# TASK 4 — Break-Even & Trailing Stop Constants
# WHY: Moving to BE at 1.8R was too late — price often reverses
# before hitting 1.8R. Moving to 1.5R gives more protection.
# Partial exit at 2R locks in profits while letting winners run.
# ══════════════════════════════════════════════════════════════
BE_TRIGGER_R              = 1.5     # CHANGED from 1.8R — move SL to entry at 1.5R
PARTIAL_EXIT_R            = 2.0     # Close 50% of position at 2R
PARTIAL_EXIT_PCT          = 0.50    # Close this fraction of position at PARTIAL_EXIT_R
TRAIL_AFTER_R             = 2.0     # After 2R, trail remaining 50% with LTF structure


# ══════════════════════════════════════════════════════════════
# TASK 5 — Correlation Groups
# WHY: BTC, ETH, SOL all move together. If BTC is long,
# adding ETH long is NOT diversification — it's doubling down.
# We only allow one position per correlation group.
# ══════════════════════════════════════════════════════════════
CORRELATION_GROUPS = [
    ["BTC/USDT", "ETH/USDT", "SOL/USDT"],   # group 1 — high correlation (crypto majors)
    ["BNB/USDT", "XRP/USDT"],               # group 2 — moderate correlation
]


# ══════════════════════════════════════════════════════════════
# TASK 6 — Setup Quality Scoring Thresholds
# WHY: Not all setups are equal. A setup with HTF SFP + session
# confluence + trend alignment deserves a larger position.
# Variable sizing amplifies winners and protects against losers.
# ══════════════════════════════════════════════════════════════
SCORE_THRESHOLDS = {
    "full"  : {"min_score": 9,  "risk_pct": 1.00},   # $50 risk (full size)
    "medium": {"min_score": 6,  "risk_pct": 0.70},   # $35 risk (70%)
    "small" : {"min_score": 4,  "risk_pct": 0.50},   # $25 risk (50%)
    "skip"  : {"min_score": 0,  "risk_pct": 0.00},   # skip (score too low)
}
BASE_RISK_USDT            = 50.0    # Full risk per trade in USD


# ══════════════════════════════════════════════════════════════
# TASK 8 — Drawdown Halt Constants
# WHY: A fixed floor ($4700) doesn't account for profits made.
# If we grow to $5500 then drop $300, we HAVE given back real
# profits — a trailing halt from peak prevents this.
# ══════════════════════════════════════════════════════════════
TRAILING_HALT_PCT         = 0.06    # 6% drawdown from peak triggers halt
MAX_WEEKLY_LOSS_PCT       = 0.08    # 8% weekly loss triggers pause until Monday


# ══════════════════════════════════════════════════════════════
# TASK 9 — Time-Based Trade Invalidation
# WHY: A setup that hasn't moved 1R in our favour after many bars
# is probably a failed setup. Holding it longer just ties up
# capital that could be used on better setups.
# ══════════════════════════════════════════════════════════════
MAX_TRADE_DURATION = {
    "15m": 48,   # 48 x 15M bars = 12 hours max hold
    "5m" : 96,   # 96 x 5M bars  = 8 hours max hold
}


# ══════════════════════════════════════════════════════════════
# TASK 10 — Live vs Backtest Validation Constants
# WHY: If our live results drift far from backtest, something
# has changed in the market or our execution. We must detect
# this early before significant damage is done.
# ══════════════════════════════════════════════════════════════
LIVE_BACKTEST_CHECK_N     = 20      # Run comparison every 20 completed live trades
LIVE_BACKTEST_WARN_PCT    = 0.15    # 15% deviation → log warning
LIVE_BACKTEST_HALT_PCT    = 0.25    # 25% deviation → halt trading and alert


# ══════════════════════════════════════════════════════════════
# RISK PARAMETERS (all tasks combined)
# ══════════════════════════════════════════════════════════════
MAX_OPEN_TRADES           = 2
MAX_DAILY_TRADES          = 4       # matches MAX_TRADES_PER_DAY
MIN_SL_DISTANCE           = {"BTC/USDT": 0.003, "default": 0.005}
SL_BUFFER_PCT             = 0.001


# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════

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


def run_system_health_check():
    """
    TASK — System Health Check.
    Validates all config values are sensible on startup.
    This catches silly misconfigurations before they cause bad trades
    or unexpected halts mid-session.

    Checks:
    - Risk values are positive and sane
    - Session times are valid HH:MM strings
    - Correlation groups contain known symbols
    - Score thresholds are in ascending order
    - FVG expiry bars are positive integers
    - Drawdown halt is between 1% and 50%
    """
    errors   = []
    warnings = []

    # ── Risk checks ──
    if BASE_RISK_USDT <= 0:
        errors.append(f"BASE_RISK_USDT must be positive, got {BASE_RISK_USDT}")
    if BASE_RISK_USDT > ACCOUNT_SIZE * 0.05:
        warnings.append(f"BASE_RISK_USDT ({BASE_RISK_USDT}) is >5% of account — very aggressive")
    if TRAILING_HALT_PCT <= 0 or TRAILING_HALT_PCT > 0.50:
        errors.append(f"TRAILING_HALT_PCT must be between 0 and 0.50, got {TRAILING_HALT_PCT}")
    if MAX_WEEKLY_LOSS_PCT <= 0 or MAX_WEEKLY_LOSS_PCT > 0.50:
        errors.append(f"MAX_WEEKLY_LOSS_PCT must be between 0 and 0.50, got {MAX_WEEKLY_LOSS_PCT}")

    # ── SFP checks ──
    if MIN_SWEEP_BUFFER_PCT <= 0:
        errors.append(f"MIN_SWEEP_BUFFER_PCT must be positive, got {MIN_SWEEP_BUFFER_PCT}")
    if MAX_SWEEP_WAIT_BARS < 3:
        warnings.append(f"MAX_SWEEP_WAIT_BARS={MAX_SWEEP_WAIT_BARS} is very small (minimum recommended: 5)")

    # ── Session time format check ──
    import re
    time_pat = re.compile(r"^\d{2}:\d{2}$")
    for sess_name, (start, end) in SESSIONS.items():
        if not time_pat.match(start) or not time_pat.match(end):
            errors.append(f"Session '{sess_name}' has invalid time format: {start}-{end} (expected HH:MM)")

    # ── Correlation group symbols ──
    all_known = set(SYMBOLS + LIVE_WATCHLIST)
    for g_idx, group in enumerate(CORRELATION_GROUPS):
        for sym in group:
            if sym not in all_known:
                warnings.append(f"Correlation group {g_idx} contains unknown symbol: {sym}")

    # ── Score thresholds must be descending ──
    tiers = ["full", "medium", "small", "skip"]
    prev_score = 999
    for tier in tiers:
        s = SCORE_THRESHOLDS[tier]["min_score"]
        if s > prev_score:
            errors.append(f"SCORE_THRESHOLDS are not in descending order at tier '{tier}'")
        prev_score = s

    # ── FVG expiry sanity ──
    for tf, bars in FVG_EXPIRY_BARS.items():
        if bars <= 0:
            errors.append(f"FVG_EXPIRY_BARS['{tf}'] must be positive, got {bars}")

    # ── BE and partial exit consistency ──
    if BE_TRIGGER_R >= PARTIAL_EXIT_R:
        warnings.append(f"BE_TRIGGER_R ({BE_TRIGGER_R}) >= PARTIAL_EXIT_R ({PARTIAL_EXIT_R}) — BE fires at or after partial exit")

    # ── Print results ──
    print("\n" + "=" * 62)
    print("  SYSTEM HEALTH CHECK")
    print("=" * 62)
    if errors:
        for e in errors:
            print(f"  [FAIL]  {e}")
    if warnings:
        for w in warnings:
            print(f"  [WARN]  {w}")
    if not errors and not warnings:
        print("  [PASS]  All config values look sane.")
    elif not errors:
        print(f"  [PASS]  No hard errors. {len(warnings)} warning(s) above.")
    else:
        print(f"  [FAIL]  {len(errors)} error(s) must be fixed before trading.")
    print("=" * 62)

    return len(errors) == 0  # True = healthy, False = something needs fixing
