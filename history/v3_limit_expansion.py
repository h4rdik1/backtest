import pandas as pd
import numpy as np

# CONFIGURATION
SYMBOL = "BTC/USDT"
RISK_REWARD = 3.0
FIXED_SL_USDT = 25.0
BIAS_EXPIRY_BARS = 3 # Phase 3: Fresh bias
AUTO_BREAKEVEN_R = 1.5

# Milestone V3 introduced:
# - Aggressive 50% FVG Limit Orders (Consequent Encroachment)
# - Auto-Breakeven Risk Management
# - Structural Stops tucked behind swing points
# - Reduced Bias Expiry (3 bars)

print("Milestone V3: The Limit Expansion (Aggressive Entry Model).")
