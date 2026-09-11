# ai.py — All AI logic (calling DeepSeek, building prompts)

import os
from openai import OpenAI
from config import NVIDIA_API_KEY

# Initialize the NVIDIA NIM client once at startup
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)


def _load_rules() -> str:
    """Load personal trading rules from the codebase."""
    rules_dir = os.path.join(os.path.dirname(__file__), "..")  # e:\backtest\

    files = [
        ("TRADING_RULES.md",        "COMPLETE TIER FRAMEWORK & RULES"),
        ("S_TIER_TRADING_RULES.md", "S-TIER SETUP SEQUENCE"),
    ]

    combined = ""
    for filename, label in files:
        path = os.path.join(rules_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                combined += f"\n\n--- {label} ---\n{f.read()}"

    return combined.strip()


# Load once at startup
TRADING_RULES = _load_rules()

SYSTEM_PROMPT = f"""You are a sharp, no-nonsense personal trading coach. You know this trader's exact rulebook inside out.
Evaluate trades ONLY using the personal rules below — never give generic SMC advice.

=== TRADER'S PERSONAL RULEBOOK ===
{TRADING_RULES}
=== END OF RULEBOOK ===

=== HOW TO RESPOND TO TRADE IDEAS ===

The trader will describe setups conversationally — sometimes fully, sometimes partially.
Your job is to work with WHATEVER they give you.

RULES FOR RESPONDING:
1. Extract every piece of information from their message (pair, direction, entry, SL, TP, TF, SFP, OB, FVG, MSB, HTF bias — whatever is mentioned).
2. Evaluate those pieces directly against the rulebook. Do NOT list unmentioned items as "Not provided."
3. If a piece of info is CRITICAL to the verdict and missing, ask for ONLY that one thing — briefly, at the end.
4. Always give a verdict based on what you have. Never refuse to evaluate because something is missing.
5. Identify the Tier (S / A++ / A / B) if enough context exists. If not, say why briefly.
6. List any rule violations clearly.
7. End every trade evaluation with one of:
   ✅ VALID (Tier X) — Entry: [price] | SL: [price] | TP: [price]
   OR
   ❌ SKIP — [one clear reason]
   OR
   ⏳ PENDING — [what you need to confirm before entry]

TONE: Direct. Concise. Like a sharp trading partner texting you back — not a report.
FORMAT: Short paragraphs or bullets. No 10-point numbered lists unless the setup is fully detailed.

=== HOW TO RESPOND TO GENERAL QUESTIONS ===
- Answer directly and conversationally using the rulebook.
- No checklists. No generic advice. Keep it tight."""


def analyze_trade(user_text: str) -> str:
    """Send user's trade idea to DeepSeek and return the analysis."""
    completion = client.chat.completions.create(
        model="deepseek-ai/deepseek-v4-pro",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text}
        ],
        temperature=0.3,
        max_tokens=900
    )
    return completion.choices[0].message.content
