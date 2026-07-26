# handlers.py — All Telegram message handler functions

import os
import sys
import subprocess
from datetime import datetime
from telegram import Update
from telegram.ext import ContextTypes
from config import IMAGES_DIR
from ai import analyze_trade
from agent import run_agent

# Map common coin names/tickers → standard symbol for analyze_setup.py
SYMBOL_MAP = {
    "btc": "BTC/USDT",  "bitcoin": "BTC/USDT",
    "eth": "ETH/USDT",  "ethereum": "ETH/USDT",
    "bnb": "BNB/USDT",
    "sol": "SOL/USDT",  "solana": "SOL/USDT",
    "sui": "SUI/USDT",
    "xrp": "XRP/USDT",  "ripple": "XRP/USDT",
    "doge": "DOGE/USDT", "dogecoin": "DOGE/USDT",
    "avax": "AVAX/USDT", "avalanche": "AVAX/USDT",
    "link": "LINK/USDT", "chainlink": "LINK/USDT",
}

def _extract_symbol(text: str) -> str | None:
    """Return the first recognised trading symbol found in the user's message."""
    lower = text.lower()
    for keyword, symbol in SYMBOL_MAP.items():
        if keyword in lower:
            return symbol
    return None


def _run_analyze(symbol: str) -> str:
    """Run tools/analyze_setup.py <symbol> and return its printed output."""
    script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "tools", "analyze_setup.py")
    )
    try:
        result = subprocess.run(
            [sys.executable, script, symbol],
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
            errors="replace",
            cwd=os.path.dirname(script),
        )
        output = result.stdout.strip()
        if not output and result.stderr:
            output = result.stderr.strip()
        return output or "(analyze_setup.py produced no output)"
    except subprocess.TimeoutExpired:
        return "[Live scan timed out after 60 s — evaluate from trader note only]"
    except Exception as e:
        return f"[Live scan error: {e} — evaluate from trader note only]"


# Keywords that mean the user wants a live scan, not a manual setup evaluation
SCAN_KEYWORDS = [
    "scan", "signal", "any trade", "can i trade", "what can i trade",
    "check market", "any setup", "what setup", "look at", "any signal",
    "check btc", "check eth", "check bnb", "check sol",
    "analyze btc", "analyze eth", "analyze bnb", "analyze sol",
    "analyse btc", "analyse eth", "analyse bnb", "analyse sol",
]

def _is_scan_request(text: str) -> bool:
    """Detect if user wants a live scan vs evaluating their own trade idea."""
    lower = text.lower()
    return any(kw in lower for kw in SCAN_KEYWORDS)


async def _send_reply(update: Update, status_msg, reply: str):
    """Send a reply, splitting into chunks if it exceeds Telegram's 4000-char limit."""
    MAX_LEN = 4000
    if len(reply) <= MAX_LEN:
        await status_msg.edit_text(reply)
    else:
        await status_msg.delete()
        chunks = [reply[i:i+MAX_LEN] for i in range(0, len(reply), MAX_LEN)]
        for chunk in chunks:
            await update.message.reply_text(chunk)


# ─── Commands ────────────────────────────────────────────────────────────────

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hello! I am your Trading Assistant Bot.\n\n"
        "📸 Send a *chart image* → saves it to your images folder.\n"
        "💬 Describe a *trade setup* → I evaluate it against your rulebook.\n"
        "🔍 Ask *any scan question* → I run your live scanner.\n"
        "⚡ `/scan` → immediate full market scan.\n"
        "💰 `/alert BTC 64000` → price alert (coming soon!)",
        parse_mode="Markdown"
    )


async def get_id_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Returns Chat ID and Thread ID — used to configure Topics routing."""
    chat_id   = update.message.chat_id
    thread_id = update.message.message_thread_id
    await update.message.reply_text(
        f"**Chat ID:** `{chat_id}`\n**Thread ID:** `{thread_id}`",
        parse_mode="Markdown"
    )


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/scan [symbol] — immediately triggers a market scan via the agent."""
    query = "/scan " + " ".join(context.args) if context.args else "/scan full watchlist (BTC ETH BNB SOL)"
    status_msg = await update.message.reply_text("🔍 Scanning the market...")
    try:
        reply = run_agent(query)
        await _send_reply(update, status_msg, reply)
    except Exception as e:
        await status_msg.edit_text(f"❌ Scan error: {e}")


# ─── Message Handlers ────────────────────────────────────────────────────────

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Smart router: scan request → agent, trade idea → rulebook evaluator.

    If the trade idea mentions a known coin, we first run analyze_setup.py
    to pull live 4H/1H/5M data, then stitch that with the user's note and
    send the combined context to analyze_trade() for a fully-informed verdict.
    """
    user_text = update.message.text

    if _is_scan_request(user_text):
        status_msg = await update.message.reply_text("🔍 Running scan...")
        try:
            reply = run_agent(user_text)
            await _send_reply(update, status_msg, reply)
        except Exception as e:
            await status_msg.edit_text(f"❌ Scan error: {e}")
    else:
        symbol = _extract_symbol(user_text)

        if symbol:
            status_msg = await update.message.reply_text(
                f"📡 Pulling live {symbol} data..."
            )
            try:
                live_data = _run_analyze(symbol)
                enriched_text = (
                    f"=== LIVE CHART DATA FOR {symbol} ===\n"
                    f"{live_data}\n"
                    f"=== END LIVE DATA ===\n\n"
                    f"=== TRADER'S NOTE ===\n"
                    f"{user_text}"
                )
                await status_msg.edit_text("🤖 Evaluating against your rulebook...")
                reply = analyze_trade(enriched_text)
                await _send_reply(update, status_msg, reply)
            except Exception as e:
                await status_msg.edit_text(f"❌ Error: {e}")
        else:
            status_msg = await update.message.reply_text("🤖 Analyzing your trade idea...")
            try:
                reply = analyze_trade(user_text)
                await _send_reply(update, status_msg, reply)
            except Exception as e:
                await status_msg.edit_text(f"❌ Error talking to AI: {e}")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives a chart photo and saves it to the images folder."""
    photo_file = await update.message.photo[-1].get_file()

    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path  = os.path.join(IMAGES_DIR, f"chart_{timestamp}.jpg")

    await photo_file.download_to_drive(file_path)
    await update.message.reply_text(
        f"✅ Chart saved as:\n`{file_path}`",
        parse_mode="Markdown"
    )
