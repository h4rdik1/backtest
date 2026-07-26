# agent.py — Tool definitions and agent loop for the Trading Scanner Agent

import os
import json
import subprocess
from openai import OpenAI
from config import NVIDIA_API_KEY

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# Project root — all file access is sandboxed here
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ─── Tool Implementations ────────────────────────────────────────────────────

def list_codebase_files(dir: str = ".") -> str:
    """List Python files in a directory, sandboxed to PROJECT_ROOT."""
    target = os.path.abspath(os.path.join(PROJECT_ROOT, dir))
    if not target.startswith(PROJECT_ROOT):
        return "[ERROR] Access denied: path is outside the project directory."
    if not os.path.isdir(target):
        return f"[ERROR] Directory not found: {dir}"

    results = []
    for root, dirs, files in os.walk(target):
        # Skip hidden folders and cache
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        rel_root = os.path.relpath(root, PROJECT_ROOT)
        for f in files:
            if f.endswith(".py") or f.endswith(".md"):
                results.append(os.path.join(rel_root, f).replace("\\", "/"))
        if len(results) > 60:  # cap to avoid overload
            results.append("... (truncated)")
            break

    return "\n".join(results) if results else "No .py or .md files found."


def read_file(path: str) -> str:
    """Read up to 200 lines of a file, sandboxed to PROJECT_ROOT."""
    target = os.path.abspath(os.path.join(PROJECT_ROOT, path))
    if not target.startswith(PROJECT_ROOT):
        return "[ERROR] Access denied: path is outside the project directory."
    if not os.path.isfile(target):
        return f"[ERROR] File not found: {path}"

    with open(target, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    MAX_LINES = 200
    content = "".join(lines[:MAX_LINES])
    if len(lines) > MAX_LINES:
        content += f"\n... (file truncated at {MAX_LINES} lines, {len(lines)} total)"
    return content


def run_script(path: str, args: list, timeout: int = 60) -> str:
    """Run a Python script as a subprocess, sandboxed to PROJECT_ROOT."""
    # Blocked scripts — too dangerous to trigger via chat
    BLOCKED = ["challenge_trader.py", "live_scanner.py"]
    if any(os.path.basename(path) == b for b in BLOCKED):
        return f"[BLOCKED] {os.path.basename(path)} cannot be triggered via the bot (it manages live trades). Use scan_once.py for scanning."

    target = os.path.abspath(os.path.join(PROJECT_ROOT, path))
    if not target.startswith(PROJECT_ROOT):
        return "[ERROR] Access denied: path is outside the project directory."
    if not os.path.isfile(target):
        return f"[ERROR] Script not found: {path}"

    # Safe, no shell injection — args are always a discrete list
    cmd = ["python", target] + [str(a) for a in args]

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace"
        )
        output = result.stdout.strip()
        if result.returncode != 0 and result.stderr:
            output += f"\n[STDERR] {result.stderr.strip()}"
        return output if output else "(Script ran but produced no output)"
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] Script exceeded {timeout}s and was terminated."
    except Exception as e:
        return f"[ERROR] Failed to run script: {e}"


# ─── Tool dispatch ───────────────────────────────────────────────────────────

TOOL_MAP = {
    "list_codebase_files": lambda args: list_codebase_files(args.get("dir", ".")),
    "read_file":           lambda args: read_file(args["path"]),
    "run_script":          lambda args: run_script(args["path"], args.get("args", []), args.get("timeout", 60)),
}

# OpenAI-compatible tool definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_codebase_files",
            "description": "Lists Python and Markdown files in the trading project. Use this first to find the right script before running anything.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir": {
                        "type": "string",
                        "description": "Subdirectory to list, relative to project root. Default is '.' (root).",
                        "default": "."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the project. Use this to check a script's CLI arguments before running it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to project root. e.g. 'tools/analyze_setup.py'"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_script",
            "description": "Execute a Python script in the project and return its output. Never run challenge_trader.py or live_scanner.py directly — use telegramBot/scan_once.py for scans.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the script, relative to project root."
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "CLI arguments to pass to the script."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Max seconds to wait before killing the script. Default 60.",
                        "default": 60
                    }
                },
                "required": ["path", "args"]
            }
        }
    }
]

AGENT_SYSTEM_PROMPT = """You are the execution layer for a personal SMC/ICT crypto trading system.

You have access to 3 tools:
1. list_codebase_files(dir) — find scripts in the project
2. read_file(path) — check a script's arguments before running it
3. run_script(path, args, timeout) — execute a script and get its output

KNOWN SCRIPTS (prefer these — no need to search):
- scratch/scan_now.py          → one-shot full market scan, no arguments needed
- tools/analyze_setup.py <SYM> → deep analysis of a single symbol (e.g. BTC/USDT)
- scratch/check_all_4h_ranges.py → check 4H ranges across all symbols
- scratch/analyze_sui.py       → SUI-specific analysis

ROUTING RULES:
- "scan", "any signal?", "what can I trade?", "/scan" → run scratch/scan_now.py (no args)
- "analyze BTC/ETH/SOL/BNB" or "deep analysis" → run tools/analyze_setup.py with the symbol
- "backtest" → look in models/ directory first with list_codebase_files
- Unknown request → call list_codebase_files("scratch") then list_codebase_files("tools") before guessing

EXECUTION RULES:
- NEVER run challenge_trader.py or live_scanner.py (live trade managers — blocked)
- Always pass symbol as uppercase with slash: BTC/USDT not btc or BTCUSDT
- If a script errors, surface the real error — do not paraphrase it as success
- If it times out, say so clearly

OUTPUT FORMAT (phone-friendly — keep it short):
For scan results format each signal as:
📊 SYMBOL [CHAIN] — Tier X | LONG/SHORT
Entry: $X | SL: $X | TP: $X (2R) | Status: PENDING/ACTIVE

For deep analysis: 5-6 bullet points covering regime, bias, entry zone, SFPs, verdict.
If no setup found: one line saying so.
No padding, no generic advice."""


def run_agent(user_text: str) -> str:
    """Run the agent loop — AI calls tools until it has a final answer."""
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_text}
    ]

    MAX_ITERATIONS = 6  # Prevent infinite loops

    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model="deepseek-ai/deepseek-v4-pro",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=1500
        )

        msg = response.choices[0].message

        # If no tool calls → final answer
        if not msg.tool_calls:
            return msg.content or "(No response)"

        # Append assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in msg.tool_calls
            ]
        })

        # Execute each tool call and append results
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            tool_fn = TOOL_MAP.get(fn_name)

            if tool_fn:
                result = tool_fn(fn_args)
            else:
                result = f"[ERROR] Unknown tool: {fn_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })

    return "Agent hit max iterations without a final answer. Try rephrasing your request."
