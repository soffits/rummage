"""Hermes Agent plugin for rummage: recall from the user's files before each turn.

Hermes appends what `pre_llm_call` returns to the turn's user message, so the system prompt stays the same. A recall
that fails, or takes longer than RUMMAGE_RECALL_TIMEOUT seconds (default 5), adds nothing.
"""

import json
import os
import subprocess


def prompt_text(message) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        return "\n".join(p.get("text", "") for p in message if isinstance(p, dict) and p.get("type") == "text")
    return ""


def recall(session_id="", user_message="", **_):
    prompt = prompt_text(user_message)
    if not prompt.strip():
        return None
    command = os.environ.get("RUMMAGE_COMMAND", "rummage")
    timeout = float(os.environ.get("RUMMAGE_RECALL_TIMEOUT", "5"))
    event = json.dumps({"prompt": prompt, "session_id": session_id or "", "cwd": os.getcwd(), "harness": "hermes"})
    try:
        run = subprocess.run([command, "recall", "--hook"], input=event, capture_output=True, text=True,
                             timeout=timeout)
        reply = json.loads(run.stdout) if run.stdout.strip() else {}
        block = (reply.get("hookSpecificOutput") or {}).get("additionalContext", "")
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        return None
    return {"context": block} if block else None


def register(ctx):
    ctx.register_hook("pre_llm_call", recall)
