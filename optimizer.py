"""Single-agent optimizer for the v2 model.

Spawns one Claude Opus 4.7 session (max effort, unlimited turns), tells it
to improve the model under `model_repo/model/`, and lets it run until it
exits or you stop it with Ctrl+C. A PreToolUse hook blocks access to dev/val
splits. If the session ends for any reason, a new one starts automatically
— model state persists via git commits.

Logs stream to stdout and to `logs/optimizer-<ts>.jsonl`.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

_HERE = Path(__file__).resolve().parent
_LOGS = _HERE / "logs"
_LOGS.mkdir(exist_ok=True)
_PROMPT_PATH = _HERE / "optimizer_prompt.md"

# Substrings that, if they appear anywhere in a tool's input, trigger a deny.
# Covers the dev/val corpus (v1's clm/corpus/) in both relative and absolute
# path forms, plus the filenames themselves.
_BLOCKED_SUBSTRINGS = (
    "dev.txt",
    "val.txt",
    "clm/corpus",
    "/clm/",
    "../clm",
)


async def guard_hook(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: Any,
) -> dict[str, Any]:
    """Block any tool call whose string inputs reference dev/val or v1's corpus."""
    tool_input = input_data.get("tool_input") or {}
    strings = " ".join(
        str(v) for v in tool_input.values() if isinstance(v, (str, int, float))
    ).lower()
    for blocked in _BLOCKED_SUBSTRINGS:
        if blocked in strings:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"Access matching '{blocked}' is forbidden. dev.txt, val.txt, "
                        f"and anything under ../clm/corpus/ are operator-only. "
                        f"Only v2/corpus/train.txt may be read."
                    ),
                }
            }
    return {}


def _log_event(log_file, event: Any) -> None:
    """Append a structured record of one SDK event."""
    record: dict[str, Any] = {"ts": time.time(), "type": type(event).__name__}
    try:
        if isinstance(event, AssistantMessage):
            record["blocks"] = []
            for b in event.content:
                if isinstance(b, TextBlock):
                    record["blocks"].append({"t": "text", "v": b.text})
                elif isinstance(b, ThinkingBlock):
                    record["blocks"].append({"t": "thinking", "v": b.thinking})
                elif isinstance(b, ToolUseBlock):
                    record["blocks"].append(
                        {"t": "tool_use", "name": b.name, "input": b.input}
                    )
        elif isinstance(event, UserMessage):
            record["blocks"] = []
            for b in event.content:
                if isinstance(b, ToolResultBlock):
                    record["blocks"].append(
                        {
                            "t": "tool_result",
                            "is_error": b.is_error,
                            "content": str(b.content)[:4000],
                        }
                    )
        elif isinstance(event, ResultMessage):
            record["cost_usd"] = getattr(event, "total_cost_usd", None)
            record["num_turns"] = getattr(event, "num_turns", None)
            record["stop_reason"] = getattr(event, "stop_reason", None)
        else:
            record["repr"] = repr(event)[:500]
    except Exception as e:
        record["log_error"] = str(e)
    log_file.write(json.dumps(record, default=str) + "\n")
    log_file.flush()


def _print_event(event: Any) -> None:
    """Render key events to stdout for tail -f visibility."""
    try:
        if isinstance(event, AssistantMessage):
            for b in event.content:
                if isinstance(b, TextBlock):
                    print(f"[assistant] {b.text}")
                elif isinstance(b, ThinkingBlock):
                    snippet = (b.thinking or "").strip().replace("\n", " ")[:250]
                    print(f"[thinking] {snippet}")
                elif isinstance(b, ToolUseBlock):
                    inp = json.dumps(b.input, default=str)[:300]
                    print(f"[tool:{b.name}] {inp}")
        elif isinstance(event, UserMessage):
            for b in event.content:
                if isinstance(b, ToolResultBlock):
                    content_str = str(b.content).strip().replace("\n", " ")[:300]
                    tag = "tool-error" if b.is_error else "tool-result"
                    print(f"[{tag}] {content_str}")
        elif isinstance(event, ResultMessage):
            cost = getattr(event, "total_cost_usd", None)
            turns = getattr(event, "num_turns", None)
            print(f"[session-end] turns={turns} cost=${cost}")
        sys.stdout.flush()
    except Exception as e:
        print(f"[print-error] {e}")
        sys.stdout.flush()


async def run_session(log_file) -> None:
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    options = ClaudeAgentOptions(
        model="claude-opus-4-7",
        system_prompt=system_prompt,
        cwd=str(_HERE),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob"],
        permission_mode="bypassPermissions",
        max_turns=None,  # unlimited
        effort="max",
        hooks={
            "PreToolUse": [HookMatcher(matcher=None, hooks=[guard_hook])],
        },
    )
    async with ClaudeSDKClient(options=options) as client:
        await client.query(
            "Improve the model. Lower train BPC AND make samples feel like "
            "real Shakespeare when prompted with realistic prefixes. Commit "
            "progress to git (`git add -A && git commit`) as soon as an "
            "improvement lands. Run indefinitely. Begin."
        )
        async for event in client.receive_response():
            _log_event(log_file, event)
            _print_event(event)


async def main() -> None:
    log_path = _LOGS / f"optimizer-{int(time.time())}.jsonl"
    print(f"[optimizer] logging to {log_path}")
    print(f"[optimizer] system prompt: {_PROMPT_PATH}")
    with open(log_path, "a", encoding="utf-8") as log_file:
        backoff = 10
        while True:
            print(f"[optimizer] session starting at {time.strftime('%H:%M:%S')}")
            try:
                await run_session(log_file)
                backoff = 10  # reset after a clean session
            except KeyboardInterrupt:
                print("[optimizer] interrupted, exiting")
                return
            except Exception as e:
                _log_event(log_file, {"outer_error": repr(e)})
                print(f"[optimizer] session error: {e}")
                backoff = min(backoff * 2, 300)
            print(
                f"[optimizer] session ended — restarting in {backoff}s "
                f"(Ctrl+C to stop)"
            )
            try:
                await asyncio.sleep(backoff)
            except KeyboardInterrupt:
                print("[optimizer] interrupted during sleep, exiting")
                return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
