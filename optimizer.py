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
import random
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

# How often to inject a structural-reflection nudge into the session.
_NUDGE_INTERVAL_SEC = 1800  # 30 minutes

# Nudge messages rotated through to break the agent out of tuning grooves and
# push it toward structural moves. Each is phrased as user guidance, not a
# specific prescription — the agent chooses the concrete action.
_NUDGES: list[str] = [
    (
        "Pause and zoom out. Look at your last ~10 commits — are they mostly "
        "scale/constant tuning and dictionary expansion, or are they adding "
        "new structural capability (new state fields, new pipeline stages, "
        "new predict layers)? If the first, force yourself to a structural "
        "move next. Run a fresh sample at length 400 with a realistic prefix, "
        "read it critically, and pick the ONE biggest gap between it and real "
        "Shakespeare. Build state + pipeline logic that targets that gap."
    ),
    (
        "Reality-check break. Generate a sample with prefix \"HAMLET:\\nTo be "
        "or not to be, \" length 400, read it aloud in your head, and list "
        "three specific failures — not general impressions, concrete failures "
        "like 'no multi-word phrase coherence beyond 2 words', 'verse rhythm "
        "is char-counted not syllable-counted', 'no awareness of which "
        "character is speaking'. Pick the failure with the highest leverage "
        "and invent a state field + pipeline stage to close it."
    ),
    (
        "Diversity check. Look at what axes your state currently captures "
        "(local char context, word completion, line counters, speaker-label "
        "FSM, basic POS, line-length buckets). Now list axes it DOESN'T "
        "capture (multi-word memory, syntactic role, clause depth, scene "
        "mood, verse prosody via syllables, addressee tracking, formulaic "
        "phrase progress, rhyme position). Pick an unexplored axis and build "
        "state for it. Don't reach for another scale tweak — the low-hanging "
        "fruit there is picked."
    ),
    (
        "The flow tier of state was supposed to capture mood, register, "
        "cadence, imagery density — the *feel* of text, not just "
        "bookkeeping. Most current flow fields are mechanical counters. "
        "Add at least one field that captures actual text-texture "
        "(scene register, emotional intensity, formulaic-phrase progress, "
        "rhyme-position, archaic-density, tonal_weight, something like "
        "that) and wire a predict layer that reads it."
    ),
    (
        "Ambitious move time. Is there a structural capability the model "
        "lacks that would require a new pipeline stage or a meaningful "
        "rewrite of an existing one? Examples: syllable counting instead "
        "of char counting for verse, a recent-words tuple instead of just "
        "last_completed_word, clause-depth tracking, current-speaker "
        "memory across lines. Pick one. Even if the first pass drops BPC "
        "by less than your recent tweaks, a new capability unlocks future "
        "wins that scale tuning can't."
    ),
    (
        "Semantic-coherence gap. Zoom in on a sample line-by-line. You'll "
        "find phrases like 'my mother is niece', 'throne of treasure', "
        "'phantom anent there' — locally grammatical but semantically "
        "absurd. Every word-transition is plausible; the problem is that "
        "successive words don't belong to compatible semantic frames. No "
        "amount of bigram/trigram/phrase expansion will fix this, because "
        "the failure is cross-word semantic incompatibility, not rare word "
        "co-occurrence. Add state that captures coarse semantic class — "
        "e.g. noun-class tags (KINSHIP / ROYALTY / BODY / EMOTION / NATURE "
        "/ ABSTRACT / WEAPON / PLACE / TIME / ...), verb-argument "
        "expectation (what noun class does the verb want as object?), "
        "topic-field persistence. Use these to gate next-word biases: "
        "after a KINSHIP noun, bias toward KINSHIP continuations or "
        "compatible modifiers, not arbitrary concrete nouns. Even a 10-15 "
        "class tagger with ~100 dictionary entries each would meaningfully "
        "reduce semantic drift and improve sample quality substantially — "
        "this is the gap between 'Shakespeare-shaped' and 'Shakespeare'."
    ),
]


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
        betas=["context-1m-2025-08-07"],  # 1M context, fewer session restarts
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

        nudge_order = list(range(len(_NUDGES)))
        random.shuffle(nudge_order)
        nudge_cursor = 0

        async def nudge_loop() -> None:
            nonlocal nudge_cursor
            while True:
                await asyncio.sleep(_NUDGE_INTERVAL_SEC)
                msg = _NUDGES[nudge_order[nudge_cursor % len(nudge_order)]]
                nudge_cursor += 1
                _log_event(log_file, {"type": "nudge_injected", "msg": msg})
                print(f"[nudge] injecting: {msg[:140]}...")
                sys.stdout.flush()
                try:
                    await client.query(msg)
                except Exception as e:
                    print(f"[nudge-error] {e}")
                    sys.stdout.flush()
                    return

        nudge_task = asyncio.create_task(nudge_loop())
        try:
            async for event in client.receive_response():
                _log_event(log_file, event)
                _print_event(event)
        finally:
            nudge_task.cancel()
            try:
                await nudge_task
            except (asyncio.CancelledError, Exception):
                pass


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
