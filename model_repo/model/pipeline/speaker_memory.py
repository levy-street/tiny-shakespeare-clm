"""Tier 2/3 — speaker memory across dialogue turns.

Reads the `last_speaker_label` field (captured by the linguistic stage
at the 2->3 speaker-FSM transition, i.e., the exact moment the ":"
closes a speaker label) and maintains a rolling window of the most
recent distinct speakers.

`recent_speakers` is a tuple of up to 7 canonical uppercase names,
most-recent first. Element [0] is the current speaker (last label
seen). Element [1] is the previous speaker. Etc. Seven is the empirically
optimal capacity: enough to hold the full cast of an ensemble scene
(often 4-6 speakers alternating with a couple of minor voices); more
than that, old cross-scene speakers start polluting the partner-letter
vote and the recency signal degrades.

The detection heuristic: `last_speaker_label` changes only at the
moment a new label closes. We compare against the previous state's
`recent_speakers[0]` (if any) to detect the transition. When a new
speaker is detected, we prepend it; if it was already in the window,
we just move it to front.

This capability unlocks:
  - biasing the speaker-label prediction to favor recently-seen
    speakers (a scene usually has 2-4 alternating characters)
  - penalizing immediate self-repeat (a speaker almost never follows
    themselves with another label)
  - gently conditioning dialogue vocabulary on who's speaking in the
    future (not implemented here — future consumer)
"""

from __future__ import annotations

from ..state import ModelState

_MAX_RECENT = 7


def update_speaker_memory(state: ModelState, token_id: int) -> ModelState:
    lbl = state.last_speaker_label
    rs = state.recent_speakers

    # No active speaker label captured yet.
    if not lbl:
        return state

    # Check if this label is already the head of recent_speakers.
    # If so, nothing changed — same speaker still talking.
    if rs and rs[0] == lbl:
        return state

    # A new speaker just closed their label. Prepend and dedupe.
    new_rs = [lbl]
    for s in rs:
        if s != lbl and len(new_rs) < _MAX_RECENT:
            new_rs.append(s)
    return state.model_copy(update={"recent_speakers": tuple(new_rs)})
