"""Tier 3 — flow / mood / register state updates.

Read Tier 1 and Tier 2 fields (so this stage can behave differently
depending on what the linguistic stage decided) and the incoming token.
Update fields that capture tone, cadence, urgency, formality drift,
emotional arc, imagery density — the *feel* of the text.

This is the stage to use when your rule is pointing at something a
reader senses but that doesn't have a crisp linguistic label.

Starts as a no-op. Split into sub-stages (cadence.py, register.py,
emotion.py, imagery.py, etc.) as the logic grows; call them in sequence
from `update_flow`.
"""

from __future__ import annotations

from ..state import ModelState


def update_flow(state: ModelState, token_id: int) -> ModelState:
    return state
