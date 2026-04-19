"""Predict layer — imperative-chain sentence-start bias.

Reads `state.imperative_chain_count`. When count >= 2, the recent
sentences have formed an imperative chain ("Speak! Attend!") and
the next sentence is more likely to open with another imperative
head verb. Boost the first-letter of the next sentence toward
imperative openers.

Imperative openers (capitals at sentence-start; lowercase at
line-internal sentence-start):
  G - Go, Give, Grant, Gird, Guard
  C - Come, Call, Cease, Cleanse
  S - Speak, Stand, Stay, See, Send, Sit, Set, Shame, Shun
  T - Tell, Take, Try, Turn
  H - Hear, Hold, Hark, Help, Haste, Hie
  A - Away, Attend, Arm, Avoid, Answer
  L - Let, Look, Live, Leave, List, Lead, Lay, Lend, Lift
  M - Mark, Mind, Make, Meet, Mend
  B - Be, Begone, Bring, Behold, Bid, Break, Bend
  F - Fly, Forbear, Follow, Fight, Fetch, Fear, Fie
  P - Peace, Pardon, Pray, Put, Push, Persist
  O - Open, Out
  W - Watch, Walk, Win, Wait, Withhold
  R - Rouse, Run, Return, Rest, Rise, Remember
  D - Do, Draw, Die, Deliver
  E - End, Endure, Enter
  K - Keep, Know, Kneel, Kill

Fires ONLY at sentence-start (first word of a fresh sentence). Scales
with count (2 → moderate; 3+ → stronger).
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_IMPER_CAPITALS: dict[str, float] = {
    # Weights are relative — letter gets bump * scale below.
    "G": 1.0,   # Go, Give, Grant
    "C": 1.0,   # Come, Call, Cease
    "S": 1.3,   # Speak, Stand, Stay, See
    "T": 1.0,   # Tell, Take, Try, Turn
    "H": 1.2,   # Hear, Hold, Hark, Help, Haste
    "A": 1.1,   # Away, Attend, Arm
    "L": 0.9,   # Let, Look, Live, Leave
    "M": 0.9,   # Mark, Mind, Make, Meet
    "B": 1.0,   # Be, Begone, Bring, Behold
    "F": 1.0,   # Fly, Forbear, Follow, Fight
    "P": 0.9,   # Peace, Pardon, Pray
    "O": 0.7,   # Open, Out
    "W": 0.8,   # Watch, Walk, Win
    "R": 0.8,   # Rouse, Run, Return, Rise
    "D": 0.8,   # Do, Draw, Die
    "E": 0.6,   # End, Endure, Enter
    "K": 0.7,   # Keep, Know, Kneel, Kill
}


def imperative_chain_start_bias(
    imperative_chain_count: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a sentence-start first-letter bias vector, or None if
    the chain isn't active. Gated by speaker_label_state."""
    if speaker_label_state != 0:
        return None
    if imperative_chain_count < 2:
        return None

    # Scale: 2 → 0.25; 3 → 0.40; 4+ → 0.50.
    if imperative_chain_count == 2:
        scale = 0.25
    elif imperative_chain_count == 3:
        scale = 0.40
    else:
        scale = 0.50

    vec = [0.0] * VOCAB_SIZE
    for ch, w in _IMPER_CAPITALS.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += scale * w
        # Also mirror at lowercase at mid-line sentence-starts (e.g.
        # after ". " mid-line). Weaker — capitals are the dominant
        # case at sentence-start.
        low = ch.lower()
        low_idx = VOCAB_INDEX.get(low)
        if low_idx is not None:
            vec[low_idx] += scale * w * 0.35
    return vec
