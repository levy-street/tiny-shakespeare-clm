"""Speaker recency bias.

Reads `state.recent_speakers` (the rolling window of up to 4 most-
recent distinct canonical speaker names, head-most-recent) and, when
we're currently mid-speaker-label (state 2) with a partial
`speaker_buffer`, biases the next character toward:

  - the letters of recently-seen speakers *other than* the current one
    (they are the plausible dialogue partners in this scene)
  - AWAY from the immediately-previous speaker's name (they almost
    never follow themselves with another label)

The bias is combined with the existing speaker_trie layer — this is a
*recency* modulation on top of the full canonical-names trie. It
doesn't invent new matches; it just tilts the existing trie toward
in-scene names.

Only fires when:
  - speaker_label_state in (1, 2)
  - speaker_buffer is a prefix of at least one recent_speakers[i]
  - recent_speakers has at least one element

Returns None when no recent-speaker prefix match is found.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def speaker_recency_bias(
    speaker_buffer: str,
    recent_speakers: tuple[str, ...],
) -> list[float] | None:
    if not speaker_buffer or not recent_speakers:
        return None
    buf = speaker_buffer.upper()

    # Separate: the current (self) speaker is element [0]; the in-scene
    # partners are [1:].
    self_sp = recent_speakers[0]
    partners = recent_speakers[1:]

    # Find partners whose uppercase form starts with the buffer.
    matching_partners = [s for s in partners if s.startswith(buf)]
    # Does the self-speaker match the buffer too?
    self_matches = self_sp.startswith(buf)

    if not matching_partners and not self_matches:
        return None

    vec = [0.0] * VOCAB_SIZE

    # Positive boost for partner continuation letters.
    # Each partner contributes a bump on the next char of its name
    # after the buffer. Also ":" / " " if the buffer IS the complete
    # partner name.
    PARTNER_BUMP = 1.5
    for sp in matching_partners:
        if len(sp) > len(buf):
            nxt = sp[len(buf)]
            # Upper-letter or space
            if nxt in VOCAB_INDEX:
                vec[VOCAB_INDEX[nxt]] += PARTNER_BUMP
            low = nxt.lower()
            if low != nxt and low in VOCAB_INDEX:
                vec[VOCAB_INDEX[low]] += PARTNER_BUMP * 0.35
        else:
            # Buffer equals partner's full name; next char is ":" or space.
            if ":" in VOCAB_INDEX:
                vec[VOCAB_INDEX[":"]] += PARTNER_BUMP

    # Negative nudge for the self-speaker continuation: they almost
    # never speak two labels in a row. Only apply if the buffer is
    # already at least 1 char and matches self.
    SELF_PENALTY = -0.8
    if self_matches and len(self_sp) > len(buf):
        nxt = self_sp[len(buf)]
        if nxt in VOCAB_INDEX:
            vec[VOCAB_INDEX[nxt]] += SELF_PENALTY
        low = nxt.lower()
        if low != nxt and low in VOCAB_INDEX:
            vec[VOCAB_INDEX[low]] += SELF_PENALTY * 0.35

    return vec
