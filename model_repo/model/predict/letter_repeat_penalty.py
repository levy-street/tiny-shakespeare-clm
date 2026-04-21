"""Letter-repetition-in-buffer penalty.

Off-trie gibberish words often show high single-letter dominance:

    "rarevrear"   — r × 4 of 9 (44%)
    "yeereeree"   — e × 5 of 9 (55%)
    "ecceeree"    — e × 4 of 8 (50%)

Real English words have single-letter frequency capped near ~25-35%
except in very short words (nn, ss, ee) or deliberate reduplication
("tomato" has o × 2 / 6 = 33%, "remember" r × 2 / 8 = 25%). At 40%+
single-letter dominance mid-word, the buffer is almost certainly
drifting.

This layer reads `word_buffer` directly. At letter_run_len >= 5 and
off-trie, for each vocab letter it counts how often that exact letter
already appears in the buffer and applies an escalating penalty to
the letter's logit — so if 'r' appears 3+ times, another 'r' is
discouraged.

The penalty is letter-specific rather than global so we don't hurt
the many real words where doubling/tripling is legitimate (little,
common, letter, stiff, knell, ruff, etc.). It also interacts
naturally with the "4th consecutive identical letter" scenario
without resort to a separate doubled-letter field.

Gated on speaker_label_state == 0 and not on_word_trie.

No corpus statistics — the dominance thresholds come from prior
knowledge of English letter distributions within words.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def letter_repeat_penalty_bias(
    word_buffer: str,
    letter_run_len: int,
    on_word_trie: bool,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if on_word_trie:
        return None
    if letter_run_len < 5:
        return None
    if not word_buffer:
        return None

    # Count each letter in buffer (case-folded).
    counts: dict[str, int] = {}
    for ch in word_buffer:
        if "a" <= ch <= "z" or "A" <= ch <= "Z":
            lc = ch.lower()
            counts[lc] = counts.get(lc, 0) + 1

    if not counts:
        return None

    vec = [0.0] * VOCAB_SIZE

    # For each letter that has already appeared, penalize emitting it
    # again. The penalty escalates with the current count AND with a
    # dominance ratio relative to the buffer length.
    blen = letter_run_len  # use letter_run_len as the denominator
    for lc, c in counts.items():
        if c < 2:
            continue  # 1 occurrence — no penalty
        ratio = c / max(blen, 1)
        # Base penalty grows with count:
        #  c=2 → 0.10 (mild nudge)
        #  c=3 → 0.35 (moderate)
        #  c=4 → 0.75 (strong)
        #  c=5+ → 1.30 (very strong)
        if c == 2:
            pen = 0.10
        elif c == 3:
            pen = 0.35
        elif c == 4:
            pen = 0.75
        else:
            pen = 1.30
        # Amplify if the ratio is high (dominance in buffer).
        if ratio >= 0.50:
            pen *= 1.6
        elif ratio >= 0.40:
            pen *= 1.35
        elif ratio >= 0.30:
            pen *= 1.10

        # Apply penalty to both lowercase and uppercase form.
        if lc in VOCAB_INDEX:
            vec[VOCAB_INDEX[lc]] -= pen
        uc = lc.upper()
        if uc in VOCAB_INDEX:
            vec[VOCAB_INDEX[uc]] -= pen

    return vec
