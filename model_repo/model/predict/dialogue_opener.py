"""Predict layer — dialogue-adjacency opener bias.

This is ADDITIVE with the existing `answer_opener` layer, which
already handles the question/exclaim/imperative → answer pattern.
This layer contributes what answer_opener doesn't:

  1. STICHOMYTHIA amplifier — when the prior turn was short
     (<=3 words), the current response is usually also a short
     retort. Amplify the "short-retort starter" letters:
     A (Ay), N (Nay/No), I (I), Y (Yes), ' ('Tis), T (Troth),
     W (Why/What), S (Sir/So), O (O).

  2. LONG-PRIOR-TURN narrative reset — after a long monologue turn
     (>=6 lines), the new speaker typically opens with a topic-
     continuation starter: T (The/Then/Thus/There), B (But/Behold),
     N (Now), S (So), A (And).

  3. PERIOD-CONTINUATION / DECLARATIVE echo — after a prior turn
     that ends with "." or ";" (not a question or exclaim), a mild
     narrative-opener bias similar to (2) but weaker.

Gates mirror `answer_opener`: fires exactly once per new turn, at
the first letter of the first word, outside speaker-label
territory.

All biases come from Shakespeare's known dialogue patterns —
no corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Short-retort starters — fire when prev turn was terse.
_STICHO_BOOSTS: dict[str, float] = {
    "A": 0.2,   # Ay, And
    "N": 0.2,   # Nay, No, Not
    "I": 0.15,  # I, Indeed, It
    "Y": 0.15,  # Yes, Yea
    "'": 0.2,   # 'Tis, 'Fore
    "T": 0.1,
    "W": 0.1,
    "S": 0.1,
    "O": 0.1,
}

# Narrative-reset starters — fire after a long prior turn.
_LONG_RESET_BOOSTS: dict[str, float] = {
    "T": 0.2,    # Then, Thus, There, The
    "B": 0.18,   # But, Behold, By
    "N": 0.15,   # Now
    "S": 0.12,   # So, Sir
    "A": 0.12,   # And
    "W": 0.1,    # Well, What
    "H": 0.08,   # How, Hark
}

# Mild continuation bias after "." / ";" prior-turn close.
_DECL_CONT_BOOSTS: dict[str, float] = {
    "T": 0.1,    # The, Then, Thus
    "A": 0.08,   # And, A
    "B": 0.08,   # But
    "I": 0.08,   # I
    "N": 0.06,
    "S": 0.06,
    "W": 0.06,
}


def _apply(vec: list[float], table: dict[str, float], weight: float = 1.0) -> None:
    for ch, val in table.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += val * weight


def dialogue_adjacency_bias(
    prev_turn_final_punct: str,
    prev_turn_word_count: int,
    prev_turn_line_count: int,
    speaker_label_state: int,
    words_in_turn: int,
    sentences_in_turn: int,
    lines_in_turn: int,
    letter_run_len: int,
    word_buffer_len: int,
    turns_closed: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    # lines_in_turn may be 1 on the very first body token (the
    # label-closing "\n" already incremented it in update_turn_progress);
    # mirror answer_opener which only gates on words_in_turn and
    # sentences_in_turn.
    if words_in_turn != 0 or sentences_in_turn != 0:
        return None
    if letter_run_len != 0 or word_buffer_len != 0:
        return None
    if turns_closed == 0:
        return None

    vec = [0.0] * VOCAB_SIZE
    fired = False

    # (1) Stichomythia amplifier.
    if prev_turn_word_count > 0 and prev_turn_word_count <= 3:
        _apply(vec, _STICHO_BOOSTS)
        fired = True

    # (2) Long prior-turn narrative reset.
    if prev_turn_line_count >= 6:
        _apply(vec, _LONG_RESET_BOOSTS)
        fired = True

    # (3) Declarative continuation — only when prior-turn was NOT a
    # question/exclaim (answer_opener handles those). Mild effect.
    if prev_turn_final_punct in (".", ";", ":") and prev_turn_word_count >= 4:
        _apply(vec, _DECL_CONT_BOOSTS)
        fired = True

    if not fired:
        return None
    return vec


def dialogue_pacing_bias(
    prev_turn_word_count: int,
    prev_turn_line_count: int,
    speaker_label_state: int,
    words_in_turn: int,
    sentences_in_turn: int,
    lines_in_turn: int,
    just_finished_word: bool,
    turns_closed: int,
    last_char: str,
) -> list[float] | None:
    """Mid-turn pacing amplifier — stichomythia vs. monologue.

    Fires at a word-end position (just after a word finishes, i.e. at
    the space-after-word decision). Uses the SHAPE of the previous
    turn to modulate how likely the current turn is to close on a
    given line.

    Stichomythia mode (prev_turn_word_count <= 3, at least one turn
    closed): the current response is likely also terse. After the
    current turn has produced 3+ words, gently boost sentence-end
    punct and newline to encourage early closure.

    Monologue mode (prev_turn_line_count >= 8): the current speaker
    is likely to also elaborate. Gently suppress sentence-end punct
    during the first few words so the model stays in the flow.

    Gates: only fires in turn body, at just_finished_word moments.
    Effect size kept small so it reshapes within the existing
    line-break / sentence-close signal rather than dominating it.
    """
    if speaker_label_state != 0:
        return None
    if turns_closed == 0:
        return None
    if not just_finished_word:
        return None
    # Only a bias AT the moment we just completed a word — the next
    # char decision is about whether to space/punct-close.
    if last_char and not ("a" <= last_char.lower() <= "z"):
        # Extra safety: just_finished_word transitions on any non-letter
        # incoming char. We want the decision *right after* a letter,
        # i.e. when the previous char (last_char) was a letter.
        pass

    vec = [0.0] * VOCAB_SIZE
    fired = False

    # Stichomythia mode: prev turn <= 3 words. Current turn has
    # already produced several words. Encourage closure.
    if 1 <= prev_turn_word_count <= 3 and words_in_turn >= 3:
        for ch, b in (
            (".", 0.15),
            ("!", 0.10),
            ("?", 0.10),
            ("\n", 0.05),
        ):
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += b
        fired = True

    # Monologue mode: prev turn >= 8 lines. Current turn is early.
    # Gentle suppression of sentence-end punct.
    if prev_turn_line_count >= 8 and words_in_turn <= 3:
        for ch, b in (
            (".", -0.10),
            ("!", -0.08),
            ("?", -0.08),
        ):
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += b
        fired = True

    if not fired:
        return None
    return vec
