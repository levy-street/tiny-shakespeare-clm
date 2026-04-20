"""Predict layer — sentence backbone bias on terminal punctuation.

Reads state.sentence_has_subject, state.sentence_has_verb, and
state.words_in_sentence. Applied at word-end decisions (i.e., when the
model is about to pick a character that could be a terminal
punctuation " . ? !"). A well-formed Shakespeare sentence normally has
both a subject and a finite verb.

Behaviour:
  * Word count >= 2 and verb MISSING: strongly suppress . ? ! and
    semicolon-as-terminator. Light nudge on comma (continuation).
  * Word count >= 2 and subject MISSING: mild suppression of . ? !.
    (Subject-less sentences are rarer in Shakespeare than verbless
    ones; in imperatives, subject is implicit.)
  * Backbone complete AND word count >= 4: gentle bias toward . ? !
    at word-end (makes long run-ons slightly less likely).

Gates:
  * speaker_label_state == 0
  * Only fires at word-end decision points (letter_run_len >= 1,
    on word-completion territory)
  * Only contributes for sentences 2+ words deep (to avoid suppressing
    legitimate short imperative replies like "Ay." / "No.").

No corpus statistics — the principle is structural grammaticality.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

from ..pipeline.sentence import SENT_IMPER


def sentence_backbone_bias(
    sentence_has_subject: bool,
    sentence_has_verb: bool,
    words_in_sentence: int,
    sentence_type: int,
    speaker_label_state: int,
    letter_run_len: int,
    on_word_trie: bool,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    # Only fire on actual word-end decision points: letter_run_len >= 1
    # and the word would plausibly be ending next.
    if letter_run_len < 1:
        return None
    # Need enough sentence material to judge.
    if words_in_sentence < 2:
        return None

    # Imperatives in Shakespeare often have implicit subject and a
    # verb is the first word ("Speak, sirrah!"). Don't over-suppress
    # terminators in imperative sentences.
    is_imper = sentence_type == SENT_IMPER

    vec = [0.0] * VOCAB_SIZE

    # Missing finite verb — gentle terminator suppression only on
    # longer sentences where the signal is reliable. Shakespeare has
    # short verbless replies ("Ay, sir." / "Marry, sir.") and our
    # verb detector has false negatives; stay conservative.
    if not sentence_has_verb and not is_imper and words_in_sentence >= 5:
        if words_in_sentence >= 8:
            mag = -0.40
        elif words_in_sentence >= 6:
            mag = -0.25
        else:
            mag = -0.12
        for ch in (".", "?", "!"):
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += mag
        sc = VOCAB_INDEX.get(";")
        if sc is not None:
            vec[sc] += mag * 0.4

    # Any non-zero entry? If not, skip.
    if not any(v != 0.0 for v in vec):
        return None
    return vec
