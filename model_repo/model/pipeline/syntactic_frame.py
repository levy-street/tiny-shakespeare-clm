"""Tier 2 — syntactic-frame role projection.

Runs after `update_pos` (needs last_word_pos and prev_word_pos) and
after `update_clause_slot` (needs clause_slot) and `update_np_head`
(needs np_open). Sets `expected_next_role` and `frame_confidence`:
a forward projection of what role the NEXT word is likely to fill.

Design goal: in the samples, trigrams collapse because the third word
has no forward constraint — the first letter of the 3rd word gets
chosen for local letter-fit, not for syntactic role. This stage turns
the existing backward-looking POS tags into a forward-looking role
expectation that predict/syntactic_frame.py can consume.

Uses only hand-coded English phrase-structure transitions. No corpus
statistics.

Fires once per token. Projection is updated AT WORD-COMPLETION (when
`just_finished_word` is true), because that's when `last_word_pos`
has changed and a new projection becomes meaningful.
"""

from __future__ import annotations

from ..state import ModelState


# Frame-role enum. Must match the schema comment.
FRAME_ANY = 0
FRAME_NOUN = 1
FRAME_ADJ_OR_NOUN = 2
FRAME_NOUN_ONLY = 3
FRAME_DET_OR_POSS = 4
FRAME_VERB_FAMILY = 5
FRAME_VERB_ONLY = 6
FRAME_PREP_OR_CONJ = 7
FRAME_OBJ = 8
FRAME_SUBJ = 9
FRAME_ADV_OR_PREP = 10


# POS enum (mirrored from pipeline/pos.py — don't import to avoid a
# circular import).
POS_UNKNOWN = 0
POS_ARTICLE = 1
POS_PRONOUN = 2
POS_POSSESSIVE = 3
POS_PREPOSITION = 4
POS_CONJUNCTION = 5
POS_AUX_VERB = 6
POS_MODAL = 7
POS_INTERJECTION = 8
POS_NEGATION = 9
POS_ADVERB = 10
POS_VERB_ING = 11
POS_VERB_ED = 12
POS_NOUN = 13
POS_ADJECTIVE = 14
POS_PROPER_NOUN = 15
POS_VERB = 16
POS_NUMBER = 17
POS_WH = 18


# Subject-pronoun subset of POS_PRONOUN (which in pipeline/pos.py
# covers all cases). These are ones likely to act as clause subject.
# Others (me, him, thee, us, them) are object-position.
_SUBJ_PRONOUN_WORDS = frozenset({
    "i", "thou", "he", "she", "it", "we", "ye", "you", "they",
})


def update_syntactic_frame(state: ModelState, token_id: int) -> ModelState:
    # Projection is meaningful at WORD BOUNDARIES. Between words we
    # carry whatever the last projection set.
    if not state.just_finished_word:
        return state

    # Reset on sentence-end / turn boundary. These are captured by
    # clause_slot transitioning back to 0 (FRESH) — handled via the
    # default path below. Also, if we're inside a speaker label, no
    # projection.
    if state.speaker_label_state != 0:
        if state.expected_next_role == FRAME_ANY and state.frame_confidence == 0.0:
            return state
        return state.model_copy(
            update={"expected_next_role": FRAME_ANY, "frame_confidence": 0.0}
        )

    # Fast-path: sentence-fresh (clause_slot == 0, FRESH). Sentence is
    # about to start (or just started). Project FRAME_SUBJ.
    cs = state.clause_slot
    last_pos = state.last_word_pos
    prev_pos = state.prev_word_pos
    last_word = state.last_completed_word

    pp_pos = state.prev_prev_word_pos

    # --- Three-word-aware transitions (highest specificity) ---------

    # After PREP + DET + NOUN ("of the king", "to the land") → the PP
    # is complete; the next word is typically a VERB (if clause_slot
    # HAS_SUBJ with an overdue verb) or a CONJ/PREP continuing.
    if (
        pp_pos == POS_PREPOSITION
        and prev_pos in (POS_ARTICLE, POS_POSSESSIVE)
        and last_pos in (POS_NOUN, POS_PROPER_NOUN)
    ):
        if cs == 1:
            role, conf = FRAME_VERB_FAMILY, 0.55
        else:
            role, conf = FRAME_PREP_OR_CONJ, 0.45
    # After DET + ADJ + NOUN ("the fair king", "a gentle lord") → NP
    # complete; expect VERB (if subject) or PREP.
    elif (
        pp_pos in (POS_ARTICLE, POS_POSSESSIVE)
        and prev_pos == POS_ADJECTIVE
        and last_pos in (POS_NOUN, POS_PROPER_NOUN)
    ):
        if cs == 1:
            role, conf = FRAME_VERB_FAMILY, 0.60
        else:
            role, conf = FRAME_PREP_OR_CONJ, 0.45
    # After SUBJ-PRON + AUX + NEGATION ("I am not", "thou art not") →
    # verb family (strong).
    elif (
        pp_pos == POS_PRONOUN
        and prev_pos == POS_AUX_VERB
        and last_pos == POS_NEGATION
    ):
        role, conf = FRAME_VERB_FAMILY, 0.75
    # After SUBJ-PRON + MODAL + NEGATION ("I will not", "thou shalt
    # not") → bare-infinitive verb (strong).
    elif (
        pp_pos == POS_PRONOUN
        and prev_pos == POS_MODAL
        and last_pos == POS_NEGATION
    ):
        role, conf = FRAME_VERB_ONLY, 0.80

    # --- Two-word-aware transitions ---------------------------------

    # After DET + ADJECTIVE → strongly expect NOUN.
    elif prev_pos == POS_ARTICLE and last_pos == POS_ADJECTIVE:
        role, conf = FRAME_NOUN_ONLY, 0.85

    # After POSSESSIVE + ADJECTIVE → strongly expect NOUN.
    elif prev_pos == POS_POSSESSIVE and last_pos == POS_ADJECTIVE:
        role, conf = FRAME_NOUN_ONLY, 0.80

    # After PREPOSITION + ARTICLE → ADJ/NOUN.
    elif prev_pos == POS_PREPOSITION and last_pos == POS_ARTICLE:
        role, conf = FRAME_ADJ_OR_NOUN, 0.70

    # After PREPOSITION + POSSESSIVE → ADJ/NOUN.
    elif prev_pos == POS_PREPOSITION and last_pos == POS_POSSESSIVE:
        role, conf = FRAME_ADJ_OR_NOUN, 0.70

    # After AUX + NEGATION ("is not", "do not") → verb family.
    elif prev_pos == POS_AUX_VERB and last_pos == POS_NEGATION:
        role, conf = FRAME_VERB_FAMILY, 0.70

    # After MODAL + NEGATION ("shall not", "will not") → VERB_ONLY.
    elif prev_pos == POS_MODAL and last_pos == POS_NEGATION:
        role, conf = FRAME_VERB_ONLY, 0.75

    # After PRONOUN + AUX ("I am", "thou art", "he hath") → verb-ing /
    # verb-ed / participle / noun.
    elif prev_pos == POS_PRONOUN and last_pos == POS_AUX_VERB:
        role, conf = FRAME_VERB_FAMILY, 0.60

    # After PRONOUN + MODAL ("I will", "thou shalt") → VERB_ONLY.
    elif prev_pos == POS_PRONOUN and last_pos == POS_MODAL:
        role, conf = FRAME_VERB_ONLY, 0.70

    # After CONJUNCTION + PRONOUN ("and I", "but thou") → verb family.
    elif prev_pos == POS_CONJUNCTION and last_pos == POS_PRONOUN:
        role, conf = FRAME_VERB_FAMILY, 0.55

    # --- One-word transitions --------------------------------------

    # After ARTICLE / POSSESSIVE alone → ADJ/NOUN.
    elif last_pos == POS_ARTICLE or last_pos == POS_POSSESSIVE:
        role, conf = FRAME_ADJ_OR_NOUN, 0.70

    # After PREPOSITION alone → DET / POSSESSIVE / PRONOUN / NOUN.
    elif last_pos == POS_PREPOSITION:
        role, conf = FRAME_DET_OR_POSS, 0.55

    # After MODAL alone → VERB (bare infinitive).
    elif last_pos == POS_MODAL:
        role, conf = FRAME_VERB_ONLY, 0.75

    # After AUX alone → VERB / VERB_ING / VERB_ED / NEGATION / NOUN.
    elif last_pos == POS_AUX_VERB:
        role, conf = FRAME_VERB_FAMILY, 0.50

    # After a subject-PRONOUN (I, thou, he, she, we, ye, they, you) →
    # verb family dominant.
    elif last_pos == POS_PRONOUN and last_word in _SUBJ_PRONOUN_WORDS:
        role, conf = FRAME_VERB_FAMILY, 0.55

    # After CONJUNCTION alone → SUBJ.
    elif last_pos == POS_CONJUNCTION:
        role, conf = FRAME_SUBJ, 0.45

    # After NEGATION ("not", "no") → verb family or noun.
    elif last_pos == POS_NEGATION:
        role, conf = FRAME_VERB_FAMILY, 0.35

    # After INTERJECTION ("O", "alas", "hark") → vocative noun or subject.
    elif last_pos == POS_INTERJECTION:
        role, conf = FRAME_SUBJ, 0.45

    # After a VERB / VERB_ING / VERB_ED → object/complement or PP.
    elif last_pos in (POS_VERB, POS_VERB_ING, POS_VERB_ED):
        # clause_slot HAS_VERB (2) + np_open False: object expected.
        if cs == 2 and not state.np_open:
            role, conf = FRAME_OBJ, 0.55
        else:
            role, conf = FRAME_OBJ, 0.40

    # After NOUN / PROPER_NOUN / PRONOUN (non-subject) → verb or prep.
    elif last_pos in (POS_NOUN, POS_PROPER_NOUN):
        # If clause_slot is HAS_SUBJ (1), a verb is overdue.
        if cs == 1:
            role, conf = FRAME_VERB_FAMILY, 0.50
        else:
            role, conf = FRAME_PREP_OR_CONJ, 0.35

    elif last_pos == POS_ADJECTIVE:
        # ADJ alone → expect NOUN next (phrase continuing).
        role, conf = FRAME_NOUN, 0.65

    elif last_pos == POS_ADVERB:
        # ADVERB → verb or adjective next, soft bias.
        role, conf = FRAME_VERB_FAMILY, 0.30

    elif last_pos == POS_WH:
        # WH ("who", "what", "when") → subject or verb.
        role, conf = FRAME_SUBJ, 0.40

    else:
        # Unknown / number / misc: no confident projection.
        role, conf = FRAME_ANY, 0.0

    # Reset-driven adjustments — at clause break (FRESH) strengthen
    # toward SUBJ; at POST_OBJ (3) weaken (sentence winding down).
    if cs == 0:
        role, conf = FRAME_SUBJ, 0.50
    elif cs == 3:
        conf *= 0.55  # sentence closing; weaker projection

    # No-op update short-circuit.
    if role == state.expected_next_role and abs(conf - state.frame_confidence) < 1e-6:
        return state
    return state.model_copy(
        update={"expected_next_role": role, "frame_confidence": conf}
    )
