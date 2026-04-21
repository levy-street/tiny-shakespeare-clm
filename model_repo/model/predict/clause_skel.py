"""Predict layer — next-word first-letter bias based on clause skeleton.

Reads `state.clause_skel` (0-5) and `state.clause_skel_age` (words
since last reset). Fires at word-start (letter_run_len == 0, after
space or after mid-punct).

Behavior by state:
  0 EMPTY         — no signal (sentence/clause just began).
  1 SUBJ_OPEN     — NP is forming. Push noun/adj starter letters;
                    suppress verbs, preps, conjunctions.
  2 SUBJ_DONE     — predicate owed. Push verb/aux/modal letters;
                    suppress another det/adj/noun opener.
  3 VERB_DONE     — optional object/complement or adjunct. Modest
                    push on determiner/pronoun/prep starters to
                    build the object NP/PP; also allow terminators
                    (intransitive-verb clauses can close here).
  4 COMP_DUE      — complement NP/PP in progress. Push noun/adj
                    starters (need a head); suppress verb/conj.
  5 CLAUSE_DONE   — clause has a full predicate. At word-end this
                    state licenses terminator; at word-start we
                    gently push adjunct-opening prepositions or
                    conjunction to let another clause begin.

Pressure scales with clause_skel_age — the longer a clause has been
waiting for its missing element, the stronger the push.

No corpus statistics. First-letter lists are hand-graded from common
Shakespeare / Early Modern English vocabulary.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# --- Starter letter inventories ---
# NOUN/ADJECTIVE head-starter letters (content words that form NP heads).
_NP_HEAD_STARTS: dict[str, float] = {
    "m": 0.85, "h": 0.90, "l": 0.80, "d": 0.80, "f": 0.85, "g": 0.80,
    "w": 0.75, "t": 0.65, "s": 0.80, "n": 0.55, "b": 0.80, "p": 0.65,
    "c": 0.70, "k": 0.50, "r": 0.60, "e": 0.50, "a": 0.55, "v": 0.35,
    "y": 0.35, "o": 0.30,
}

# VERB / AUX / MODAL starters.
_VERB_STARTS: dict[str, float] = {
    "s": 0.95, "t": 0.90, "l": 0.85, "g": 0.85, "c": 0.85, "f": 0.80,
    "m": 0.80, "b": 0.75, "h": 0.80, "r": 0.70, "p": 0.65, "w": 0.80,
    "k": 0.60, "d": 0.85, "a": 0.55, "i": 0.55, "o": 0.35, "n": 0.30,
}

# DETERMINER / POSSESSIVE / PRONOUN starters (NP openers).
_DET_PRON_STARTS: dict[str, float] = {
    "t": 1.00, "a": 0.90, "m": 0.75, "h": 0.80, "o": 0.70, "y": 0.65,
    "s": 0.55, "e": 0.40, "n": 0.40, "w": 0.55, "i": 0.90,
}

# PREPOSITION / COORDINATOR starters (adjunct openers).
_PREP_CONJ_STARTS: dict[str, float] = {
    "o": 0.90, "t": 0.85, "i": 0.80, "w": 0.95, "f": 0.80, "b": 0.90,
    "a": 0.85, "u": 0.75, "s": 0.55, "n": 0.50, "y": 0.45, "e": 0.30,
}


def _build_vec(
    primary: dict[str, float],
    penalize: dict[str, float] | None,
    scale: float,
    pen_frac: float = 0.35,
) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch, w in primary.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += scale * w
        up = ch.upper()
        if up in VOCAB_INDEX:
            # Mid-line cap is less likely; small share.
            vec[VOCAB_INDEX[up]] += scale * w * 0.15
    if penalize:
        for ch, w in penalize.items():
            if ch not in VOCAB_INDEX:
                continue
            pw = primary.get(ch, 0.0)
            if w > pw + 0.10:
                net = -(w - pw) * scale * pen_frac
                vec[VOCAB_INDEX[ch]] += net
                up = ch.upper()
                if up in VOCAB_INDEX:
                    vec[VOCAB_INDEX[up]] += net * 0.15
    return vec


def clause_skel_bias(
    clause_skel: int,
    clause_skel_age: int,
    letter_run_len: int,
    last_char_class: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0:
        return None
    if last_char_class not in (1, 7):
        return None
    if clause_skel == 0:
        return None

    # Age-based scale escalation. Longer wait = stronger push.
    age = clause_skel_age
    base = 0.60
    if age >= 4:
        scale = base * 1.8
    elif age >= 2:
        scale = base * 1.3
    else:
        scale = base

    if clause_skel == 1:
        # SUBJ_OPEN — NP forming. Push noun/adj starters; suppress
        # prep/conj/verb which would break the NP.
        return _build_vec(
            primary=_NP_HEAD_STARTS,
            penalize={
                **{k: v * 0.8 for k, v in _PREP_CONJ_STARTS.items()},
                **{k: v * 0.4 for k, v in _VERB_STARTS.items()},
            },
            scale=scale,
            pen_frac=0.30,
        )
    elif clause_skel == 2:
        # SUBJ_DONE — predicate owed. Push verb/aux/modal, suppress
        # another det/adj/pronoun (would fragment the clause) and
        # suppress prep/conj (no predicate yet).
        # Also tag "i" as AUX ("is"), "h" as AUX ("hath/has/have"),
        # "a" as AUX ("art/am/are"), "d" as AUX ("doth/dost/did"),
        # "w" as MODAL ("will/would"), "s" as MODAL ("shall/should").
        return _build_vec(
            primary=_VERB_STARTS,
            penalize={
                **{k: v * 0.7 for k, v in _DET_PRON_STARTS.items()},
                **{k: v * 0.5 for k, v in _PREP_CONJ_STARTS.items()},
            },
            scale=scale * 1.10,  # slightly stronger push than others
            pen_frac=0.30,
        )
    elif clause_skel == 3:
        # VERB_DONE — complement / adjunct coming. Allow DET/PRON (to
        # open object NP) and PREP (to open PP adjunct). Suppress
        # another VERB start (verb chain).
        combined: dict[str, float] = {}
        for k, v in _DET_PRON_STARTS.items():
            combined[k] = max(combined.get(k, 0.0), v)
        for k, v in _PREP_CONJ_STARTS.items():
            combined[k] = max(combined.get(k, 0.0), v * 0.9)
        return _build_vec(
            primary=combined,
            penalize={k: v * 0.6 for k, v in _VERB_STARTS.items()},
            scale=scale * 0.9,
            pen_frac=0.25,
        )
    elif clause_skel == 4:
        # COMP_DUE — need an NP head. Push noun/adj/pronoun starters;
        # suppress verb/conj.
        combined = {}
        for k, v in _NP_HEAD_STARTS.items():
            combined[k] = max(combined.get(k, 0.0), v)
        for k, v in _DET_PRON_STARTS.items():
            combined[k] = max(combined.get(k, 0.0), v * 0.7)
        return _build_vec(
            primary=combined,
            penalize={
                **{k: v * 0.6 for k, v in _VERB_STARTS.items()},
                **{k: v * 0.5 for k, v in _PREP_CONJ_STARTS.items()},
            },
            scale=scale,
            pen_frac=0.30,
        )
    elif clause_skel == 5:
        # CLAUSE_DONE — prefer adjunct openers (PREP) or coordinator
        # ("and/but/or/nor"). Allow a fresh NP too (apposition).
        return _build_vec(
            primary=_PREP_CONJ_STARTS,
            penalize=None,
            scale=scale * 0.65,
        )
    return None
