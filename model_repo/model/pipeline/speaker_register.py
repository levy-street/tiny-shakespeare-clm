"""Tier 2/3 — speaker-register classification.

Reads the head of `state.recent_speakers` and maps it to one of seven
registers (plus UNKNOWN=0). The register is a hand-curated categorical
encoding of each character's dramatic type — noble-tragic, comic
prose, royal-formal, villain, feminine-lover, brief-servant,
supernatural — distilled from prior knowledge of Shakespeare's plays.

Downstream `predict.speaker_register_bias` consumers read the register
and condition word-start vocabulary on it. This closes the loop in
which `recent_speakers[0]` was being maintained but never influenced
prediction: HAMLET, ESCALUS, and ABRAHAM all received identical
priors. Now they don't.

Also maintains `register_age` — the number of tokens since the last
register change — so biases can taper during the first few tokens of
a new turn (which are usually the speaker label itself and a line
break, not yet the speaker's words).

No corpus statistics — purely prior knowledge of these characters.
"""

from __future__ import annotations

from ..state import ModelState


# Register codes (match the comments in state/schema.py).
REG_UNKNOWN = 0
REG_TRAGIC_NOBLE = 1
REG_COMIC_PROSE = 2
REG_ROYAL_FORMAL = 3
REG_VILLAIN = 4
REG_LOVER_FEMININE = 5
REG_SERVANT_BRIEF = 6
REG_SUPERNATURAL = 7


# Character name → register. Names are stored UPPERCASE (how
# speaker_memory normalizes labels). Keep entries tight — only hard
# assignments. Anything not listed falls through to heuristic checks.
_EXACT: dict[str, int] = {
    # Tragic nobles / protagonists
    "HAMLET": REG_TRAGIC_NOBLE,
    "LEAR": REG_TRAGIC_NOBLE,
    "KING LEAR": REG_TRAGIC_NOBLE,
    "MACBETH": REG_TRAGIC_NOBLE,
    "OTHELLO": REG_TRAGIC_NOBLE,
    "ROMEO": REG_TRAGIC_NOBLE,
    "BRUTUS": REG_TRAGIC_NOBLE,
    "CORIOLANUS": REG_TRAGIC_NOBLE,
    "MARCIUS": REG_TRAGIC_NOBLE,
    "TIMON": REG_TRAGIC_NOBLE,
    "ANTONY": REG_TRAGIC_NOBLE,
    "MARK ANTONY": REG_TRAGIC_NOBLE,
    "CASSIUS": REG_TRAGIC_NOBLE,
    "TROILUS": REG_TRAGIC_NOBLE,
    "HECTOR": REG_TRAGIC_NOBLE,
    "POSTHUMUS": REG_TRAGIC_NOBLE,
    "BENEDICK": REG_TRAGIC_NOBLE,
    # Comic prose / clowns / fools
    "FOOL": REG_COMIC_PROSE,
    "CLOWN": REG_COMIC_PROSE,
    "LAUNCE": REG_COMIC_PROSE,
    "LAUNCELOT": REG_COMIC_PROSE,
    "BOTTOM": REG_COMIC_PROSE,
    "DOGBERRY": REG_COMIC_PROSE,
    "TOUCHSTONE": REG_COMIC_PROSE,
    "FESTE": REG_COMIC_PROSE,
    "COSTARD": REG_COMIC_PROSE,
    "TRINCULO": REG_COMIC_PROSE,
    "STEPHANO": REG_COMIC_PROSE,
    "POMPEY": REG_COMIC_PROSE,
    "SPEED": REG_COMIC_PROSE,
    "GRAVEDIGGER": REG_COMIC_PROSE,
    "FIRST GRAVEDIGGER": REG_COMIC_PROSE,
    "SECOND GRAVEDIGGER": REG_COMIC_PROSE,
    "NURSE": REG_COMIC_PROSE,
    # Royal / formal
    "KING": REG_ROYAL_FORMAL,
    "KING HENRY": REG_ROYAL_FORMAL,
    "KING HENRY IV": REG_ROYAL_FORMAL,
    "KING HENRY V": REG_ROYAL_FORMAL,
    "KING HENRY VI": REG_ROYAL_FORMAL,
    "KING HENRY VIII": REG_ROYAL_FORMAL,
    "KING RICHARD": REG_ROYAL_FORMAL,
    "KING RICHARD II": REG_ROYAL_FORMAL,
    "KING RICHARD III": REG_ROYAL_FORMAL,
    "KING EDWARD": REG_ROYAL_FORMAL,
    "KING JOHN": REG_ROYAL_FORMAL,
    "QUEEN": REG_ROYAL_FORMAL,
    "QUEEN MARGARET": REG_ROYAL_FORMAL,
    "QUEEN ELIZABETH": REG_ROYAL_FORMAL,
    "DUKE": REG_ROYAL_FORMAL,
    "DUKE SENIOR": REG_ROYAL_FORMAL,
    "DUKE VINCENTIO": REG_ROYAL_FORMAL,
    "PRINCE": REG_ROYAL_FORMAL,
    "PRINCE HENRY": REG_ROYAL_FORMAL,
    "PRINCE HAL": REG_ROYAL_FORMAL,
    "PROSPERO": REG_ROYAL_FORMAL,
    "CAESAR": REG_ROYAL_FORMAL,
    "JULIUS CAESAR": REG_ROYAL_FORMAL,
    "OCTAVIUS": REG_ROYAL_FORMAL,
    "OCTAVIUS CAESAR": REG_ROYAL_FORMAL,
    "ESCALUS": REG_ROYAL_FORMAL,
    "POLONIUS": REG_ROYAL_FORMAL,
    "GLOUCESTER": REG_ROYAL_FORMAL,
    "CLAUDIUS": REG_ROYAL_FORMAL,
    "GAUNT": REG_ROYAL_FORMAL,
    "JOHN OF GAUNT": REG_ROYAL_FORMAL,
    # Villains
    "IAGO": REG_VILLAIN,
    "EDMUND": REG_VILLAIN,
    "AARON": REG_VILLAIN,
    "ANGELO": REG_VILLAIN,
    "DON JOHN": REG_VILLAIN,
    "SHYLOCK": REG_VILLAIN,
    "TYBALT": REG_VILLAIN,
    "MACBETH LADY": REG_VILLAIN,
    "LADY MACBETH": REG_VILLAIN,
    # Feminine / lover roles
    "JULIET": REG_LOVER_FEMININE,
    "VIOLA": REG_LOVER_FEMININE,
    "ROSALIND": REG_LOVER_FEMININE,
    "PORTIA": REG_LOVER_FEMININE,
    "MIRANDA": REG_LOVER_FEMININE,
    "DESDEMONA": REG_LOVER_FEMININE,
    "OPHELIA": REG_LOVER_FEMININE,
    "CORDELIA": REG_LOVER_FEMININE,
    "PERDITA": REG_LOVER_FEMININE,
    "IMOGEN": REG_LOVER_FEMININE,
    "HELENA": REG_LOVER_FEMININE,
    "HERMIA": REG_LOVER_FEMININE,
    "CRESSIDA": REG_LOVER_FEMININE,
    "CELIA": REG_LOVER_FEMININE,
    "BEATRICE": REG_LOVER_FEMININE,
    "ISABELLA": REG_LOVER_FEMININE,
    "OLIVIA": REG_LOVER_FEMININE,
    "HERO": REG_LOVER_FEMININE,
    "MARIANA": REG_LOVER_FEMININE,
    # Servants / brief roles (often by number)
    "FIRST CITIZEN": REG_SERVANT_BRIEF,
    "SECOND CITIZEN": REG_SERVANT_BRIEF,
    "THIRD CITIZEN": REG_SERVANT_BRIEF,
    "FOURTH CITIZEN": REG_SERVANT_BRIEF,
    "CITIZEN": REG_SERVANT_BRIEF,
    "FIRST SERVANT": REG_SERVANT_BRIEF,
    "SECOND SERVANT": REG_SERVANT_BRIEF,
    "SERVANT": REG_SERVANT_BRIEF,
    "MESSENGER": REG_SERVANT_BRIEF,
    "FIRST MESSENGER": REG_SERVANT_BRIEF,
    "SECOND MESSENGER": REG_SERVANT_BRIEF,
    "OFFICER": REG_SERVANT_BRIEF,
    "SOLDIER": REG_SERVANT_BRIEF,
    "FIRST SOLDIER": REG_SERVANT_BRIEF,
    "SECOND SOLDIER": REG_SERVANT_BRIEF,
    "GUARD": REG_SERVANT_BRIEF,
    "SAILOR": REG_SERVANT_BRIEF,
    "WATCHMAN": REG_SERVANT_BRIEF,
    "SHERIFF": REG_SERVANT_BRIEF,
    "PAGE": REG_SERVANT_BRIEF,
    "BOY": REG_SERVANT_BRIEF,
    "HERALD": REG_SERVANT_BRIEF,
    "LORD": REG_SERVANT_BRIEF,  # ambient lord voice is usually brief
    "FIRST LORD": REG_SERVANT_BRIEF,
    "SECOND LORD": REG_SERVANT_BRIEF,
    "GENTLEMAN": REG_SERVANT_BRIEF,
    "FIRST GENTLEMAN": REG_SERVANT_BRIEF,
    "SECOND GENTLEMAN": REG_SERVANT_BRIEF,
    "ABRAHAM": REG_SERVANT_BRIEF,
    "SAMPSON": REG_SERVANT_BRIEF,
    "GREGORY": REG_SERVANT_BRIEF,
    # Supernatural
    "GHOST": REG_SUPERNATURAL,
    "WITCH": REG_SUPERNATURAL,
    "FIRST WITCH": REG_SUPERNATURAL,
    "SECOND WITCH": REG_SUPERNATURAL,
    "THIRD WITCH": REG_SUPERNATURAL,
    "ALL WITCHES": REG_SUPERNATURAL,
    "APPARITION": REG_SUPERNATURAL,
    "HECATE": REG_SUPERNATURAL,
    "FAIRY": REG_SUPERNATURAL,
    "ARIEL": REG_SUPERNATURAL,
    "PUCK": REG_SUPERNATURAL,
    "OBERON": REG_SUPERNATURAL,
    "TITANIA": REG_SUPERNATURAL,
    "SPIRIT": REG_SUPERNATURAL,
    "ORACLE": REG_SUPERNATURAL,
}


def _classify(label: str) -> int:
    """Map a canonical UPPERCASE speaker label to a register code.

    Exact match wins; otherwise fall back to a handful of substring
    heuristics for compound labels. Returns REG_UNKNOWN when nothing
    matches.
    """
    if not label:
        return REG_UNKNOWN
    if label in _EXACT:
        return _EXACT[label]
    # Fall-back heuristics.
    if label.startswith("KING "):
        return REG_ROYAL_FORMAL
    if label.startswith("QUEEN "):
        return REG_ROYAL_FORMAL
    if label.startswith("PRINCE "):
        return REG_ROYAL_FORMAL
    if label.startswith("DUKE ") or label.startswith("DUCHESS "):
        return REG_ROYAL_FORMAL
    if label.startswith("LORD ") or label.startswith("LADY "):
        # A specific Lady X / Lord X is more often a named noble.
        return REG_ROYAL_FORMAL
    if "CITIZEN" in label or "MESSENGER" in label or "SERVANT" in label:
        return REG_SERVANT_BRIEF
    if "SOLDIER" in label or "OFFICER" in label or "GUARD" in label:
        return REG_SERVANT_BRIEF
    if "WITCH" in label or "SPIRIT" in label or "APPARITION" in label:
        return REG_SUPERNATURAL
    return REG_UNKNOWN


def update_speaker_register(state: ModelState, token_id: int) -> ModelState:
    rs = state.recent_speakers
    head = rs[0] if rs else ""
    new_reg = _classify(head)
    # If register didn't change, just increment age (lightly — saturate
    # at 255 to keep the integer bounded and avoid infinite churn in
    # the frozen-state copy path).
    if new_reg == state.speaker_register:
        if state.register_age >= 255:
            return state
        return state.model_copy(update={"register_age": state.register_age + 1})
    return state.model_copy(update={
        "speaker_register": new_reg,
        "register_age": 0,
    })
