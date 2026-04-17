"""Speaker-label trie bias.

Shakespearean speaker labels look like:

    First Citizen:
    MENENIUS:
    KING HENRY IV:
    LADY MACBETH:

When our FSM says we're inside a speaker label, bias the distribution
strongly toward the next characters of known canonical speaker names.

This layer activates whenever `state.speaker_buffer` is non-empty and
matches a known prefix; it returns a letter/space/colon bias vector.
"""

from __future__ import annotations

import math

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# Canonical speaker labels from Shakespeare's works. Mixed-case here for
# readability; we uppercase before building the trie so they can match
# the speaker_buffer (which is uppercase).
_SPEAKERS: tuple[str, ...] = (
    # Coriolanus
    "MENENIUS", "CORIOLANUS", "MARCIUS", "CAIUS MARCIUS",
    "BRUTUS", "SICINIUS", "COMINIUS", "TITUS LARTIUS", "LARTIUS",
    "VOLUMNIA", "VIRGILIA", "VALERIA",
    # Hamlet
    "HAMLET", "CLAUDIUS", "KING CLAUDIUS", "GERTRUDE", "QUEEN GERTRUDE",
    "POLONIUS", "LAERTES", "OPHELIA", "HORATIO", "FORTINBRAS",
    "ROSENCRANTZ", "GUILDENSTERN", "OSRIC", "MARCELLUS", "BERNARDO",
    "FRANCISCO", "REYNALDO",
    # Othello
    "OTHELLO", "IAGO", "DESDEMONA", "CASSIO", "EMILIA", "RODERIGO",
    "BRABANTIO", "MONTANO", "LODOVICO", "BIANCA",
    # Macbeth
    "MACBETH", "LADY MACBETH", "MACDUFF", "LADY MACDUFF", "BANQUO",
    "DUNCAN", "KING DUNCAN", "MALCOLM", "DONALBAIN", "LENNOX",
    "ROSS", "ANGUS", "FLEANCE", "HECATE", "SIWARD", "YOUNG SIWARD",
    # King Lear
    "KING LEAR", "LEAR", "CORDELIA", "GONERIL", "REGAN", "EDMUND",
    "EDGAR", "KENT", "GLOUCESTER", "ALBANY", "CORNWALL", "OSWALD",
    "FOOL",
    # Romeo and Juliet
    "ROMEO", "JULIET", "MERCUTIO", "TYBALT", "BENVOLIO", "PARIS",
    "FRIAR LAURENCE", "FRIAR JOHN", "NURSE", "CAPULET", "LADY CAPULET",
    "MONTAGUE", "LADY MONTAGUE", "ESCALUS", "PRINCE", "SAMPSON",
    "GREGORY", "ABRAHAM", "BALTHASAR", "PETER",
    # Henry plays & others
    "KING HENRY", "KING HENRY IV", "KING HENRY V", "KING HENRY VI",
    "KING RICHARD", "KING RICHARD II", "KING RICHARD III",
    "KING EDWARD", "KING EDWARD IV", "KING JOHN",
    "PRINCE HENRY", "PRINCE HAL", "HENRY BOLINGBROKE",
    "PRINCE EDWARD", "HOTSPUR", "FALSTAFF", "POINS", "BARDOLPH",
    "PISTOL", "NYM", "MISTRESS QUICKLY", "DOLL TEARSHEET",
    "GLENDOWER", "OWEN GLENDOWER", "NORTHUMBERLAND",
    "WESTMORELAND", "WORCESTER", "DOUGLAS", "MORTIMER", "VERNON",
    "BLUNT", "PERCY", "LADY PERCY", "PRINCE JOHN", "GAUNT", "YORK",
    "AUMERLE", "SURREY", "CARLISLE", "MOWBRAY", "BUSHY", "BAGOT",
    "GREEN", "ARCHBISHOP", "BISHOP",
    # Antony and Cleopatra
    "ANTONY", "MARK ANTONY", "CLEOPATRA", "OCTAVIUS CAESAR",
    "CAESAR", "ENOBARBUS", "LEPIDUS", "POMPEY", "MENAS", "EROS",
    "CHARMIAN", "IRAS", "ALEXAS", "MARDIAN",
    # Julius Caesar
    "JULIUS CAESAR", "BRUTUS", "CASSIUS", "MARK ANTONY", "CASCA",
    "OCTAVIUS", "CALPURNIA", "PORTIA", "LUCIUS", "MESSALA",
    # The Tempest
    "PROSPERO", "MIRANDA", "CALIBAN", "ARIEL", "FERDINAND",
    "ALONSO", "SEBASTIAN", "GONZALO", "TRINCULO", "STEPHANO",
    # Much Ado / Twelfth Night / As You Like It / others
    "BEATRICE", "BENEDICK", "CLAUDIO", "HERO", "LEONATO",
    "DON PEDRO", "DON JOHN", "DOGBERRY", "VERGES",
    "ORSINO", "VIOLA", "OLIVIA", "SEBASTIAN", "MALVOLIO",
    "SIR TOBY", "SIR ANDREW", "MARIA", "FESTE",
    "ROSALIND", "ORLANDO", "CELIA", "DUKE SENIOR", "DUKE FREDERICK",
    "JAQUES", "TOUCHSTONE", "PHEBE", "SILVIUS", "AUDREY", "ADAM",
    # Merchant of Venice
    "SHYLOCK", "ANTONIO", "BASSANIO", "PORTIA", "NERISSA",
    "JESSICA", "LORENZO", "GRATIANO", "LAUNCELOT",
    # Midsummer Night's Dream
    "THESEUS", "HIPPOLYTA", "OBERON", "TITANIA", "PUCK",
    "HERMIA", "HELENA", "LYSANDER", "DEMETRIUS", "EGEUS",
    "BOTTOM", "QUINCE", "FLUTE", "SNOUT", "STARVELING", "SNUG",
    # Richard II / III
    "QUEEN MARGARET", "QUEEN ELIZABETH", "DUCHESS OF YORK",
    "DUCHESS OF GLOUCESTER",
    # Generic crowd & small parts
    "ALL", "BOTH", "CHORUS", "PROLOGUE", "EPILOGUE",
    "FIRST CITIZEN", "SECOND CITIZEN", "THIRD CITIZEN", "FOURTH CITIZEN",
    "FIRST SERVANT", "SECOND SERVANT", "THIRD SERVANT",
    "FIRST MURDERER", "SECOND MURDERER", "THIRD MURDERER",
    "FIRST LORD", "SECOND LORD", "THIRD LORD",
    "FIRST GENTLEMAN", "SECOND GENTLEMAN", "THIRD GENTLEMAN",
    "FIRST SOLDIER", "SECOND SOLDIER", "THIRD SOLDIER",
    "FIRST OFFICER", "SECOND OFFICER", "THIRD OFFICER",
    "FIRST WATCHMAN", "SECOND WATCHMAN", "THIRD WATCHMAN",
    "FIRST SENATOR", "SECOND SENATOR", "THIRD SENATOR",
    "FIRST MESSENGER", "SECOND MESSENGER",
    "FIRST WITCH", "SECOND WITCH", "THIRD WITCH",
    "LORD", "LADY", "SERVANT", "MESSENGER", "CAPTAIN",
    "GENTLEMAN", "GENTLEWOMAN", "LORD MAYOR", "MAYOR",
    "KING", "QUEEN", "PRINCE", "PRINCESS", "DUKE", "DUCHESS",
    "EARL", "COUNT", "FRIAR", "NURSE", "CITIZEN", "WATCH", "WATCHMAN",
    "DOCTOR", "PRIEST", "JAILER", "HERALD", "OFFICER",
    "SOLDIER", "SENATOR", "TRIBUNE", "COUNTRYMAN",
    "SCRIVENER", "MUSICIAN", "GARDENER", "GROOM", "PAGE",
    "POET", "PAINTER", "JEWELLER", "MERCHANT",
)


# Build trie: prefix -> {next_char: count}
_TRIE: dict[str, dict[str, int]] = {}


def _add(word: str) -> None:
    for i in range(len(word) + 1):
        prefix = word[:i]
        _TRIE.setdefault(prefix, {})
        if i < len(word):
            _TRIE[prefix][word[i]] = _TRIE[prefix].get(word[i], 0) + 1
        else:
            # Terminator: ":" ends the label.
            _TRIE[prefix][":"] = _TRIE[prefix].get(":", 0) + 5


for _s in _SPEAKERS:
    _add(_s)


def _bias_for(prefix: str) -> list[float] | None:
    if prefix not in _TRIE:
        return None
    nexts = _TRIE[prefix]
    if not nexts:
        return None
    n = len(prefix)
    scale = min(0.3 + 0.6 * n, 3.5)
    total = sum(nexts.values())
    vec = [0.0] * VOCAB_SIZE
    # Slight negative on any upper-case letter not in our next-set.
    neg = -0.4 * min(scale, 2.0)
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] = neg
    for ch, w in nexts.items():
        if ch not in VOCAB_INDEX:
            continue
        frac = w / total
        bias = scale * math.log((frac + 0.02) / 0.05)
        vec[VOCAB_INDEX[ch]] = bias
    return vec


def _precompute() -> dict[str, list[float]]:
    return {p: _bias_for(p) for p in _TRIE if _bias_for(p) is not None}


_PREFIX_BIAS: dict[str, list[float]] = _precompute()


def speaker_trie_bias(buffer: str) -> list[float] | None:
    if not buffer:
        return None
    return _PREFIX_BIAS.get(buffer)
