"""Tier 2/3 — play-family lock.

Shakespeare scenes never mix characters from different plays. Our
samples do (HAMLET prefix produces NERISSA / LEONATO / AUMERLE mid-
text). This stage infers the play-family from completed speaker
labels and locks it so predict/play_family.py can bias speaker-label
letter choices toward in-family names.

Runs after update_speaker_memory (which captures last_speaker_label
and recent_speakers). Fires only at the moment a new speaker label
has just closed, signaled by state.last_speaker_label having just
changed (tracked via speaker_label_state transition 2→3). We detect
this by checking state.last_char == ':' AND state.speaker_label_state
in {3} AND prev state was 2 (encoded as: consecutive colon-close).

Family enumeration (duplicated from schema docstring for clarity):
  0 UNKNOWN  1 HAMLET_DANE  2 ROMAN  3 ENGLISH_HISTORY
  4 OTHER_TRAGEDY  5 COMEDY_PROSE  6 ROMANCE

Unknown / generic speakers (MESSENGER, SERVANT, etc.) are NEUTRAL —
they do not overwrite an existing lock.

No corpus statistics: speaker→family assignments come from known
Shakespeare canon.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


PF_UNKNOWN = 0
PF_HAMLET_DANE = 1
PF_ROMAN = 2
PF_ENGLISH_HISTORY = 3
PF_OTHER_TRAGEDY = 4
PF_COMEDY_PROSE = 5
PF_ROMANCE = 6


# Canonical-name → family. Keys are UPPERCASE (speaker labels are
# stored uppercased). Names that appear in multiple plays are
# assigned to the play where they're the major character (e.g.
# MARGARET -> ENGLISH_HISTORY because she's in H6/R3; ANTONIO ->
# COMEDY_PROSE because he's in Merchant/Twelfth).
_SPEAKER_FAMILY: dict[str, int] = {
    # ----- Hamlet only -----
    "HAMLET": PF_HAMLET_DANE,
    "HORATIO": PF_HAMLET_DANE,
    "OPHELIA": PF_HAMLET_DANE,
    "POLONIUS": PF_HAMLET_DANE,
    "LAERTES": PF_HAMLET_DANE,
    "CLAUDIUS": PF_HAMLET_DANE,
    "GERTRUDE": PF_HAMLET_DANE,
    "FORTINBRAS": PF_HAMLET_DANE,
    "ROSENCRANTZ": PF_HAMLET_DANE,
    "GUILDENSTERN": PF_HAMLET_DANE,
    "OSRIC": PF_HAMLET_DANE,
    "MARCELLUS": PF_HAMLET_DANE,
    "BERNARDO": PF_HAMLET_DANE,
    "VOLTEMAND": PF_HAMLET_DANE,
    "REYNALDO": PF_HAMLET_DANE,
    "GHOST": PF_HAMLET_DANE,
    # ----- Roman plays (JC, A&C, Coriolanus, Titus, Timon) -----
    "CAESAR": PF_ROMAN,
    "BRUTUS": PF_ROMAN,   # disambig vs H6 Brutus — both are Roman-ish
    "CASSIUS": PF_ROMAN,
    "ANTONY": PF_ROMAN,
    "CLEOPATRA": PF_ROMAN,
    "OCTAVIUS": PF_ROMAN,
    "ENOBARBUS": PF_ROMAN,
    "CHARMIAN": PF_ROMAN,
    "IRAS": PF_ROMAN,
    "OCTAVIA": PF_ROMAN,
    "POMPEY": PF_ROMAN,
    "LEPIDUS": PF_ROMAN,
    "CORIOLANUS": PF_ROMAN,
    "MENENIUS": PF_ROMAN,
    "VOLUMNIA": PF_ROMAN,
    "COMINIUS": PF_ROMAN,
    "AUFIDIUS": PF_ROMAN,
    "VIRGILIA": PF_ROMAN,
    "SICINIUS": PF_ROMAN,
    "TITUS": PF_ROMAN,
    "AARON": PF_ROMAN,
    "TAMORA": PF_ROMAN,
    "LAVINIA": PF_ROMAN,
    "CHIRON": PF_ROMAN,
    "TIMON": PF_ROMAN,
    "APEMANTUS": PF_ROMAN,
    "FLAVIUS": PF_ROMAN,
    "ALCIBIADES": PF_ROMAN,
    "PORTIA": PF_ROMAN,   # JC Portia overlaps with Merchant Portia; pick Roman
    "CALPURNIA": PF_ROMAN,
    "CASCA": PF_ROMAN,
    "CICERO": PF_ROMAN,
    "TREBONIUS": PF_ROMAN,
    "METELLUS": PF_ROMAN,
    "LIGARIUS": PF_ROMAN,
    "POPILIUS": PF_ROMAN,
    "PUBLIUS": PF_ROMAN,
    # ----- English histories (H4/5/6/8, R2/3, King John) -----
    "HENRY": PF_ENGLISH_HISTORY,
    "KING HENRY": PF_ENGLISH_HISTORY,
    "RICHARD": PF_ENGLISH_HISTORY,
    "KING RICHARD": PF_ENGLISH_HISTORY,
    "HAL": PF_ENGLISH_HISTORY,
    "PRINCE": PF_ENGLISH_HISTORY,
    "PRINCE HENRY": PF_ENGLISH_HISTORY,
    "HOTSPUR": PF_ENGLISH_HISTORY,
    "FALSTAFF": PF_ENGLISH_HISTORY,
    "POINS": PF_ENGLISH_HISTORY,
    "YORK": PF_ENGLISH_HISTORY,
    "LANCASTER": PF_ENGLISH_HISTORY,
    "BOLINGBROKE": PF_ENGLISH_HISTORY,
    "NORTHUMBERLAND": PF_ENGLISH_HISTORY,
    "PERCY": PF_ENGLISH_HISTORY,
    "WARWICK": PF_ENGLISH_HISTORY,
    "GLOUCESTER": PF_ENGLISH_HISTORY,  # shared with Lear; more common in histories
    "BUCKINGHAM": PF_ENGLISH_HISTORY,
    "HASTINGS": PF_ENGLISH_HISTORY,
    "SALISBURY": PF_ENGLISH_HISTORY,
    "OXFORD": PF_ENGLISH_HISTORY,
    "MOWBRAY": PF_ENGLISH_HISTORY,
    "AUMERLE": PF_ENGLISH_HISTORY,
    "CATESBY": PF_ENGLISH_HISTORY,
    "RATCLIFFE": PF_ENGLISH_HISTORY,
    "TYRRELL": PF_ENGLISH_HISTORY,
    "STANLEY": PF_ENGLISH_HISTORY,
    "MARGARET": PF_ENGLISH_HISTORY,
    "ELIZABETH": PF_ENGLISH_HISTORY,
    "ANNE": PF_ENGLISH_HISTORY,
    "CLARENCE": PF_ENGLISH_HISTORY,
    "EDWARD": PF_ENGLISH_HISTORY,
    "EDWARD IV": PF_ENGLISH_HISTORY,
    "SUFFOLK": PF_ENGLISH_HISTORY,
    "TALBOT": PF_ENGLISH_HISTORY,
    "PISTOL": PF_ENGLISH_HISTORY,
    "BARDOLPH": PF_ENGLISH_HISTORY,
    "NYM": PF_ENGLISH_HISTORY,
    "MORTIMER": PF_ENGLISH_HISTORY,
    "DOUGLAS": PF_ENGLISH_HISTORY,
    "KATHARINE": PF_ENGLISH_HISTORY,
    "KING JOHN": PF_ENGLISH_HISTORY,
    "CONSTANCE": PF_ENGLISH_HISTORY,
    "FAULCONBRIDGE": PF_ENGLISH_HISTORY,
    "WOLSEY": PF_ENGLISH_HISTORY,
    "CRANMER": PF_ENGLISH_HISTORY,
    "CROMWELL": PF_ENGLISH_HISTORY,
    # ----- Other tragedies (Lear, Macbeth, Othello, R&J) -----
    "LEAR": PF_OTHER_TRAGEDY,
    "CORDELIA": PF_OTHER_TRAGEDY,
    "REGAN": PF_OTHER_TRAGEDY,
    "GONERIL": PF_OTHER_TRAGEDY,
    "EDMUND": PF_OTHER_TRAGEDY,
    "EDGAR": PF_OTHER_TRAGEDY,
    "KENT": PF_OTHER_TRAGEDY,
    "OSWALD": PF_OTHER_TRAGEDY,
    "ALBANY": PF_OTHER_TRAGEDY,
    "CORNWALL": PF_OTHER_TRAGEDY,
    "FOOL": PF_OTHER_TRAGEDY,
    "MACBETH": PF_OTHER_TRAGEDY,
    "LADY MACBETH": PF_OTHER_TRAGEDY,
    "BANQUO": PF_OTHER_TRAGEDY,
    "DUNCAN": PF_OTHER_TRAGEDY,
    "MALCOLM": PF_OTHER_TRAGEDY,
    "MACDUFF": PF_OTHER_TRAGEDY,
    "ROSS": PF_OTHER_TRAGEDY,
    "FLEANCE": PF_OTHER_TRAGEDY,
    "LENNOX": PF_OTHER_TRAGEDY,
    "ANGUS": PF_OTHER_TRAGEDY,
    "OTHELLO": PF_OTHER_TRAGEDY,
    "IAGO": PF_OTHER_TRAGEDY,
    "DESDEMONA": PF_OTHER_TRAGEDY,
    "CASSIO": PF_OTHER_TRAGEDY,
    "RODERIGO": PF_OTHER_TRAGEDY,
    "EMILIA": PF_OTHER_TRAGEDY,
    "BIANCA": PF_OTHER_TRAGEDY,   # also in Taming, but Othello more salient
    "BRABANTIO": PF_OTHER_TRAGEDY,
    "MONTANO": PF_OTHER_TRAGEDY,
    "LODOVICO": PF_OTHER_TRAGEDY,
    "ROMEO": PF_OTHER_TRAGEDY,
    "JULIET": PF_OTHER_TRAGEDY,
    "MERCUTIO": PF_OTHER_TRAGEDY,
    "TYBALT": PF_OTHER_TRAGEDY,
    "BENVOLIO": PF_OTHER_TRAGEDY,
    "NURSE": PF_OTHER_TRAGEDY,
    "FRIAR LAURENCE": PF_OTHER_TRAGEDY,
    "FRIAR": PF_OTHER_TRAGEDY,
    "PARIS": PF_OTHER_TRAGEDY,
    "CAPULET": PF_OTHER_TRAGEDY,
    "MONTAGUE": PF_OTHER_TRAGEDY,
    "LADY CAPULET": PF_OTHER_TRAGEDY,
    "WITCH": PF_OTHER_TRAGEDY,
    # ----- Comedies -----
    "THESEUS": PF_COMEDY_PROSE,
    "HIPPOLYTA": PF_COMEDY_PROSE,
    "OBERON": PF_COMEDY_PROSE,
    "TITANIA": PF_COMEDY_PROSE,
    "PUCK": PF_COMEDY_PROSE,
    "LYSANDER": PF_COMEDY_PROSE,
    "DEMETRIUS": PF_COMEDY_PROSE,  # also Titus but MND/MoV use shared
    "HELENA": PF_COMEDY_PROSE,
    "HERMIA": PF_COMEDY_PROSE,
    "BOTTOM": PF_COMEDY_PROSE,
    "QUINCE": PF_COMEDY_PROSE,
    "FLUTE": PF_COMEDY_PROSE,
    "SNOUT": PF_COMEDY_PROSE,
    "STARVELING": PF_COMEDY_PROSE,
    "SNUG": PF_COMEDY_PROSE,
    "ORSINO": PF_COMEDY_PROSE,
    "VIOLA": PF_COMEDY_PROSE,
    "OLIVIA": PF_COMEDY_PROSE,
    "SEBASTIAN": PF_COMEDY_PROSE,
    "FESTE": PF_COMEDY_PROSE,
    "MALVOLIO": PF_COMEDY_PROSE,
    "SIR TOBY": PF_COMEDY_PROSE,
    "SIR ANDREW": PF_COMEDY_PROSE,
    "MARIA": PF_COMEDY_PROSE,
    "ANTONIO": PF_COMEDY_PROSE,
    "BEATRICE": PF_COMEDY_PROSE,
    "BENEDICK": PF_COMEDY_PROSE,
    "CLAUDIO": PF_COMEDY_PROSE,
    "HERO": PF_COMEDY_PROSE,
    "DON PEDRO": PF_COMEDY_PROSE,
    "DON JOHN": PF_COMEDY_PROSE,
    "LEONATO": PF_COMEDY_PROSE,
    "DOGBERRY": PF_COMEDY_PROSE,
    "VERGES": PF_COMEDY_PROSE,
    "ROSALIND": PF_COMEDY_PROSE,
    "CELIA": PF_COMEDY_PROSE,
    "ORLANDO": PF_COMEDY_PROSE,
    "JAQUES": PF_COMEDY_PROSE,
    "TOUCHSTONE": PF_COMEDY_PROSE,
    "AUDREY": PF_COMEDY_PROSE,
    "DUKE SENIOR": PF_COMEDY_PROSE,
    "OLIVER": PF_COMEDY_PROSE,
    "NERISSA": PF_COMEDY_PROSE,
    "BASSANIO": PF_COMEDY_PROSE,
    "SHYLOCK": PF_COMEDY_PROSE,
    "JESSICA": PF_COMEDY_PROSE,
    "LORENZO": PF_COMEDY_PROSE,
    "LAUNCELOT": PF_COMEDY_PROSE,
    "GRATIANO": PF_COMEDY_PROSE,
    "LAUNCE": PF_COMEDY_PROSE,
    "SPEED": PF_COMEDY_PROSE,
    "PROTEUS": PF_COMEDY_PROSE,
    "VALENTINE": PF_COMEDY_PROSE,
    "JULIA": PF_COMEDY_PROSE,
    "SILVIA": PF_COMEDY_PROSE,
    "KATE": PF_COMEDY_PROSE,
    "KATHARINA": PF_COMEDY_PROSE,
    "BAPTISTA": PF_COMEDY_PROSE,
    "PETRUCHIO": PF_COMEDY_PROSE,
    "LUCENTIO": PF_COMEDY_PROSE,
    "TRANIO": PF_COMEDY_PROSE,
    "GRUMIO": PF_COMEDY_PROSE,
    "HORTENSIO": PF_COMEDY_PROSE,
    "VINCENTIO": PF_COMEDY_PROSE,
    "FORD": PF_COMEDY_PROSE,
    "PAGE": PF_COMEDY_PROSE,
    "SHALLOW": PF_COMEDY_PROSE,
    "SLENDER": PF_COMEDY_PROSE,
    "EVANS": PF_COMEDY_PROSE,
    "HOST": PF_COMEDY_PROSE,
    "HELEN": PF_COMEDY_PROSE,
    "BERTRAM": PF_COMEDY_PROSE,
    "PAROLLES": PF_COMEDY_PROSE,
    "COUNTESS": PF_COMEDY_PROSE,
    "LAFEW": PF_COMEDY_PROSE,
    "ANGELO": PF_COMEDY_PROSE,
    "ISABELLA": PF_COMEDY_PROSE,
    "ESCALUS": PF_COMEDY_PROSE,
    "LUCIO": PF_COMEDY_PROSE,
    "VINCENT": PF_COMEDY_PROSE,
    "ABRAHAM": PF_COMEDY_PROSE,
    "BEROWNE": PF_COMEDY_PROSE,
    "BIRON": PF_COMEDY_PROSE,
    "ADRIANA": PF_COMEDY_PROSE,
    "LUCIANA": PF_COMEDY_PROSE,
    "DROMIO": PF_COMEDY_PROSE,
    # ----- Romances -----
    "PROSPERO": PF_ROMANCE,
    "MIRANDA": PF_ROMANCE,
    "ARIEL": PF_ROMANCE,
    "CALIBAN": PF_ROMANCE,
    "FERDINAND": PF_ROMANCE,
    "GONZALO": PF_ROMANCE,
    "STEPHANO": PF_ROMANCE,
    "TRINCULO": PF_ROMANCE,
    "ALONSO": PF_ROMANCE,
    "LEONTES": PF_ROMANCE,
    "HERMIONE": PF_ROMANCE,
    "PERDITA": PF_ROMANCE,
    "PAULINA": PF_ROMANCE,
    "AUTOLYCUS": PF_ROMANCE,
    "POLIXENES": PF_ROMANCE,
    "FLORIZEL": PF_ROMANCE,
    "CAMILLO": PF_ROMANCE,
    "PERICLES": PF_ROMANCE,
    "MARINA": PF_ROMANCE,
    "THAISA": PF_ROMANCE,
    "SIMONIDES": PF_ROMANCE,
    "CLEON": PF_ROMANCE,
    "DIONYZA": PF_ROMANCE,
    "CERIMON": PF_ROMANCE,
    "IMOGEN": PF_ROMANCE,
    "POSTHUMUS": PF_ROMANCE,
    "CLOTEN": PF_ROMANCE,
    "CYMBELINE": PF_ROMANCE,
    "IACHIMO": PF_ROMANCE,
    "PISANIO": PF_ROMANCE,
    "BELARIUS": PF_ROMANCE,
    "GUIDERIUS": PF_ROMANCE,
    "ARVIRAGUS": PF_ROMANCE,
    "PHILARIO": PF_ROMANCE,
}


def classify_speaker(name: str) -> int:
    """Lookup a speaker name -> play_family. Handles name normalization
    for numeral speakers ('FIRST', 'SECOND', 'THIRD') and compound
    titles ('KING HENRY' → HENRY). Returns PF_UNKNOWN for unmapped /
    generic speakers (MESSENGER, SERVANT, CITIZEN, GENTLEMAN, LORD,
    LADY, OFFICER, SOLDIER, CAPTAIN, KING, QUEEN, DUKE alone, etc.).
    """
    if not name:
        return PF_UNKNOWN
    name = name.strip().upper()
    if name in _SPEAKER_FAMILY:
        return _SPEAKER_FAMILY[name]
    # Compound titles: try stripping common prefixes.
    for prefix in ("KING ", "QUEEN ", "DUKE ", "DUCHESS ",
                   "LORD ", "LADY ", "SIR ", "PRINCE ", "PRINCESS ",
                   "DON ", "FRIAR ", "FATHER "):
        if name.startswith(prefix):
            tail = name[len(prefix):]
            if tail in _SPEAKER_FAMILY:
                return _SPEAKER_FAMILY[tail]
    # Try first word only (KING HENRY IV → KING → no, HENRY → history).
    parts = name.split()
    if len(parts) >= 2 and parts[0] not in ("FIRST", "SECOND", "THIRD", "FOURTH"):
        if parts[0] in _SPEAKER_FAMILY:
            return _SPEAKER_FAMILY[parts[0]]
    return PF_UNKNOWN


def update_play_family(state: ModelState, token_id: int) -> ModelState:
    # Fire on the character that closes a speaker label: the colon.
    # Prefer the FSM-confirmed path (state 2→3, last_speaker_label set
    # by linguistic). But ALSO fire at any ":" preceded by an
    # uppercase letter that just completed a word — this catches the
    # initial-prefix case where the FSM never entered state 1 because
    # no \n\n preceded the label (how every sample run begins).
    ch = VOCAB[token_id]
    if ch != ":":
        return state

    name = ""
    # Path 1: FSM confirmed label close — last_speaker_label is set.
    if state.speaker_label_state == 3 and state.last_speaker_label:
        name = state.last_speaker_label
    # Path 2: ":" right after a just-completed word whose final char
    # was uppercase. We uppercase last_completed_word and look it up.
    # If the all-caps form matches a canonical speaker name, trust it.
    elif (
        state.just_finished_word
        and state.prev_char
        and state.prev_char.isupper()
        and state.last_completed_word
    ):
        candidate = state.last_completed_word.upper().strip()
        if candidate in _SPEAKER_FAMILY:
            name = candidate
        else:
            return state
    else:
        return state

    fam = classify_speaker(name)
    if fam == PF_UNKNOWN:
        # Generic / unrecognized speaker — do NOT overwrite a prior
        # lock. Keep the existing family.
        return state

    if fam != state.play_family:
        return state.model_copy(update={"play_family": fam})
    return state
