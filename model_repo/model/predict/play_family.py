"""Predict layer — play-family bias on speaker-label letters.

At speaker_label_state in {1, 2}, when state.play_family is LOCKED
(non-zero), bias the letter distribution toward first-letters of
speakers in that family and away from first-letters that are only
in OTHER families.

Fires only inside a speaker label (state 1 or 2) so the mid-speech
text is unaffected. Inside a label, the current character could
be any of: the first letter of the label (state 1 just ending), or
a continuation letter (state 2). We tilt both positions — with the
strongest signal at state 1 (first-letter commit).

No gibberish-catch role here (speaker_offtrie handles that); this
is purely a cross-play mixing filter.

Weights are from the per-family first-letter frequency distribution
over the speaker list in pipeline/play_family.py — NOT counted from
the corpus, but hand-aggregated. Rough distributions:

  HAMLET_DANE     : H O P L C G F R M B V
  ROMAN           : C A B V O M P T S I F
  ENGLISH_HISTORY : H R Y L B N P W G C M E A S T F
  OTHER_TRAGEDY   : L C R G E M B D O I T F J K N A P W
  COMEDY_PROSE    : B C D H L M O P R S T V A E F G J K N I
  ROMANCE         : P M A C L H F G T S I D E B
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Set of first letters (uppercase) known to begin in-family speaker
# names. Derived from the _SPEAKER_FAMILY table by aggregating first
# letters of each name, then manually listed here.
_FAMILY_FIRST_LETTERS: dict[int, dict[str, float]] = {
    # 1 HAMLET_DANE
    1: {
        "H": 1.0,   # HAMLET, HORATIO
        "O": 0.9,   # OPHELIA, OSRIC
        "P": 0.9,   # POLONIUS
        "L": 0.9,   # LAERTES
        "C": 0.8,   # CLAUDIUS
        "G": 0.8,   # GERTRUDE, GUILDENSTERN, GHOST
        "F": 0.7,   # FORTINBRAS
        "R": 0.9,   # ROSENCRANTZ, REYNALDO
        "M": 0.8,   # MARCELLUS
        "B": 0.7,   # BERNARDO
        "V": 0.5,   # VOLTEMAND
    },
    # 2 ROMAN
    2: {
        "C": 1.0,   # CAESAR, CASSIUS, CLEOPATRA, CORIOLANUS, COMINIUS, CHARMIAN, CICERO, CASCA, CALPURNIA
        "A": 1.0,   # ANTONY, AUFIDIUS, AARON, APEMANTUS, ALCIBIADES
        "B": 0.9,   # BRUTUS
        "V": 0.9,   # VOLUMNIA, VIRGILIA
        "O": 0.8,   # OCTAVIUS, OCTAVIA
        "M": 0.8,   # MENENIUS, METELLUS
        "P": 1.0,   # POMPEY, PORTIA, PUBLIUS, POPILIUS
        "T": 0.9,   # TITUS, TAMORA, TREBONIUS
        "S": 0.7,   # SICINIUS
        "I": 0.7,   # IRAS
        "F": 0.7,   # FLAVIUS
        "L": 0.8,   # LEPIDUS, LAVINIA, LIGARIUS
        "E": 0.7,   # ENOBARBUS
        "C_chiron": 0.0,  # CHIRON already covered by C
    },
    # 3 ENGLISH_HISTORY
    3: {
        "H": 1.0,   # HENRY, HAL, HOTSPUR, HASTINGS
        "R": 1.0,   # RICHARD, RATCLIFFE
        "Y": 0.9,   # YORK
        "L": 0.9,   # LANCASTER
        "B": 1.0,   # BOLINGBROKE, BUCKINGHAM, BARDOLPH
        "N": 0.9,   # NORTHUMBERLAND, NYM
        "P": 1.0,   # PERCY, POINS, PRINCE, PISTOL
        "W": 0.9,   # WARWICK, WOLSEY
        "G": 0.9,   # GLOUCESTER
        "C": 1.0,   # CATESBY, CROMWELL, CRANMER, CLARENCE, CONSTANCE
        "M": 0.9,   # MARGARET, MOWBRAY, MORTIMER
        "E": 0.9,   # ELIZABETH, EDWARD
        "A": 0.8,   # AUMERLE, ANNE
        "S": 0.9,   # SALISBURY, SUFFOLK, STANLEY
        "T": 0.8,   # TALBOT, TYRRELL
        "F": 0.9,   # FALSTAFF, FAULCONBRIDGE
        "D": 0.7,   # DOUGLAS
        "K": 0.8,   # KATHARINE, KING JOHN
        "O": 0.7,   # OXFORD
    },
    # 4 OTHER_TRAGEDY
    4: {
        "L": 1.0,   # LEAR, LENNOX, LADY MACBETH/CAPULET, LODOVICO
        "C": 0.9,   # CORDELIA, CASSIO, CAPULET
        "R": 1.0,   # REGAN, ROMEO, RODERIGO, ROSS
        "G": 0.9,   # GONERIL
        "E": 1.0,   # EDMUND, EDGAR, EMILIA
        "M": 1.0,   # MACBETH, MACDUFF, MERCUTIO, MALCOLM, MONTAGUE, MONTANO
        "B": 1.0,   # BANQUO, BENVOLIO, BIANCA, BRABANTIO
        "D": 1.0,   # DUNCAN, DESDEMONA
        "O": 1.0,   # OTHELLO, OSWALD
        "I": 1.0,   # IAGO
        "T": 0.9,   # TYBALT
        "F": 0.9,   # FLEANCE, FRIAR, FOOL
        "J": 1.0,   # JULIET
        "K": 0.7,   # KENT
        "N": 0.8,   # NURSE
        "A": 0.8,   # ALBANY, ANGUS
        "P": 0.8,   # PARIS
        "W": 0.7,   # WITCH
    },
    # 5 COMEDY_PROSE
    5: {
        "B": 1.0,   # BEATRICE, BENEDICK, BASSANIO, BOTTOM, BAPTISTA, BEROWNE, BIRON, BERTRAM, BIANCA
        "C": 1.0,   # CLAUDIO, CELIA, COUNTESS
        "D": 1.0,   # DOGBERRY, DEMETRIUS, DON PEDRO, DON JOHN, DROMIO, DUKE SENIOR
        "H": 1.0,   # HERO, HIPPOLYTA, HELENA, HERMIA, HELEN, HORTENSIO, HOST
        "L": 1.0,   # LEONATO, LYSANDER, LAUNCELOT, LORENZO, LAUNCE, LUCENTIO, LAFEW, LUCIO, LUCIANA
        "M": 1.0,   # MALVOLIO, MARIA, MISTRESS
        "O": 1.0,   # OLIVIA, ORSINO, OBERON, ORLANDO, OLIVER
        "P": 1.0,   # PORTIA, PUCK, PETRUCHIO, PROTEUS, PAROLLES, PAGE
        "R": 0.9,   # ROSALIND
        "S": 1.0,   # SEBASTIAN, SHYLOCK, SILVIA, SHALLOW, SLENDER, SPEED
        "T": 1.0,   # TITANIA, TOUCHSTONE, TRANIO, THESEUS
        "V": 1.0,   # VIOLA, VALENTINE, VERGES, VINCENTIO, VINCENT
        "A": 1.0,   # ANTONIO, ANGELO, ADRIANA, AUDREY, ABRAHAM
        "E": 0.8,   # ESCALUS, EVANS
        "F": 0.9,   # FESTE, FORD, FLUTE
        "G": 0.9,   # GRATIANO, GRUMIO
        "J": 1.0,   # JAQUES, JESSICA, JULIA
        "K": 0.9,   # KATE, KATHARINA
        "N": 0.9,   # NERISSA
        "I": 0.8,   # ISABELLA
    },
    # 6 ROMANCE
    6: {
        "P": 1.0,   # PROSPERO, PERDITA, PAULINA, POLIXENES, PERICLES, POSTHUMUS, PISANIO, PHILARIO
        "M": 1.0,   # MIRANDA, MARINA
        "A": 1.0,   # ARIEL, AUTOLYCUS, ALONSO, ARVIRAGUS
        "C": 1.0,   # CALIBAN, CAMILLO, CLEON, CYMBELINE, CLOTEN, CERIMON
        "L": 1.0,   # LEONTES
        "H": 1.0,   # HERMIONE
        "F": 1.0,   # FERDINAND, FLORIZEL
        "G": 1.0,   # GONZALO, GUIDERIUS
        "T": 1.0,   # TRINCULO, THAISA
        "S": 1.0,   # STEPHANO, SIMONIDES
        "I": 1.0,   # IMOGEN, IACHIMO
        "D": 0.9,   # DIONYZA
        "E": 0.6,
        "B": 1.0,   # BELARIUS
    },
}


def _build_letter_vec(family: int, scale: float) -> list[float]:
    """Build a bias vector that boosts in-family first letters and
    penalizes letters that are NOT in that family's speaker set."""
    vec = [0.0] * VOCAB_SIZE
    letters = _FAMILY_FIRST_LETTERS.get(family, {})
    if not letters:
        return vec

    # Boost in-family letters (both upper and lower-case, though
    # speaker labels are overwhelmingly uppercase at the first-letter
    # position).
    for letter, w in letters.items():
        if len(letter) != 1 or not letter.isalpha():
            continue
        upper = letter.upper()
        if upper in VOCAB_INDEX:
            vec[VOCAB_INDEX[upper]] += scale * w
        lower = letter.lower()
        if lower in VOCAB_INDEX:
            vec[VOCAB_INDEX[lower]] += scale * w * 0.3

    # Penalize letters that are NOT in the family (excluding non-alpha).
    # The scale here is lighter — we don't want to make any letter
    # have a log-prob below validity.
    all_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    family_upper_set = {l.upper() for l in letters if len(l) == 1 and l.isalpha()}
    for ch in all_upper:
        if ch not in family_upper_set:
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] -= scale * 0.45
    return vec


_FAMILY_VEC: dict[int, list[float]] = {
    f: _build_letter_vec(f, scale=0.55) for f in (1, 2, 3, 4, 5, 6)
}


def play_family_bias(
    play_family: int,
    speaker_label_state: int,
    speaker_buffer: str,
) -> list[float] | None:
    """Bias the first-letter-of-name position within a speaker label.

    Fires only at FSM states 1 (just saw \\n\\n, awaiting cap) and 2
    (inside label). Returns None otherwise or if family == UNKNOWN.

    At state 1, this is the FIRST-LETTER commit — strongest signal.
    At state 2 with empty speaker_buffer, still first-letter (buffer
    hasn't captured it yet).
    At state 2 with speaker_buffer length >= 1, we're mid-name — this
    layer's signal attenuates because the first letter is already
    committed and the speaker_trie takes over.
    """
    if play_family == 0:
        return None
    if speaker_label_state not in (1, 2):
        return None

    base = _FAMILY_VEC.get(play_family)
    if base is None:
        return None

    # Attenuate based on how far into the label we are.
    buf_len = len(speaker_buffer or "")
    if buf_len == 0:
        # Full strength — first letter about to be chosen.
        return list(base)
    if buf_len == 1:
        # Slight attenuation — second letter might still help weed
        # out (e.g., ML of MALCOLM vs ME of MERCUTIO).
        return [x * 0.35 for x in base]
    # Beyond 2 letters: speaker_trie handles name completion.
    return None
