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
    # Additional named characters (historical plays, comedies, late plays)
    "BUCKINGHAM", "DUKE OF BUCKINGHAM",
    "CATESBY", "RATCLIFF", "TYRREL", "HASTINGS",
    "WARWICK", "CLARENCE", "GLOUCESTER",
    "DUKE OF YORK", "DUKE OF NORFOLK", "DUKE OF SUFFOLK",
    "DUKE OF SOMERSET", "DUKE OF EXETER",
    "LORD STANLEY", "LORD HASTINGS", "LORD RIVERS",
    "LADY ANNE", "LADY GREY",
    "RICHMOND", "HENRY RICHMOND",
    "LIEUTENANT", "SHERIFF", "TUTOR", "KEEPER",
    "SCRIBE", "LAWYER", "USHER", "TAILOR",
    "HOSTESS", "HOSTESS QUICKLY",
    "TALBOT", "YOUNG TALBOT", "JOAN LA PUCELLE", "PUCELLE",
    "JACK CADE", "CADE", "DICK", "SMITH",
    "JOURDAIN", "SOUTHWELL", "BOLINGBROKE",
    "SIMPCOX", "WHITMORE", "LIEUTENANT",
    "GOWER", "FLUELLEN", "MACMORRIS", "JAMY",
    "CHORUS", "EPILOGUE", "PROLOGUE",
    "CAMILLO", "CLEOMENES", "DION", "ANTIGONUS",
    "MOPSA", "DORCAS", "PERDITA", "HERMIONE",
    "LEONTES", "POLIXENES", "FLORIZEL",
    "ARCHIDAMUS", "MAMILLIUS", "EMILIA",
    "BELARIUS", "GUIDERIUS", "ARVIRAGUS", "POSTHUMUS",
    "IACHIMO", "CLOTEN", "IMOGEN", "PISANIO", "CORNELIUS",
    "PHILARIO",
    "SIMONIDES", "PERICLES", "CERIMON", "LYSIMACHUS",
    "THAISA", "MARINA", "BOULT", "LEONINE",
    "TIMON", "APEMANTUS", "ALCIBIADES", "FLAVIUS", "FLAMINIUS",
    "SERVILIUS", "LUCULLUS", "VENTIDIUS",
    "TITUS ANDRONICUS", "AARON", "SATURNINUS", "BASSIANUS",
    "LAVINIA", "MARCUS", "TAMORA", "DEMETRIUS", "CHIRON",
    "LUCIUS", "YOUNG LUCIUS", "QUINTUS", "MARTIUS", "MUTIUS",
    "HELICANUS", "ESCANES",
    "ULYSSES", "AGAMEMNON", "MENELAUS", "AJAX", "NESTOR",
    "DIOMEDES", "PATROCLUS", "THERSITES",
    "PRIAM", "HECTOR", "TROILUS", "PARIS", "DEIPHOBUS",
    "HELENUS", "CRESSIDA", "CASSANDRA", "ANDROMACHE",
    "KATHARINE", "KATHARINA", "PETRUCHIO", "BIANCA",
    "BAPTISTA", "GREMIO", "HORTENSIO", "TRANIO", "LUCENTIO",
    "BIONDELLO", "GRUMIO", "VINCENTIO",
    "OBERON", "PUCK", "BOTTOM", "FLUTE", "EGEUS",
    "NICK BOTTOM", "PETER QUINCE",
    "LAUNCE", "SPEED", "VALENTINE", "PROTEUS", "JULIA",
    "SILVIA", "LUCETTA", "THURIO",
    "BEROWNE", "LONGAVILLE", "DUMAINE", "ROSALINE",
    "FERDINAND", "KING OF NAVARRE",
    "MOTH", "ARMADO", "COSTARD", "HOLOFERNES", "NATHANIEL",
    "DULL", "JAQUENETTA",
    # More Antony & Cleopatra / Julius Caesar
    "AGRIPPA", "PROCULEIUS", "DOLABELLA", "THIDIAS",
    "MENECRATES", "CANIDIUS", "PHILO", "DERCETAS",
    "METELLUS", "CINNA", "TREBONIUS", "DECIUS",
    "ARTEMIDORUS", "FLAVIUS", "MARULLUS", "VOLUMNIUS",
    "VARRO", "CLITUS", "DARDANIUS", "STRATO", "TITINIUS",
    "PINDARUS", "LEPIDUS", "YOUNG CATO",
    # Measure for Measure
    "ISABELLA", "ANGELO", "ESCALUS", "CLAUDIO", "MARIANA",
    "LUCIO", "POMPEY", "ELBOW", "FROTH", "ABHORSON",
    "BARNARDINE", "OVERDONE", "MISTRESS OVERDONE",
    "PROVOST", "JULIET",
    # All's Well That Ends Well
    "HELENA", "BERTRAM", "PAROLLES", "LAFEU", "COUNTESS",
    "LAVACH", "LAVATCH",
    # Much Ado About Nothing extras
    "LEONATO", "ANTONIO", "BORACHIO", "CONRADE",
    "SEACOAL", "VERGES", "DON JOHN", "DON PEDRO",
    "URSULA", "MARGARET",
    # Merry Wives of Windsor
    "FORD", "MISTRESS FORD", "PAGE", "MISTRESS PAGE",
    "SHALLOW", "SLENDER", "EVANS", "DOCTOR CAIUS",
    "CAIUS", "FENTON", "ANNE PAGE", "ROBIN",
    "WILLIAM", "PISTOL", "SIMPLE", "RUGBY",
    # Winter's Tale / Cymbeline more
    "ANTIGONUS", "EMILIA", "AUTOLYCUS", "CLOWN",
    "OLD SHEPHERD", "SHEPHERD", "YOUNG SHEPHERD",
    "MARINER", "TIME", "FIRST SHEPHERD",
    # Henry VIII
    "WOLSEY", "CARDINAL WOLSEY", "CRANMER", "CROMWELL",
    "SURVEYOR", "CHAMBERLAIN", "LORD CHAMBERLAIN",
    "GARDINER", "LOVELL", "BRANDON", "CAMPEIUS",
    "CAPUCHIUS", "GRIFFITH", "PATIENCE",
    # Two Gentlemen / extras
    "LAUNCE", "SPEED", "PANTHINO", "EGLAMOUR",
    # Comedy of Errors
    "ANTIPHOLUS", "DROMIO", "SOLINUS", "ANGELO",
    "LUCIANA", "ADRIANA", "AEMILIA",
    # Pericles extras
    "GOWER", "DIONYZA", "CLEON", "PHILEMON",
    # Henry IV / V extras
    "FANG", "SNARE", "DAVY", "FEEBLE", "WART",
    "MOULDY", "BULLCALF", "SHADOW", "PETO",
    "CHIEF JUSTICE", "LORD CHIEF JUSTICE",
    "DAUPHIN", "CONSTABLE", "MOUNTJOY", "ERPINGHAM",
    "RAMBURES", "GRANDPRE", "GOVERNOR",
    # Henry VI parts 1-3
    "VERNON", "BASSET", "MARGARET", "QUEEN MARGARET",
    "LEWIS", "KING LEWIS", "KING LEWIS XI",
    "REIGNIER", "ALENCON", "CHARLES", "BASTARD",
    "BEVIS", "GEORGE", "HOLLAND", "MICHAEL",
    "YOUNG CLIFFORD", "SAY", "LORD SAY",
    # Richard III extras
    "DUCHESS", "DUCHESS OF YORK", "QUEEN ELIZABETH",
    "KING EDWARD IV", "BRAKENBURY", "LORD MAYOR",
    "BLUNT", "BREAKENBURY", "DIGHTON", "FORREST",
    # King Lear extras
    "CURAN", "OLD MAN",
    # Tempest extras
    "BOATSWAIN", "MASTER", "MARINERS", "IRIS",
    "CERES", "JUNO", "NYMPHS", "REAPERS",
    # Othello extras
    "GRATIANO",
    # Much Ado / LLL extras
    "BENEDICK", "BEATRICE", "HERO", "BALTHASAR",
    # Hamlet extras
    "GHOST", "PLAYER KING", "PLAYER QUEEN",
    "FIRST PLAYER", "SECOND PLAYER",
    "FIRST CLOWN", "SECOND CLOWN",
    "FIRST GRAVEDIGGER",
    "FIRST AMBASSADOR",
    "VOLTIMAND", "CORNELIUS",
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


def _bias_for(prefix: str, lower: bool) -> list[float] | None:
    """Build the bias vector for ``prefix``. If ``lower`` is True, the
    positive bias is applied to LOWERCASE letter variants (for mixed-
    case speakers like "First Citizen"). Negative bias on non-next-set
    CASE-MATCHED letters is applied similarly. The ":" terminator is
    always boosted in its canonical single-codepoint form.
    """
    if prefix not in _TRIE:
        return None
    nexts = _TRIE[prefix]
    if not nexts:
        return None
    n = len(prefix)
    scale = min(0.3 + 0.6 * n, 3.5)
    total = sum(nexts.values())
    vec = [0.0] * VOCAB_SIZE
    # Mild negative on any case-matched letter not in our next-set.
    neg = -0.4 * min(scale, 2.0)
    alphabet = (
        "abcdefghijklmnopqrstuvwxyz" if lower
        else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    for ch in alphabet:
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] = neg
    for ch, w in nexts.items():
        frac = w / total
        bias = scale * math.log((frac + 0.02) / 0.05)
        # Trie keys are uppercase letters or ":" / " ". Route letters
        # to the case-matched vocab entry; route ":" / " " unchanged.
        if ch.isalpha():
            key = ch.lower() if lower else ch.upper()
            if key in VOCAB_INDEX:
                vec[VOCAB_INDEX[key]] = bias
        else:
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] = bias
    return vec


def _precompute() -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    upper = {}
    lower = {}
    for p in _TRIE:
        u = _bias_for(p, lower=False)
        if u is not None:
            upper[p] = u
        l = _bias_for(p, lower=True)
        if l is not None:
            lower[p] = l
    return upper, lower


_PREFIX_BIAS_UPPER, _PREFIX_BIAS_LOWER = _precompute()


def speaker_trie_bias(
    buffer: str, saw_lower: bool = False
) -> list[float] | None:
    if not buffer:
        return None
    # When the label has already emitted a lowercase letter (e.g.,
    # "First Cit" — the 'irst' and 'it' were lowercase), route the
    # positive continuation bias to LOWERCASE letter variants. This
    # closes the long-standing mismatch where the trie biased only
    # uppercase letters even though mixed-case speaker labels
    # ("First Citizen", "Third Servingman") append lowercase chars.
    if saw_lower:
        return _PREFIX_BIAS_LOWER.get(buffer)
    return _PREFIX_BIAS_UPPER.get(buffer)


def is_speaker_prefix(buffer: str) -> bool:
    """True iff `buffer` is a prefix of at least one canonical speaker
    name (including the empty prefix and complete names)."""
    return buffer in _TRIE
