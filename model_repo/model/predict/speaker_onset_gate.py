"""Speaker-label onset phonotactic gate.

Shakespeare speaker names are English (or classical) proper names.
Their FIRST TWO LETTERS obey English onset phonotactics: either a
vowel-start, a vowel+vowel diphthong, or a consonant followed by a
vowel/Y, or one of a small set of legal onset consonant clusters.

When the speaker buffer's first two letters form an ILLEGAL onset —
e.g., "HT" (HTIGE), "PM" (PMUTS), "NC" (NCHIG), "RF", "DM" — the label
is almost certainly a phantom the sampler hallucinated. The existing
vowel gate catches all-consonant runs AFTER position 2; the off-trie
drift penalty scales only gradually. Neither acts at position 2, so
labels whose first two letters are already structurally impossible can
sail through to ":" closure.

This layer fires at speaker_label_state == 2 with buffer length >= 2
and applies:
  * strong penalty on ":" (prevent phantom closure)
  * moderate penalty on further letters (discourage extension)
  * boost on "\\n" (escape hatch to end the phantom run)

Escalates with buffer length — a 2-letter illegal onset is a warning;
a 4-letter illegal-onset buffer is a near-certain phantom.

No corpus statistics — the legal-onset set is pure phonological prior.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_VOWELS: frozenset[str] = frozenset("AEIOU")
_CONSONANTS: frozenset[str] = frozenset("BCDFGHJKLMNPQRSTVWXZ")
# Y is vowel-like at position 2 (Tybalt, Henry), consonant at position 1
# (York, Young).

# Legal 2-consonant English name-onset clusters (uppercased). Derived
# from the English onset inventory (and classical/biblical borrowings
# that appear in Shakespeare): BL BR CH CL CR DR DW FL FR GH GL GN GR
# KH KN PH PL PR PS PT RH SC SH SK SL SM SN SP ST SW TH TR TS TW VL VR
# WH WR. Any other consonant-consonant start is illegal.
_LEGAL_CC: frozenset[str] = frozenset({
    "BL", "BR",
    "CH", "CL", "CR",
    "DR", "DW",
    "FL", "FR",
    "GH", "GL", "GN", "GR",
    "KH", "KN",
    "PH", "PL", "PR", "PS", "PT",
    "RH",
    "SC", "SH", "SK", "SL", "SM", "SN", "SP", "ST", "SW",
    "TH", "TR", "TS", "TW",
    "VL", "VR",
    "WH", "WR",
})

# Legal 2-vowel English name-onset diphthongs. English proper-name
# vowel-vowel openings are selective: AE (Aemilia), AI (Aimee),
# AO (Aoife), AU (Augustus, Aumerle), EA (Eamon), EI (Einhard),
# EO (Eostre), EU (Eugene), IA (Iago), IE (rare), IO (Iona),
# OA (Oates), OE (Oenone), OI (Oisin), OU (rare), UE (rare),
# Doubled-vowel starts: AA (Aaron), EE rare, II rare, OO rare, UU rare.
# UA, UI, IO, UO, etc. are rare — we allow them to avoid false
# positives. Only clearly-impossible vowel sequences would be flagged;
# this set is the safe whitelist.
_LEGAL_VV: frozenset[str] = frozenset({
    "AA", "AE", "AI", "AO", "AU",
    "EA", "EE", "EI", "EO", "EU",
    "IA", "IE", "IO",
    "OA", "OE", "OI", "OU",
    # UA / UE / UI / UO at English name-start are essentially nonexistent
    # (no Shakespeare speaker begins with these). Leave them out so
    # phantom "UA..." / "UE..." labels get penalized.
})


def _is_illegal_onset(ab: str) -> bool:
    """Return True when ab is a clearly-illegal English name onset."""
    if len(ab) < 2:
        return False
    a, b = ab[0], ab[1]
    # Route both characters through uppercase (speaker_buffer is
    # already uppercased but guard defensively).
    a = a.upper()
    b = b.upper()
    if not ("A" <= a <= "Z" and "A" <= b <= "Z"):
        return False

    # Y-first (York, Young) — always legal.
    if a == "Y":
        return False

    # Vowel-first: check vowel-vowel diphthong; vowel+consonant or
    # vowel+Y always legal.
    if a in _VOWELS:
        if b in _VOWELS:
            return (a + b) not in _LEGAL_VV
        # Vowel + consonant (or Y) is legal.
        return False

    # Consonant-first.
    if a in _CONSONANTS:
        # Consonant + vowel or Y — legal.
        if b in _VOWELS or b == "Y":
            return False
        # Consonant + consonant — must be in whitelist.
        return (a + b) not in _LEGAL_CC

    return False


def speaker_onset_gate_bias(
    speaker_label_state: int,
    speaker_buffer: str,
) -> list[float] | None:
    """Return a bias vec suppressing phantom speaker labels whose first
    two letters violate English name-onset phonotactics."""
    if speaker_label_state != 2:
        return None
    n = len(speaker_buffer)
    if n < 2:
        return None
    if not _is_illegal_onset(speaker_buffer[:2]):
        return None

    # Escalation schedule by buffer length.
    # n=2: early warning — don't hard-commit yet (vowel might still
    #      rescue via a separate channel, but we lean toward bail).
    # n=3: strong — 3 chars into an illegal-onset label is very
    #      unlikely to be real.
    # n=4+: extreme — near-certain phantom.
    if n == 2:
        colon_pen = -1.4
        letter_pen = -0.30
        nl_boost = 0.25
    elif n == 3:
        colon_pen = -2.8
        letter_pen = -0.70
        nl_boost = 0.80
    elif n == 4:
        colon_pen = -4.5
        letter_pen = -1.20
        nl_boost = 1.50
    else:  # n >= 5
        colon_pen = -6.0
        letter_pen = -1.70
        nl_boost = 2.30

    vec = [0.0] * VOCAB_SIZE

    idx = VOCAB_INDEX.get(":")
    if idx is not None:
        vec[idx] += colon_pen

    # Penalize further letters — both cases. The FSM routes case by
    # saw_lower but applying to both is safe: any letter continuation
    # is discouraged regardless.
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
        i = VOCAB_INDEX.get(ch)
        if i is not None:
            vec[i] += letter_pen

    # Newline escape hatch — lets the FSM exit label territory cleanly.
    nl = VOCAB_INDEX.get("\n")
    if nl is not None:
        vec[nl] += nl_boost

    return vec
