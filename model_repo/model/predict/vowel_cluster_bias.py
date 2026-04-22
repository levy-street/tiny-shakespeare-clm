"""Vowel-cluster phonotactic bias.

Reads `state.vowel_run_letters` — the literal lowercase vowel string
since the last consonant or word start within the current word.

Complements `post_vowel_cluster` (which handles consonant codas) with
the symmetric vowel-side logic: given the vowel cluster currently
forming, penalize NEXT-VOWEL choices that would extend it into a
phonotactically illegal English vowel sequence.

Examples the existing bigram / cv-alternation layers miss:
  * After "o" — "oe" is legal (toe, foe) but "oa" is also legal
    (boat, goat), while "oo" is legal (boot). Yet after "oe", the
    cluster is essentially closed — a third vowel ("oei", "oea") is
    not a legal English cluster at word-interior position.
  * After "i" — "ia" (Maria), "ie" (pie), "io" (lion) are legal, but
    "iu" is rare (only in proper nouns). After "ie", a third vowel
    would make "iea"/"ieo"/"iei" — essentially none of these are
    legal interior clusters (exception: "ieu" in "lieu").
  * After "e" — "ea", "ee", "ei" (rare), "eo" (rare), "eu", "ew"
    are legal. "ey" word-final is legal. "eau" as 3-cluster is legal
    (beauty). Most other 3-vowel extensions are not.
  * Any 4+ vowel cluster is essentially never legal mid-word.

Rules encoded here:

  * len == 1: permit any vowel that forms a legal 2-vowel cluster;
    penalize those that form illegal 2-vowel clusters (aa, ii, uu,
    yy — actually aa/ii/uu are already caught by illegal_vowel_double,
    but cross-pair illegalities like "ao", "iu", "uy" are covered
    here with a gentler penalty and positive bias on legal pairs).

  * len == 2: the cluster is already a legal 2-vowel. Extending to
    3 vowels is ALMOST always illegal; permit only a whitelisted few
    (eau, iou, ieu, aye, yeu). Push consonants. Penalize all
    non-whitelisted vowel extensions.

  * len >= 3: no legal English interior 4-vowel cluster. Strong push
    to consonant; strong penalty on any vowel.

Gates:
  * speaker_label_state == 0 (labels have loose phonotactics).
  * vowel_run_letters must be non-empty.

No corpus statistics — derived from prior knowledge of English
vowel phonotactics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_VOWEL_CHARS: tuple[str, ...] = ("a", "e", "i", "o", "u", "y")
_CONS_LOWER: str = "bcdfghjklmnpqrstvwxz"

# Legal English 2-vowel interior clusters. Generous: includes all
# common diphthongs and some rarer but attested ones (eo in "people",
# eu in "feud", ua in "guard", uo in "quorum"). Excludes:
#   aa, ii, uu, yy (already handled by illegal_vowel_double, but we
#      also suppress them here for reinforcement)
#   ao (no English word: rare — *cacao*)
#   oa is legal (boat)
#   iu (rare — "Brutius" only as proper noun)
#   uy (rare — only at word-final "guy", "buy")
# Keys are the 2-letter string; value is 1 (legal).
_LEGAL_2VOWEL: frozenset[str] = frozenset({
    # a-lead
    "ae",        # aerial, aesop
    "ai", "au", "aw", "ay",
    # e-lead
    "ea", "ee", "ei", "eo", "eu", "ew", "ey",
    # i-lead
    "ia", "ie", "io",
    # o-lead
    "oa", "oe", "oi", "oo", "ou", "ow", "oy",
    # u-lead
    "ua", "ue", "ui", "uo",
    # y-lead (y-as-first-vowel is rare; "ye" in "yes" the y is
    # consonant, but internal like "tyee", "byes")
    "ye",
})

# Legal English 3-vowel interior clusters (all attested). Very short
# whitelist — essentially these are the only interior triples.
_LEGAL_3VOWEL: frozenset[str] = frozenset({
    "eau",       # beauty, beau, beauteous
    "eou",       # beauteous (rare)
    "iou",       # curious, various, previous
    "ieu",       # lieu, adieu
    "uou",       # vacuous, fatuous
    "aye",       # aye (word-final usually)
    "oey",       # NOT usually legal — omit
    "eye",       # eye, eyes (full word)
    "yie",       # dyeing — actually d-y-e-i-n-g
})


def _build_vecs() -> tuple[list[float], list[float], list[float],
                            list[float], list[float], dict[str, list[float]],
                            dict[str, list[float]]]:
    """Precompute the static bias vectors.

    Returns a tuple of:
      * vowel_push_mild, vowel_push_strong — consonant-push scaffolds
      * cluster2 maps str -> per-vowel-penalty vector
      * cluster3 maps str -> per-vowel-penalty vector (strong)
    """
    # len==2 push: gentle consonants, penalize all vowels except
    # whitelist (this will be rebuilt per-prefix).
    # len==3: strong consonants, strong all-vowel penalty.
    cons_push_mild = [0.0] * VOCAB_SIZE
    cons_push_strong = [0.0] * VOCAB_SIZE
    cons_push_extreme = [0.0] * VOCAB_SIZE
    vowel_pen_mild = [0.0] * VOCAB_SIZE  # unused direct
    vowel_pen_strong = [0.0] * VOCAB_SIZE  # unused direct

    for ch in _CONS_LOWER:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            cons_push_mild[idx] += 0.30
            cons_push_strong[idx] += 0.75
            cons_push_extreme[idx] += 1.35

    # For each existing prefix length (1 or 2), build per-prefix
    # penalty vectors. The idea: for every vowel v not forming a
    # legal cluster with prefix, put a per-letter penalty on v's
    # vocab index.
    prefix_vecs_2: dict[str, list[float]] = {}
    # Length-1 prefixes → for each next-vowel, check (prefix+v) in
    # _LEGAL_2VOWEL. If not, penalize v.
    for prefix in "aeiouy":
        vec = [0.0] * VOCAB_SIZE
        any_set = False
        for v in _VOWEL_CHARS:
            pair = prefix + v
            if pair in _LEGAL_2VOWEL:
                # Legal: tiny positive to reinforce
                idx = VOCAB_INDEX.get(v)
                if idx is not None:
                    vec[idx] += 0.12
                    any_set = True
            else:
                # Illegal 2-vowel cluster. Penalize the vowel (but
                # don't duplicate what illegal_vowel_double catches
                # for doubles — gentler there).
                idx = VOCAB_INDEX.get(v)
                if idx is not None:
                    if pair[0] == pair[1]:
                        vec[idx] += -0.20  # gentle (other layers cover)
                    else:
                        vec[idx] += -0.60  # full cross-pair illegality
                    any_set = True
        if any_set:
            prefix_vecs_2[prefix] = vec

    prefix_vecs_3: dict[str, list[float]] = {}
    # Length-2 prefixes → for each next-vowel, check (prefix+v) in
    # _LEGAL_3VOWEL. If not, strong penalty.
    for a in "aeiouy":
        for b in "aeiouy":
            pfx = a + b
            if pfx not in _LEGAL_2VOWEL:
                # Prefix itself illegal — shouldn't occur because
                # the coda-tracker upstream would have penalized it,
                # but for safety fire anyway.
                vec = [0.0] * VOCAB_SIZE
                for v in _VOWEL_CHARS:
                    idx = VOCAB_INDEX.get(v)
                    if idx is not None:
                        vec[idx] += -1.10
                # Plus push consonants.
                for ch in _CONS_LOWER:
                    idx = VOCAB_INDEX.get(ch)
                    if idx is not None:
                        vec[idx] += 0.60
                prefix_vecs_3[pfx] = vec
                continue
            vec = [0.0] * VOCAB_SIZE
            any_set = False
            for v in _VOWEL_CHARS:
                triple = pfx + v
                if triple in _LEGAL_3VOWEL:
                    idx = VOCAB_INDEX.get(v)
                    if idx is not None:
                        vec[idx] += 0.30
                        any_set = True
                else:
                    idx = VOCAB_INDEX.get(v)
                    if idx is not None:
                        vec[idx] += -1.10
                        any_set = True
            # Always push consonants at len==2 to encourage closure.
            for ch in _CONS_LOWER:
                idx = VOCAB_INDEX.get(ch)
                if idx is not None:
                    vec[idx] += 0.35
                    any_set = True
            if any_set:
                prefix_vecs_3[pfx] = vec

    return (cons_push_mild, cons_push_strong, cons_push_extreme,
            vowel_pen_mild, vowel_pen_strong,
            prefix_vecs_2, prefix_vecs_3)


(_CONS_PUSH_MILD, _CONS_PUSH_STRONG, _CONS_PUSH_EXTREME,
 _VP_MILD, _VP_STRONG, _PFX2, _PFX3) = _build_vecs()


def vowel_cluster_bias(
    vowel_run_letters: str,
    speaker_label_state: int,
    on_word_trie: bool,
    letters_off_trie: int,
    letter_run_len: int,
) -> list[float] | None:
    """Return a bias vector modulating next-letter choice based on
    vowel-cluster phonotactics. Returns None when no signal applies.
    """
    if speaker_label_state != 0:
        return None
    if not vowel_run_letters:
        return None
    if letter_run_len < 1:
        return None

    n = len(vowel_run_letters)

    # While fully on-trie, the trie signal dominates. Keep the fire
    # gentle — scale down rather than silence, so we don't fight the
    # trie on valid "eau"/"iou"/etc. words.
    on_trie_scale = 0.4 if on_word_trie else 1.0
    # More drift off-trie → stronger signal.
    drift = max(0, min(letters_off_trie, 5))
    drift_scale = 1.0 + 0.12 * drift

    scale = on_trie_scale * drift_scale

    if n == 1:
        vec = _PFX2.get(vowel_run_letters)
        if vec is None:
            return None
        return [x * scale for x in vec]

    if n == 2:
        vec = _PFX3.get(vowel_run_letters)
        if vec is None:
            # Unknown 2-prefix (shouldn't happen; 'yy' etc.). Extreme push.
            return [x * scale for x in _CONS_PUSH_EXTREME]
        return [x * scale for x in vec]

    # n >= 3: any further vowel is almost certainly gibberish.
    # Strong consonant push + strong vowel penalty.
    out = [0.0] * VOCAB_SIZE
    for ch in _CONS_LOWER:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            out[idx] += 1.35 * scale
    for v in _VOWEL_CHARS:
        idx = VOCAB_INDEX.get(v)
        if idx is not None:
            out[idx] += -1.40 * scale
    return out
