"""Tier 2 — illegal-bigram phonotactic accumulator.

Maintains `bad_bigram_count` — the number of letter pairs within the
current word-buffer that are phonotactically illegal in English.

Illegal bigrams are pairs that virtually never appear adjacent within
an English / Shakespearean word. They come from the phonotactic
constraints of the language: certain consonant-consonant sequences
never syllabify, whether within a syllable or across a boundary.

Each time a new letter is appended to the word_buffer, we test the
(previous-letter, incoming-letter) pair against `_ILLEGAL_BIGRAMS`
and bump the counter. The counter is reset on any non-letter char
(word boundary). Apostrophes are tolerated as continuers and do NOT
themselves seed a bigram (we treat them as invisible to the
phonotactic test — "can't" shouldn't flag "n't").

Speaker-label territory is skipped because proper names have looser
phonotactics (e.g., "PTOLEMY" starts with "pt", a legal Greek-origin
cluster English borrows). We reset-and-skip there.

No corpus statistics — the illegal set comes from well-known English
phonotactic constraints.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


_LETTERS: frozenset[str] = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_VOWELS: frozenset[str] = frozenset("aeiouyAEIOUY")
_CONSONANTS: frozenset[str] = frozenset("bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ")


def _lc(ch: str) -> str:
    return ch.lower() if ch.isupper() else ch


# Legal English 3-consonant clusters. When three consonants in a row
# occur in the buffer, the cluster must match one of these (anywhere
# in the sequence — onsets like "spr" at word-start, or codas like
# "nct" at word-end, or syllable-boundary clusters like "str" in
# "astronaut"). If not, it's a phonotactic violation.
#
# Compiled from well-known English onset/coda inventories.
_LEGAL_CCC: frozenset[str] = frozenset({
    # Onsets (word-initial 3-consonant clusters in English)
    "spr", "str", "scr", "spl", "skr", "skl",
    "shr", "thr",
    "sph", "sch",
    # Common codas / medials (mostly arising at syllable boundaries)
    "nth", "nst", "rst", "nct", "mpt", "pts", "cts",
    "rld", "rth", "rts", "rds", "rns", "rks", "rms", "rps",
    "rls", "lth", "lds", "lts", "lks", "lms", "lps",
    "nds", "nts", "ngs", "nks", "ncs",
    "cks", "sks", "sps", "sts",
    "fts", "fth",
    "pth", "xth", "ghs", "gth",
    # Cross-syllable medials common in English
    "ngl", "nkl", "rpl", "rbl", "rkl", "rgl", "rsl", "rtl",
    "ndl", "ntl", "mpl", "nkl", "stl", "scl",
    "ndr", "ntr", "mpr", "mbr", "nkr", "ngr", "str",
    "ldr", "lgr", "lfr",
    # Preceded by 's': strl? no. Keep tight.
    "mbl", "ndl", "rdl", "rml", "rnl",
    # Less common but attested medials with 'h'
    "rch", "lch", "nch", "rsh", "lsh", "nsh",
    "rph", "lph", "nph",
    # Archaic Shakespeare forms with apostrophe-'d converted: here 'd
    # is a separate char so not a 3-cluster. OK.
})

# Pre-enumerated rare-but-legal 3-consonant clusters that occur in
# recognizable English words; add a few more to prevent false positives:
_LEGAL_CCC = _LEGAL_CCC | frozenset({
    "chr", "phr", "shr", "thr",  # christian, phrase, shrew, three
    "scl", "spl", "squ",           # ACT+ual → ct+u not CCC, skip
    "bst", "bts",                  # abst-/debts
    "cts",
    "dth", "bth", "gth",
    "wth",
    "lv", "rv",  # not CCC; ignore
    "xpl", "xpr", "xtr", "xcl",    # express, explain, extreme, exclaim
    "nthr", "mblr",  # monster-like (not CCC, skip)
    "ght", "lst",
    "rtch", "ntch", "stch", "ltch", "rdsh",
    # The above are 4-letter; they cover 3-letter subsequences too:
    # rtc, tch; nrt; ngt. Add:
    "rtc", "ntc", "stc",
    "rgh", "lgh", "ngh",  # burgh, sigh, aught
})


# Three-vowel runs are mostly illegal in English except for a handful
# like "eau" (beauteous), "iou" (adventurous), "uee" (payee), "eie"
# (marquee? no). Treat 3+ vowels as violation UNLESS in the allowed
# set.
_LEGAL_VVV: frozenset[str] = frozenset({
    "eau", "iou", "uee", "iau", "iai", "oue", "eue", "uie",
    "iei", "uea", "aia", "oui", "aie", "uae", "eia",
})


# English phonotactically illegal letter bigrams. These are pairs
# that virtually never appear adjacent inside a real English word,
# in any syllable position. Compiled from prior phonological
# knowledge; we deliberately keep this tight so it fires rarely on
# real words and consistently on gibberish.
#
# Categories (most illegal cases):
#   * q followed by non-u
#   * j followed by consonant
#   * x followed by most consonants (xt/xh allowed)
#   * z followed by problematic consonant
#   * v followed by consonant (except vr rare, vl rare)
#   * Certain consonants immediately followed by h (h is almost
#     always the LEADING letter of a digraph; the cases ch/gh/ph/
#     rh/sh/th/wh/zh are legal; others like bh/dh/fh/jh/kh/lh/mh/
#     nh/qh/vh/xh/yh are not)
#   * Certain rare cross-consonant pairs: tv, db, pd, bd, gf, …
_ILLEGAL_BIGRAMS: frozenset[str] = frozenset({
    # q + non-u
    "qb", "qc", "qd", "qe", "qf", "qg", "qh", "qi", "qj", "qk",
    "ql", "qm", "qn", "qo", "qp", "qr", "qs", "qt", "qv", "qw",
    "qx", "qy", "qz",
    # j + consonant (j is almost always followed by a vowel)
    "jb", "jc", "jd", "jf", "jg", "jh", "jk", "jl", "jm", "jn",
    "jp", "jq", "jr", "js", "jt", "jv", "jw", "jx", "jz",
    # x + problematic consonant (xt, xh legal; others not)
    "xb", "xc", "xd", "xf", "xg", "xj", "xk", "xl", "xm", "xn",
    "xq", "xr", "xs", "xv", "xw", "xz",
    # z + problematic consonant
    "zb", "zc", "zd", "zf", "zg", "zj", "zk", "zp", "zq", "zr",
    "zs", "zv", "zw", "zx",
    # v + consonant (vr rare but possible; vl rare; rest illegal)
    "vb", "vc", "vd", "vf", "vg", "vh", "vj", "vk", "vm", "vn",
    "vp", "vq", "vs", "vt", "vw", "vx", "vz",
    # Consonant + h (only ch/gh/ph/rh/sh/th/wh/zh are legal digraphs)
    "bh", "dh", "fh", "jh", "kh", "lh", "mh", "nh", "qh", "vh",
    "xh", "yh",
    # C + non-matching C (a selection of high-confidence illegal
    # cross-consonant pairs that appear in observed gibberish)
    "tv", "td", "tb", "tm", "tn", "tk",
    "dq", "db", "dp", "dk", "dg", "dm", "dn",
    "pb", "pd", "pk", "pg", "pm", "pn", "pj",
    "bk", "bd", "bf", "bg", "bj", "bm", "bn", "bp", "bt", "bv",
    "gd", "gf", "gj", "gk", "gm", "gn", "gp", "gt",
    "kb", "kd", "kf", "kg", "kj", "km", "kn", "kp", "kt", "kv",
    "fb", "fd", "fg", "fj", "fk", "fm", "fn", "fp", "fs", "ft",
    "mb",
    "ms", "mt",
    # ^ note: "mb" is actually allowed (dumb, lamb, climb); remove it
    # (keep mentally). Actually let's remove ambiguous ones.
    "cj", "cb", "cd", "cf", "cg", "ck",
    # ck is COMMON (luck, back). Remove.
    # Legal drops: keep conservative.
})

# Strip known-legal ones from the set.
_ILLEGAL_BIGRAMS = _ILLEGAL_BIGRAMS - frozenset({
    "mb",        # lamb, dumb
    "ck",        # luck, back
    "cb", "cd", "cf", "cg",  # rare but can occur in compounds / names
    "ms", "mt",  # isthmus, warmth ok; remove to be safe
    "qe", "qi", "qo",  # can appear in "burqa" stretches, rare proper names; drop
    "fs", "ft",  # ft (left, lift) is common
    "bt",        # subtle, debt
    "bp", "bm", "bn",  # rare but submarine etc
    "dm", "dn",  # admit, kidney
    "gm", "gn", "gs",  # paradigm, agnostic, dogs
    "km",        # rare ok
    "mb",
    "kn",        # knight, kneel
    "ph",        # wait we removed via digraph list... ph is LEGAL.
    # Drop any with 'h' second that are actually legal digraphs (we
    # didn't add them, but guard):
})


def update_phonotactic(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-label: don't account. Reset if anything is set.
    if state.speaker_label_state != 0:
        if state.bad_bigram_count != 0 or state.bad_trigram_count != 0:
            return state.model_copy(update={
                "bad_bigram_count": 0,
                "bad_trigram_count": 0,
            })
        return state

    # Word-boundary: non-letter (apostrophe is a continuer so it
    # doesn't boundary the word, but we also don't update bigrams on
    # apostrophes — handled below).
    if ch not in _LETTERS and ch != "'":
        if state.bad_bigram_count != 0 or state.bad_trigram_count != 0:
            return state.model_copy(update={
                "bad_bigram_count": 0,
                "bad_trigram_count": 0,
            })
        return state

    # Apostrophe: continuer but doesn't participate in bigram.
    if ch == "'":
        return state

    # Letter: evaluate the (prev-letter, this-letter) bigram.
    # word_buffer already includes the incoming char (update_linguistic
    # ran before us), so wb[-2] is the prior letter, wb[-1] is this one.
    wb = state.word_buffer
    if len(wb) < 2:
        return state
    prev_ch = wb[-2]
    cur_ch = wb[-1]
    if prev_ch == "'":
        # Skip bigrams that cross an apostrophe (e.g. "'t", "'s", "'d").
        return state

    pair = _lc(prev_ch) + _lc(cur_ch)
    updates: dict = {}
    if pair in _ILLEGAL_BIGRAMS:
        updates["bad_bigram_count"] = state.bad_bigram_count + 1

    # Trigram check: three consonants in a row whose cluster isn't
    # a legal English CCC pattern, or three vowels in a row whose
    # sequence isn't in the attested set.
    if len(wb) >= 3:
        a, b, c = _lc(wb[-3]), _lc(wb[-2]), _lc(wb[-1])
        if a in _CONSONANTS and b in _CONSONANTS and c in _CONSONANTS:
            tri = a + b + c
            if tri not in _LEGAL_CCC:
                updates["bad_trigram_count"] = state.bad_trigram_count + 1
        elif a in _VOWELS and b in _VOWELS and c in _VOWELS:
            tri = a + b + c
            if tri not in _LEGAL_VVV:
                updates["bad_trigram_count"] = state.bad_trigram_count + 1

    if updates:
        return state.model_copy(update=updates)
    return state
