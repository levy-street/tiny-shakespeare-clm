"""Predict layer — preemptive illegal-bigram penalty.

Complements `phonotactic_close_bias` which fires AFTER an illegal
bigram has already been committed to the buffer. This layer fires
BEFORE: given the last emitted letter, penalize next-letter choices
that would form an English-phonotactically-illegal bigram.

Example: after "et", "tv" is illegal — so penalize 'v' as a next
letter. In the observed samples this was the step that produced
"etvst": at the v-choice moment there was no explicit bias against
"tv" (bigram.py has no entry, so default 0). This layer fills that
gap with strong, targeted preemptive penalties drawn from the same
hand-compiled illegal-bigram set used by the reactive close-out.

Gates:
  * last_char must be an ASCII letter (lowercase; uppercase lowercased)
  * speaker_label_state == 0 (proper names tolerate rarer clusters)
  * word_buffer non-empty (we're mid-word)
  * letter_run_len >= 1 (default: active inside any word)

No corpus statistics — derived from prior knowledge of English
phonotactics (the illegal set from pipeline/phonotactic.py).
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Re-declare the illegal bigram set here to avoid a cross-package
# import at predict time. Kept in sync with pipeline/phonotactic.py's
# final _ILLEGAL_BIGRAMS set (after the legal-drops subtraction).
_ILLEGAL: frozenset[str] = frozenset({
    # q + non-u
    "qb", "qc", "qd", "qe", "qf", "qg", "qh", "qi", "qj", "qk",
    "ql", "qm", "qn", "qo", "qp", "qr", "qs", "qt", "qv", "qw",
    "qx", "qy", "qz",
    # j + consonant
    "jb", "jc", "jd", "jf", "jg", "jh", "jk", "jl", "jm", "jn",
    "jp", "jq", "jr", "js", "jt", "jv", "jw", "jx", "jz",
    # x + problematic consonant
    "xb", "xc", "xd", "xf", "xg", "xj", "xk", "xl", "xm", "xn",
    "xq", "xr", "xs", "xv", "xw", "xz",
    # z + problematic consonant
    "zb", "zc", "zd", "zf", "zg", "zj", "zk", "zp", "zq", "zr",
    "zs", "zv", "zw", "zx",
    # v + consonant
    "vb", "vc", "vd", "vf", "vg", "vh", "vj", "vk", "vm", "vn",
    "vp", "vq", "vs", "vt", "vw", "vx", "vz",
    # Consonant + h (not legal digraph)
    "bh", "dh", "fh", "jh", "kh", "lh", "mh", "nh", "qh", "vh",
    "xh", "yh",
    # Illegal doubled letters
    "hh", "jj", "kk", "qq", "uu", "vv", "ww", "xx", "yy", "zz",
    # High-confidence illegal cross-consonant pairs
    "tv", "td", "tb", "tm", "tn", "tk",
    "dq", "dp", "dk", "dg",
    "pb", "pd", "pk", "pg", "pm", "pn", "pj",
    "bk", "bd", "bf", "bg", "bj", "bp", "bt", "bv",
    "gd", "gf", "gj", "gk", "gp", "gt",
    "kb", "kd", "kf", "kg", "kj", "kp", "kt", "kv",
    "fb", "fd", "fg", "fj", "fk", "fm", "fn", "fp",
    "cj",
})


def _lc(ch: str) -> str:
    return ch.lower() if ch.isupper() else ch


# Precompute per-prev-letter vector of per-next-letter penalties.
# Index by ord(prev_letter_lc). For each next letter forming an
# illegal pair, add a strong negative log-bias.
_PRE_VECS: dict[str, list[float]] = {}
_PENALTY = -3.2  # strong but not infinite; other biases may outweigh legitimately


def _build() -> None:
    for prev in "abcdefghijklmnopqrstuvwxyz":
        vec = [0.0] * VOCAB_SIZE
        any_set = False
        for nxt in "abcdefghijklmnopqrstuvwxyz":
            if (prev + nxt) in _ILLEGAL:
                idx = VOCAB_INDEX.get(nxt)
                if idx is not None:
                    vec[idx] += _PENALTY
                    any_set = True
                # Also penalize the uppercase variant (mid-word
                # uppercase is already blocked, but defensive).
                idx_u = VOCAB_INDEX.get(nxt.upper())
                if idx_u is not None:
                    vec[idx_u] += _PENALTY
                    any_set = True
        if any_set:
            _PRE_VECS[prev] = vec


_build()


def illegal_bigram_preempt_bias(
    last_char: str,
    speaker_label_state: int,
    word_buffer: str,
    letter_run_len: int,
) -> list[float] | None:
    """Return preemptive penalty vector for next letter, or None.

    Fires only outside speaker-label territory, inside a word, when
    the previous char is a letter with known illegal continuations.
    """
    if speaker_label_state != 0:
        return None
    if not word_buffer:
        return None
    if letter_run_len < 1:
        return None
    if not last_char:
        return None
    prev = _lc(last_char)
    return _PRE_VECS.get(prev)
