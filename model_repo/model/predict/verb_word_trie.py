"""Verb-focused word-trie — mid-word bias toward verb completions.

A structural move. The existing verb-overdue bias (in compose.py) is
applied ONLY at word-start: when `clause_slot == HAS_SUBJ` and
`words_since_verb >= 1`, it nudges the first letter toward verb-
starter letters (h, a, i, w, d, s, c, m, b, ...). Once the first
letter has been emitted, that bias is silent — and mid-word letter
choice falls back to the general word-trie / n-gram priors, which
don't know that a VERB is syntactically expected.

This layer closes that gap. It holds a trie of common Shakespearean
verb / auxiliary / modal forms (hand-curated from prior knowledge of
English and Early Modern English — no corpus counting). When the
clause is waiting for a verb AND the current word_buffer is a prefix
of some verb in the trie, it:

  (a) boosts letters that continue on the verb-trie, so "h" → "a"
      toward "hath" / "have", "wi" → "l" toward "will", "sha" →
      "l" toward "shall", etc.
  (b) when the buffer IS a complete verb in our list, boosts
      terminator characters (space, comma, period) so the verb
      closes rather than drifting ("hath" → " " instead of
      "hath" → "s" for "haths" / "hatha").

Gate:
  - speaker_label_state == 0 (not inside a NAME: label)
  - clause_slot == SLOT_HAS_SUBJ (1)
  - words_since_verb >= 1 (a verb is overdue)
  - letter_run_len >= 1 (we're mid-word)
  - word_buffer is a prefix of some verb in VERB_WORDS

Scale grows with words_since_verb — the longer we've waited for a
verb, the stronger the pull.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# --- Verb / auxiliary / modal inventory (lowercase, hand-curated) ---
#
# Organized by family for readability. Includes Early-Modern-English
# forms: -eth, -est, -st, -(e)dst. NOT a corpus count — these are
# forms I know Shakespeare uses frequently.
VERB_WORDS: frozenset[str] = frozenset({
    # be-verb
    "am", "art", "is", "are", "was", "wast", "were", "wert",
    "be", "been", "being",
    # have-verb
    "have", "has", "had", "hath", "hast", "hadst", "having",
    # do-verb
    "do", "does", "did", "done", "doing", "doth", "dost", "didst",
    # modals (incl. EME)
    "will", "wilt", "would", "wouldst", "wouldest",
    "shall", "shalt", "should", "shouldst", "shouldest",
    "can", "canst", "could", "couldst",
    "may", "mayst", "mayest", "might", "mightst", "mightest",
    "must", "mustst",
    "ought", "oughtest",
    "need", "needst",
    "dare", "darest", "durst",
    # go
    "go", "goes", "went", "gone", "going",
    "goest", "goeth",
    # come
    "come", "comes", "came", "coming",
    "comest", "cometh",
    # see
    "see", "sees", "saw", "seen", "seeing",
    "seest", "seeth",
    # know
    "know", "knows", "knew", "known", "knowing",
    "knowest", "knoweth",
    # think
    "think", "thinks", "thought", "thinking",
    "thinkest", "thinketh",
    # speak
    "speak", "speaks", "spake", "spoke", "spoken", "speaking",
    "speakest", "speaketh",
    # say
    "say", "says", "said", "saying",
    "sayest", "saith", "sayeth",
    # tell
    "tell", "tells", "told", "telling",
    "tellest", "telleth",
    # hear
    "hear", "hears", "heard", "hearing",
    "hearest", "heareth",
    # make
    "make", "makes", "made", "making",
    "makest", "maketh",
    # take
    "take", "takes", "took", "taken", "taking",
    "takest", "taketh",
    # give
    "give", "gives", "gave", "given", "giving",
    "givest", "giveth",
    # find
    "find", "finds", "found", "finding",
    "findest", "findeth",
    # stand
    "stand", "stands", "stood", "standing",
    "standest", "standeth",
    # seem
    "seem", "seems", "seemed", "seeming",
    "seemest", "seemeth",
    # love
    "love", "loves", "loved", "loving",
    "lovest", "loveth",
    # live
    "live", "lives", "lived", "living",
    "livest", "liveth",
    # look
    "look", "looks", "looked", "looking",
    "lookest", "looketh",
    # feel
    "feel", "feels", "felt", "feeling",
    "feelest", "feeleth",
    # hold
    "hold", "holds", "held", "holding",
    "holdest", "holdeth",
    # keep
    "keep", "keeps", "kept", "keeping",
    "keepest", "keepeth",
    # let
    "let", "lets", "letting",
    "lettest", "letteth",
    # get
    "get", "gets", "got", "gotten", "getting",
    "gettest",
    # fall
    "fall", "falls", "fell", "fallen", "falling",
    "fallest", "falleth",
    # bear
    "bear", "bears", "bore", "borne", "born", "bearing",
    "bearest", "beareth",
    # bring
    "bring", "brings", "brought", "bringing",
    "bringest", "bringeth",
    # sit
    "sit", "sits", "sat", "sitting",
    "sittest", "sitteth",
    # lie
    "lie", "lies", "lay", "lain", "lying",
    # walk
    "walk", "walks", "walked", "walking",
    # run
    "run", "runs", "ran", "running",
    # turn
    "turn", "turns", "turned", "turning",
    # play
    "play", "plays", "played", "playing",
    # lose
    "lose", "loses", "lost", "losing",
    # win
    "win", "wins", "won", "winning",
    # die
    "die", "dies", "died", "dying",
    # kill
    "kill", "kills", "killed", "killing",
    # swear
    "swear", "swears", "swore", "sworn", "swearing",
    "swearest", "sweareth",
    # cry
    "cry", "cries", "cried", "crying",
    # hate
    "hate", "hates", "hated", "hating",
    # fear
    "fear", "fears", "feared", "fearing",
    "fearest", "feareth",
    # hope
    "hope", "hopes", "hoped", "hoping",
    # wish
    "wish", "wishes", "wished", "wishing",
    # leave
    "leave", "leaves", "left", "leaving",
    # stay
    "stay", "stays", "stayed", "staying",
    # flee
    "flee", "flees", "fled", "fleeing",
    # fly
    "fly", "flies", "flew", "flown", "flying",
    # lead
    "lead", "leads", "led", "leading",
    # seek
    "seek", "seeks", "sought", "seeking",
    "seekest", "seeketh",
    # weep
    "weep", "weeps", "wept", "weeping",
    "weepest", "weepeth",
    # laugh
    "laugh", "laughs", "laughed", "laughing",
    # sleep
    "sleep", "sleeps", "slept", "sleeping",
    "sleepest", "sleepeth",
    # wake
    "wake", "wakes", "woke", "waked", "waking",
    # break
    "break", "breaks", "broke", "broken", "breaking",
    # fight
    "fight", "fights", "fought", "fighting",
    # draw
    "draw", "draws", "drew", "drawn", "drawing",
    # rise
    "rise", "rises", "rose", "risen", "rising",
    # send
    "send", "sends", "sent", "sending",
    # bid
    "bid", "bids", "bade", "bidden", "bidding",
    "biddest", "biddeth",
    # grant
    "grant", "grants", "granted", "granting",
    # pray
    "pray", "prays", "prayed", "praying",
    "prayest", "prayeth", "prithee",
    # call
    "call", "calls", "called", "calling",
    "callest", "calleth",
    # cast
    "cast", "casts", "casting",
    # put
    "put", "puts", "putting",
    # set
    "set", "sets", "setting",
    "settest", "setteth",
    # show
    "show", "shows", "showed", "shown", "showing",
    # meet
    "meet", "meets", "met", "meeting",
    # speak-like: cry, shout
    "shout", "shouts", "shouted", "shouting",
    # serve
    "serve", "serves", "served", "serving",
    # mean
    "mean", "means", "meant", "meaning",
    # read
    "read", "reads", "reading",
    # write
    "write", "writes", "wrote", "written", "writing",
    # answer
    "answer", "answers", "answered", "answering",
    # ask
    "ask", "asks", "asked", "asking",
    # wait
    "wait", "waits", "waited", "waiting",
    # follow
    "follow", "follows", "followed", "following",
    # seize
    "seize", "seizes", "seized", "seizing",
    # strike
    "strike", "strikes", "struck", "stricken", "striking",
    # remain
    "remain", "remains", "remained", "remaining",
    # return
    "return", "returns", "returned", "returning",
    # believe
    "believe", "believes", "believed", "believing",
    # forget
    "forget", "forgets", "forgot", "forgotten", "forgetting",
    # remember
    "remember", "remembers", "remembered", "remembering",
    # forgive
    "forgive", "forgives", "forgave", "forgiven", "forgiving",
    # deserve
    "deserve", "deserves", "deserved", "deserving",
    # pardon
    "pardon", "pardons", "pardoned", "pardoning",
    # beg
    "beg", "begs", "begged", "begging",
    # swear-related
    "vow", "vows", "vowed", "vowing",
    # obey
    "obey", "obeys", "obeyed", "obeying",
    # command
    "command", "commands", "commanded", "commanding",
    # conquer
    "conquer", "conquers", "conquered", "conquering",
    # prove
    "prove", "proves", "proved", "proven", "proving",
    # promise
    "promise", "promises", "promised", "promising",
    # part
    "part", "parts", "parted", "parting",
    # kneel
    "kneel", "kneels", "knelt", "kneeled", "kneeling",
    # teach
    "teach", "teaches", "taught", "teaching",
    "teachest", "teacheth",
    # trust
    "trust", "trusts", "trusted", "trusting",
    "trustest", "trusteth",
    # fear-class: doubt
    "doubt", "doubts", "doubted", "doubting",
    # wonder
    "wonder", "wonders", "wondered", "wondering",
    # mourn
    "mourn", "mourns", "mourned", "mourning",
    # bleed
    "bleed", "bleeds", "bled", "bleeding",
    # tremble
    "tremble", "trembles", "trembled", "trembling",
    # hang
    "hang", "hangs", "hung", "hanged", "hanging",
})


def _build_verb_trie() -> dict:
    """Build a nested-dict trie over VERB_WORDS.

    Each node is a dict; a special key "$" marks the end of a word.
    """
    root: dict = {}
    for w in VERB_WORDS:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node["$"] = True
    return root


_VERB_TRIE = _build_verb_trie()


def _descend(buf: str) -> dict | None:
    """Descend the verb-trie by `buf` (lowercase). Return the node at
    which we land, or None if buf isn't a prefix of any verb word.
    """
    node = _VERB_TRIE
    for ch in buf:
        if ch not in node:
            return None
        node = node[ch]
    return node


def verb_word_trie_bias(
    word_buffer: str,
    letter_run_len: int,
    clause_slot: int,
    words_since_verb: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a bias vector pushing toward verb-word completions.

    Active only when:
      - speaker_label_state == 0
      - clause_slot == 1 (HAS_SUBJ)
      - words_since_verb >= 1 (verb overdue)
      - letter_run_len >= 1 AND word_buffer is a prefix of some verb
    """
    if speaker_label_state != 0:
        return None
    if clause_slot != 1:  # HAS_SUBJ
        return None
    if words_since_verb < 1:
        return None
    if letter_run_len < 1:
        return None
    if not word_buffer:
        return None

    # Case: uppercase start is fine ("Hath" / "Have") — match lower.
    buf = word_buffer.lower()
    # Only letters in the buffer mean anything here.
    if not buf.isalpha():
        return None

    node = _descend(buf)
    if node is None:
        return None

    # Overdue-scale: grow with words_since_verb, capped.
    wsv = min(words_since_verb, 4)
    # Modest absolute scale — we're augmenting a word-start bias and
    # riding alongside the general word_trie. Too strong would shove
    # probability onto "hath" when the corpus has a legitimate
    # non-verb "h..." word.
    base = 0.30 + 0.25 * wsv  # wsv=1: 0.55, wsv=4: 1.30

    vec = [0.0] * VOCAB_SIZE

    # (a) Boost letters that continue on the verb-trie. Weight each
    # continuation by whether it leads to an end-of-word within a
    # few letters (prefers short verb completions like "am"/"is" over
    # long ones).
    any_bias = False
    for ch, child in node.items():
        if ch == "$":
            continue
        if not isinstance(child, dict):
            continue
        # Measure how close this branch is to a verb-end (1..4 letters).
        proximity = _nearest_end(child, depth=1, limit=4)
        if proximity is None:
            # No end within the limit — still boost a little.
            lean = 0.35
        else:
            # proximity 1 (very close verb-end) → strong; 4 → modest.
            lean = 1.0 - 0.15 * (proximity - 1)
            lean = max(lean, 0.40)
        idx_lo = VOCAB_INDEX.get(ch)
        if idx_lo is not None:
            vec[idx_lo] += base * lean
            any_bias = True
        # Capital is rarely emitted mid-word, skip.

    # (b) If buf itself is a complete verb, gently favor word-enders.
    # Only fire at letter_run_len >= 2 to avoid over-firing on single-
    # letter hypothetical completions.
    if node.get("$") and letter_run_len >= 2:
        term_scale = 0.35 + 0.20 * wsv  # wsv=1: 0.55; wsv=4: 1.15
        for ch, w in ((" ", 1.0), (",", 0.55), (".", 0.45),
                      (";", 0.30), ("\n", 0.35)):
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += term_scale * w
                any_bias = True

    if not any_bias:
        return None
    return vec


def _nearest_end(node: dict, depth: int, limit: int) -> int | None:
    """BFS-style: return the shortest additional-letter distance to a
    verb-end ("$") within this subtree, or None if beyond `limit`.
    """
    if depth > limit:
        return None
    if node.get("$"):
        return depth
    best: int | None = None
    for ch, child in node.items():
        if ch == "$":
            continue
        if not isinstance(child, dict):
            continue
        d = _nearest_end(child, depth + 1, limit)
        if d is not None and (best is None or d < best):
            best = d
    return best
