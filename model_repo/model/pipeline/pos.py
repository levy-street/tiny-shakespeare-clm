"""POS-tagging pipeline stage.

Classifies `state.last_completed_word` into a coarse part-of-speech tag
using hand-specified closed-class word lists and morphological suffix
patterns. The tag is written to `state.last_word_pos` and the previous
tag is preserved in `state.prev_word_pos`, enabling downstream layers
to condition on a short POS context.

Closed-class words (pronouns, articles, prepositions, etc.) are
identified by explicit membership; open-class words (nouns, verbs,
adjectives, adverbs) are guessed from suffixes. The tagger is
deliberately approximate — its job is to give the predict layer a
useful prior over what kind of word comes next, not to be a formal
linguistic analyzer.
"""

from __future__ import annotations

from ..state import ModelState

# Tag enumeration.
POS_UNKNOWN = 0
POS_ARTICLE = 1        # the, a, an
POS_PRONOUN = 2        # I, me, my, thou, thee, thy, he, her, etc.
POS_POSSESSIVE = 3     # my, thy, his, her, our, your, their, mine, thine
POS_PREPOSITION = 4    # of, to, in, on, with, for, by, at, as, from, upon
POS_CONJUNCTION = 5    # and, but, or, nor, yet, so, if, though, when
POS_AUX_VERB = 6       # is, are, was, were, be, been, am, hath, doth
POS_MODAL = 7          # shall, will, would, should, could, may, might, must
POS_INTERJECTION = 8   # o, oh, ah, alas, hark, lo, fie
POS_NEGATION = 9       # not, no, nay, never, nor
POS_ADVERB = 10        # -ly words, now, then, here, there
POS_VERB_ING = 11      # -ing
POS_VERB_ED = 12       # -ed, -ied, -'d past tense
POS_NOUN = 13          # guessed: -tion, -ness, -ment, -er, -or, -ship
POS_ADJECTIVE = 14     # guessed: -ous, -ful, -less, -able, -ible, -ive
POS_PROPER_NOUN = 15   # starts with uppercase (we store lowercased, so
                       # detected differently; see below)
POS_VERB = 16          # plain verb (guessed)
POS_NUMBER = 17        # one, two, ... (also covers ordinals)
POS_WH = 18            # who, what, when, where, why, how, which

N_POS = 19

# Closed-class word membership.
_ARTICLES = frozenset({"the", "a", "an"})
_PRONOUNS = frozenset({
    "i", "me", "myself",
    "thou", "thee", "thyself",
    "he", "him", "himself",
    "she", "herself",
    "it", "itself",
    "we", "us", "ourselves",
    "ye", "you", "yourself", "yourselves",
    "they", "them", "themselves",
    "one", "ones",
})
_POSSESSIVES = frozenset({
    "my", "mine",
    "thy", "thine",
    "his", "hers",
    "her",  # note: also pronoun-object; classify as possessive by default
    "our", "ours",
    "your", "yours",
    "their", "theirs",
    "its",
})
_PREPOSITIONS = frozenset({
    "of", "to", "in", "on", "with", "for", "by", "at", "as", "from",
    "into", "unto", "upon", "onto", "till", "until", "through", "throughout",
    "against", "toward", "towards", "beneath", "beside", "besides",
    "between", "beyond", "below", "above", "within", "without", "behind",
    "amid", "amidst", "among", "amongst", "about", "after", "before",
    "across", "around", "along", "down", "off", "over", "under",
    "o'er", "'gainst", "'tween",
})
_CONJUNCTIONS = frozenset({
    "and", "but", "or", "nor", "yet", "so",
    "if", "though", "although", "because", "since", "when", "whenever",
    "while", "whilst", "where", "wherever", "than", "then", "whereas",
    "unless", "until", "ere", "lest", "as",
})
_AUX_VERBS = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "am", "art",
    "hath", "doth", "hast", "dost", "wert",
    "have", "has", "had", "having",
    "do", "does", "did", "done", "doing",
})
_MODALS = frozenset({
    "shall", "will", "would", "should", "could", "may", "might", "must",
    "can", "shalt", "wilt", "wouldst", "shouldst", "couldst", "mayst",
    "canst", "didst", "ought",
})
_INTERJECTIONS = frozenset({
    "o", "oh", "ah", "alas", "hark", "lo", "fie", "pshaw", "tush",
    "marry", "zounds", "nay", "yea", "aye", "ay",
})
_NEGATIONS = frozenset({
    "not", "no", "never", "none", "nothing", "nobody", "nowhere", "neither",
})
_WH = frozenset({
    "who", "whom", "whose", "what", "when", "where", "why", "how",
    "which", "whither", "whence", "whereof", "wherefore",
})
_NUMBERS = frozenset({
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth",
    "once", "twice", "thrice",
    "half", "both", "several",
})
_ADVERBS = frozenset({
    "now", "then", "here", "there", "hither", "thither", "whither",
    "hence", "thence", "whence", "ever", "never", "always", "often",
    "soon", "late", "early", "yet", "still", "already",
    "very", "too", "also", "just", "only", "even", "quite", "rather",
    "almost", "nearly", "perhaps", "maybe", "surely", "truly", "indeed",
    "forsooth", "verily", "methinks", "perchance", "peradventure",
    "anon", "away", "back", "forth", "on", "off", "out", "down", "up",
    "in", "so", "well", "ill",
})
# Common plain verbs (base forms and common inflections not captured by
# suffixes). Kept short — suffix patterns handle most of the long tail.
_VERBS = frozenset({
    "go", "goes", "gone", "went",
    "come", "comes", "came",
    "know", "knows", "knew", "known",
    "think", "thinks", "thought",
    "say", "says", "said",
    "see", "sees", "seen", "saw",
    "make", "makes", "made",
    "take", "takes", "took", "taken",
    "tell", "tells", "told",
    "give", "gives", "gave", "given",
    "find", "finds", "found",
    "leave", "leaves", "left",
    "bring", "brings", "brought",
    "hold", "holds", "held",
    "speak", "speaks", "spoke", "spoken",
    "stand", "stands", "stood",
    "hear", "hears", "heard",
    "keep", "keeps", "kept",
    "let", "lets",
    "set", "sets",
    "put", "puts",
    "cut", "cuts",
    "run", "runs", "ran",
    "sit", "sits", "sat",
    "eat", "eats", "ate", "eaten",
    "get", "gets", "got",
    "read", "reads",
    "seek", "seeks", "sought",
    "meet", "meets", "met",
    "write", "writes", "wrote", "written",
    "buy", "buys", "bought",
    "send", "sends", "sent",
    "spend", "spends", "spent",
    "lose", "loses", "lost",
    "win", "wins", "won",
    "lay", "lays", "laid",
    "die", "dies", "died",
    "lie", "lies", "lay",
    "kill", "kills",
    "love", "loves",
    "hate", "hates",
    "fear", "fears",
    "pray", "prays",
    "weep", "weeps", "wept",
    "sleep", "sleeps", "slept",
    "fly", "flies", "flew", "flown",
    "grow", "grows", "grew", "grown",
    "blow", "blows", "blew", "blown",
    "show", "shows", "shown",
    "throw", "throws", "threw", "thrown",
    "break", "breaks", "broke", "broken",
    "speak", "spoke", "spoken",
    "wake", "wakes", "woke",
    "rise", "rises", "rose", "risen",
    "fall", "falls", "fell", "fallen",
    "fight", "fights", "fought",
    "want", "wants",
    "need", "needs",
    "feel", "feels", "felt",
    "look", "looks",
    "seem", "seems",
    "turn", "turns",
    "call", "calls",
    "work", "works",
    "play", "plays",
    "move", "moves",
    "live", "lives",
    "seem", "seems",
    "stay", "stays",
    "try", "tries",
    "use", "uses",
    "ask", "asks",
    "wish", "wishes",
    "bear", "bears", "bore", "borne",
    "beat", "beats",
    "swear", "swears", "swore", "sworn",
    "wear", "wears", "wore", "worn",
    "tear", "tears", "tore", "torn",
    "art", "wilt", "shalt",  # archaic verb forms
})


def _classify_by_suffix(w: str) -> int:
    # Archaic past with "'d" — e.g. disbench'd.
    if w.endswith("'d") or w.endswith("ed") or w.endswith("d"):
        if w.endswith("ed") and len(w) > 3:
            return POS_VERB_ED
        if w.endswith("'d") and len(w) > 2:
            return POS_VERB_ED
    if w.endswith("ing") and len(w) > 4:
        return POS_VERB_ING
    if w.endswith("ly") and len(w) > 3:
        return POS_ADVERB
    # Noun-like suffixes.
    for suf in ("tion", "sion", "ness", "ment", "ship", "hood", "ance",
                "ence", "ity", "dom"):
        if w.endswith(suf) and len(w) > len(suf) + 1:
            return POS_NOUN
    # Adjective-like suffixes.
    for suf in ("ous", "ful", "less", "able", "ible", "ive", "ish",
                "al", "ic", "ial", "ous"):
        if w.endswith(suf) and len(w) > len(suf) + 1:
            return POS_ADJECTIVE
    # Agent/plural-y nouns.
    if w.endswith("er") and len(w) > 3:
        return POS_NOUN
    if w.endswith("or") and len(w) > 3:
        return POS_NOUN
    if w.endswith("ist") and len(w) > 3:
        return POS_NOUN
    # Plural noun (simple heuristic).
    if w.endswith("s") and len(w) > 3:
        # Could be verb 3rd person singular; leave ambiguous as NOUN.
        return POS_NOUN
    return POS_UNKNOWN


def classify(word: str) -> int:
    """Return POS tag for a lowercased completed word (may contain ')."""
    if not word:
        return POS_UNKNOWN
    w = word
    # Handle leading apostrophe ('tis, 'gainst, 'em) — strip for
    # lookup but keep the full word for suffix checks where relevant.
    core = w.lstrip("'")
    if core != w:
        # Archaic contracted forms.
        if core in {"tis"}:
            return POS_AUX_VERB  # 'tis = it is
        if core in {"em"}:
            return POS_PRONOUN
        if core in {"gainst", "neath", "twixt", "tween"}:
            return POS_PREPOSITION
    if w in _ARTICLES:
        return POS_ARTICLE
    if w in _POSSESSIVES:
        return POS_POSSESSIVE
    if w in _PRONOUNS:
        return POS_PRONOUN
    if w in _PREPOSITIONS:
        return POS_PREPOSITION
    if w in _CONJUNCTIONS:
        return POS_CONJUNCTION
    if w in _AUX_VERBS:
        return POS_AUX_VERB
    if w in _MODALS:
        return POS_MODAL
    if w in _INTERJECTIONS:
        return POS_INTERJECTION
    if w in _NEGATIONS:
        return POS_NEGATION
    if w in _WH:
        return POS_WH
    if w in _NUMBERS:
        return POS_NUMBER
    if w in _ADVERBS:
        return POS_ADVERB
    if w in _VERBS:
        return POS_VERB
    # Suffix-based guess.
    return _classify_by_suffix(w)


_CONTENT_TAGS = frozenset({
    POS_NOUN,
    POS_VERB,
    POS_VERB_ING,
    POS_VERB_ED,
    POS_ADJECTIVE,
    POS_ADVERB,
    POS_PROPER_NOUN,
    POS_UNKNOWN,  # unknown words are usually open-class content
})
_CONTENT_WORDS_CAP = 8

# POS tags that are *transparent* to recent_pos_backbone — they pass
# through without being pushed into the tuple. These are closed-class
# "glue" words that rarely change the content-level syntactic picture:
# function words, light verbs are NOT transparent (they matter); but
# interjections, conjunctions, negations, WH words, pronouns,
# articles, possessives are.
_BACKBONE_TRANSPARENT = frozenset({
    POS_INTERJECTION,
    POS_CONJUNCTION,
    POS_NEGATION,
    POS_WH,
    POS_ARTICLE,
    POS_POSSESSIVE,
    POS_PRONOUN,
})
_BACKBONE_CAP = 4

# Main-verb POS tags for verb-chain tracking. These are open-class
# verbs — chains of two in a row are ungrammatical. AUX/MODAL are
# transparent to this counter (legitimate aux+verb chains).
_MAIN_VERB_TAGS = frozenset({
    POS_VERB,
    POS_VERB_ING,
    POS_VERB_ED,
})
# Tags that are transparent to verb_chain_len (don't count, don't reset).
# Adverbs and negations sit between verbs ("Go not there", "runs swiftly").
_VERB_CHAIN_TRANSPARENT = frozenset({
    POS_ADVERB,
    POS_NEGATION,
    POS_AUX_VERB,
    POS_MODAL,
})


def update_pos(state: ModelState, token_id: int) -> ModelState:
    # Only recompute when a word just completed; otherwise hold steady.
    if not state.just_finished_word:
        return state
    word = state.last_completed_word
    if not word:
        return state
    new_tag = classify(word)

    # Maintain a rolling content-words tuple (most-recent first,
    # capped at CAP). Only words classified as content are kept.
    # Skip single-letter "words" (likely stray chars like "a"/"i"
    # from speaker-label residue or prefix drift).
    if new_tag in _CONTENT_TAGS and len(word) >= 2:
        # Avoid duplicate at the head so immediate repetition ("O, O")
        # doesn't flood the tuple.
        if state.content_words and state.content_words[0] == word:
            new_content = state.content_words
        else:
            new_content = (word,) + state.content_words
            if len(new_content) > _CONTENT_WORDS_CAP:
                new_content = new_content[:_CONTENT_WORDS_CAP]
    else:
        new_content = state.content_words

    # Update content-backbone POS tuple (filtered by transparent set).
    if new_tag in _BACKBONE_TRANSPARENT:
        new_backbone = state.recent_pos_backbone
    else:
        # Skip repeated identical heads to avoid noun-repetition flooding.
        if state.recent_pos_backbone and state.recent_pos_backbone[0] == new_tag:
            # Still rotate — same POS back-to-back is meaningful
            # (e.g. two nouns = compound/apposition).
            new_backbone = (new_tag,) + state.recent_pos_backbone
        else:
            new_backbone = (new_tag,) + state.recent_pos_backbone
        if len(new_backbone) > _BACKBONE_CAP:
            new_backbone = new_backbone[:_BACKBONE_CAP]

    # Update verb-chain length.
    if new_tag in _MAIN_VERB_TAGS:
        new_vcl = state.verb_chain_len + 1
        # Cap to prevent unbounded growth across transparent glue.
        if new_vcl > 5:
            new_vcl = 5
    elif new_tag in _VERB_CHAIN_TRANSPARENT:
        # Transparent — preserve current count.
        new_vcl = state.verb_chain_len
    else:
        # Non-verb content / function-word: reset.
        new_vcl = 0

    # Shift current → prev, prev → prev_prev.
    return state.model_copy(
        update={
            "prev_prev_word_pos": state.prev_word_pos,
            "prev_word_pos": state.last_word_pos,
            "last_word_pos": new_tag,
            "content_words": new_content,
            "recent_pos_backbone": new_backbone,
            "verb_chain_len": new_vcl,
        }
    )
