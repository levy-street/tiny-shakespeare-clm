"""Tier 3 — flow / mood / register state updates.

Reads Tier 1 (base counters) and Tier 2 (linguistic) fields and the
incoming token, and updates *flow*-level fields that capture the
texture of the emerging text — whether the current word has gone
off the known-vocabulary trie, how long the current line has run, how
overdue the next sentence-ending mark is, and whether the last word
was a short closed-class function word.

These flow fields are later consumed by the predict layer, where they
modulate biases toward line-ending, sentence-ending, or content-word
continuations.
"""

from __future__ import annotations

from ..state import ModelState
from ..predict.word_trie import COMPLETE_WORDS, is_on_trie
from .pos import (
    POS_ADJECTIVE,
    POS_ADVERB,
    POS_AUX_VERB,
    POS_MODAL,
    POS_NOUN,
    POS_PROPER_NOUN,
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
)

# Short closed-class words whose follow-up is almost always a
# content word (noun / verb / adjective), not another function word.
_FUNCTION_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "my", "thy", "his", "her", "our", "your", "their",
    "this", "that", "these", "those",
    "of", "to", "in", "on", "with", "for", "by", "at", "as", "from",
    "into", "unto", "upon", "o'er", "'gainst",
    "and", "but", "or", "nor", "so", "yet", "for",
    "i", "thou", "he", "she", "we", "ye", "they", "you", "me", "thee",
    "him", "us", "them",
    "is", "are", "was", "were", "be", "been", "am", "art", "hath", "doth",
    "hast", "dost", "shall", "will", "would", "should", "could", "may",
    "might", "must", "do", "did", "does", "have", "has", "had",
    "not", "no", "nay", "yea",
    "if", "when", "where", "while", "though", "than", "then", "now",
    "here", "there",
})


def _line_length_bucket(chars_since_newline: int) -> int:
    if chars_since_newline < 20:
        return 0
    if chars_since_newline < 35:
        return 1
    if chars_since_newline < 50:
        return 2
    return 3


def _sent_distance_bucket(chars_since_sentence_end: int) -> int:
    if chars_since_sentence_end < 40:
        return 0
    if chars_since_sentence_end < 80:
        return 1
    return 2


_VOWELS_SET = frozenset("aeiouAEIOU")


def _count_syllables(word: str) -> int:
    """Count vowel groups in a word (proxy for syllables). Strips
    apostrophes, handles silent-final-e in multi-syllable words, and
    returns at least 1 for any letter-bearing input. Approximate but
    robust enough for the 1-vs-multi distinction."""
    if not word:
        return 0
    s = word.lower().replace("'", "")
    if not s:
        return 0
    count = 0
    prev_vowel = False
    for ch in s:
        is_v = ch in "aeiouy"
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    # Silent final 'e' (except -le which is syllabic): "love", "made",
    # "time" → 1 syllable not 2. Only strip when we have >= 2 counted
    # groups, to avoid reducing e.g. "be" to 0.
    if count > 1 and s.endswith("e") and not s.endswith("le"):
        count -= 1
    return max(1, count)


# Words whose appearance marks an archaic / early-modern register.
# Not a frequency list — a hand-picked set of lexical markers that
# unambiguously signal "this scene is in archaic mode". Each bumps
# archaic_density by _ARCHAIC_BUMP (or _STRONG_BUMP for strong markers).
_ARCHAIC_STRONG: frozenset[str] = frozenset({
    "thou", "thee", "thy", "thine", "hath", "doth", "hast", "dost",
    "wilt", "shalt", "art", "wert", "canst", "didst", "wouldst",
    "couldst", "shouldst", "mayst", "mightst",
    "prithee", "methinks", "forsooth", "wherefore", "whence",
    "hither", "thither", "whither", "anon", "alack", "ere",
    "marry", "sirrah", "zounds", "quoth", "mayhap",
    "'tis", "'twas", "'twere", "'gainst",
})
_ARCHAIC_MILD: frozenset[str] = frozenset({
    "nay", "yea", "ay", "fie", "oft", "mine",
    "o'er", "ne'er", "e'er", "e'en",
    "unto", "upon",
})
_ARCHAIC_STRONG_BUMP = 0.28
_ARCHAIC_MILD_BUMP = 0.10
_ARCHAIC_DECAY = 0.985  # per completed word

# Emotional-intensity markers.
_EMO_WORDS_STRONG: frozenset[str] = frozenset({
    "o", "oh", "alas", "alack", "fie", "ah", "ay", "woe",
    "ha", "hey", "ho", "aha", "out", "die", "dead", "slain",
    "murder", "treason", "villain", "monster", "traitor",
})
_EMO_WORDS_MILD: frozenset[str] = frozenset({
    "god", "heaven", "love", "death", "dear", "pray", "sweet",
    "heart", "tears", "poor", "cruel", "fair", "hark",
    "blood", "grave", "curse", "hate", "rage", "hell",
    "bitter", "fear", "grief", "horror", "fury", "devil",
})
# Modern-only markers that gently pull density down (explicitly
# *not* archaic — we saw a modern form that suggests the register
# is drifting toward modern). Small effect.
_MODERN_MARKERS: frozenset[str] = frozenset({
    "okay", "really",  # won't appear in Shakespeare; kept empty-ish
})

# --- Tonal texture: dark/heavy vs light/hopeful ---
# Word classes for the rolling tonal_weight field. Each completed
# word bumps the rolling float toward +1 (light) or -1 (dark) by
# the class-specific amount. The field decays toward 0 per word.
_TONAL_STRONG_DARK: frozenset[str] = frozenset({
    "death", "dead", "die", "dies", "died", "dying",
    "blood", "bloody", "murder", "slain", "slay", "slays",
    "grief", "griefs", "sorrow", "sorrows", "woe", "woes",
    "hell", "grave", "tomb", "coffin", "corpse", "corse",
    "tears", "weep", "weeping", "wept", "mourn", "mourning",
    "hate", "hatred", "rage", "wrath", "fury", "curse", "cursed",
    "fear", "fears", "dread", "horror", "horrors", "terror",
    "devil", "fiend", "monster", "traitor", "villain", "villains",
    "poison", "poisoned", "dagger", "sword", "wound", "wounds",
    "dark", "darkness", "night", "black", "bleak", "cold",
    "cruel", "cruelty", "bitter", "foul", "vile", "loathsome",
    "despair", "anguish", "pain", "suffering", "plague",
    "war", "battle", "slaughter", "tyranny", "tyrant",
    "ghost", "spectre", "damn", "damned", "sin", "sins",
    "betray", "betrayed", "betrayal",
})
_TONAL_MILD_DARK: frozenset[str] = frozenset({
    "pale", "dim", "sad", "heavy", "weary", "faint", "sick",
    "silent", "still", "hollow", "lost", "gone", "fall", "fallen",
    "broken", "break", "broke", "shadow", "shadows",
    "lonely", "alone", "cruel", "cold",
    "poor", "dead", "deaf", "blind", "mad",
    "forgot", "forget", "forgotten",
    "wild", "fierce", "harsh",
})
_TONAL_MILD_LIGHT: frozenset[str] = frozenset({
    "bright", "warm", "gentle", "soft", "smile", "smiling",
    "laugh", "laughter", "kind", "kindly", "kindness",
    "good", "fair", "peace", "peaceful", "quiet",
    "pure", "noble", "noblest", "gracious", "mercy",
    "calm", "glad", "sweet", "sweetly",
    "music", "song", "sing", "singing",
    "friend", "friends", "friendship",
})
_TONAL_STRONG_LIGHT: frozenset[str] = frozenset({
    "love", "loves", "loved", "loving", "beloved", "lover",
    "joy", "joys", "joyful", "bliss", "delight", "delights",
    "fairest", "sweetest", "beauty", "beauteous",
    "heart", "hearts",  # associated with affection here
    "grace", "graceful", "blessing", "blessed", "bless",
    "angel", "heaven", "heavens", "heavenly",
    "light", "lights", "dawn", "sun", "sunshine", "golden",
    "hope", "hopes", "hopeful",
    "peace", "mirth", "merry", "cheer", "cheerful",
})
_TONAL_STRONG_BUMP = 0.30
_TONAL_MILD_BUMP = 0.12
_TONAL_DECAY = 0.96

# --- Imagery density: concrete/sensory vs abstract ---
# Words that paint pictures: body parts, instruments, weather,
# nature objects, colors, textures, gestures. Distinct from tonal
# lexicon (which carries valence, not imagery).
_IMAGERY_STRONG: frozenset[str] = frozenset({
    # Body
    "eye", "eyes", "ear", "ears", "hand", "hands", "head", "face",
    "faces", "hair", "brow", "brows", "lip", "lips", "tongue",
    "cheek", "cheeks", "throat", "breast", "breasts", "bosom",
    "bosoms", "arm", "arms", "leg", "legs", "foot", "feet", "knee",
    "knees", "finger", "fingers", "skin", "flesh", "bone", "bones",
    "blood", "tears", "tear", "breath", "pulse", "heart",
    # Weapons / objects
    "sword", "swords", "dagger", "daggers", "blade", "blades",
    "knife", "knives", "spear", "spears", "arrow", "arrows",
    "bow", "bows", "crown", "crowns", "throne", "thrones", "ring",
    "rings", "chain", "chains", "mirror", "mirrors", "cup", "cups",
    "goblet", "goblets", "letter", "letters", "cloak", "cloaks",
    "robe", "robes", "gown", "gowns", "mask", "masks", "key",
    "keys", "coin", "coins",
    # Sky / nature concrete
    "moon", "sun", "star", "stars", "cloud", "clouds",
    "fire", "flame", "flames", "smoke", "ash", "ashes",
    "wind", "storm", "rain", "snow", "frost", "dew", "mist",
    "lightning", "thunder", "sea", "wave", "waves", "shore",
    "river", "stream", "tide", "flood",
    # Plants / earth
    "rose", "roses", "flower", "flowers", "leaf", "leaves", "tree",
    "trees", "bough", "boughs", "grass", "thorn", "thorns",
    "stone", "stones", "rock", "rocks", "earth", "dust", "mud",
    "gold", "silver", "pearl", "pearls", "gem", "gems",
    # Animals
    "horse", "horses", "dog", "dogs", "wolf", "wolves", "lion",
    "serpent", "snake", "snakes", "worm", "worms", "bird", "birds",
    "crow", "crows", "dove", "doves", "fly", "flies", "bee", "bees",
    # Light / color / shadow
    "shadow", "shadows", "light", "lights", "dark", "darkness",
    "gleam", "shine", "shines", "blush", "red", "white", "black",
    "pale", "bright",
    # Gesture / motion concrete
    "kiss", "kisses", "smile", "smiles", "touch", "touches",
    "blow", "blows", "wound", "wounds", "cut", "cuts", "stab",
    "stabs",
})
_IMAGERY_MILD: frozenset[str] = frozenset({
    # Abstract but still sense-adjacent / bodied
    "sound", "sounds", "voice", "voices", "music", "song", "songs",
    "scent", "smell", "taste", "silence", "noise",
    "warm", "cold", "soft", "hard", "sharp", "smooth", "rough",
    "sweet", "bitter", "sour",
    "green", "blue", "gray", "grey",
    "shape", "form", "forms", "line", "edge",
    "house", "houses", "door", "doors", "window", "windows",
    "gate", "gates", "wall", "walls", "tower", "towers", "bed",
    "beds", "field", "fields", "garden", "road", "roads", "path",
    "paths", "ship", "ships", "boat", "boats",
    "day", "night", "morning", "evening", "dawn", "dusk",
})
# Abstract nouns that pull imagery_density down (text is moving
# toward argument/abstraction rather than image).
_IMAGERY_ABSTRACT: frozenset[str] = frozenset({
    "thought", "thoughts", "matter", "matters", "case", "cases",
    "fact", "facts", "manner", "manners", "reason", "reasons",
    "cause", "causes", "sake", "sakes", "kind", "kinds", "sort",
    "sorts", "state", "thing", "things", "whit", "ought", "naught",
    "issue", "issues", "purpose", "purposes", "means", "mean",
    "argument", "arguments", "opinion", "opinions", "notion",
    "notions", "sense", "senses", "truth", "truths", "way", "ways",
    "word", "words", "name", "names", "speech", "speeches",
    "promise", "promises",
})
_IMAGERY_STRONG_BUMP = 0.28
_IMAGERY_MILD_BUMP = 0.10
_IMAGERY_ABSTRACT_BUMP = -0.10
_IMAGERY_DECAY = 0.95  # per completed word

# --- 2nd-person addressing register ---
# Shakespeare characters lock into one of two 2nd-person pronoun
# registers within a turn: thou-register (thou/thee/thy/thine/thyself
# and their archaic verb forms) vs. you-register (you/your/yours/
# yourself/ye). Crossing mid-speech is rare. We push addressing_register
# positive on thou-forms and negative on you-forms; decay per word;
# dampen on speaker-turn change (next speaker may inherit or flip).
_THOU_FORMS: frozenset[str] = frozenset({
    "thou", "thee", "thy", "thine", "thyself",
})
_YOU_FORMS: frozenset[str] = frozenset({
    "you", "your", "yours", "yourself", "ye",
})
_ADDR_BUMP = 1.1
_ADDR_DECAY = 0.92  # per completed word
_ADDR_MAX = 3.0

# --- Invocation mode (rhetorical / declamatory texture) ---
# Canonical sentence-opener invocation words. Detected via
# words_in_sentence == 1 (first completed word of sentence) and
# match against this set.
_INVOC_OPENERS_STRONG: frozenset[str] = frozenset({
    "o", "oh", "alas", "alack", "ah", "ay",
    "hark", "behold", "hail", "hear", "lo",
    "heavens", "gods",
})
# Rhetorical WH-openers signal declamatory voice (though weaker).
_INVOC_OPENERS_WH: frozenset[str] = frozenset({
    "what", "why", "whence", "wherefore", "how", "when",
})
# Vocative nouns that in invocation mode reinforce it.
_INVOC_VOCATIVES: frozenset[str] = frozenset({
    "lord", "lords", "god", "gods", "heaven", "heavens",
    "death", "fortune", "muse", "fate", "love", "nature",
    "time", "night", "day", "sun", "moon", "world", "soul",
})
_INVOC_STRONG_BUMP = 0.45
_INVOC_WH_BUMP = 0.18
_INVOC_VOCATIVE_BUMP = 0.10
_INVOC_EXCLAIM_BUMP = 0.25
_INVOC_DECAY = 0.92

# --- Sonority level (phonetic texture) ---
# Per-letter bumps categorized by phonetic class. Vowels and liquids
# push positive (melodic); hard stops and harsh consonants push negative
# (percussive).
_SONORITY_VOWEL = 0.035
_SONORITY_LIQUID = 0.020  # l m n r
_SONORITY_APPROX = 0.015  # w y
_SONORITY_FRIC_VL = 0.005  # f h s (voiceless fricatives)
_SONORITY_VOICED_C = -0.010  # v c
_SONORITY_HARD_STOP = -0.025  # k t p
_SONORITY_VOICED_STOP = -0.018  # g b d
_SONORITY_RARE_HARSH = -0.035  # j q x z
_SONORITY_DECAY_LETTER = 0.985
_SONORITY_DECAY_NONLETTER = 0.97

_LIQUIDS = frozenset("lmnrLMNR")
_APPROX = frozenset("wyWY")
_FRIC_VL = frozenset("fhsFHS")
_VOICED_C = frozenset("vcVC")
_HARD_STOP = frozenset("ktpKTP")
_VOICED_STOP = frozenset("gbdGBD")
_RARE_HARSH = frozenset("jqxzJQXZ")


def update_flow(state: ModelState, token_id: int) -> ModelState:
    # Linguistic updates have already run; use the post-update state.
    wb = state.word_buffer
    if not wb:
        # No partial word in progress.
        on_trie = True
    else:
        on_trie = is_on_trie(wb)

    line_length_bucket = _line_length_bucket(state.chars_since_newline)
    sent_distance_bucket = _sent_distance_bucket(state.chars_since_sentence_end)

    # Did we just complete a function word?
    after_function_word = (
        state.just_finished_word
        and state.last_completed_word in _FUNCTION_WORDS
    )

    # Heuristic: we're in a prose line if we've seen enough chars since
    # the last newline without hitting a colon-newline (speaker label)
    # boundary recently. A simple proxy: chars_since_newline > 55 suggests
    # prose (most verse lines are shorter). This is soft.
    in_prose_line = state.chars_since_newline > 55

    # --- Phonotactic tracking inside the current word ---
    # letters_off_trie: letters written since the buffer first went
    # off-trie. 0 while on-trie or when no word is in progress.
    # consonants_since_vowel: consecutive consonants since the last
    # vowel in the current word (resets at word end, at vowel).
    # vowels_in_word: number of vowels seen in the current word.
    if not wb:
        letters_off_trie = 0
        offtrie_depart_pos = 0
        consonants_since_vowel = 0
        vowels_in_word = 0
        vowels_since_consonant = 0
        has_seen_complete = False
        letters_past_complete = 0
    else:
        last_ch = state.last_char
        is_letter = len(last_ch) == 1 and (
            ("a" <= last_ch <= "z") or ("A" <= last_ch <= "Z")
        )
        if on_trie:
            letters_off_trie = 0
            offtrie_depart_pos = 0
        elif is_letter:
            letters_off_trie = state.letters_off_trie + 1
            # Record departure position on the letter that took us
            # off-trie (the first off-trie letter). Persist afterward.
            if state.letters_off_trie == 0:
                offtrie_depart_pos = state.letter_run_len
            else:
                offtrie_depart_pos = state.offtrie_depart_pos
        else:
            letters_off_trie = state.letters_off_trie
            offtrie_depart_pos = state.offtrie_depart_pos
        if is_letter:
            if last_ch in _VOWELS_SET:
                consonants_since_vowel = 0
                vowels_in_word = state.vowels_in_word + 1
                vowels_since_consonant = state.vowels_since_consonant + 1
            else:
                consonants_since_vowel = state.consonants_since_vowel + 1
                vowels_in_word = state.vowels_in_word
                vowels_since_consonant = 0
        else:
            consonants_since_vowel = state.consonants_since_vowel
            vowels_in_word = state.vowels_in_word
            vowels_since_consonant = state.vowels_since_consonant

        # Word-trie drift recovery tracking.
        # wb is the buffer including the just-added letter (or apostrophe).
        # If wb itself is a complete known word, we're AT a viable stop
        # right now — reset the drift counter and mark has_seen_complete.
        # Otherwise, if we've previously seen a complete form in this
        # word, increment the drift counter (only on real letters; non-
        # letters don't extend the word). If we've never seen a complete
        # form, the counter stays 0 and predict won't fire.
        wb_is_complete = wb in COMPLETE_WORDS
        if wb_is_complete:
            has_seen_complete = True
            letters_past_complete = 0
        else:
            has_seen_complete = state.has_seen_complete
            if has_seen_complete and is_letter:
                letters_past_complete = state.letters_past_complete + 1
            else:
                letters_past_complete = state.letters_past_complete

    # Verse-mode rolling score. Update only when a line has just
    # completed (newline emitted that terminated a non-blank line).
    # We read from the updated `prev_line_length` (the just-finished
    # line's length). The score decays toward 0 on blank lines so that
    # speaker-label blanks don't force any particular mode.
    verse_score = state.verse_score
    if state.last_char == "\n":
        ln = state.prev_line_length
        if 1 < ln < 60:  # verse-shaped line
            delta = 0.7 if 20 <= ln <= 52 else 0.3
            verse_score = min(3.0, verse_score + delta)
        elif ln >= 70:  # prose-shaped line
            verse_score = max(-3.0, verse_score - 0.9)
        elif ln >= 60:
            verse_score = max(-3.0, verse_score - 0.4)
        else:
            # blank or very short: mild decay toward 0
            verse_score *= 0.9

    # Archaic register density: a rolling [0, 1] float.
    # On each completed word, decay + bump based on the word.
    archaic_density = state.archaic_density
    if state.just_finished_word:
        archaic_density *= _ARCHAIC_DECAY
        w = state.last_completed_word
        if w:
            if w in _ARCHAIC_STRONG:
                archaic_density = min(1.0, archaic_density + _ARCHAIC_STRONG_BUMP)
            elif w in _ARCHAIC_MILD:
                archaic_density = min(1.0, archaic_density + _ARCHAIC_MILD_BUMP)
    # Reset to 0 at start of a new speaker's dialogue (post-label
    # newline + double-newline would give us a fresh scene context).
    # Concretely, reset when we just emitted a blank line after a
    # label (consecutive_newlines == 2). This lets each speaker's
    # register develop fresh but preserves continuity within a speech.
    if state.consecutive_newlines >= 2 and state.last_char == "\n":
        # Preserve a fraction so scene-wide register isn't fully lost.
        archaic_density *= 0.6

    # Emotional intensity: bumped on "!", "?", and O-vocatives / emo
    # interjections; decayed per completed word.
    emo = state.emotional_intensity
    lc = state.last_char
    if lc == "!":
        emo = min(1.0, emo + 0.45)
    elif lc == "?":
        emo = min(1.0, emo + 0.30)
    if state.just_finished_word:
        emo *= 0.97  # per-word decay (slow: emotion lingers)
        w = state.last_completed_word
        if w in _EMO_WORDS_STRONG:
            emo = min(1.0, emo + 0.35)
        elif w in _EMO_WORDS_MILD:
            emo = min(1.0, emo + 0.18)
    # Fresh speaker: damp emotional carryover.
    if state.consecutive_newlines >= 2 and lc == "\n":
        emo *= 0.5

    # Tonal texture: rolling [-1, +1] capturing dark/heavy vs
    # light/hopeful mood of the recent lexicon. Decays toward 0
    # per completed word, then bumps by word class.
    tonal_weight = state.tonal_weight
    if state.just_finished_word:
        tonal_weight *= _TONAL_DECAY
        w = state.last_completed_word
        if w:
            if w in _TONAL_STRONG_DARK:
                tonal_weight = max(-1.0, tonal_weight - _TONAL_STRONG_BUMP)
            elif w in _TONAL_STRONG_LIGHT:
                tonal_weight = min(1.0, tonal_weight + _TONAL_STRONG_BUMP)
            elif w in _TONAL_MILD_DARK:
                tonal_weight = max(-1.0, tonal_weight - _TONAL_MILD_BUMP)
            elif w in _TONAL_MILD_LIGHT:
                tonal_weight = min(1.0, tonal_weight + _TONAL_MILD_BUMP)
    # Fresh speaker: damp carryover (scene tone is speaker-specific).
    if state.consecutive_newlines >= 2 and lc == "\n":
        tonal_weight *= 0.55

    # Imagery density: concrete/sensory vs abstract register.
    # Decays per completed word, then bumps by imagery class.
    imagery_density = state.imagery_density
    if state.just_finished_word:
        imagery_density *= _IMAGERY_DECAY
        w = state.last_completed_word
        if w:
            if w in _IMAGERY_STRONG:
                imagery_density = min(1.0, imagery_density + _IMAGERY_STRONG_BUMP)
            elif w in _IMAGERY_MILD:
                imagery_density = min(1.0, imagery_density + _IMAGERY_MILD_BUMP)
            elif w in _IMAGERY_ABSTRACT:
                imagery_density = max(0.0, imagery_density + _IMAGERY_ABSTRACT_BUMP)
    # Fresh speaker: damp carryover.
    if state.consecutive_newlines >= 2 and lc == "\n":
        imagery_density *= 0.55

    # Addressing register: push positive on thou-forms, negative on
    # you-forms. Decay per word; dampen across speaker-turn boundary.
    addressing_register = state.addressing_register
    if state.just_finished_word:
        addressing_register *= _ADDR_DECAY
        w = state.last_completed_word
        if w in _THOU_FORMS:
            addressing_register = min(
                _ADDR_MAX, addressing_register + _ADDR_BUMP
            )
        elif w in _YOU_FORMS:
            addressing_register = max(
                -_ADDR_MAX, addressing_register - _ADDR_BUMP
            )
    if state.consecutive_newlines >= 2 and lc == "\n":
        addressing_register *= 0.35

    # Cadence: staccato (-) ↔ flowing (+). Bumped at word completion
    # by the completed word's length, and at clausal punctuation.
    # Captures the *texture* of delivery: tight short bursts vs.
    # sweeping long phrases. Distinct from chars_since_comma (which
    # measures distance); cadence tracks the *feel* of the recent text.
    cadence = state.cadence
    if state.just_finished_word:
        cadence *= 0.95  # decay toward neutral each word
        w = state.last_completed_word
        if w:
            L = len(w)
            if L <= 3:
                # Short word: staccato pull. Exclude single-letter
                # "words" which are often noise or bare "I"/"O".
                if L >= 2:
                    cadence = max(-1.0, cadence - 0.08)
            elif L >= 10:
                cadence = min(1.0, cadence + 0.20)  # +0.12 + 0.08
            elif L >= 7:
                cadence = min(1.0, cadence + 0.12)
    # Clausal punctuation bump: commas / semicolons are the defining
    # marker of staccato rhythm. Applies on the exact token.
    if lc == "," or lc == ";":
        cadence = max(-1.0, cadence - 0.14)
    # Sentence-end: gentle decay toward neutral — the clause closed.
    if lc in ".!?":
        cadence *= 0.75
    # Fresh speaker: damp carryover (each speaker has their own cadence).
    if state.consecutive_newlines >= 2 and lc == "\n":
        cadence *= 0.4

    # Sonority level: rolling [-1, +1] phonetic texture. Updated on
    # every character emission. Decays faster on non-letters.
    sonority_level = state.sonority_level
    if len(lc) == 1:
        if ("a" <= lc <= "z") or ("A" <= lc <= "Z"):
            sonority_level *= _SONORITY_DECAY_LETTER
            if lc in _VOWELS_SET:
                sonority_level += _SONORITY_VOWEL
            elif lc in _LIQUIDS:
                sonority_level += _SONORITY_LIQUID
            elif lc in _APPROX:
                sonority_level += _SONORITY_APPROX
            elif lc in _FRIC_VL:
                sonority_level += _SONORITY_FRIC_VL
            elif lc in _VOICED_C:
                sonority_level += _SONORITY_VOICED_C
            elif lc in _HARD_STOP:
                sonority_level += _SONORITY_HARD_STOP
            elif lc in _VOICED_STOP:
                sonority_level += _SONORITY_VOICED_STOP
            elif lc in _RARE_HARSH:
                sonority_level += _SONORITY_RARE_HARSH
            # Clamp.
            if sonority_level > 1.0:
                sonority_level = 1.0
            elif sonority_level < -1.0:
                sonority_level = -1.0
        else:
            sonority_level *= _SONORITY_DECAY_NONLETTER
    if state.consecutive_newlines >= 2 and lc == "\n":
        sonority_level *= 0.30

    # Invocation mode: rolling [0, 1] tracking rhetorical/declamatory
    # voice. Bumped by invocation openers at sentence start, WH
    # rhetorical openers, "!" at sentence end, and vocatives when
    # mode is already warm. Decays per completed word.
    invocation_mode = state.invocation_mode
    if state.just_finished_word:
        invocation_mode *= _INVOC_DECAY
        w = state.last_completed_word
        if w:
            # First word of the sentence? words_in_sentence == 1 means
            # this was the opener.
            if state.words_in_sentence == 1:
                if w in _INVOC_OPENERS_STRONG:
                    invocation_mode = min(1.0, invocation_mode + _INVOC_STRONG_BUMP)
                elif w in _INVOC_OPENERS_WH:
                    invocation_mode = min(1.0, invocation_mode + _INVOC_WH_BUMP)
            # Vocatives reinforce when already in mode.
            if invocation_mode > 0.2 and w in _INVOC_VOCATIVES:
                invocation_mode = min(1.0, invocation_mode + _INVOC_VOCATIVE_BUMP)
    # "!" at sentence end bumps for the *next* sentence.
    if lc == "!":
        invocation_mode = min(1.0, invocation_mode + _INVOC_EXCLAIM_BUMP)
    elif lc in ".?":
        invocation_mode *= 0.92  # mild attenuation
    # Speaker turn: new speaker, damp carryover.
    if state.consecutive_newlines >= 2 and lc == "\n":
        invocation_mode *= 0.25

    # Ornament density: rolling [0, 1] tracking adjective/adverb
    # stacking richness vs. spare action-verb-driven diction.
    ornament_density = state.ornament_density
    if state.just_finished_word:
        ornament_density *= 0.96  # per-word decay
        pos = state.last_word_pos
        if pos == POS_ADJECTIVE:
            ornament_density = min(1.0, ornament_density + 0.18)
        elif pos == POS_ADVERB:
            ornament_density = min(1.0, ornament_density + 0.08)
        elif pos in (POS_NOUN, POS_PROPER_NOUN):
            ornament_density = max(0.0, ornament_density - 0.10)
        elif pos in (POS_VERB, POS_AUX_VERB, POS_MODAL,
                     POS_VERB_ING, POS_VERB_ED):
            ornament_density = max(0.0, ornament_density - 0.06)
    # Sentence-end: partial reset.
    if lc in ".!?":
        ornament_density *= 0.85
    # Speaker turn: dampen carryover.
    if state.consecutive_newlines >= 2 and lc == "\n":
        ornament_density *= 0.4

    # --- monosyllabic_run ---
    # Increment on completing a 1-syllable word; reset otherwise.
    # Reset on sentence-end and speaker-turn boundary.
    monosyllabic_run = state.monosyllabic_run
    if state.just_finished_word:
        w = state.last_completed_word
        if w:
            syl = _count_syllables(w)
            if syl == 1:
                monosyllabic_run = min(12, monosyllabic_run + 1)
            else:
                monosyllabic_run = 0
    if lc in ".!?":
        monosyllabic_run = 0
    if state.consecutive_newlines >= 2 and lc == "\n":
        monosyllabic_run = 0

    return state.model_copy(
        update={
            "on_word_trie": on_trie,
            "line_length_bucket": line_length_bucket,
            "sent_distance_bucket": sent_distance_bucket,
            "after_function_word": after_function_word,
            "in_prose_line": in_prose_line,
            "letters_off_trie": letters_off_trie,
            "offtrie_depart_pos": offtrie_depart_pos,
            "has_seen_complete": has_seen_complete,
            "letters_past_complete": letters_past_complete,
            "consonants_since_vowel": consonants_since_vowel,
            "vowels_in_word": vowels_in_word,
            "vowels_since_consonant": vowels_since_consonant,
            "verse_score": verse_score,
            "archaic_density": archaic_density,
            "emotional_intensity": emo,
            "tonal_weight": tonal_weight,
            "imagery_density": imagery_density,
            "addressing_register": addressing_register,
            "cadence": cadence,
            "ornament_density": ornament_density,
            "invocation_mode": invocation_mode,
            "sonority_level": sonority_level,
            "monosyllabic_run": monosyllabic_run,
        }
    )
