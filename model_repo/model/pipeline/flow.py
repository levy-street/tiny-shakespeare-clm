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
        elif is_letter:
            letters_off_trie = state.letters_off_trie + 1
        else:
            letters_off_trie = state.letters_off_trie
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

    return state.model_copy(
        update={
            "on_word_trie": on_trie,
            "line_length_bucket": line_length_bucket,
            "sent_distance_bucket": sent_distance_bucket,
            "after_function_word": after_function_word,
            "in_prose_line": in_prose_line,
            "letters_off_trie": letters_off_trie,
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
        }
    )
