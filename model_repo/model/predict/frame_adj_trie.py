"""Predict layer — mid-word adjective bias when the syntactic frame
projects FRAME_ADJ_OR_NOUN.

Existing role-specific word-tries cover:
  - subject_word_trie  → clause_slot FRESH (subject opener)
  - verb_word_trie     → clause_slot HAS_SUBJ + verb overdue
  - object_word_trie   → clause_slot HAS_VERB (object expected)
  - post_obj_word_trie → clause_slot POST_OBJ (conjunctive continuation)

Adjectives — critically the "fair / gentle / noble / sweet / brave"
class that opens DETERMINER-ADJECTIVE-NOUN patterns — are NOT covered
by any of these. They land inside HAS_SUBJ or HAS_VERB slots where the
role-specific trie is either "verb" or "object" (which biases pronoun
/ det vocab, not adjectives).

Gate:
  - speaker_label_state == 0
  - expected_next_role == FRAME_ADJ_OR_NOUN (after DET / POSS / PREP+DET)
  - frame_confidence >= 0.4
  - letter_run_len >= 1 (mid-word only — first-letter handled by
    syntactic_frame_start_bias)
  - word_buffer is a prefix of some adjective in the list

This layer is POSITIVE-ONLY on continuations matching the adjective
vocabulary; when the buffer already IS a complete adjective, it also
boosts word-terminators so the adjective closes cleanly.

No corpus statistics — the adjective list is hand-compiled from prior
knowledge of Shakespearean / Early Modern English vocabulary.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Frame enum (mirrored from pipeline/syntactic_frame.py).
FRAME_ADJ_OR_NOUN = 2


# --- Adjective inventory (hand-curated) --------------------------------
_ADJ_WORDS: tuple[str, ...] = (
    # Physical / perceptual
    "fair", "foul", "pure", "vile", "base", "rich", "poor", "pale",
    "dark", "bright", "light", "heavy", "high", "low", "deep",
    "shallow", "tall", "short", "long", "brief", "broad", "narrow",
    "thick", "thin", "sharp", "blunt", "soft", "hard", "smooth",
    "rough", "clean", "foul", "fresh", "stale", "warm", "cold",
    "hot", "cool",
    # Moral / characterological
    "good", "great", "gentle", "grave", "gracious", "noble", "royal",
    "loyal", "faithful", "false", "true", "honest", "just", "unjust",
    "kind", "cruel", "mild", "wild", "fierce", "tame", "wise",
    "foolish", "mad", "sober", "sane",
    "brave", "bold", "meek", "coward", "cowardly", "valiant",
    "worthy", "unworthy", "virtuous", "wicked", "holy", "sacred",
    "cursed", "blessed", "damned", "heavenly", "hellish", "devilish",
    "human", "inhuman", "humane", "beastly", "bestial",
    # Emotive / state
    "sweet", "sour", "bitter", "tender", "harsh", "rude", "blunt",
    "proud", "humble", "sad", "happy", "merry", "glad", "sorry",
    "sorrowful", "grievous", "mournful", "joyful", "joyous", "jolly",
    "weary", "tired", "sick", "ill", "well", "healthy",
    "heartsick", "heartbroken", "distracted", "disturbed",
    "silent", "quiet", "loud", "still", "mute", "dumb",
    "calm", "fierce", "gentle", "furious", "wrathful", "angry",
    "mad", "raging", "peaceful",
    # Age / time
    "old", "new", "young", "aged", "ancient", "fresh", "modern",
    "former", "latter", "present", "past", "future",
    # Light / weight / force
    "dim", "bright", "dark", "shining", "glittering", "sparkling",
    "heavy", "light", "weighty", "strong", "weak", "mighty", "feeble",
    "tough", "frail", "firm", "fragile", "hardy",
    # Shakespearean favorites
    "vain", "idle", "proud", "arrogant", "pretty", "beauteous",
    "beautiful", "lovely", "handsome", "comely", "ugly", "hideous",
    "monstrous", "wondrous", "marvellous", "marvelous", "glorious",
    "famous", "infamous", "renowned",
    "swift", "slow", "quick", "sudden", "nimble", "lazy",
    "bold", "timid", "nervous",
    "grateful", "thankful", "ungrateful",
    "rich", "wealthy", "opulent", "barren", "fertile",
    "green", "white", "black", "red", "golden", "silver",
    "bloody", "bloodless", "stained", "pure", "chaste", "unchaste",
    "modest", "shameless",
    "secret", "hidden", "open", "plain",
    "simple", "complex", "tangled", "knotty",
    "living", "dying", "dead", "slain", "fallen", "lost", "found",
    "broken", "torn", "bruised", "wounded", "hurt", "healed",
    "certain", "uncertain", "sure", "unsure", "doubtful",
    "strange", "familiar", "common", "rare", "precious",
    "earthly", "worldly", "heavenly", "mortal", "immortal",
    "fatal", "mortal", "deadly", "lethal",
    "unhappy", "unlucky", "lucky", "fortunate", "unfortunate",
    "blessed", "cursed",
    "idle", "busy", "diligent",
    "full", "empty", "void", "replete",
    "free", "bound", "captive", "imprisoned",
    "naked", "bare", "clothed", "armed", "unarmed",
    "single", "double", "sole", "only", "alone",
    "strange", "odd", "queer",
    "chief", "principal", "prime", "first", "last", "former",
    "next", "due",
    "sovereign", "supreme", "paramount",
    # -ly adverbs masquerading as adjectives rarely; skip adverb list.
)


def _build_trie() -> dict:
    root: dict = {}
    for w in _ADJ_WORDS:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node["$"] = True  # terminal marker
    return root


_TRIE: dict = _build_trie()


def _descend(buf: str) -> dict | None:
    node = _TRIE
    for ch in buf:
        nxt = node.get(ch)
        if nxt is None:
            return None
        node = nxt
    return node


def frame_adj_midword_bias(
    expected_next_role: int,
    frame_confidence: float,
    word_buffer: str,
    letter_run_len: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if expected_next_role != FRAME_ADJ_OR_NOUN:
        return None
    if frame_confidence < 0.4:
        return None
    if letter_run_len < 1:
        return None
    if not word_buffer:
        return None

    # Only match on the tail of word_buffer equal to letters emitted
    # since the last word-boundary — which is exactly letter_run_len
    # chars.
    tail = word_buffer[-letter_run_len:].lower()
    node = _descend(tail)
    if node is None:
        return None

    # Collect next-char options and whether buffer is a complete adj.
    is_terminal = node.get("$", False)
    next_chars = [ch for ch in node.keys() if ch != "$"]

    if not next_chars and not is_terminal:
        return None

    # Scale grows with confidence. Modest ceiling — stacks with general
    # word_trie + syntactic_frame first-letter.
    scale = 1.0 * frame_confidence
    if scale <= 0.0:
        return None

    vec = [0.0] * VOCAB_SIZE

    # Push next-letter continuations.
    # Weight per letter: if the letter leads to a completion soon
    # (within 2-3 chars), weight higher. Simpler: equal weight across
    # next_chars with a modest magnitude.
    if next_chars:
        per = scale * 0.60
        for ch in next_chars:
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += per

    # If the buffer IS a complete adjective, push word-terminators.
    # We only push if letter_run_len >= 3 (avoid closing "a", "ox"
    # accidentally at run 1-2).
    if is_terminal and letter_run_len >= 3:
        term_scale = scale * 1.10
        for ch, w in (
            (" ", 1.0),
            (",", 0.35),
            (";", 0.20),
            (":", 0.15),
            (".", 0.25),
            ("\n", 0.20),
        ):
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += w * term_scale

    return vec
