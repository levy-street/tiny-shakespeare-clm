"""Proper-noun scene rolodex — tracks recurring capitalized content words.

Shakespeare is thick with repeated proper nouns: once "Rome" is
introduced, it reappears dozens of times in the same scene; likewise
"Coriolanus", "Volsces", "Antium", "Northumberland", "Ophelia". The
model already has a word-trie and a speaker-memory, but no general
memory of non-speaker proper nouns once seen.

This stage maintains two fields:

  current_word_started_cap  bool  — set True when the first char of
                                    the current letter-run was UPPER
                                    (outside speaker-label territory);
                                    reset at word completion.

  proper_nouns_seen        tuple[str,...]  — up to 10 lowercased words
                                    that appeared capitalized at a
                                    non-sentence-start position in
                                    recent history. Most-recent first.

Why "non-sentence-start": all sentence-initial words are capital
regardless of whether they're proper nouns. We only want to learn
from true proper-noun-style capitals (mid-sentence or after a
vocative/title). `sentence_start_pending` at the moment the first
letter emitted lets us filter.

Detection logic runs at two moments per token:

  1. When letter_run_len transitions 0 -> 1 with an UPPER first char,
     record `word_cap_mid_sentence` (a local derivation of whether
     this word was mid-sentence when it was capitalized).

  2. When `just_finished_word` fires, if current_word_started_cap is
     True AND the word was mid-sentence AND it's not a short common
     capitalized interjection (I, O, Ay, Nay, etc.), prepend the
     lowercased word to proper_nouns_seen (dedup keeping most-recent).

The rolodex is never hard-reset; it rolls naturally with 10 entries.
A scene-break reset would be cleaner but is hard to detect purely
from character stream.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB
from .linguistic import NEWLINE, SPACE, UPPER

# Lowercased forms of single-letter / short capitalized words that
# appear capitalized for REASONS OTHER than proper-noun status:
#   - "i" / "o" are always capital even mid-phrase
#   - "ay" / "nay" are interjections sometimes rendered cap at start
#     of a reply but not proper nouns
# We skip these so the rolodex stays focused on names / places.
_SKIP_LOWERED = frozenset({"i", "o", "ay", "nay", "oh", "ah", "la"})

_MAX_PN = 10


def update_proper_noun_memory(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]
    updates: dict = {}

    # --- 1. Update current_word_started_cap based on this token. ---
    # When we just emitted the FIRST letter of a new letter run, check
    # if it was UPPER. Outside speaker label only.
    started_cap = state.current_word_started_cap
    if state.speaker_label_state in (2, 3):
        # Inside a speaker label — don't use this machinery for labels.
        new_started_cap = False
    elif state.letter_run_len == 1 and state.last_char_class == UPPER:
        # First letter of a new letter-run just became UPPER. Decide
        # whether this is a MID-SENTENCE capital (a likely proper
        # noun) or a FORCED capital (line-initial, sentence-initial,
        # or in speaker-label residue). Only mid-sentence counts.
        #
        # The previous char (just before this letter) identifies the
        # context. We read `prev_char` (the char before last_char).
        pc = state.prev_char
        if pc == "\n":
            # Line-initial capital — in verse, every line starts cap
            # regardless of sentence boundary. NOT a proper-noun signal.
            new_started_cap = False
        elif pc == " ":
            # After a space: check if the space was a sentence-start
            # or just a mid-clause space. chars_since_sentence_end at
            # the time of the emitted cap tells us: if it's 2 (the
            # space counted 1, the cap counted 1), then period was
            # 2 chars back → sentence-start. Otherwise mid-sentence.
            if state.chars_since_sentence_end <= 2:
                new_started_cap = False
            else:
                new_started_cap = True
        else:
            # Other punctuation (quote-open, dash, etc.): treat as
            # mid-sentence.
            new_started_cap = True
    elif state.letter_run_len >= 1:
        # Continuing within a letter run: preserve.
        new_started_cap = started_cap
    else:
        # No letter run active (just a terminator): preserve for one
        # step so `just_finished_word` can still read it, then the
        # word-completion branch below resets.
        new_started_cap = started_cap

    # --- 2. At word completion, maybe add to rolodex. ---
    new_pn = state.proper_nouns_seen
    if state.just_finished_word:
        word = state.last_completed_word
        add = False
        if (
            started_cap
            and word
            and len(word) >= 2
            and word.lower() not in _SKIP_LOWERED
            and not state.sentence_start_pending
            # Not inside a speaker label context.
            and state.speaker_label_state == 0
            # Don't flood from the very first token of the corpus.
            and state.words_in_sentence >= 1
            # Exclude all-apostrophe / non-alpha words.
            and any(c.isalpha() for c in word)
        ):
            add = True
        if add:
            w = word.lower().strip("'")
            if w and len(w) >= 2 and w not in _SKIP_LOWERED:
                # Dedup: if already at head, no-op; else prepend.
                if not new_pn or new_pn[0] != w:
                    deduped = [w]
                    for existing in new_pn:
                        if existing != w and len(deduped) < _MAX_PN:
                            deduped.append(existing)
                    new_pn = tuple(deduped)
        # Reset the cap flag at word completion.
        new_started_cap = False

    if new_started_cap != state.current_word_started_cap:
        updates["current_word_started_cap"] = new_started_cap
    if new_pn is not state.proper_nouns_seen:
        updates["proper_nouns_seen"] = new_pn

    if not updates:
        return state
    return state.model_copy(update=updates)
