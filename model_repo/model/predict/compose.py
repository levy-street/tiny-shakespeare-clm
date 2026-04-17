"""Compose predict layers into the final distribution.

Order:
  1. Start from unigram log-probs (prior).
  2. Add context-class biases (from context.py).
  3. If last char is a letter, add letter-bigram biases (from bigram.py).
  4. If last char is a space or single newline (word start), add
     startword biases (from startword.py).
  5. Apply speaker-label FSM specific boosts.
  6. Log-softmax renormalize.
"""

from __future__ import annotations

import math

from ..pipeline.linguistic import (
    APOSTROPHE,
    LOWER_CONS,
    LOWER_VOWEL,
    NEWLINE,
    SPACE,
    UPPER,
)
from ..pipeline.pos import (
    _ARTICLES,
    _AUX_VERBS,
    _CONJUNCTIONS,
    _MODALS,
    _NEGATIONS,
    _POSSESSIVES,
    _PREPOSITIONS,
)
from ..state import ModelState
from ..vocab import VOCAB, VOCAB_INDEX, VOCAB_SIZE
from .bigram import bigram_bias
from .context import CTX_BIAS_VECTORS, context_key
from .letter3 import letter3_bias
from .letter4 import letter4_bias
from .next_word import next_word_bias
from .pos_next import pos_next_bias
from .speaker_trie import speaker_trie_bias
from .start4gram import start4gram_bias
from .start5gram import start5gram_bias
from .startbigram import startbigram_bias
from .starttrigram import starttrigram_bias
from .startword import START_BIAS
from .trigram import trigram_bias
from .unigram import UNIGRAM_LOGPROBS
from .word_trie import COMPLETE_WORDS, FORCE_END_BIAS, is_on_trie, word_trie_bias


def _log_softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    logz = m + math.log(z)
    return [x - logz for x in logits]


def predict(state: ModelState) -> list[float]:
    # Layer 1: unigram.
    logits = list(UNIGRAM_LOGPROBS)

    # Layer 2: context-class biases.
    ctx = context_key(state)
    ctx_bias = CTX_BIAS_VECTORS[ctx]
    for i in range(VOCAB_SIZE):
        logits[i] += ctx_bias[i]

    last_cls = state.last_char_class

    # Layer 3: letter-bigram biases (only inside letter runs).
    if state.last_char and last_cls in (UPPER, LOWER_VOWEL, LOWER_CONS):
        bi = bigram_bias(state.last_char)
        if bi is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += bi[i]

    # Layer 3b: trigram digraph biases (last two letters).
    if state.last_char and state.prev_char:
        tg = trigram_bias(state.prev_char, state.last_char)
        if tg is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += tg[i]

    # Layer 3b2: letter-trigram bias (last 3 letters → next).
    if state.word_buffer:
        l3 = letter3_bias(state.word_buffer)
        if l3 is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += l3[i]

    # Layer 3b3: word-start bigram bias — at letter_run_len == 1, the
    # second letter is heavily conditioned on the first (word-start
    # distributions differ from mid-word). Applies only when speaker
    # label is not constraining things.
    if (
        state.letter_run_len == 1
        and state.last_char
        and state.speaker_label_state not in (2,)
    ):
        sb = startbigram_bias(state.last_char)
        if sb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += sb[i]

    # Layer 3b4: word-start trigram bias — at letter_run_len == 2, the
    # third letter is conditioned on the first two letters of the fresh
    # word. Word-start 3-letter distributions differ substantially from
    # mid-word. Applies only outside speaker-label territory.
    if (
        state.letter_run_len == 2
        and len(state.word_buffer) == 2
        and state.speaker_label_state not in (2,)
    ):
        stt = starttrigram_bias(state.word_buffer)
        if stt is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += stt[i]

    # Layer 3b5: word-start 4-gram bias — at letter_run_len == 3, the
    # fourth letter is conditioned on the first three letters of the
    # fresh word.
    if (
        state.letter_run_len == 3
        and len(state.word_buffer) == 3
        and state.speaker_label_state not in (2,)
    ):
        s4 = start4gram_bias(state.word_buffer)
        if s4 is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += s4[i]

    # Layer 3b6: word-start 5-gram bias — at letter_run_len == 4, the
    # fifth letter is conditioned on the first four letters of the
    # fresh word. Many 4-letter word starts uniquely determine the next
    # letter (ther→e, woul→d, migh→t, ligh→t, ough→t, whic→h, frie→n).
    if (
        state.letter_run_len == 4
        and len(state.word_buffer) == 4
        and state.speaker_label_state not in (2,)
    ):
        s5 = start5gram_bias(state.word_buffer)
        if s5 is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += s5[i]

    # Layer 3c: word-trie completion bias.
    if state.word_buffer:
        wt = word_trie_bias(state.word_buffer)
        if wt is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += wt[i]

    # Layer 3d: speaker-label trie bias.
    if state.speaker_buffer:
        st = speaker_trie_bias(state.speaker_buffer)
        if st is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += st[i]

    # Layer 4: start-of-word bias (after space or single newline).
    if last_cls == SPACE or (
        last_cls == NEWLINE and state.consecutive_newlines == 1
    ):
        for i in range(VOCAB_SIZE):
            logits[i] += START_BIAS[i]

        # Layer 4b: next-word (word-bigram) first-letter bias.
        if state.last_completed_word:
            nw = next_word_bias(state.last_completed_word)
            if nw is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += nw[i]
            else:
                # Fallback: POS-based next-letter bias.
                pn = pos_next_bias(state.last_word_pos)
                if pn is not None:
                    for i in range(VOCAB_SIZE):
                        logits[i] += pn[i]

        # Layer 4c: at a sentence start (post ". ", post "? ", post "! "
        # or post a double-newline blank line), strongly boost capital
        # letters relative to lowercase. The training corpus always
        # starts new sentences with a capital (outside of mid-sentence
        # continuations).
        is_sentence_start = (
            state.prev_char_class == 6  # PUNCT_END — . ? !
            and last_cls == SPACE
        ) or (
            last_cls == NEWLINE and state.consecutive_newlines == 1
            and state.chars_since_sentence_end <= 2
        )
        if is_sentence_start:
            for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] += 1.2
            for ch in "abcdefghijklmnopqrstuvwxyz":
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] -= 0.5

        # Verse-line-start capital boost: after a *single* newline that
        # terminated a VERSE-length line (typically 15-55 chars),
        # Shakespeare almost always begins the next line with a capital
        # letter even when no sentence-ending punctuation is present.
        # Skip when already handled by is_sentence_start; skip when the
        # previous line was very short (blank/label) or very long (prose).
        on_verse_line_start = (
            last_cls == NEWLINE
            and state.consecutive_newlines == 1
            and not is_sentence_start
            and 1 <= state.prev_line_length <= 80
        )
        if on_verse_line_start:
            for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] += 3.0
            for ch in "abcdefghijklmnopqrstuvwxyz":
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] -= 1.2

        # Post-speaker-label newline: prev_line_length is small (the
        # speaker label itself) and the line ended with ":". Dialogue
        # starts here, almost always with a capital letter.
        on_post_label_start = (
            last_cls == NEWLINE
            and state.consecutive_newlines == 1
            and not is_sentence_start
            and not on_verse_line_start
            and 1 < state.prev_line_length < 15
            and state.prev_char_class == 7  # PUNCT_MID (the ":" of the label)
        )
        if on_post_label_start:
            for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] += 3.0
            for ch in "abcdefghijklmnopqrstuvwxyz":
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] -= 1.2
        # Additionally, at BOTH sentence-start and verse-line-start,
        # bias specific common starting capitals: T, A, W, I, O, B, H,
        # S, M, N, F, C, L, P, G, D, R, Y. Skip speaker-label context.
        if (is_sentence_start or on_verse_line_start) and state.speaker_label_state == 0:
            line_start_caps = {
                "T": 1.2,  # The, That, This, To, Thou, Then, There, Though, Tell
                "A": 1.0,  # And, A, As, At, All, Art, After, Above
                "W": 1.0,  # When, Who, What, With, We, Were, Will, Which
                "I": 0.8,  # I, In, Is, It, If
                "O": 0.7,  # O, Oh, Or, Of, On, Our
                "B": 0.8,  # But, By, Be, Before, Behold
                "H": 0.8,  # He, His, Her, Have, Hath, How, Here
                "S": 0.7,  # So, She, Shall, Should, Still
                "M": 0.7,  # My, Me, Must, Most, Must
                "N": 0.6,  # Now, No, Not, Never, Nay
                "F": 0.6,  # For, From, Father
                "C": 0.5,  # Come, Could
                "L": 0.4,  # Let, Look, Like, Love
                "P": 0.4,  # Pray, Peace, Poor
                "G": 0.4,  # Good, God, Go
                "D": 0.3,  # Do, Doth, Did, Dear
                "R": 0.3,  # Rome, Rich
                "Y": 0.5,  # You, Your, Yet, Ye, Yes
            }
            for ch, b in line_start_caps.items():
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] += b

    # Layer 5: speaker-label-specific boosts.
    if state.speaker_label_state == 3:
        # After ":" closing a label: strongly expect \n.
        logits[VOCAB_INDEX["\n"]] += 3.5
    elif state.speaker_label_state == 2 and state.upper_run_len >= 3:
        # Name is long enough to plausibly end; boost ":"
        logits[VOCAB_INDEX[":"]] += 3.0

    # After sentence-ending punctuation (. ? !) at a verse-line-length
    # position, newline is a far more likely continuation than space.
    # Shakespeare's verse lines end with "." + newline very often.
    if (
        last_cls == 6  # PUNCT_END
        and state.speaker_label_state == 0
    ):
        csn = state.chars_since_newline
        if csn >= 50:
            logits[VOCAB_INDEX["\n"]] += 7.5
        elif csn >= 40:
            logits[VOCAB_INDEX["\n"]] += 6.0
        elif csn >= 30:
            logits[VOCAB_INDEX["\n"]] += 4.5
        elif csn >= 20:
            logits[VOCAB_INDEX["\n"]] += 3.0
        elif csn >= 15:
            logits[VOCAB_INDEX["\n"]] += 2.0
        elif csn >= 10:
            logits[VOCAB_INDEX["\n"]] += 1.2
        # Very short lines ending with PUNCT_END — typical in post-
        # speaker-label interjections like "Ay." or "O!". Boost \n when
        # this line appears to be a standalone utterance (previous line
        # was a speaker label or blank).
        elif csn >= 3 and state.prev_line_length < 25:
            logits[VOCAB_INDEX["\n"]] += 3.0

    # After ".\n" (sentence-end + single newline), the next character is
    # very often another newline (blank line before next speaker).
    if (
        last_cls == NEWLINE
        and state.consecutive_newlines == 1
        and state.prev_char_class == 6  # PUNCT_END before the \n
    ):
        logits[VOCAB_INDEX["\n"]] += 6.0

    # After mid-clause punctuation (, ; :) at a long-line position,
    # newline is also somewhat more likely (enjambment after comma).
    if (
        last_cls == 7  # PUNCT_MID
        and state.speaker_label_state == 0
    ):
        csn = state.chars_since_newline
        if csn >= 50:
            logits[VOCAB_INDEX["\n"]] += 7.0
        elif csn >= 40:
            logits[VOCAB_INDEX["\n"]] += 5.5
        elif csn >= 30:
            logits[VOCAB_INDEX["\n"]] += 3.5
        elif csn >= 20:
            logits[VOCAB_INDEX["\n"]] += 1.8


    # After apostrophe, specifically boost common contraction letters.
    if last_cls == APOSTROPHE:
        for ch, boost in (("s", 2.0), ("d", 1.5), ("t", 1.5), ("l", 1.0),
                          ("r", 0.8), ("v", 0.8)):
            logits[VOCAB_INDEX[ch]] += boost

    # After a contraction letter (apostrophe two characters back, with a
    # letter in between — e.g. "I'l", "he's", "I'v"), the next char is
    # very often either another contraction letter (for "'ll"/"'ve") or
    # a word terminator (space/comma/period).
    if (
        state.prev_char_class == APOSTROPHE
        and state.last_char_class in (LOWER_CONS, LOWER_VOWEL)
    ):
        # Boost space/terminator to close out the contraction.
        logits[VOCAB_INDEX[" "]] += 5.5
        # Specific letter doublings: 'll / 've.
        if state.last_char == "l":
            logits[VOCAB_INDEX["l"]] += 3.5
        if state.last_char == "v":
            logits[VOCAB_INDEX["e"]] += 4.0
        if state.last_char == "r":
            logits[VOCAB_INDEX["e"]] += 3.0  # 're (you're)

    # Layer 5c: phonotactic vowel enforcement. English very rarely
    # admits 4+ consecutive consonants without a vowel (the longest
    # real cluster is ~3-4: "str", "scr", "schr"). When the count
    # reaches 4+, bias toward vowels/word-enders. Gentle at 4,
    # sharper at 5+. This catches gibberish like "nlrntd".
    if (
        state.consonants_since_vowel >= 4
        and state.word_buffer
        and state.speaker_label_state == 0
    ):
        c = state.consonants_since_vowel
        if c == 4:
            vbump = 0.6
            cpen = 0.25
            tbump = 0.4  # terminator bump (space/comma/period/nl)
        elif c == 5:
            vbump = 1.5
            cpen = 0.8
            tbump = 0.9
        else:
            vbump = 2.6
            cpen = 1.5
            tbump = 1.4
        for ch in "aeiou":
            if ch in VOCAB_INDEX:
                logits[VOCAB_INDEX[ch]] += vbump
        if "y" in VOCAB_INDEX:
            logits[VOCAB_INDEX["y"]] += vbump * 0.4
        for ch in "bcdfghjklmnpqrstvwxz":
            if ch in VOCAB_INDEX:
                logits[VOCAB_INDEX[ch]] -= cpen
        for ch in " ,.;\n":
            if ch in VOCAB_INDEX:
                logits[VOCAB_INDEX[ch]] += tbump

    # Layer 5d: vowel repetition penalty. 3+ consecutive vowels is
    # almost always nonsense ("aeere", "oeeeore", "uou"). English
    # allows rare 3-vowel sequences (beautiful, queue, iou) but
    # 4+ is practically never. Kicks in at 3+ to gently steer away,
    # stronger at 4+.
    if (
        state.vowels_since_consonant >= 3
        and state.word_buffer
        and state.speaker_label_state == 0
    ):
        v = state.vowels_since_consonant
        if v == 3:
            vpen = 1.0
            cbump = 0.4
            tbump = 0.4
        elif v == 4:
            vpen = 2.2
            cbump = 1.0
            tbump = 1.2
        else:
            vpen = 3.2
            cbump = 1.8
            tbump = 1.8
        for ch in "aeiou":
            if ch in VOCAB_INDEX:
                logits[VOCAB_INDEX[ch]] -= vpen
        for ch in "bcdfghjklmnpqrstvwxyz":
            if ch in VOCAB_INDEX:
                logits[VOCAB_INDEX[ch]] += cbump
        for ch in " ,.;\n":
            if ch in VOCAB_INDEX:
                logits[VOCAB_INDEX[ch]] += tbump

    # Layer 6: line-position / flow-aware modulations.
    # Only apply when we're outside speaker-label territory.
    if state.speaker_label_state == 0:
        llb = state.line_length_bucket
        sdb = state.sent_distance_bucket
        # At end-of-word position (letter_run >= 2 AND on_word_trie)
        # on progressively longer lines, newline becomes more likely as
        # the word's terminator. Training verse wraps ~30-50 chars;
        # prose ~60-80.
        if (
            (state.letter_run_len >= 2 and state.on_word_trie)
            or (state.letter_run_len == 1
                and state.word_buffer in COMPLETE_WORDS)
        ):
            csn = state.chars_since_newline
            if csn >= 60:
                logits[VOCAB_INDEX["\n"]] += 9.0
            elif csn >= 50:
                logits[VOCAB_INDEX["\n"]] += 7.5
            elif csn >= 40:
                logits[VOCAB_INDEX["\n"]] += 6.0
            elif csn >= 35:
                logits[VOCAB_INDEX["\n"]] += 4.5
            elif csn >= 25:
                logits[VOCAB_INDEX["\n"]] += 3.0
            elif csn >= 20:
                logits[VOCAB_INDEX["\n"]] += 2.0
        # Off-trie end-of-word: still some newline boost but weaker
        # (proper nouns, archaic forms that escape our trie).
        elif state.letter_run_len >= 3 and not state.on_word_trie:
            csn = state.chars_since_newline
            if csn >= 60:
                logits[VOCAB_INDEX["\n"]] += 6.5
            elif csn >= 50:
                logits[VOCAB_INDEX["\n"]] += 5.2
            elif csn >= 40:
                logits[VOCAB_INDEX["\n"]] += 4.0
            elif csn >= 30:
                logits[VOCAB_INDEX["\n"]] += 2.5
            elif csn >= 22:
                logits[VOCAB_INDEX["\n"]] += 1.2
        # Sentence-type-dependent end-punct ratios. Summed ratios stay
        # near the original (1.0 + 0.3 + 0.3 = 1.6) so total end-punct
        # mass stays calibrated; we reshuffle within the budget.
        st_type = state.sentence_type
        if st_type == 2:  # INTERROGATIVE
            ratio_period, ratio_q, ratio_excl = 0.35, 1.15, 0.15
        elif st_type == 3:  # EXCLAMATIVE
            ratio_period, ratio_q, ratio_excl = 0.55, 0.20, 0.90
        else:  # DECL or UNKNOWN: original shape
            ratio_period, ratio_q, ratio_excl = 1.0, 0.3, 0.3

        # Overdue sentence end: at word-end on-trie, boost sentence-end
        # punctuation so the model actually closes sentences.
        if (
            (state.letter_run_len >= 2 and state.on_word_trie)
            or (state.letter_run_len == 1
                and state.word_buffer in COMPLETE_WORDS)
        ):
            csse = state.chars_since_sentence_end
            if csse >= 100:
                bump = 7.5
            elif csse >= 80:
                bump = 6.5
            elif csse >= 60:
                bump = 5.5
            elif csse >= 45:
                bump = 4.5
            elif csse >= 30:
                bump = 2.5
            else:
                bump = 0.0
            if bump > 0.0:
                logits[VOCAB_INDEX["."]] += bump * ratio_period
                if "?" in VOCAB_INDEX:
                    logits[VOCAB_INDEX["?"]] += bump * ratio_q
                if "!" in VOCAB_INDEX:
                    logits[VOCAB_INDEX["!"]] += bump * ratio_excl
                if "," in VOCAB_INDEX:
                    logits[VOCAB_INDEX[","]] += bump * 0.6
                if ";" in VOCAB_INDEX:
                    logits[VOCAB_INDEX[";"]] += bump * 0.45
        # Mirror off-trie (weaker, min-length 4) — archaic/proper words
        # can also close an overdue sentence.
        elif state.letter_run_len >= 4 and not state.on_word_trie:
            csse = state.chars_since_sentence_end
            if csse >= 100:
                bump = 5.5
            elif csse >= 80:
                bump = 4.5
            elif csse >= 60:
                bump = 3.5
            elif csse >= 45:
                bump = 2.8
            elif csse >= 30:
                bump = 1.5
            else:
                bump = 0.0
            if bump > 0.0:
                logits[VOCAB_INDEX["."]] += bump * ratio_period
                if "?" in VOCAB_INDEX:
                    logits[VOCAB_INDEX["?"]] += bump * ratio_q
                if "!" in VOCAB_INDEX:
                    logits[VOCAB_INDEX["!"]] += bump * ratio_excl
                if "," in VOCAB_INDEX:
                    logits[VOCAB_INDEX[","]] += bump * 0.5
                if ";" in VOCAB_INDEX:
                    logits[VOCAB_INDEX[";"]] += bump * 0.4
        # Shakespeare is comma-heavy. Mid-sentence (before reaching the
        # "sentence is overdue" state handled above), bias toward commas
        # and semicolons at word-end-on-trie positions.
        is_complete_word = (
            state.word_buffer in COMPLETE_WORDS
            and state.on_word_trie
        )
        if (
            (
                (state.letter_run_len >= 2 and state.on_word_trie)
                or (state.letter_run_len == 1 and is_complete_word)
            )
            and state.chars_since_sentence_end < 40
        ):
            if "," in VOCAB_INDEX:
                logits[VOCAB_INDEX[","]] += 5.5
            if ";" in VOCAB_INDEX:
                logits[VOCAB_INDEX[";"]] += 2.5
        # Also off-trie with a longer min-length: archaic/proper words
        # can certainly be followed by ",".
        elif (
            state.letter_run_len >= 4
            and not state.on_word_trie
            and state.chars_since_sentence_end < 40
        ):
            if "," in VOCAB_INDEX:
                logits[VOCAB_INDEX[","]] += 4.5
            if ";" in VOCAB_INDEX:
                logits[VOCAB_INDEX[";"]] += 2.0

        # At word-end-on-trie, space is the most likely next char.
        # Boost it to reflect natural word-break frequency.
        if state.letter_run_len >= 2 and state.on_word_trie:
            logits[VOCAB_INDEX[" "]] += 0.6
        # For single-letter complete words (a/I/O), space is even more
        # certain — they almost always end right there.
        elif (
            state.letter_run_len == 1
            and state.word_buffer in COMPLETE_WORDS
        ):
            logits[VOCAB_INDEX[" "]] += 3.6




        # POS-aware terminal punctuation bias. At word-end on-trie,
        # the POS that JUST completed (captured at just_finished_word
        # time, but here we rely on the fact that state.word_buffer is
        # itself an active word still — so use a lookahead classifier
        # on the buffer to predict what kind of word will have just
        # completed). We approximate by using last_word_pos at the
        # moment the space arrives, which happens in the next advance.
        # For *this* boundary, use the buffer's suffix as a cue.
        if (
            state.letter_run_len >= 2
            and state.on_word_trie
            and state.word_buffer
        ):
            buf = state.word_buffer
            # Function-word-ish buffer: after it, sentence end is very
            # unlikely (penalize period/? /!). Check for short buffers
            # matching known closed-class words.
            is_closed = (
                buf in _ARTICLES
                or buf in _POSSESSIVES
                or buf in _PREPOSITIONS
                or buf in _CONJUNCTIONS
                or buf in _AUX_VERBS
                or buf in _MODALS
            )
            if is_closed:
                # After function words, period/?/! are very unlikely;
                # space is the clear winner.
                logits[VOCAB_INDEX["."]] -= 3.0
                if "?" in VOCAB_INDEX:
                    logits[VOCAB_INDEX["?"]] -= 2.5
                if "!" in VOCAB_INDEX:
                    logits[VOCAB_INDEX["!"]] -= 2.5
                if ":" in VOCAB_INDEX:
                    logits[VOCAB_INDEX[":"]] -= 1.5
                # Next char is almost always a space (preceding next word).
                logits[VOCAB_INDEX[" "]] += 1.2
                # Newline almost never happens after a function word.
                logits[VOCAB_INDEX["\n"]] -= 1.8

    return _log_softmax(logits)
