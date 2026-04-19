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
from .address import address_midword_bias, address_start_bias
from .alliteration import alliteration_start_bias
from .anaphora import anaphora_midword_bias, anaphora_start_bias
from .antithesis import antithesis_closure_bias, antithesis_pivot_bias
from .archaic import archaic_midword_bias, archaic_start_bias
from .meditative import meditative_midword_bias, meditative_start_bias
from .bigram import bigram_bias
from .cadence import cadence_wordend_bias
from .enjambment import enjambment_wordend_bias
from .meter import pentameter_wordend_bias
from .polysyllable import polysyllable_midword_bias
from .context import CTX_BIAS_VECTORS, context_key
from .letter3 import letter3_bias
from .match_count import match_count_bias
from .letter4 import letter4_bias
from .line_coherence import line_coherence_wordend_bias
from .list_bias import list_start_bias, list_wordend_comma_bias
from .next_word import next_word_bias
from .word_bigram_continue import word_bigram_continue_bias
from .phrase_continue import phrase_continue_bias
from .phrase_trigram_continue import phrase_trigram_continue_bias
from .word_integrity import word_integrity_bias
from .sensory_charge import sensory_charge_start_bias
from .oath_mode import oath_mode_start_bias, oath_mode_close_bias
from .np_head import np_head_start_bias
from .ornament import ornament_start_bias
from .parallel import parallel_start_bias
from .proper_noun import proper_noun_start_bias
from .proper_noun_memory import (
    proper_noun_memory_mid_bias,
    proper_noun_memory_start_bias,
)
from .phrase_bigram import phrase_bigram_bias
from .phrase_trigram import phrase_trigram_bias
from .pos_next import pos_next_bias
from .pos_bigram_next import pos_bigram_next_bias
from .repetition import repetition_start_bias
from .rhyme import rhyme_midword_bias
from .rhythm import rhythm_wordend_bias
from .sonority import sonority_midword_bias
from .suffix_completion import suffix_completion_bias
from .verb_word_trie import verb_word_trie_bias
from .object_word_trie import object_word_trie_bias
from .post_obj_word_trie import post_obj_word_trie_bias
from .subject_word_trie import subject_word_trie_bias
from .subord import subord_midword_bias, subord_word_end_bias
from .tense import tense_midword_bias, tense_start_bias
from .caesura import caesura_bias
from .clause_rhythm import clause_rhythm_comma_bias
from .urgency import urgency_word_end_bias, urgency_long_word_bias
from .dependent_clause import dependent_clause_bias
from .offtrie_depart import offtrie_depart_bias
from .mid_departure import mid_departure_bias
from .cv_alternation import cv_alternation_bias
from .discourse_rhythm import discourse_rhythm_start_bias
from .slot_next import slot_start_bias
from .speaker_recency import speaker_recency_bias
from .register_commit_bias import register_commit_start_bias
from .speaker_register_bias import speaker_register_start_bias
from .sentence_length_prior import sentence_length_prior_bias
from .speaker_trie import speaker_trie_bias
from .start4gram import start4gram_bias
from .tonal import tonal_start_bias
from .turn_opener import TURN_OPENER_START_BIAS
from .answer_opener import answer_opener_start_bias
from .answer_expectation import answer_expectation_start_bias
from .dialogue_opener import dialogue_adjacency_bias, dialogue_pacing_bias
from .start5gram import start5gram_bias
from .startbigram import startbigram_bias
from .onset_cluster import onset_cluster_bias
from .dash_aside import dash_aside_open_bias, dash_aside_close_bias
from .starttrigram import starttrigram_bias
from .startword import START_BIAS
from .formula import formula_midword_bias, formula_start_bias
from .iambic import iambic_word_start_bias
from .imagery import imagery_start_bias
from .scene_topic import scene_topic_midword_bias, scene_topic_start_bias
from .invocation import (
    invocation_sentence_end_bias,
    invocation_sentence_start_bias,
    invocation_word_start_bias,
)
from .next_sentence_bias import next_sentence_start_bias
from .doubt import doubt_sentence_end_bias, doubt_word_start_bias
from .sentence_anaphora import sentence_anaphora_start_bias
from .topic import content_repeat_bias, topic_bias, topic_midword_bias
from .addressee import addressee_midword_bias, addressee_start_bias
from .adjacent_repeat import adjacent_repeat_bias
from .transitivity import transitivity_midword_bias, transitivity_start_bias
from .turn_punct_texture import (
    turn_punct_texture_bias,
    turn_punct_texture_sentence_start_bias,
)
from .word_form import word_form_midword_bias, word_form_start_bias
from .trie_recovery import trie_recovery_bias
from .word_end_bigram import word_end_bigram_bias
from .referent import referent_start_bias
from .verb_agreement import verb_agreement_bias, verb_agreement_start_bias
from .verb_chain import verb_chain_bias
from .verb_object_class import verb_object_class_start_bias
from .clause_depth import clause_depth_close_bias
from .double_cons_start import double_consonant_penalty
from .red_flags import red_flags_close_bias
from .negation import negation_start_bias
from .case_slot import case_slot_start_bias
from .lament import lament_start_bias, lament_sentence_start_bias
from .tenderness import tenderness_start_bias, tenderness_sentence_start_bias
from .fury import fury_end_bias, fury_start_bias
from .gravitas import gravitas_start_bias, gravitas_sentence_start_bias
from .drift_recovery import drift_recovery_bias, drift_recovery_midword_bias
from .gibberish_hardcap import gibberish_hardcap_bias
from .verb_complement import verb_complement_start_bias
from .line_break_bias import line_break_newline_bias
from .trigram import trigram_bias
from .vocative import VOCATIVE_START_BIAS
from .unigram import UNIGRAM_LOGPROBS
from .word_trie import COMPLETE_WORDS, FORCE_END_BIAS, is_on_trie, word_trie_bias


def _log_softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    logz = m + math.log(z)
    return [x - logz for x in logits]


def _log_softmax_smoothed(logits: list[float], floor: float) -> list[float]:
    """Log-softmax with a uniform-probability floor (label smoothing).

    After normalization, blends with a uniform distribution:
      p' = (1 - V*floor) * p + floor
    ensuring every token has at least `floor` probability. Improves BPC
    on tail/surprise tokens where layer biases would otherwise drive
    the posterior to near-zero.
    """
    V = len(logits)
    if floor <= 0.0:
        return _log_softmax(logits)
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    c = 1.0 - V * floor
    return [math.log(c * (e / z) + floor) for e in exps]


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
                logits[i] += bi[i] * 1.8

    # Layer 3a: mid-word uppercase penalty. English / Shakespeare text
    # virtually never has uppercase letters in the interior of a word
    # — capitals appear only at word-start (sentence/line start, proper
    # nouns) or throughout a speaker-label. After a lowercase letter
    # outside speaker-label territory, strongly penalize all uppercase.
    # The word_trie and startword biases already vote for the right
    # case at start positions; this closes the loop at continuation.
    if (
        last_cls in (LOWER_VOWEL, LOWER_CONS)
        and state.speaker_label_state == 0
        and state.letter_run_len >= 1
    ):
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if ch in VOCAB_INDEX:
                logits[VOCAB_INDEX[ch]] -= 8.0

    # Layer 3b: trigram digraph biases (last two letters).
    if state.last_char and state.prev_char:
        tg = trigram_bias(state.prev_char, state.last_char)
        if tg is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += tg[i] * 2.0

    # Layer 3b2: letter-trigram bias (last 3 letters → next). Apply
    # only off-trie, where the word_trie doesn't already give signal.
    if state.word_buffer and not state.on_word_trie:
        l3 = letter3_bias(state.word_buffer)
        if l3 is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += l3[i] * 3.0

    # Layer 3b2-L4: letter-4gram suffix bias (last 4 letters → next).
    # Only fires for hand-specified high-confidence English suffixes
    # ("ough→t", "tion→ ", "ness→ ", "ight→ ", etc.). Applied off-trie
    # to rescue words whose 5+-letter form our trie doesn't carry —
    # on-trie, the word_trie already dominates and l4 collides with it.
    if (
        state.word_buffer
        and state.letter_run_len >= 4
        and not state.on_word_trie
    ):
        l4 = letter4_bias(state.word_buffer)
        if l4 is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += l4[i] * 2.5

    # Layer 3b2b: word-form mid-word bias. When WFE_PAST_PART is
    # active, tilt letters 3-5 of the word toward -en/-n/-ed ending
    # trajectories (seen/taken/given/borne/drawn/loved/feared).
    if state.word_buffer and state.letter_run_len >= 3:
        wfm = word_form_midword_bias(
            state.word_form_expectation,
            state.wfe_wait_words,
            state.word_buffer,
            state.letter_run_len,
            state.speaker_label_state,
        )
        if wfm is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += wfm[i]

    # Layer 3b2b-poly: polysyllable-density mid-word bias. When the
    # recent passages have been polysyllabic, slightly discourage
    # space in the letter_run_len [3, 6] decision zone — keep
    # extending. When they've been monosyllabic, nudge space — close
    # the word early. Rhythm modulator with small amplitude.
    if state.word_buffer and 3 <= state.letter_run_len <= 6:
        pmb = polysyllable_midword_bias(
            state.polysyllable_density,
            state.letter_run_len,
            state.on_word_trie,
            state.speaker_label_state,
        )
        if pmb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += pmb[i]

    # (Layer 3b2c: sonority-texture bias — disabled; letter-n-gram
    # priors already capture phonotactic regularities better than a
    # smoothed sonority field can tilt them. The sonority_level field
    # is still maintained as state for possible use at word-start
    # positions where n-gram signal is weaker.)






    # Layer 3b3: word-start bigram bias — at letter_run_len == 1, the
    # second letter is heavily conditioned on the first (word-start
    # distributions differ from mid-word). Applies only when speaker
    # label is not constraining things, AND only when we're actually at
    # the START of a word (buffer is exactly the one letter). Skips
    # contractions like "I'l" where the buffer has an apostrophe and
    # the letter is a mid-word continuation, not a word start.
    if (
        state.letter_run_len == 1
        and state.last_char
        and state.speaker_label_state not in (2,)
        and len(state.word_buffer) == 1
    ):
        sb = startbigram_bias(state.last_char)
        if sb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += sb[i] * 1.2
        # Double-consonant word-start penalty: "f" → "ff", "r" → "rr",
        # etc. are all implausible English word-starts. Suppresses
        # gibberish like "frr-", "tt-", "ss-".
        dc = double_consonant_penalty(state.last_char)
        if dc is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += dc[i]

    # Onset-cluster legality (letter_run_len in {1, 2}): penalize
    # next-letter choices that would produce phonotactically illegal
    # English word onsets (e.g., "rs-", "mn-", "tb-", "dl-", triple
    # onsets like "strn-"). Broader coverage than startbigram — fires
    # at letter 2 AND letter 3 of a consonant-only prefix.
    if (
        state.speaker_label_state not in (2,)
        and state.letter_run_len in (1, 2)
    ):
        oc = onset_cluster_bias(
            state.word_buffer,
            state.letter_run_len,
            state.speaker_label_state,
        )
        if oc is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += oc[i] * 3.5

    # Extend double-consonant check to mid-cluster positions. Doubling
    # the last consonant inside a consonant cluster is almost always
    # gibberish ("frr", "stt", "prr", "strr"). Skip when previous letter
    # was a vowel — "let"→"tt" (letter), "app"→"pp" (apple) are common.
    if (
        state.last_char
        and state.speaker_label_state not in (2,)
        and state.consonants_since_vowel >= 2
        and state.word_buffer
    ):
        dc = double_consonant_penalty(state.last_char)
        if dc is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += dc[i]

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
                logits[i] += stt[i] * 1.3

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
                logits[i] += s4[i] * 1.3

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
                logits[i] += s5[i] * 1.7

    # Layer 3c: word-trie completion bias. Scaled by letter position
    # AND by trie_match_count — when the completion set is tight
    # (1-3 words), the remaining bias vector is a sharp near-unique
    # prediction; push it a little harder so the model commits.
    if state.word_buffer:
        wt = word_trie_bias(state.word_buffer)
        if wt is not None:
            # Scale slightly up at longer prefixes — fewer completions
            # remain, so the bias is more discriminating.
            # Scale per letter_run_len: longer prefixes are more
            # discriminating. Asymmetric: slightly down-weight very
            # short prefixes (1-2 letters) since word_trie massively
            # over-weights common 2-letter prefix branches relative
            # to the residual context-class/bigram priors.
            rl = state.letter_run_len
            if rl <= 1:
                wt_scale = 1.28
            elif rl == 2:
                wt_scale = 1.40
            elif rl == 3:
                wt_scale = 1.46
            elif rl == 4:
                wt_scale = 1.50
            else:
                wt_scale = 1.54
            for i in range(VOCAB_SIZE):
                logits[i] += wt[i] * wt_scale

    # Layer 3c-WBC: word-bigram CONTINUATION bias. Given the previous
    # completed word and the current buffer (letters 1-4), bias the
    # next char toward the canonical continuations after that prior
    # word: after "to" + "b" → "e" (be); after "I" + "a" → "m" (am);
    # after "my" + "l" → "o" (lord); after "the" + "k" → "i" (king).
    # This extends next_word_bias beyond the first letter, sharpening
    # mid-word decisions with word-bigram context.
    if state.word_buffer and state.last_completed_word:
        wbc = word_bigram_continue_bias(
            state.last_completed_word,
            state.word_buffer,
            state.letter_run_len,
            state.speaker_label_state,
            state.on_word_trie,
        )
        if wbc is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += wbc[i]

    # Layer 3c-PHC: two-word phrase CONTINUATION bias. Conditions the
    # next char on (prev_prev_completed_word, last_completed_word) plus
    # the current buffer. Sharper than word_bigram_continue because it
    # knows "I have" + "b" → "e" (been), vs "to have" + "b" is looser.
    if (
        state.word_buffer
        and state.last_completed_word
        and state.prev_completed_word
    ):
        phc = phrase_continue_bias(
            state.prev_completed_word,
            state.last_completed_word,
            state.word_buffer,
            state.letter_run_len,
            state.speaker_label_state,
        )
        if phc is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += phc[i]

    # Layer 3c-PHC3: three-word phrase CONTINUATION bias. Conditions the
    # next char on (prev_prev, prev, last) completed words plus the
    # current buffer. Sharper than PHC because 3-word context disambiguates
    # continuations that 2-word cannot — "I have been" + "g" → "o" (gone),
    # "thou shalt not" + "d" → "i" (die), "to be or" + "n" → "o" (not),
    # "God save the" + "k" → "i" (king).
    if (
        state.word_buffer
        and state.last_completed_word
        and state.prev_completed_word
        and state.prev_prev_completed_word
    ):
        ph3 = phrase_trigram_continue_bias(
            state.prev_prev_completed_word,
            state.prev_completed_word,
            state.last_completed_word,
            state.word_buffer,
            state.letter_run_len,
            state.speaker_label_state,
        )
        if ph3 is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += ph3[i]

    # Layer 3c-WI: word-integrity termination push. When the current
    # word_buffer has collapsed into phonotactic gibberish (low
    # word_integrity, no trie match, likely long consonant run or no
    # vowel), strongly boost terminator chars (space/punct) so the
    # model bails out of the nonsense run. Structural gibberish-
    # terminator targeted at the single biggest sample-quality gap.
    wi = word_integrity_bias(
        state.word_integrity,
        state.letter_run_len,
        state.on_word_trie,
        state.speaker_label_state,
        state.buffer_consonant_run,
    )
    if wi is not None:
        for i in range(VOCAB_SIZE):
            logits[i] += wi[i]

    # Layer 3c-MC: graded trie-match-count bias. Complements the
    # binary on_word_trie. Three regimes:
    #   * count == 0 (just dropped from >=1): strong terminator push.
    #   * count == 1: unique completion — discourage premature end.
    #   * count in {2,3}: narrow set — gentle anti-term nudge.
    if state.word_buffer:
        mc = match_count_bias(
            state.trie_match_count,
            state.prev_trie_match_count,
            state.letter_run_len,
            state.on_word_trie,
            len(state.word_buffer),
            state.speaker_label_state,
        )
        if mc is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += mc[i]

    # Layer 3c-PNM: proper-noun rolodex mid-word continuation bias.
    # When we're inside a capitalized word that started mid-sentence
    # (a probable proper noun) AND the buffer matches a prefix of a
    # rolodex entry, strongly bias the next letter to continue that
    # name. This is where the rolodex earns its keep: "Ro" -> "m" to
    # continue "Rome"; "Cor" -> "i" to continue "Coriolanus".
    if state.proper_nouns_seen and state.current_word_started_cap:
        pnm_mid = proper_noun_memory_mid_bias(
            state.proper_nouns_seen,
            state.speaker_label_state,
            state.current_word_started_cap,
            state.word_buffer,
            state.letter_run_len,
        )
        if pnm_mid is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += pnm_mid[i]

    # Layer 3c1: topic-midword bias — when content_words indicate an
    # active topical cluster (DARK/LIGHT/ROYAL), tilt mid-word letter
    # choice toward completions that stay within the cluster. This
    # rides atop word_trie (which votes for all known words) and
    # nudges equipoised completions toward topically-coherent ones
    # ("bl" after dark content → bleed/blood over bless).
    if (
        state.word_buffer
        and state.content_words
        and state.speaker_label_state == 0
    ):
        tmw = topic_midword_bias(state.word_buffer, state.content_words)
        if tmw is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += tmw[i]
        # Scene-topic mid-word bias: when a scene_topics cluster is
        # dominant and we've drifted off-trie, nudge letter choice
        # toward topic-characteristic continuation letters.
        stm = scene_topic_midword_bias(
            state.scene_topics,
            state.speaker_label_state,
            state.letter_run_len,
            state.letters_off_trie,
            state.on_word_trie,
        )
        if stm is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += stm[i]
        # Content-word repetition: direct prefix-match boost for recent
        # content words (motif repetition). Runs alongside topic-midword
        # which is distribution-averaged; this is sharper.
        crb = content_repeat_bias(state.word_buffer, state.content_words)
        if crb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += crb[i]

    # Layer 3c1a: trie-drift recovery. When the buffer has drifted
    # OFF the word-trie AND had a complete-word prefix earlier in
    # this word, escalate bias toward word-terminators and word-
    # ending letters. Fires only off-trie so long legitimate words
    # like "therefore" (on-trie throughout) are untouched.
    if (
        state.word_buffer
        and not state.on_word_trie
        and state.speaker_label_state == 0
        and (state.letters_past_complete >= 1 or state.letters_off_trie >= 2)
    ):
        tr = trie_recovery_bias(
            state.has_seen_complete,
            state.letters_past_complete,
            state.letters_off_trie,
        )
        if tr is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += tr[i]

        # Layer 3c1a-depart: departure-position aware bias. Reads
        # offtrie_depart_pos, the letter_run_len at which the word
        # LEFT the trie. Late departures (>= 5) mean we had a known
        # 5-letter prefix and are extending it into nonsense — push
        # hard toward word-end. Early departures (1-2) mean the word
        # was gibberish from the start — also push hard, with a
        # stronger gibberish-letter penalty.
        od = offtrie_depart_bias(
            state.offtrie_depart_pos,
            state.letters_off_trie,
            state.speaker_label_state,
        )
        if od is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += od[i] * 1.5

        # Layer 3c1a-mid: mid-departure (pos 3-4) extension bias.
        # offtrie_depart_bias explicitly skips depart_pos in {3, 4}
        # and trie_recovery has term_boost=0; this fills the gap.
        # Pushes end-letter and terminator probability in the 5-10
        # char gibberish window after a plausible 3-4 letter prefix.
        mdb = mid_departure_bias(
            state.mid_departure_extension,
            state.speaker_label_state,
        )
        if mdb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += mdb[i]

        # Layer 3c1a-cv: C-V alternation push inside polysyllabic
        # off-trie interiors. When 3+ consonants have been stacked
        # without a vowel (or 3+ vowels without a consonant),
        # overextended clusters get a phonotactic correction.
        cv = cv_alternation_bias(
            state.syllables_in_word,
            state.letter_run_len,
            state.consonants_since_vowel,
            state.vowels_since_consonant,
            state.on_word_trie,
            state.letters_off_trie,
            state.speaker_label_state,
        )
        if cv is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += cv[i] * 2.8
        # Word-end bigram plausibility: look at the last 2 letters of
        # the off-trie buffer and decide whether the suffix looks like
        # a real English word ending (boost " ") or clearly mid-word
        # (penalize " "). Complements trie_recovery's drift signal
        # with a fine-grained suffix-shape check.
        web = word_end_bigram_bias(
            state.word_buffer,
            state.letter_run_len,
            state.on_word_trie,
            state.letters_off_trie,
            state.speaker_label_state,
        )
        if web is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += web[i]

        # Morphological-suffix completion: when the off-trie tail
        # matches the early part of a productive suffix (-ing, -eth,
        # -ness, -ment, -tion, -ly, -ous, -ful, -less, -able, ...),
        # nudge the next letter toward the suffix-completing letter,
        # and once the suffix is complete, gently boost terminators.
        sc = suffix_completion_bias(
            state.word_buffer,
            state.letter_run_len,
            state.on_word_trie,
            state.letters_off_trie,
            state.speaker_label_state,
        )
        if sc is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += sc[i]

        # Phonotactic red-flag closure: when word_shape has counted
        # 2+ phonotactic warnings in the current word (persistent
        # across letter steps), boost terminators to force close.
        rf = red_flags_close_bias(
            state.word_red_flags,
            state.letter_run_len,
            state.speaker_label_state,
        )
        if rf is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += rf[i]

        # Scene-drift mid-word terminator push. When drift_streak >= 2
        # AND the current word is also off-trie AND has extended 4+
        # letters, push aggressively toward terminator / safe ending
        # letters to cut the gibberish short.
        dmw = drift_recovery_midword_bias(
            state.drift_streak,
            state.letter_run_len,
            state.letters_off_trie,
            state.on_word_trie,
            state.speaker_label_state,
        )
        if dmw is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += dmw[i]

        # Hard cap: exponentially growing terminator bias once a word
        # has extended off-trie past letter 10. Complements offtrie_depart
        # (which caps its scaling at letter 8) by continuing the
        # pressure to truly unbounded off-trie lengths. Forces gibberish
        # to close by letter 13-15 at the latest.
        ghc = gibberish_hardcap_bias(
            state.letter_run_len,
            state.on_word_trie,
            state.letters_off_trie,
            state.speaker_label_state,
        )
        if ghc is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += ghc[i]


    # Layer 3c1-verb: verb-word-trie mid-word bias. When the clause
    # has a subject but no verb yet, AND the current word_buffer is
    # a prefix of some verb / aux / modal in our inventory, nudge the
    # next letter toward a verb-completion branch. When the buffer
    # IS a complete verb, gently boost terminators so the verb closes
    # rather than drifting. This layer rides alongside the general
    # word_trie and closes the gap between word-start verb-overdue
    # bias (letter_run_len==0) and word-end drift recovery. Active
    # both on-trie and off-trie — its signal is about SYNTAX, not
    # vocabulary coverage.
    if (
        state.word_buffer
        and state.speaker_label_state == 0
        and state.clause_slot == 1
        and state.words_since_verb >= 1
    ):
        vwt = verb_word_trie_bias(
            state.word_buffer,
            state.letter_run_len,
            state.clause_slot,
            state.words_since_verb,
            state.speaker_label_state,
        )
        if vwt is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += vwt[i]

    # Layer 3c1-obj: object-phrase word-trie mid-word bias. Parallel
    # to verb_word_trie but fires at clause_slot == HAS_VERB — the
    # post-verb slot where a determiner / short pronoun / preposition
    # / complement-opener is the likely next word. Closes the gap
    # where "is se...", "gave hi...", "speak to...", etc., drift into
    # improbable completions.
    if (
        state.word_buffer
        and state.speaker_label_state == 0
        and state.clause_slot == 2
    ):
        owt = object_word_trie_bias(
            state.word_buffer,
            state.letter_run_len,
            state.clause_slot,
            state.speaker_label_state,
        )
        if owt is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += owt[i]

    # Layer 3c1-post: post-object word-trie bias. Third corner of the
    # clause-FSM-aware predict family. Fires at clause_slot == POST_OBJ
    # and biases toward coordinating / subordinating conjunctions,
    # relatives, chain adverbs, and PP-extending prepositions.
    if (
        state.word_buffer
        and state.speaker_label_state == 0
        and state.clause_slot == 3
    ):
        pwt = post_obj_word_trie_bias(
            state.word_buffer,
            state.letter_run_len,
            state.clause_slot,
            state.speaker_label_state,
        )
        if pwt is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += pwt[i]

    # Layer 3c1-subj: subject / clause-opener word-trie bias. Fourth
    # corner of the clause-FSM-aware predict family. Fires at
    # clause_slot == FRESH (shortly after a sentence / clausal break)
    # and biases toward subject pronouns, determiners opening subject
    # NPs, wh-words, interjections, temporal/conditional openers, and
    # short negations.
    if (
        state.word_buffer
        and state.speaker_label_state == 0
        and state.clause_slot == 0
        and state.chars_since_sentence_end <= 20
    ):
        swt = subject_word_trie_bias(
            state.word_buffer,
            state.letter_run_len,
            state.clause_slot,
            state.speaker_label_state,
            state.chars_since_sentence_end,
        )
        if swt is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += swt[i]

    # Layer 3c1-subord: subordinate-clause mid-word bias. Inside a
    # subord at HAS_SUBJ, lean toward -eth / -est completions.
    if (
        state.word_buffer
        and state.speaker_label_state == 0
        and state.subord_depth >= 1
    ):
        sbm = subord_midword_bias(
            state.subord_depth,
            state.subord_words_since_open,
            state.letter_run_len,
            state.word_buffer,
            state.clause_slot,
            state.speaker_label_state,
        )
        if sbm is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += sbm[i]

    # Layer 3c1-trans: transitivity mid-word bias. When an object is
    # expected AND the current buffer is a short determiner-prefix
    # ("t"/"th"/"m"/"my"/"h"/"hi"/"a"/"an"/"y"/"yo"), nudge the
    # continuation toward completing the determiner.
    if (
        state.word_buffer
        and state.speaker_label_state == 0
        and state.verb_transitivity != 0
    ):
        trmb = transitivity_midword_bias(
            state.verb_transitivity,
            state.vt_wait_words,
            state.word_buffer,
            state.letter_run_len,
            state.speaker_label_state,
        )
        if trmb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += trmb[i]

    # Layer 3c2: subject-verb agreement morphology bias. When a
    # subject has been identified (clause_slot == HAS_SUBJ), the
    # upcoming verb's suffix is morphologically constrained — "thou
    # X-est / X-st", "he/she X-eth / X-s". Fires mid/end-of-word.
    va = verb_agreement_bias(
        state.verb_agreement,
        state.clause_slot,
        state.speaker_label_state,
        state.word_buffer,
        state.letter_run_len,
    )
    if va is not None:
        for i in range(VOCAB_SIZE):
            logits[i] += va[i]

    # Layer 3c1-addr: addressee-memory mid-word bias. When vocative
    # expectation is active AND the current buffer is a prefix of the
    # last vocative noun this speaker used, boost the continuing
    # letter so we complete to the same noun.
    if (
        state.word_buffer
        and state.speaker_label_state == 0
        and state.vocative_expectation
        and state.last_vocative
        and state.turn_vocative_count >= 1
    ):
        amw = addressee_midword_bias(
            state.last_vocative,
            state.turn_vocative_count,
            state.word_buffer,
            state.vocative_expectation,
            state.speaker_label_state,
        )
        if amw is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += amw[i]

    # Layer 3c1-adj: adjacent-word-repeat blocker. "of of", "the the",
    # "and and" never happen. Narrow rule; fires at word-start, mid-
    # word on matching prefix, and at terminator if buffer equals
    # last_completed_word exactly.
    ar = adjacent_repeat_bias(
        state.word_buffer,
        state.last_completed_word,
        state.letter_run_len,
        state.speaker_label_state,
        state.consecutive_newlines,
    )
    if ar is not None:
        for i in range(VOCAB_SIZE):
            logits[i] += ar[i]

    # Layer 3c1b: formulaic-phrase mid-word bias. When we're positioned
    # inside a known multi-word formula and the current buffer is a
    # prefix of an expected next word, bias toward the continuation
    # letter. This gives multi-word lookahead that phrase_bigram can't.
    if (
        state.word_buffer
        and state.formula_node != 0
        and state.speaker_label_state == 0
    ):
        fmw = formula_midword_bias(state.formula_node, state.word_buffer)
        if fmw is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += fmw[i]


    # Layer 3c2-tense: sentence-tense suffix-completion tilt. Reads
    # sentence_tense and tilts the buffer's suffix decision toward
    # tense-consistent endings (-ed for PAST, -s/-eth for PRESENT).
    tmw = tense_midword_bias(
        state.sentence_tense,
        state.sentence_tense_age,
        state.word_buffer,
        state.letter_run_len,
        state.speaker_label_state,
        state.on_word_trie,
    )
    if tmw is not None:
        for i in range(VOCAB_SIZE):
            logits[i] += tmw[i]

    # Layer 3c2: archaic mid-word disambiguation — when buffer matches
    # a prefix shared by archaic and modern words, lean toward the
    # archaic completion in proportion to archaic_density.
    if state.word_buffer and state.archaic_density > 0.0:
        am = archaic_midword_bias(state.word_buffer, state.archaic_density)
        if am is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += am[i]

    # Layer 3c2e: antithesis pivot / closure bias. Reads the state
    # maintained by pipeline/antithesis.py. At word-start, when we're
    # in OPENER_SEEN state (a "not/nor/neither/rather" opener has
    # fired and no pivot yet), nudge toward pivot-opener letters
    # (b/o/n/t/y/e). When PIVOTED and we're deep in the complement
    # half, gently elevate closing punctuation at between-word
    # positions. This captures a uniquely Shakespearean rhetorical
    # axis — the two-part contrast rhythm — that no existing layer
    # sees.
    if (
        state.letter_run_len == 0
        and state.speaker_label_state == 0
        and state.antithesis_state == 1  # OPENER_SEEN
        and state.antithesis_words_since_opener >= 2
    ):
        ap = antithesis_pivot_bias(state.antithesis_words_since_opener)
        for i in range(VOCAB_SIZE):
            logits[i] += ap[i]
    if (
        state.letter_run_len == 0
        and state.speaker_label_state == 0
        and state.antithesis_state == 2  # PIVOTED
        and state.antithesis_words_since_pivot >= 3
    ):
        ac = antithesis_closure_bias(
            state.antithesis_words_since_pivot, state.letter_run_len
        )
        for i in range(VOCAB_SIZE):
            logits[i] += ac[i]

    # Layer 3c3: meditative-register word-start bias. Reads the flow-tier
    # meditative_register (rises on philosophical/abstract vocabulary like
    # think/soul/mind/dream/nature/reason, decays per word). At word-start
    # (letter_run_len == 0 AND not in a speaker label), leans toward
    # first letters that begin meditative-lexicon words (t, m, s, d, w, r,
    # p, f, i, h) and gently away from concrete-battlefield opener letters
    # (b, k, g). This is a genuine flow-texture consumer: soliloquy vs.
    # battle cry.
    if (
        state.letter_run_len == 0
        and state.speaker_label_state == 0
        and state.meditative_register > 0.08
    ):
        ms = meditative_start_bias(state.meditative_register)
        for i in range(VOCAB_SIZE):
            logits[i] += ms[i]

    # Layer 3c3b: meditative mid-word disambiguation. When the buffer is
    # a prefix shared by a meditative word and a non-meditative word,
    # lean toward the meditative completion in proportion to register.
    if state.word_buffer and state.meditative_register > 0.15:
        mm = meditative_midword_bias(
            state.word_buffer, state.meditative_register
        )
        if mm is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += mm[i]

    # Layer 3c2d: monosyllabic-run rhythm word-end bias. Reads
    # state.monosyllabic_run (a flow-tier texture field) and, when
    # we're deep in a monosyllabic-run cadence AND the current word
    # buffer has reached a complete known word (has_seen_complete),
    # nudges toward word-ending characters (space/punct) over further
    # letters. This closes the word before it drifts polysyllabic and
    # breaks the percussive rhythm. Gated on has_seen_complete AND
    # letters_past_complete <= 1 so we only fire at legitimate close
    # points, not mid-word at positions that may still have letters.
    if (
        state.word_buffer
        and state.speaker_label_state == 0
        and state.has_seen_complete
        and state.letters_past_complete <= 1
    ):
        rh = rhythm_wordend_bias(
            state.monosyllabic_run,
            state.letter_run_len,
            state.speaker_label_state,
            state.on_word_trie,
        )
        if rh is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += rh[i]

    # Layer 3c2d-rhythm: clause-rhythm comma-pressure. When the current
    # comma-less run has grown long AND we're sitting at a clean
    # word-end boundary on the word trie, nudge toward comma to
    # reproduce Shakespeare's comma-heavy clausal cadence. Fires only
    # at on-trie complete-word positions so we don't inject punctuation
    # into gibberish.
    if (
        state.word_buffer
        and state.speaker_label_state == 0
        and state.chars_since_comma >= 20
    ):
        crc = clause_rhythm_comma_bias(
            state.chars_since_comma,
            state.chars_since_sentence_end,
            state.word_buffer,
            state.on_word_trie,
            state.letter_run_len,
            state.speaker_label_state,
            state.has_seen_complete,
            state.letters_past_complete,
        )
        if crc is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += crc[i]

    # Layer 3c2b-dep: dependent-clause closer. When inside an active
    # dependent clause (opened by "if/when/though/which/that/..."),
    # push comma at word-end (close dep) and penalize sentence-enders
    # (main clause is still pending).
    if state.in_dependent_clause and state.speaker_label_state == 0:
        dc = dependent_clause_bias(
            state.in_dependent_clause,
            state.words_in_subordinate,
            state.clause_slot,
            state.chars_since_sentence_end,
            state.chars_since_comma,
            state.word_buffer,
            state.on_word_trie,
            state.letter_run_len,
            state.speaker_label_state,
            COMPLETE_WORDS,
        )
        if dc is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += dc[i]

    # Layer 3c2b-caesura: caesura-gap suppression. After a caesura
    # fired in this line, suppress a second comma/semicolon at the
    # exact same syllable (gap == 0 — "choppy" doubled break).
    if state.speaker_label_state == 0 and state.has_caesura_this_line:
        cb = caesura_bias(
            state.has_caesura_this_line,
            state.caesura_syllable,
            state.syllables_in_line,
            state.verse_score,
            state.verse_line_run,
            state.prev_line_syllables,
            state.speaker_label_state,
            state.consecutive_newlines,
            state.chars_since_sentence_end,
            state.letter_run_len,
            state.on_word_trie,
            state.word_buffer,
            COMPLETE_WORDS,
        )
        if cb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += cb[i]

    # Layer 3c2b-urgency: flow-driven urgency/tempo bias.
    # Reads the urgency_tempo flow field (frantic ↔ languid), pushing
    # "!"/"." and away from "," at word-end in high-urgency contexts,
    # and pushing space inside long words to keep the tempo tight.
    if state.speaker_label_state == 0:
        ub = urgency_word_end_bias(
            state.urgency_tempo,
            state.letter_run_len,
            state.on_word_trie,
            state.word_buffer,
            COMPLETE_WORDS,
            state.speaker_label_state,
            state.consecutive_newlines,
            state.chars_since_sentence_end,
            state.words_in_sentence,
        )
        if ub is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += ub[i]
        ulw = urgency_long_word_bias(
            state.urgency_tempo,
            state.letter_run_len,
            state.on_word_trie,
            state.word_buffer,
            COMPLETE_WORDS,
            state.speaker_label_state,
        )
        if ulw is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += ulw[i]

    # Layer 3c2c: rhyme-position mid-word bias. When we're in a verse
    # run and approaching line-end, nudge the next letter toward the
    # previous line's rhyme letter.
    if (
        state.word_buffer
        and state.speaker_label_state == 0
        and state.verse_line_run >= 1
    ):
        rb = rhyme_midword_bias(
            state.prev_line_tail,
            state.prev_prev_line_tail,
            state.verse_line_run,
            state.chars_since_newline,
            state.word_buffer,
        )
        if rb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += rb[i]

    # Layer 3c2b: anaphora mid-word continuation. When an anaphora
    # pattern is active AND we're mid-way through the first word of
    # a fresh line, push toward completing the anaphora-repeated word.
    if (
        state.word_buffer
        and state.speaker_label_state == 0
        and state.words_completed_on_line == 0
        and state.recent_line_starters
    ):
        amw = anaphora_midword_bias(
            state.recent_line_starters,
            state.word_buffer,
            state.chars_since_newline,
        )
        if amw is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += amw[i]

    # Layer 3c3: 2nd-person addressing-register mid-word bias. When
    # buffer is a prefix of a pronoun in the currently-established
    # register, push the next letter toward completion of that
    # register's pronoun. Skip inside speaker-label territory.
    if (
        state.word_buffer
        and state.speaker_label_state == 0
        and abs(state.addressing_register) > 0.5
    ):
        ab = address_midword_bias(
            state.addressing_register, state.word_buffer
        )
        if ab is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += ab[i]



    # Layer 3d: speaker-label trie bias.
    if state.speaker_buffer:
        st = speaker_trie_bias(state.speaker_buffer)
        if st is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += st[i]
        elif (
            state.speaker_label_state == 2
            and len(state.speaker_buffer) >= 3
        ):
            # Off-trie drift inside a speaker label. The offtrie_run
            # counter tells us how deep we've drifted. Short runs are
            # plausibly a minor/unknown name (keep a small ":" close-
            # nudge). Very long runs (6+) are likely phantom labels
            # the sampler hallucinated — cap the boost and give a
            # mild newline lift so the FSM can escape.
            drift = min(len(state.speaker_buffer) - 2, 4)
            run = state.speaker_label_offtrie_run
            if ":" in VOCAB_INDEX:
                logits[VOCAB_INDEX[":"]] += 0.15 * drift
            if run >= 4:
                esc = min(run - 3, 6)
                if "\n" in VOCAB_INDEX:
                    logits[VOCAB_INDEX["\n"]] += 0.4 * esc

    # Layer 3d1: speaker recency bias — tilt the speaker trie toward
    # characters recently in-scene, and penalize immediate self-repeat.
    if state.speaker_buffer and state.recent_speakers:
        sr = speaker_recency_bias(state.speaker_buffer, state.recent_speakers)
        if sr is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += sr[i]

    # Layer 4: start-of-word bias (after space or single newline).
    if last_cls == SPACE or (
        last_cls == NEWLINE and state.consecutive_newlines == 1
    ):
        for i in range(VOCAB_SIZE):
            logits[i] += START_BIAS[i]

        # Layer 4-SC: sensory-charge register start bias. When the
        # recent-words window has been corporeal / sensory (tragic
        # lyric), boost first letters of sensory vocabulary. When it
        # has been abstract / discursive (court argument), boost
        # first letters of reasoning vocabulary. Flow-tier texture.
        scb = sensory_charge_start_bias(
            state.sensory_charge,
            state.speaker_label_state,
        )
        if scb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += scb[i]

        # Layer 4-OATH-S: oath-mode word-start bias. After "by" /
        # "upon" / "my" / "his" / "thy" / "our" with hot oath_mode,
        # bias first letters toward oath-object vocabulary (heaven,
        # troth, soul, faith, honour, God, sword, life, crown, grave).
        omsb = oath_mode_start_bias(
            state.oath_mode,
            state.last_completed_word,
            state.speaker_label_state,
            state.letter_run_len,
            state.word_buffer,
        )
        if omsb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += omsb[i]

        # Layer 4-OATH-C: oath-mode closure bias. When the just-
        # completed word is a canonical oath object (heaven/soul/
        # troth/faith/sword/honour/etc.) with oath_mode still warm,
        # bias next char toward "," (formulaic closure), "." , or
        # "!" over starting another word.
        omcb = oath_mode_close_bias(
            state.oath_mode,
            state.last_completed_word,
            state.speaker_label_state,
            state.word_buffer,
            state.letter_run_len,
            state.chars_since_sentence_end,
        )
        if omcb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += omcb[i]

        # Layer 4-PN: proper-noun slot bias. At mid-sentence word-
        # starts, penalize phantom capitals when no title / vocative
        # / quote-lead signal has raised the PN slot; boost capitals
        # when a proper-name context is established.
        pnb = proper_noun_start_bias(
            state.proper_noun_slot,
            state.speaker_label_state,
            state.sentence_start_pending,
            state.chars_since_sentence_end,
            state.words_in_sentence,
            state.consecutive_newlines,
        )
        if pnb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += pnb[i]

        # Layer 4-PNM: proper-noun scene rolodex bias. Recently-seen
        # capitalized content words (Rome, Coriolanus, Volsces, etc.)
        # get their initial letter boosted when a proper-noun context
        # is plausible, collapsing the huge first-letter uncertainty
        # over unknown names into the handful the scene has already
        # introduced.
        pnm = proper_noun_memory_start_bias(
            state.proper_nouns_seen,
            state.speaker_label_state,
            state.proper_noun_slot,
            state.sentence_start_pending,
            state.letter_run_len,
            state.word_buffer,
        )
        if pnm is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += pnm[i]

        # Layer 4-DR: discourse-rhythm sentence-first-letter bias. Reads
        # recent_sentence_types (rolling tuple of last 4 closed sentence
        # types) to detect discourse patterns — question chains,
        # exclamation chains, declarative flow — and shape the next
        # sentence's first letter accordingly. Only fires at a true
        # sentence-first-letter position (words_in_sentence == 0,
        # empty word_buffer, letter_run_len == 0).
        drb = discourse_rhythm_start_bias(
            state.recent_sentence_types,
            state.words_in_sentence,
            state.letter_run_len,
            state.word_buffer,
            state.speaker_label_state,
        )
        if drb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += drb[i]

        # Layer 4-SREG: speaker-register word-start tilt. Conditions
        # vocabulary on the current speaker's dramatic archetype
        # (tragic-noble, comic-prose, royal-formal, villain, lover,
        # servant, supernatural). Only fires after the register has
        # settled (register_age >= 3) so the speaker-label tokens
        # themselves aren't affected.
        srb = speaker_register_start_bias(
            state.speaker_register,
            state.register_age,
            state.letter_run_len,
            state.speaker_label_state,
        )
        if srb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += srb[i]

        # Layer 4-TVC: thou/you address-register commit tilt. Once
        # the speaker has used a T-form or V-form pronoun this turn,
        # keep their address register consistent for the rest of the
        # turn (Early Modern English grammar — switching mid-turn is
        # jarring). Reinforces verb_agreement's clause-local signal
        # with a turn-level prior that persists across sentence ends.
        tvb = register_commit_start_bias(
            state.thou_thee_commit,
            state.letter_run_len,
            state.speaker_label_state,
            state.case_slot,
            state.case_wait_words,
        )
        if tvb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += tvb[i]

        # Layer 4-TENSE: sentence-level tense register word-start tilt.
        # Reads sentence_tense (set by pipeline/tense.py at first finite
        # verb). Biases next verb's first letter toward tense-consistent
        # choices — keeps PAST-leaning sentences pulling past-tense verbs,
        # PRESENT-leaning sentences pulling present, etc.
        tsb = tense_start_bias(
            state.sentence_tense,
            state.sentence_tense_age,
            state.speaker_label_state,
            state.letter_run_len,
            state.word_buffer,
        )
        if tsb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += tsb[i]

        # Layer 4-LIST: list-parallelism first-letter bias. When we're
        # in a comma-separated list, bias toward (a) the same first
        # letter as the previous items (alliterative parallelism),
        # (b) closing conjunctions "and/or/nor/but" after 2+ commas,
        # and (c) POS-consistent starter letters if the first list
        # item's POS was a noun/verb/adjective.
        lsb = list_start_bias(
            state.commas_since_sent_end,
            state.list_item_pending,
            state.list_last_item_first_letter,
            state.list_parallel_run,
            state.list_first_item_pos,
            state.speaker_label_state,
        )
        if lsb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += lsb[i]

        # Layer 4b: next-word (word-bigram) first-letter bias.
        if state.last_completed_word:
            nw = next_word_bias(state.last_completed_word)
            if nw is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += nw[i]
                # Also add a small pos_next signal on top of next_word,
                # to pick up POS-class-level patterns that next_word's
                # per-word bigram doesn't capture.
                pn = pos_next_bias(state.last_word_pos)
                if pn is not None:
                    for i in range(VOCAB_SIZE):
                        logits[i] += 0.5 * pn[i]
            else:
                # Fallback: POS-based next-letter bias.
                pn = pos_next_bias(state.last_word_pos)
                if pn is not None:
                    for i in range(VOCAB_SIZE):
                        logits[i] += pn[i]

            # Layer 4b-bigram: 2-back POS-bigram → first-letter bias.
            # Reads both prev_word_pos and last_word_pos. Picks up
            # content-level bigram patterns ((DET, NOUN) → VERB next,
            # (PRONOUN, AUX) → predicate next, etc.) that 1-back POS
            # context can't see.
            pbn = pos_bigram_next_bias(
                state.prev_word_pos,
                state.last_word_pos,
            )
            if pbn is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += pbn[i]

            # Layer 4b2: subject-verb agreement word-start bias.
            # When a thou-subject has been identified and the verb
            # role is still unfilled, boost typical thou-verb
            # starter letters.
            vas = verb_agreement_start_bias(
                state.verb_agreement,
                state.clause_slot,
                state.speaker_label_state,
                state.letter_run_len,
                last_cls,
            )
            if vas is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += vas[i]

            # Layer 4b2a: verb-chain suppression. If a main verb just
            # filled the current slot, penalize first letters of main
            # verbs to prevent verb-after-verb-after-verb chains like
            # "Sail roar endanger" that never occur in real Shakespeare.
            # AUX/MODAL are transparent to verb_chain_len so legitimate
            # "had gone", "would speak", etc. are unaffected.
            vc = verb_chain_bias(
                state.verb_chain_len,
                state.clause_slot,
                state.speaker_label_state,
            )
            if vc is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += vc[i]


        # Layer 4b0-subord: subordinate-clause word-end bias.
        # Inside a subord (depth >= 1), suppress sentence-end punct
        # (the main clause still needs closing) and boost comma /
        # semicolon once the subord has run 4+ words.
        if state.subord_depth >= 1 and state.word_buffer:
            sbe = subord_word_end_bias(
                state.subord_depth,
                state.subord_words_since_open,
                state.letter_run_len,
                state.word_buffer,
                state.on_word_trie,
                state.chars_since_sentence_end,
                state.speaker_label_state,
            )
            if sbe is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += sbe[i]

        # Layer 4b1: word-repetition bias. Shakespeare's emotional
        # climaxes chain identical words with comma separators
        # ("Never, never, never, never, never!"; "O, O, O, O!";
        # "Why, why, why, why?"; "No, no, no, no!"). We trigger only
        # when the two previous words are identical AND the recent
        # char context shows a comma-space separator (so we're in a
        # repetition rhythm, not a random accidental repeat).
        if (
            state.last_completed_word
            and state.last_completed_word == state.prev_completed_word
            and state.prev_char_class == 7  # PUNCT_MID before the space
        ):
            w = state.last_completed_word
            if w and w[0] in VOCAB_INDEX:
                logits[VOCAB_INDEX[w[0]]] += 2.4
                up = w[0].upper()
                if up != w[0] and up in VOCAB_INDEX:
                    logits[VOCAB_INDEX[up]] += 1.4

        # Layer 4b1a: verb-overdue bias. When the clause has a subject
        # but no verb yet, and we're at a word-start position, bias
        # toward verb/aux/modal-starter letters.
        if (
            state.clause_slot == 1  # HAS_SUBJ
            and state.speaker_label_state == 0
            and state.words_since_verb >= 1
        ):
            wsv = min(state.words_since_verb, 4)
            vb_scale = 0.05 + 0.10 * min(wsv, 3)
            # Verb-starter first letters (aux, modal, main).
            verb_letters: dict[str, float] = {
                "h": 1.0,  # have/hath/hast/had
                "a": 0.9,  # am/art/are
                "i": 0.8,  # is
                "w": 1.0,  # will/would/was/were
                "d": 0.8,  # do/did/dost/doth/didst
                "s": 0.7,  # shall/should/shalt/shouldst
                "c": 0.6,  # can/canst/could/couldst
                "m": 0.6,  # may/mayst/might/must
                "b": 0.5,  # be/been/being
                "g": 0.4,  # go/gave/got
                "t": 0.4,  # take/tell/think
                "l": 0.3,  # look/let/love/live
                "k": 0.3,  # know/kill
                "f": 0.3,  # feel/find/fall/fight
            }
            for ch, lean in verb_letters.items():
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] += vb_scale * lean
                up = ch.upper()
                if up in VOCAB_INDEX:
                    logits[VOCAB_INDEX[up]] += vb_scale * lean * 0.6

        # Layer 4b1b: vocative-noun first-letter bias when the state
        # machine has detected a vocative-construction lead-in (e.g.
        # "my dear ___" / "O sweet ___"). Only active at word-start
        # outside speaker-label territory.
        if state.vocative_expectation and state.speaker_label_state == 0:
            for i in range(VOCAB_SIZE):
                logits[i] += VOCATIVE_START_BIAS[i]
            # Layer 4b1c: addressee-memory bias — if we've already
            # committed to a specific vocative noun this turn, bias
            # strongly toward repeating it.
            ab = addressee_start_bias(
                state.last_vocative,
                state.turn_vocative_count,
                state.vocative_expectation,
                state.speaker_label_state,
            )
            if ab is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += ab[i]

        # Layer 4b3: tonal-texture bias — at word-start outside speaker
        # labels, nudge first-letter choice toward the lexicon consistent
        # with the rolling tonal_weight (dark vs light register). This
        # is a flow-texture layer: lexical coherence bleeds across word
        # boundaries. Scale is proportional to |tonal_weight|.
        if state.speaker_label_state == 0:
            tw = state.tonal_weight
            if tw != 0.0:
                tb = tonal_start_bias(tw)
                if tb is not None:
                    for i in range(VOCAB_SIZE):
                        logits[i] += tb[i]

        # Layer 4b3a: invocation-mode mid-sentence word-start bias.
        # When the speaker is in declamatory voice, boost vocative-lead
        # modifier starters (my/thy/good/sweet/noble/dear/fair/holy).
        # Distinct from sentence-start invocation bias (which boosts
        # opening capitals like O/Alas/Hark); this fires at every
        # word-start within the invocation passage.
        iwb = invocation_word_start_bias(
            state.invocation_mode, state.speaker_label_state
        )
        if iwb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += iwb[i]

        # Layer 4b3b: imagery-density bias — at word-start outside
        # speaker labels, when imagery_density has risen above baseline,
        # nudge next-word first letter toward concrete/sensory starter
        # letters (b=blood/body/blade, e=eye/ear/earth, s=sword/sun/
        # star, h=hand/heart/head, f=face/fire/flower, r=rose/river,
        # t=tear/tongue/throne, c=cheek/chain/cloud, m=moon/mouth, ...).
        # Orthogonal to tonal_weight (valence) and topic clusters
        # (categorical) — this is a continuous imagistic-texture axis.
        if state.speaker_label_state == 0 and state.imagery_density > 0.0:
            ib = imagery_start_bias(state.imagery_density)
            if ib is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += ib[i]

        # Layer 4b3c: scene-topic cluster bias. When one of the 8
        # semantic clusters (war/love/death/royalty/nature/body/faith/
        # fortune) has been clearly activated by recent content words
        # (dominance threshold + margin), bias the next word's first
        # letter toward that cluster's characteristic starter letters.
        # Orthogonal to tonal_weight (valence) and imagery_density
        # (concreteness): this captures *categorical topic*.
        stb = scene_topic_start_bias(
            state.scene_topics, state.speaker_label_state
        )
        if stb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += stb[i]

        # Layer 4b3c+: doubt-register word-start bias. When the rolling
        # doubt_register float has drifted into doubt (+) or assertion
        # (-) territory, nudge first letter toward the matching lexical
        # family (perhaps/may/methinks vs verily/surely/indeed).
        dwsb = doubt_word_start_bias(
            state.doubt_register, state.speaker_label_state
        )
        if dwsb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += dwsb[i]

        # Layer 4b3c-neg: negation-scope word-start bias. When a
        # negation-class word (not/no/nor/never/neither/...) fired
        # recently inside this sentence, boost "n"/"b"/"y" starters
        # for characteristic Shakespearean continuations:
        #   "nor X nor Y", "not X but Y", "neither X nor Y",
        #   "never X, never Y". The specific last_negation_word
        #   targets the strongest of these patterns ("neither" →
        #   almost-certain "nor" next).
        nsb = negation_start_bias(
            state.negation_count,
            state.words_since_negation,
            state.last_negation_word,
            state.speaker_label_state,
        )
        if nsb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += nsb[i]

        # Layer 4b3c-case: pronoun case-slot word-start bias. After a
        # preposition or transitive verb (CASE_OBJ) or at clause start
        # (CASE_SUBJ), bias pronoun-starter letters toward the expected
        # case: subject (I, h-e, s-he, t-hou/they, w-e, y-e) vs
        # object (m-e, t-hee/them, h-im/her, u-s, y-ou). Signals are
        # modest because many of these letters open non-pronoun words
        # too; the relative lift is what matters.
        csb = case_slot_start_bias(
            state.case_slot,
            state.case_wait_words,
            state.speaker_label_state,
        )
        if csb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += csb[i]

        # Layer 4b3c-lament: lament-register word-start bias. When the
        # rolling grief texture (lament_register) is high (>= 0.35),
        # boost first letters of the grief lexicon (woe/sorrow/grief/
        # alas/tears/mourn/pity/heart/heavy/death/dread/lament). This
        # is a Tier 3 flow axis — distinct from tonal_weight (valence)
        # by targeting the *specific* lexicon of mourning rather than
        # generic dark words. At sentence-start, additionally lift
        # "O"/"A"/"W" (the apostrophe of grief: "O woe!", "Alas!",
        # "Woe is me").
        lsb = lament_start_bias(
            state.lament_register,
            state.speaker_label_state,
        )
        if lsb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += lsb[i]

        # Sentence-first-letter extra boost for "O"/"A"/"W" capitals
        # when lament is active.
        if (
            state.words_in_sentence == 0
            and not state.word_buffer
            and state.letter_run_len == 0
        ):
            lssb = lament_sentence_start_bias(
                state.lament_register,
                state.speaker_label_state,
            )
            if lssb is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += lssb[i]

        # Layer 4b3c-tender: tenderness-register word-start bias. The
        # complementary pole to lament: love/sweet/fair/dear/gentle
        # lexicon. A Tier 3 flow axis distinct from tonal_weight
        # (generic valence), imagery_density (concreteness), and
        # lament_register (grief). Samples in Shakespeare's romantic
        # scenes want "my dear lady", "sweet love", "fair flower".
        tsb = tenderness_start_bias(
            state.tenderness_register,
            state.speaker_label_state,
        )
        if tsb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += tsb[i]

        if (
            state.words_in_sentence == 0
            and not state.word_buffer
            and state.letter_run_len == 0
        ):
            tssb = tenderness_sentence_start_bias(
                state.tenderness_register,
                state.speaker_label_state,
            )
            if tssb is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += tssb[i]

        # Layer 4b3c-gravitas: gravitas-register word-start bias. A
        # Tier 3 flow axis capturing moral / philosophical / cosmic
        # weight — distinct from lament (grief), tenderness (love),
        # and tonal_weight (dark/light). Rises on abstract-moral
        # lexicon (honour, virtue, soul, duty, conscience, heaven,
        # fate) and decays otherwise. When register is high, boosts
        # word-start letters for the same cloud (v, j, h, e, c, g,
        # f, s, d, p, m, r, n). Shakespeare's soliloquies — Hamlet,
        # Lear, Brutus — thicken this register and pull content
        # words out of the abstract-philosophical cloud.
        gsb = gravitas_start_bias(
            state.gravitas_register,
            state.speaker_label_state,
        )
        if gsb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += gsb[i]

        # Layer 4-FURY: rage/wrath texture register.
        # Tier-3 flow field tracking angry speech TOWARD another
        # (curses, threats, invective). Distinct from gravitas
        # (controlled), lament (passive-grieving), tonal (event-driven).
        # Rises on rage/curse/insult lexicon and "!" amplification;
        # decays quickly; mostly reset on turn boundary.
        # At word-start, boosts fury-cluster letter starters
        # (d/h/w/v/c/r/f/p) and mildly discourages tender ones (l/g).
        fsb = fury_start_bias(
            state.fury_register,
            state.letter_run_len,
            state.speaker_label_state,
        )
        if fsb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += fsb[i]

        # At word-end with fury high, bias sentence termination
        # toward "!" over "." / ";".
        feb = fury_end_bias(
            state.fury_register,
            state.letter_run_len,
            state.word_buffer,
            state.speaker_label_state,
            state.words_in_sentence,
        )
        if feb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += feb[i]

        if (
            state.words_in_sentence == 0
            and not state.word_buffer
            and state.letter_run_len == 0
        ):
            gssb = gravitas_sentence_start_bias(
                state.gravitas_register,
                state.speaker_label_state,
            )
            if gssb is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += gssb[i]

        # Layer 4b3c-vcc: verb-complement class word-start bias. When
        # the last completed verb set an expectation (that-clause,
        # preposition, past-participle, infinitive, predicate), nudge
        # the NEXT word's first letter toward the expected complement
        # opener. Complementary to the existing pos_next / pos_bigram
        # biases, which condition only on POS — this conditions on
        # a specific verb-class-driven slot with per-slot letter
        # inventories (e.g., "hath" → s/b/g/t [seen, been, gone,
        # taken]; "shall" → b/s/h/g [be, see, have, go]; "said" →
        # t/i/w [that, I, what]).
        vcb = verb_complement_start_bias(
            state.verb_complement_class,
            state.vcc_wait_words,
            state.speaker_label_state,
            state.letter_run_len,
            state.word_buffer,
        )
        if vcb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += vcb[i]

        # Layer 4b3c-drift: scene-drift recovery. When 2+ consecutive
        # words have completed OFF the word-trie, we're in a runaway
        # letter-ngram gibberish regime. Apply a recovery bias pulling
        # word-starts toward common English starters (t/a/i/o/h/w/b/s/
        # f/m) and away from rare letters (x/z/j/q). This is a
        # STRUCTURAL safety net — a different *kind* of signal from
        # the additive lexicon biases: it fires only on detected
        # quality failure and its strength scales with streak length.
        dr = drift_recovery_bias(
            state.drift_streak,
            state.speaker_label_state,
            state.words_in_sentence,
            state.consecutive_newlines,
        )
        if dr is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += dr[i]

        # Layer 4b3b2: 2nd-person addressing-register word-start bias.
        # When the register is established (|register| > 0.5), nudge
        # first letter toward the matching pronoun series (t vs y).
        if state.speaker_label_state == 0:
            ar = state.addressing_register
            if abs(ar) > 0.5:
                asb = address_start_bias(ar)
                if asb is not None:
                    for i in range(VOCAB_SIZE):
                        logits[i] += asb[i]

        # Layer 4b3-iamb: iambic-foot stress bias. In confident
        # pentameter verse, bias the first letter of the next word
        # toward function-word starters (if next syllable falls on
        # an unstressed beat) or content-word starters (if stressed).
        # A genuinely metrical signal — orthogonal to pos_next / topic.
        ib = iambic_word_start_bias(
            state.syllables_in_line,
            state.verse_score,
            state.verse_line_run,
            state.prev_line_syllables,
            state.speaker_label_state,
        )
        if ib is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += ib[i]

        # Layer 4b3c: formulaic-phrase word-start bias. When we're
        # inside a known multi-word formula, boost first letters of
        # expected next words. This is a real multi-word lookahead
        # that sits above phrase_bigram's two-word window.
        if state.speaker_label_state == 0 and state.formula_node != 0:
            fsb = formula_start_bias(state.formula_node)
            if fsb is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += fsb[i]


        # Layer 4b4: topic bias — at word-start outside speaker labels,
        # use the rolling content_words tuple (last 4 content words) to
        # detect a dominant topical cluster (dark/war/death vs love/
        # tender vs royal/court) and tilt next-word first-letter choice
        # toward the cluster's starter lexicon. This sits alongside
        # tonal_weight (which is a single dark/light scalar) but reads
        # real words instead of a smoothed score — giving orthogonal
        # topical signal that tonal_weight's decay can't see.
        if state.speaker_label_state == 0 and state.content_words:
            tpc = topic_bias(state.content_words)
            if tpc is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += tpc[i]

        # Layer 4b4b: anti-repetition — penalize first letters of
        # non-exempt words that have already appeared in this clause.
        # Breaks echo-loop pathology ("there there there").
        if state.speaker_label_state == 0 and state.recent_clause_words:
            rpb = repetition_start_bias(state.recent_clause_words)
            if rpb is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += rpb[i]

        # Layer 4b4c-trans: verb-transitivity word-start bias. When a
        # transitive or linking verb just completed, strongly push the
        # next word's first letter toward determiner/possessive/noun
        # starters (for VT_DO_EXPECTED) or adjective starters (for
        # VT_COMP_EXPECTED). Complements np_open (which only sees
        # article/possessive/preposition openers) by catching the
        # more-common V+NP pattern where the determiner itself is the
        # expected next word.
        if state.speaker_label_state == 0:
            trb = transitivity_start_bias(
                state.verb_transitivity,
                state.vt_wait_words,
                state.speaker_label_state,
            )
            if trb is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += trb[i]

        # Layer 4b4c-wfe: word-form expectation. After an infinitive-
        # marker / modal, a perfect-auxiliary, a copula, "of", "more/
        # less", or "most", the morphological form of the next word
        # is constrained. Boost first letters most common for words
        # of the expected form. This is a morphology-aware prior
        # orthogonal to transitivity (object-slot) and np_open
        # (NP-head slot).
        if state.speaker_label_state == 0:
            wfb = word_form_start_bias(
                state.word_form_expectation,
                state.wfe_wait_words,
                state.speaker_label_state,
            )
            if wfb is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += wfb[i]

        # Layer 4b4c-voc: verb-object semantic-class bias. After a
        # recent main verb, bias first-letters toward objects that
        # are semantically compatible with the verb's class:
        # VIOLENCE → person-pronoun starters; BE_EXIST → article/
        # possessive starters; MOTION → prepositions; etc. This is a
        # semantic nudge, not a form constraint.
        if state.speaker_label_state == 0:
            voc = verb_object_class_start_bias(
                state.verb_class,
                state.vc_wait_words,
                state.speaker_label_state,
            )
            if voc is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += voc[i]

        # Layer 4b2-par: parallel-structure bias after a coordinating
        # conjunction. When last_completed_word is "and"/"or"/"nor"/
        # "but"/"yet", bias next-word first letters toward the POS
        # family of the word BEFORE the conjunction (prev_word_pos).
        if state.speaker_label_state == 0:
            parb = parallel_start_bias(
                state.last_completed_word,
                state.last_word_pos,
                state.prev_word_pos,
                state.speaker_label_state,
            )
            if parb is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += parb[i]

        # Layer 4b4b-allit: within-line alliteration boost. When the
        # last 2+ content words on this line all started with the
        # same letter, nudge the next word's first letter toward the
        # same character. Function-word transparency is handled in
        # the pipeline stage, so "full fathom of the five" still
        # reads as run=3 on 'f'.
        if state.speaker_label_state == 0 and state.line_alliteration_run >= 2:
            ab = alliteration_start_bias(
                state.line_alliteration_letter,
                state.line_alliteration_run,
                state.speaker_label_state,
            )
            if ab is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += ab[i]

        # Layer 4b4b-orn: ornament-density word-start bias. Tier 3
        # flow field reads ornament_density and nudges toward noun vs
        # adjective starters based on the ornateness groove.
        if state.speaker_label_state == 0 and state.ornament_density >= 0.10:
            ob = ornament_start_bias(
                state.ornament_density,
                state.np_open,
                state.speaker_label_state,
            )
            if ob is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += ob[i]

        # Layer 4b4c: NP-head expectation. When np_open is True we're
        # waiting for a head noun to close the current noun phrase
        # (opened by an article/possessive/preposition). Boost noun/
        # adjective first letters; penalize re-opener function-word
        # first letters.
        if state.np_open and state.speaker_label_state == 0:
            nhb = np_head_start_bias(
                state.np_open,
                state.np_wait_words,
                state.speaker_label_state,
                state.last_word_pos,
            )
            if nhb is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += nhb[i]

        # Layer 4b2: phrase bigram — given the previous TWO completed
        # words, bias next word's first letter for known 3-word formulas.
        if state.prev_completed_word and state.last_completed_word:
            pb = phrase_bigram_bias(
                state.prev_completed_word, state.last_completed_word
            )
            if pb is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += pb[i]

        # Layer 4b2-tri: phrase trigram — given the THREE most recent
        # completed words, bias next-word first-letter for known 4-word
        # formulas ("I pray thee tell", "to be or not", "or not to be",
        # "by my troth I", "now is the winter", "in the name of", ...).
        # Strictly more specific than phrase_bigram — fires rarely
        # but with high confidence when it does.
        if (
            state.prev_prev_completed_word
            and state.prev_completed_word
            and state.last_completed_word
        ):
            pt = phrase_trigram_bias(
                state.prev_prev_completed_word,
                state.prev_completed_word,
                state.last_completed_word,
            )
            if pt is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += pt[i]

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
            # Invocation-mode sentence-start: when the speaker has been
            # in declamatory voice, tilt first letter toward canonical
            # invocation openers (O/Alas/Hark/Lo/What/Why/Behold).
            isb = invocation_sentence_start_bias(
                state.invocation_mode, state.speaker_label_state
            )
            if isb is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += isb[i]
            # Cross-sentence opener bias: condition on previous
            # sentence's type (INTERROG → response openers; EXCLAM →
            # momentum openers).
            nsb = next_sentence_start_bias(
                state.prev_sentence_type, state.speaker_label_state
            )
            if nsb is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += nsb[i]
            # Sentence-level anaphora: if the previous sentence opened
            # with a chain-worthy word (And/When/Let/O/I/My/To/...),
            # boost that same first letter at this sentence's start,
            # scaled by the run length.
            sab = sentence_anaphora_start_bias(
                state.prev_sentence_first_word,
                state.sentence_anaphora_run,
                state.speaker_label_state,
            )
            if sab is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += sab[i]
            # Turn-emphasis sentence-start bias: when the current turn
            # has shown an "!"-heavy or "?"-heavy texture, tilt the
            # first letter of the new sentence toward canonical
            # exclamative openers (O/A/H) or WH-openers. Scaled down
            # because sentence-start is already heavily over-determined.
            tpt_ss = turn_punct_texture_sentence_start_bias(
                state.turn_exclam_count,
                state.turn_question_count,
                state.sentences_in_turn,
                state.speaker_label_state,
            )
            if tpt_ss is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += tpt_ss[i] * 0.30

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
        # Enjambment: prev line wrapped with a letter (mid-word/mid-phrase)
        # and was prose-length (>= 50 chars). Here the next line almost
        # always begins LOWERCASE — a continuation of the same clause,
        # e.g. "considering\nhow honour would become".
        is_enjambed = (
            on_verse_line_start
            and state.prev_line_final_class == 3
            and state.prev_line_length >= 50
        )
        if on_verse_line_start and not is_enjambed:
            for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] += 3.0
            for ch in "abcdefghijklmnopqrstuvwxyz":
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] -= 1.2
        elif is_enjambed:
            # Invert: favor lowercase continuation, penalize capitals.
            for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] -= 1.2
            for ch in "abcdefghijklmnopqrstuvwxyz":
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] += 0.6

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

        # Layer 4b5: line-starter anaphora bias. At verse-line-start
        # (NOT sentence-start — anaphora is about repeated line
        # openings within the flow, not about fresh-sentence capital
        # bias), boost the first letter shared across recent line-
        # starters. Skip post-label starts: dialogue openings aren't
        # anaphoric with the prior speaker's lines.
        if (
            on_verse_line_start
            and state.speaker_label_state == 0
            and state.recent_line_starters
        ):
            ab = anaphora_start_bias(state.recent_line_starters)
            if ab is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += ab[i]
        # Additionally, at BOTH sentence-start and verse-line-start,
        # bias specific common starting capitals: T, A, W, I, O, B, H,
        # S, M, N, F, C, L, P, G, D, R, Y. Skip speaker-label context.
        if (is_sentence_start or (on_verse_line_start and not is_enjambed)) and state.speaker_label_state == 0:
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

        # Layer 4b6: turn-opener bias. At the very first word of a
        # speaker's turn (words_in_turn == 0 AND sentences_in_turn == 0),
        # tilt first-letter choice toward letters that disproportionately
        # open Shakespearean turns (O/Alas/Nay/Why/My/Pray/Good/...). This
        # fires on top of the generic sentence-start / line-start capital
        # boosts and reshapes WITHIN the capitals. Guarded to a genuine
        # word-start position outside speaker-label territory.
        if (
            state.speaker_label_state == 0
            and state.words_in_turn == 0
            and state.sentences_in_turn == 0
            and state.lines_in_turn <= 1
        ):
            for i in range(VOCAB_SIZE):
                logits[i] += TURN_OPENER_START_BIAS[i]

        # Layer 4b6b: cross-turn answer-opener bias. If the PRIOR
        # speaker's turn ended with "?", "!", or imperative, shape
        # this speaker's first-word first-letter toward answer /
        # reaction openers. Complements the generic turn-opener
        # bias with cross-turn semantic awareness.
        aob = answer_opener_start_bias(
            state.prev_turn_final_sent_type,
            state.words_in_turn,
            state.sentences_in_turn,
            state.speaker_label_state,
            state.letter_run_len,
            state.word_buffer,
        )
        if aob is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += aob[i]

        # Layer 4b6bb: WH-class-specific answer expectation. When the
        # pipeline identified the prior turn's closing "?" as a
        # specific WH-class (where / why / how / ...), this bias
        # targets the new turn's first letter with a class-specific
        # table that is tighter than the generic answer_opener one.
        aeb = answer_expectation_start_bias(
            state.pending_question_type,
            state.words_in_turn,
            state.sentences_in_turn,
            state.speaker_label_state,
            state.letter_run_len,
            state.word_buffer,
        )
        if aeb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += aeb[i]

        # Layer 4b6c: dialogue-adjacency amplifier.  Additive with
        # answer_opener; contributes stichomythia / long-prior-turn /
        # declarative-continuation biases that answer_opener doesn't.
        dab = dialogue_adjacency_bias(
            state.prev_turn_final_punct,
            state.prev_turn_word_count,
            state.prev_turn_line_count,
            state.speaker_label_state,
            state.words_in_turn,
            state.sentences_in_turn,
            state.lines_in_turn,
            state.letter_run_len,
            len(state.word_buffer),
            state.turns_closed,
        )
        if dab is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += dab[i]

    # Layer 4b6d: dialogue pacing — stichomythia/monologue modulation
    # of sentence-end-punct preference mid-turn.  Operates at a DIFFERENT
    # gate than the opener: fires at just_finished_word inside any turn
    # after at least one turn has closed.
    dpb = dialogue_pacing_bias(
        state.prev_turn_word_count,
        state.prev_turn_line_count,
        state.speaker_label_state,
        state.words_in_turn,
        state.sentences_in_turn,
        state.lines_in_turn,
        state.just_finished_word,
        state.turns_closed,
        state.last_char,
    )
    if dpb is not None:
        for i in range(VOCAB_SIZE):
            logits[i] += dpb[i]

    # Layer 4d: verb-agreement bias based on subject pronoun.
    # When the clause's subject is "thou", Shakespearean agreement
    # prefers "hast/dost/wilt/shalt/art" over modern "has/does/will/
    # shall/are". Similarly, when subject is "he/she/it", "hath/doth"
    # is preferred over "has/does". We apply a targeted letter-choice
    # bias at specific word-interior positions.
    subj = state.subject_pronoun
    wb = state.word_buffer
    if subj and wb and state.letter_run_len >= 1:
        if subj == "thou":
            # Archaic 2nd-person singular: -st endings strongly preferred.
            if wb == "ha":
                logits[VOCAB_INDEX["s"]] += 1.5  # hast
                logits[VOCAB_INDEX["v"]] -= 0.8  # have (modern)
                logits[VOCAB_INDEX["d"]] += 0.2
            elif wb == "has":
                logits[VOCAB_INDEX["t"]] += 1.8  # hast (complete -st)
            elif wb == "wil":
                logits[VOCAB_INDEX["t"]] += 1.5  # wilt
                logits[VOCAB_INDEX["l"]] -= 0.3  # will (modern)
            elif wb == "shal":
                logits[VOCAB_INDEX["t"]] += 1.5  # shalt
                logits[VOCAB_INDEX["l"]] -= 0.3
            elif wb == "ar":
                logits[VOCAB_INDEX["t"]] += 1.2  # art
                logits[VOCAB_INDEX["e"]] -= 0.4  # are
            elif wb == "do":
                logits[VOCAB_INDEX["s"]] += 1.0  # dost
                logits[VOCAB_INDEX["t"]] += 1.0  # doth
                logits[VOCAB_INDEX["e"]] -= 0.6  # does
            elif wb == "dos":
                logits[VOCAB_INDEX["t"]] += 1.8  # dost
            elif wb == "can":
                logits[VOCAB_INDEX["s"]] += 1.2  # canst
                logits[VOCAB_INDEX["t"]] -= 0.3  # can't less likely after thou
            elif wb == "cans":
                logits[VOCAB_INDEX["t"]] += 1.8  # canst
            elif wb == "di":
                logits[VOCAB_INDEX["d"]] += 0.4  # didst
            elif wb == "did":
                logits[VOCAB_INDEX["s"]] += 1.3  # didst
            elif wb == "dids":
                logits[VOCAB_INDEX["t"]] += 1.8  # didst
            elif wb == "ma":
                logits[VOCAB_INDEX["y"]] += 0.2
            elif wb == "may":
                logits[VOCAB_INDEX["s"]] += 1.1  # mayst
            elif wb == "mays":
                logits[VOCAB_INDEX["t"]] += 1.8  # mayst
            elif wb == "mig":
                logits[VOCAB_INDEX["h"]] += 0.2
            elif wb == "might":
                logits[VOCAB_INDEX["s"]] += 0.9  # mightst (rare)
            elif wb == "woul":
                logits[VOCAB_INDEX["d"]] += 0.5
            elif wb == "would":
                logits[VOCAB_INDEX["s"]] += 1.3  # wouldst
            elif wb == "woulds":
                logits[VOCAB_INDEX["t"]] += 1.8
            elif wb == "coul":
                logits[VOCAB_INDEX["d"]] += 0.5
            elif wb == "could":
                logits[VOCAB_INDEX["s"]] += 1.3  # couldst
            elif wb == "coulds":
                logits[VOCAB_INDEX["t"]] += 1.8
            elif wb == "shoul":
                logits[VOCAB_INDEX["d"]] += 0.5
            elif wb == "should":
                logits[VOCAB_INDEX["s"]] += 1.3  # shouldst
            elif wb == "shoulds":
                logits[VOCAB_INDEX["t"]] += 1.8
            elif wb == "wer":
                logits[VOCAB_INDEX["t"]] += 1.1  # wert (archaic)
                logits[VOCAB_INDEX["e"]] -= 0.3
        elif subj in ("he", "she", "it"):
            # 3rd singular: "hath/doth" archaic > "has/does" modern.
            if wb == "ha":
                logits[VOCAB_INDEX["t"]] += 0.9  # hath
                logits[VOCAB_INDEX["s"]] -= 0.2  # has
                logits[VOCAB_INDEX["v"]] -= 0.4  # have (agreement error)
                logits[VOCAB_INDEX["d"]] += 0.2
            elif wb == "hat":
                logits[VOCAB_INDEX["h"]] += 1.0  # hath
            elif wb == "do":
                logits[VOCAB_INDEX["t"]] += 0.8  # doth
                logits[VOCAB_INDEX["e"]] -= 0.4  # does
            elif wb == "dot":
                logits[VOCAB_INDEX["h"]] += 1.4  # doth

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

    # Line-break propriety bias: suppress \n at grammatically-bad
    # positions (mid-word, open NP, no verb yet) and mildly boost it
    # at clause-break positions. Gated to verse contexts.
    lbb = line_break_newline_bias(
        state.line_break_propriety,
        state.verse_score,
        state.chars_since_newline,
        state.speaker_label_state,
        state.in_prose_line,
    )
    if lbb is not None:
        for i in range(VOCAB_SIZE):
            logits[i] += lbb[i]

    # Parenthetical-dash aside bias: right after "--", strongly bias
    # toward newline/space + discourse-opener caps. And after 3+ words
    # inside an unclosed aside, boost "-" after space to set up "--"
    # closure. Reads state.in_dash_aside (set by pipeline/dash_aside.py).
    daob = dash_aside_open_bias(
        state.in_dash_aside,
        state.chars_since_dash_open,
        state.speaker_label_state,
    )
    if daob is not None:
        for i in range(VOCAB_SIZE):
            logits[i] += daob[i]
    dacb = dash_aside_close_bias(
        state.in_dash_aside,
        state.words_since_dash_open,
        state.letter_run_len,
        state.last_char,
        state.speaker_label_state,
    )
    if dacb is not None:
        for i in range(VOCAB_SIZE):
            logits[i] += dacb[i]


    # After apostrophe, the bias depends on what preceded the apostrophe:
    # - If preceded by a letter: it's a contraction ('s, 'd, 't, 'll, 've)
    #   or possessive. Boost common contraction letters.
    # - If preceded by space/newline: it's opening a quoted phrase (e.g.
    #   'Thanks', 'God save...', 'tis, 'gainst, 'Tis). Boost capitals +
    #   archaic contraction leaders (t, g).
    if last_cls == APOSTROPHE:
        prev_cls = state.prev_char_class
        # SPACE == 2, NEWLINE == 3 — can't import constants here without
        # existing import; use direct numeric comparison with class enum.
        opened_quote = prev_cls in (SPACE, NEWLINE) or state.tokens_seen == 1
        # Closing-quote contexts: after . ? ! , ; — the apostrophe
        # terminates a quoted phrase; expect space/newline next.
        closed_quote = prev_cls in (6, 7)  # PUNCT_END, PUNCT_MID
        if closed_quote:
            logits[VOCAB_INDEX[" "]] += 7.0
            logits[VOCAB_INDEX["\n"]] += 5.0
            for ch in "abcdefghijklmnopqrstuvwxyz":
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] -= 4.0
            for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] -= 4.0
        if opened_quote:
            # Quote-opening context — bias capitals and 't / 'g
            # (archaic contraction leaders like 'tis, 'twas, 'gainst).
            for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] += 5.0
            # Archaic contraction leaders (lowercase): 'tis, 'twas, 'gainst.
            for ch, boost in (("t", 4.0), ("g", 1.3)):
                logits[VOCAB_INDEX[ch]] += boost
        else:
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
        state.vowels_since_consonant >= 2
        and state.word_buffer
        and state.speaker_label_state == 0
    ):
        v = state.vowels_since_consonant
        if v == 2:
            # Pre-emptive: discourage a 3rd consecutive vowel.
            vpen = 0.85
            cbump = 0.30
            tbump = 0.20
        elif v == 3:
            vpen = 1.5
            cbump = 0.5
            tbump = 0.6
        elif v == 4:
            vpen = 2.6
            cbump = 1.1
            tbump = 1.3
        else:
            vpen = 3.6
            cbump = 1.9
            tbump = 1.9
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
        # prose ~60-80. verse_score in [-3, 3]: positive = verse passage
        # (shorter lines, earlier newline); negative = prose (longer lines,
        # later newline). Adjust the per-bucket bumps accordingly.
        if (
            (state.letter_run_len >= 2 and state.on_word_trie
                and state.word_buffer in COMPLETE_WORDS)
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
        elif st_type == 4:  # IMPERATIVE
            # Bare-verb imperative: "Come!", "Speak, friends.", "Hark
            # thee!". Exclamation is elevated; period stays strong
            # (reflective imperatives close with "."); question mark
            # is suppressed. Conservative shift — the cost of an
            # occasional mis-tag on a declarative is larger than the
            # gain on a correct imperative.
            ratio_period, ratio_q, ratio_excl = 0.80, 0.15, 0.65
        else:  # DECL or UNKNOWN: original shape
            ratio_period, ratio_q, ratio_excl = 1.0, 0.3, 0.3

        # Emotional-intensity modulation: when emo is high, shift
        # end-punct mass from "." toward "!". Proportional to emo so
        # modern/neutral registers are unaffected. Applied as a soft
        # ratio tilt within the calibrated total-mass budget.
        emo = state.emotional_intensity
        if emo > 0.0 and st_type != 2:  # don't override interrogatives
            shift_p = min(0.95 * emo, 0.95)
            ratio_excl += ratio_period * shift_p
            ratio_period *= (1.0 - shift_p)

        # Invocation-mode end-punctuation shift: when the speaker has
        # been in declamatory voice, tilt sentence-end mass toward "!"
        # at the expense of ".". This is a softer effect than the
        # emotional-intensity shift (which spikes on short bursts).
        # Invocation-mode !/. shift — disabled; net-negative on BPC
        # in isolation. The sentence-start capital-letter bias below
        # carries the signal.
        # inv = state.invocation_mode
        # if inv > 0.2 and st_type != 2:
        #     shift_i = min(0.08 * inv, 0.08)
        #     ratio_excl += ratio_period * shift_i
        #     ratio_period *= (1.0 - shift_i)

        # Turn-emphasis ratio shift: when the current speaker turn has
        # an established "!"-heavy or "?"-heavy punctuation texture
        # (e.g. a raging speaker vs an interrogating speaker), shift
        # end-punct mass toward the turn's dominant mode. Conservative:
        # only fires at sentences_in_turn >= 2 and scales with excess
        # over baseline proportions. This is a TURN-level texture
        # signal, distinct from emotional_intensity (short-burst) and
        # invocation_mode (decay-based declamatory register).
        _tsit = state.sentences_in_turn
        if _tsit >= 2 and st_type != 2:
            _excl_frac = state.turn_exclam_count / _tsit
            _ques_frac = state.turn_question_count / _tsit
            _excl_excess = max(0.0, _excl_frac - 0.15)
            _ques_excess = max(0.0, _ques_frac - 0.12)
            if _tsit == 2:
                _tconf = 0.50
            elif _tsit == 3:
                _tconf = 0.75
            elif _tsit == 4:
                _tconf = 0.90
            else:
                _tconf = 1.0
            if _excl_excess > 0.0:
                _shift_e = min(0.70 * _excl_excess * _tconf, 0.65)
                ratio_excl += ratio_period * _shift_e
                ratio_period *= (1.0 - _shift_e)
            if _ques_excess > 0.0:
                _shift_q = min(0.65 * _ques_excess * _tconf, 0.65)
                ratio_q += ratio_period * _shift_q
                ratio_period *= (1.0 - _shift_q)

        # Overdue sentence end: at word-end on-trie, boost sentence-end
        # punctuation so the model actually closes sentences. The
        # clause_slot state machine (FRESH=0, HAS_SUBJ=1, HAS_VERB=2,
        # POST_OBJ=3) modulates this: a clause that hasn't yet seen
        # its verb is syntactically unfinished and shouldn't close.
        if (
            (state.letter_run_len >= 2 and state.on_word_trie
                and state.word_buffer in COMPLETE_WORDS)
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
            # Clause-slot modulation: commas are more natural at
            # post-object positions (end of phrase) than immediately
            # after a fresh-subject position.
            slot = state.clause_slot
            if slot == 3:      # POST_OBJ
                slot_mul = 1.15
            elif slot == 2:    # HAS_VERB
                slot_mul = 1.05
            else:              # HAS_SUBJ / FRESH
                slot_mul = 0.85
            if "," in VOCAB_INDEX:
                logits[VOCAB_INDEX[","]] += 4.5 * slot_mul
            if ";" in VOCAB_INDEX:
                logits[VOCAB_INDEX[";"]] += 1.9 * slot_mul
            # List-parallelism extra comma boost: when we already have
            # 1+ commas and an alliterative / POS-parallel pattern is
            # running, keep the list chaining.
            lwc = list_wordend_comma_bias(
                state.commas_since_sent_end,
                state.list_parallel_run,
                state.list_first_item_pos,
                state.chars_since_sentence_end,
                state.speaker_label_state,
            )
            if lwc > 0.0:
                logits[VOCAB_INDEX[","]] += lwc
        # Also off-trie with a longer min-length: archaic/proper words
        # can certainly be followed by ",".
        elif (
            state.letter_run_len >= 4
            and not state.on_word_trie
            and state.chars_since_sentence_end < 40
        ):
            if "," in VOCAB_INDEX:
                logits[VOCAB_INDEX[","]] += 3.0
            if ";" in VOCAB_INDEX:
                logits[VOCAB_INDEX[";"]] += 1.3

        # (An imperative-first-word comma/! boost was tried here and
        # net-regressed BPC. The sentence_type end-punct-ratio branch
        # carries the imperative signal by itself; a direct boost at
        # word-end over-fires on declaratives that happen to start with
        # the same verb.)


        # Line-coherence close pressure. Reads line_ontrie_words and
        # line_offtrie_words to decide whether the current line is
        # salvageable. Failing lines (>= 2 garbage words, <= 1 real)
        # get strong newline push; healthy lines (>= 3 real, 0 garbage)
        # get mild anti-newline so a well-formed line can extend.
        lcb = line_coherence_wordend_bias(
            state.line_ontrie_words,
            state.line_offtrie_words,
            state.letter_run_len,
            state.on_word_trie,
            state.speaker_label_state,
            state.word_buffer,
            COMPLETE_WORDS,
            state.chars_since_sentence_end,
            state.consecutive_newlines,
        )
        if lcb is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += lcb[i]

        # Clause-depth close pressure: when we're nested inside a
        # subordinator clause and it's been running for several words,
        # push toward comma/semicolon (and at depth 2+, sentence-end).
        # Fires at word-end-on-trie; gated internally by speaker label
        # and word completeness.
        cd = clause_depth_close_bias(
            state.clause_depth,
            state.words_in_subordinate,
            state.letter_run_len,
            state.on_word_trie,
            state.word_buffer,
            state.speaker_label_state,
            COMPLETE_WORDS,
        )
        if cd is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += cd[i]

        # At word-end-on-trie, space is the most likely next char.
        # Boost it to reflect natural word-break frequency. But only
        # when the buffer is either a complete word OR long enough to
        # be plausibly complete (4+ chars). Short prefixes like "th",
        # "wh", "br", "sh" should NOT yet boost space — they want
        # letter extensions.
        if state.letter_run_len >= 2 and state.on_word_trie:
            if state.word_buffer in COMPLETE_WORDS:
                logits[VOCAB_INDEX[" "]] += -0.10
                # Cadence-texture bias at a genuine word-end. Staccato
                # register boosts commas/semicolons; flowing register
                # boosts space. Scaled by |cadence|.
                cb = cadence_wordend_bias(state.cadence)
                if cb is not None:
                    for i in range(VOCAB_SIZE):
                        logits[i] += cb[i]
                # Pentameter meter-aware newline bias. At word-end in
                # a verse passage, nudge \n when syllables_in_line is
                # at or past the 10-syllable target (matching the
                # prior line's length when established).
                pb = pentameter_wordend_bias(
                    state.syllables_in_line,
                    state.prev_line_syllables,
                    state.verse_score,
                    state.verse_line_run,
                    state.chars_since_newline,
                )
                if pb is not None:
                    for i in range(VOCAB_SIZE):
                        logits[i] += pb[i]
                # Enjambment-density: recent lines enjambed → boost \n
                # direct; end-stopped → boost terminal punct. Fires
                # only in verse, inside the line-end zone.
                eb = enjambment_wordend_bias(
                    state.enjambment_density,
                    state.verse_line_run,
                    state.verse_score,
                    state.speaker_label_state,
                    state.chars_since_newline,
                )
                if eb is not None:
                    for i in range(VOCAB_SIZE):
                        logits[i] += eb[i]
            elif state.letter_run_len >= 5:
                # Non-complete on-trie buffer that's gotten long (5+).
                # Less confident but still plausible to end.
                logits[VOCAB_INDEX[" "]] -= 2.0
            elif state.letter_run_len == 4:
                # Length-4 on-trie non-complete: mid-word, still
                # growing toward a longer word. Penalize space.
                logits[VOCAB_INDEX[" "]] -= 2.0
            else:
                # On-trie short buffer (len 2–3) that is NOT itself a
                # complete word: it's growing toward a longer word.
                # Penalize premature word-terminators. Catches
                # "ti/dou/tram"-style early-space mistakes.
                scale = 1.3 if state.letter_run_len == 2 else 1.0
                logits[VOCAB_INDEX[" "]] -= 2.4 * scale
                if "," in VOCAB_INDEX:
                    logits[VOCAB_INDEX[","]] -= 1.7 * scale
                if "." in VOCAB_INDEX:
                    logits[VOCAB_INDEX["."]] -= 1.7 * scale
                if ";" in VOCAB_INDEX:
                    logits[VOCAB_INDEX[";"]] -= 1.2 * scale
                if "\n" in VOCAB_INDEX:
                    logits[VOCAB_INDEX["\n"]] -= 1.2 * scale
                if "!" in VOCAB_INDEX:
                    logits[VOCAB_INDEX["!"]] -= 1.2 * scale
                if "?" in VOCAB_INDEX:
                    logits[VOCAB_INDEX["?"]] -= 1.2 * scale
                if ":" in VOCAB_INDEX:
                    logits[VOCAB_INDEX[":"]] -= 0.8 * scale
        # For single-letter complete words (A/I/O), space is even more
        # certain — they almost always end right there. BUT: lowercase
        # "i" as a standalone word never occurs in Shakespeare (always
        # "I" capitalized). state.last_char preserves case, so we can
        # distinguish:
        #   - "I", "O", "A" (uppercase, standalone): very likely word.
        #   - "a" (lowercase, as in "a bird"): likely a standalone word.
        #   - "i" (lowercase): almost always the first letter of a
        #     longer word (in/is/if/it/into/...). Do NOT boost space.
        elif (
            state.letter_run_len == 1
            and state.word_buffer in COMPLETE_WORDS
        ):
            if state.last_char == "i":
                # Lowercase "i" standalone is ungrammatical. Penalize
                # space (and any terminator) so we extend to a longer
                # word.
                logits[VOCAB_INDEX[" "]] -= 5.5
                for ch in ",.;:\n!?":
                    if ch in VOCAB_INDEX:
                        logits[VOCAB_INDEX[ch]] -= 3.0
            elif state.last_char == "o":
                # Lowercase "o" standalone is rare (except apostrophized
                # "o'er" / "o'clock" forms which have ' instead of space).
                # Weaker penalty than "i" but still bias toward extending.
                logits[VOCAB_INDEX[" "]] -= 5.0
                for ch in ",.;:\n!?":
                    if ch in VOCAB_INDEX:
                        logits[VOCAB_INDEX[ch]] -= 3.0
            else:
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

    # Final temperature calibration: scale logits by 1/T. Values > 1
    # soften the distribution (reduces over-peaked losses on surprises);
    # values < 1 sharpen. Context-dependent: different regimes have
    # different amounts of layer-bias stacking, so we calibrate per
    # context instead of one-size-fits-all.
    if state.speaker_label_state != 0:
        # Inside speaker label: very narrow distribution, bias sums
        # are meaningful, lighter softening.
        T = 1.02
    elif state.letter_run_len == 0:
        # Word-start: many layer biases stack (startword, next_word,
        # pos_next, context-class, etc.).
        T = 1.40
    elif state.on_word_trie:
        # Mid-word on trie: word_trie bias dominates and is sharp.
        # Modulate by trie_match_count — when only 1 known word
        # matches, our trie is probably wrong a large fraction of the
        # time (the real text has many rare words we don't know);
        # soften there. At broader counts, the bias is averaged
        # across many completions and we can trust it.
        tmc = state.trie_match_count
        if tmc == 1:
            T = 2.20
        elif tmc == 2:
            T = 2.00
        elif tmc <= 4:
            T = 1.80
        elif tmc <= 8:
            T = 1.64
        elif tmc <= 16:
            T = 1.56
        else:
            T = 1.48
    else:
        # Off-trie mid-word: letter n-grams + drift-recovery stack.
        # Higher T than on-trie because many strong negative biases
        # (red_flags, gibberish_hardcap, drift recovery) stack and
        # over-sharpen when the actual next char is a vowel-insert.
        T = 1.70
    if T != 1.0:
        logits = [x / T for x in logits]
    return _log_softmax_smoothed(logits, 0.8e-4)
