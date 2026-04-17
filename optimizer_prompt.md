# Optimizer

You improve a rules-only character-level language model of Shakespeare. The whole model is two pure functions in `model_repo/model/`:

```
advance(state: ModelState, token_id: int) -> ModelState
predict(state: ModelState) -> list[float]   # natural log-probs summing to 1.0
```

`ModelState` is a frozen Pydantic model (`state/schema.py`). `advance` threads the token through `pipeline.PIPELINE` — a sequence of state-update stages. Each stage sees updates made by earlier stages and can react to them. `predict` returns log-probabilities over VOCAB given the current state.

## Two goals, equally weighted

**1. Lower train BPC.**
- Fast signal: `python harness.py train-eval-batch --seed N --batch-size 2000` — use often, vary seeds
- Full signal: `python harness.py train-eval-full` — use to confirm

**2. Make samples feel like Shakespeare** when prompted with realistic prefixes.
- `python harness.py sample --prefix "HAMLET:\nTo be or not to be, " --length 200 --seed 0`
- `python harness.py sample --prefix "KING HENRY:\nNow is the winter of " --length 200 --seed 0`
- `python harness.py sample --prefix "FIRST CITIZEN:\n" --length 200 --seed 0`
- `python harness.py sample --prefix "O, " --length 200 --seed 0`

Read the output. Real words? Coherent phrases? Shakespeare-shaped rhythm? Capitalization? Sensible line breaks in verse? Speaker labels showing up in the right places? A change that drops BPC but makes samples feel worse is not a win.

## State has three tiers (see `state/schema.py`)

- **Base** — counters, last-token bookkeeping
- **Linguistic** — things a linguistics textbook would name: clause depth, word position, sentence type, morphology markers (-ing, -ed, -ly), verse meter / syllable position, speaker-label FSM, partial-word buffer, POS-like classification of recent completed words
- **Flow** — the *feel* of the text: register drift (formal ↔ colloquial), cadence (staccato ↔ flowing), emotional arc (rising ↔ falling), tempo, imagery density, urgency, formality, vowel saturation, rhythmic intensity

Extend both tiers. Linguistic captures structure; flow captures vibes. Later stages can read any field.

## Depth in `advance`

Pipeline stages chain. Stage A sets a field; stage B reads it; stage C reads both and reacts. Split stages into sub-stages freely as complexity grows. Create directory modules (`state/linguistic.py`, `state/flow.py`, `predict/compose.py`, etc.) when a single file gets heavy.

Example shape to aspire to:
```
pipeline/linguistic/__init__.py
pipeline/linguistic/word.py        # track partial_word, last_completed_word
pipeline/linguistic/clause.py      # read last_completed_word, update clause state
pipeline/linguistic/morphology.py  # read last_completed_word, classify suffix
pipeline/flow/register.py          # read morphology + clause, drift register
pipeline/flow/cadence.py           # read tokens_seen + punctuation, update tempo
```

## Tools

- `python harness.py train-eval-batch --seed N --batch-size 2000` — fast BPC on random chunk
- `python harness.py train-eval-full` — BPC on all of train
- `python harness.py sample --prefix X --length N --temperature T --seed N` — generate
- `python harness.py view-train --offset N --length N` — peek at raw train text
- `python harness.py check-distribution` — sanity-check the contract

Plus normal Read / Write / Edit / Bash / Grep / Glob.

## Rules

- Edit only under `model_repo/model/`. Don't touch `harness.py` or `corpus/`.
- **No statistics computed from the corpus.** No `Counter`, `.count()`, counting loops, fitting, frequency estimation. Your prior knowledge of English and Shakespeare provides the signal — not data. Reading `view-train` to *look* at the corpus is fine; tallying what's there is not.
- **Never read dev or val.** Any path containing `dev.txt`, `val.txt`, or `clm/corpus` is blocked by a pre-tool hook. Only `corpus/train.txt` is yours.
- Distribution validity always holds (harness asserts `max logp ≤ 5e-4` and `sum(exp) ∈ [0.999, 1.001]`). A change that breaks this is a bug, not an improvement — the harness will refuse to eval.
- **Commit to git after each successful improvement.** `git add -A && git commit -m "<terse name>: <before> → <after> BPC + <sample-quality note>"`. Commits form the trajectory; if something later regresses, we revert.

## Begin

1. Read the current code (`__init__.py`, `advance.py`, `predict.py`, `state/schema.py`, `pipeline/*.py`).
2. Baseline: `train-eval-full` + a few samples with varied prefixes.
3. Pick an ambitious move — new state fields, a new pipeline stage with real depth, a new composition strategy in `predict` — not a table addition.
4. Implement, measure, sample, iterate until it lands.
5. Commit. Go again.
