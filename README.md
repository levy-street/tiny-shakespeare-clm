# Tiny Shakespeare CLM

A rules-only character-level language model for Shakespeare-style text.

## Generation Samples

These fixed-seed snippets were generated from the current model with
`--temperature 0.6`.

```text
$ python3 harness.py sample --prefix $'HAMLET:\nTo be or not to be, ' --length 180 --temperature 0.6 --seed 0
HAMLET:
To be or not to be, the man is not inter to of the time
shall the feature our somerset in her letting.

OILETITC:
lord listen the earth, and may at hand land hand
Year. Bethought enter him and her hea
```

```text
$ python3 harness.py sample --prefix $'KING HENRY:\nNow is the winter of ' --length 180 --temperature 0.6 --seed 1
KING HENRY:
Now is the winter of france, a thousand a outlive man
in the world verona. Me from the more a sword winter,
Indeed the side heart so and the heart
The hand heart need, for a thousand heart best thou
ha
```

```text
$ python3 harness.py sample --prefix $'FIRST CITIZEN:\n' --length 180 --temperature 0.6 --seed 2
FIRST CITIZEN:
Will the poor trust a man of an hurled the worthless
thou shall the throne shall be the share the no man
until earn ye insist title, enobarbus crest
Of an inter marry.

IACHIMO:
An
```

## Code Policy Optimization

This project does not train a neural network. The model's policy, the function
that maps a rolling character state to a next-character probability
distribution, is written directly in Python.

Optimization happens by changing code instead of updating learned weights. New
state fields, pipeline stages, word-shape rules, syntactic heuristics, semantic
trackers, and logit-bias layers are added under `model_repo/model/`, then
evaluated with the harness using bits per character and fixed-seed text samples.

In other words, the codebase itself is the parameter space. Each improvement is
a policy edit: a change to how the model reads context, updates state, or
chooses the next character distribution.

## Model API

The model lives in `model_repo/model/` and exposes a deliberately small API:

```python
from model import ModelState, advance, predict

state = ModelState()
logprobs = predict(state)
state = advance(state, token_id)
```

`predict(state)` returns natural log-probabilities over the project vocabulary.
`advance(state, token_id)` returns the next immutable `ModelState`. The state is
a frozen Pydantic model, and the update path is a pure pipeline of focused
state-update stages.

## Repository Layout

- `harness.py` - evaluation and sampling CLI for the model.
- `optimizer.py` - long-running agent optimizer that edits `model_repo/model/`.
- `optimizer_prompt.md` - system prompt used by the optimizer.
- `nudges.txt` - live-editable structural prompts injected into optimizer runs.
- `corpus/train.txt` - training corpus used by the harness and vocabulary.
- `model_repo/model/` - model package.
  - `state/schema.py` defines `ModelState`.
  - `advance.py` threads tokens through `pipeline.PIPELINE`.
  - `pipeline/` contains state-update stages.
  - `predict/` contains distribution-bias layers and final composition.
  - `vocab.py` builds the character vocabulary from `corpus/train.txt`.

## Requirements

- Python 3.10+
- `pydantic` for the model state
- `claude_agent_sdk` only if you run `optimizer.py`

There is no packaged dependency file in this repo. For basic harness usage:

```bash
python3 -m pip install pydantic
```

Install the Claude agent SDK separately if you want to run the optimizer.

## Harness Usage

Run commands from the repository root.

Check the model contract:

```bash
python3 harness.py check-distribution
```

Evaluate bits per character on a random training chunk:

```bash
python3 harness.py train-eval-batch --seed 0 --batch-size 2000
```

Evaluate the full training corpus:

```bash
python3 harness.py train-eval-full
```

Generate a sample:

```bash
python3 harness.py sample --prefix $'HAMLET:\nTo be or not to be, ' --length 300 --seed 0
```

View a slice of the training corpus:

```bash
python3 harness.py view-train --offset 0 --length 1000
```

`dev-eval` and `val-eval` are operator-only commands. The dev and validation
splits live outside this directory by design and should not be accessed during
model optimization.

## Model Development

The intended edit surface is `model_repo/model/`.

Common development loop:

1. Run `python3 harness.py check-distribution`.
2. Capture a baseline with `train-eval-batch` across a few seeds.
3. Generate samples with realistic prefixes and inspect quality.
4. Add or adjust state fields, pipeline stages, or predict layers.
5. Re-run the contract check, BPC checks, and samples.

Important invariants:

- `advance()` must not mutate its input state.
- `predict()` must return one log-probability per vocabulary entry.
- `sum(exp(logp))` must stay close to `1.0`.
- Probabilities must remain valid for every state the harness can reach.

## Optimizer

`optimizer.py` starts a long-running Claude agent session with a guard hook that
blocks dev/val access. It streams logs to stdout and writes JSONL logs under
`logs/`.

```bash
python3 optimizer.py
```

The optimizer prompt asks the agent to improve both training BPC and sample
quality, then commit successful improvements. `nudges.txt` can be edited while
the optimizer is running; the process re-reads it before each scheduled nudge.
