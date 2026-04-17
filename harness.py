"""Evaluation harness for the v2 two-function model API.

Subcommands split into two groups:

  AGENT-ACCESSIBLE (the optimizer may call these freely):
      train-eval-batch     BPC on a random chunk of the train corpus.
      train-eval-full      BPC on the full train corpus.
      sample               Generate text by sampling from the model.
      view-train           Print a slice of the raw train corpus.
      check-distribution   Sanity-check the two-function contract.

  OPERATOR-ONLY (never run by the optimizer — scientific integrity):
      dev-eval             BPC on the dev split.
      val-eval             BPC on the val split.

The dev and val splits live outside this directory. The optimizer must not
read them directly, must not invoke dev-eval / val-eval, and must not cd
into any directory containing them.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "model_repo"))

from model import ModelState, advance, predict, VOCAB, VOCAB_INDEX, VOCAB_SIZE  # noqa: E402


_LN2 = math.log(2.0)

# Validity bounds enforced on every emitted distribution.
_LOGP_CEILING = 5e-4
_DIST_SUM_MIN = 0.999
_DIST_SUM_MAX = 1.001
_PROB_FLOOR = 1e-12

_TRAIN = _HERE / "corpus" / "train.txt"
# dev.txt and val.txt live outside v2/ by design. Operator-only.
_DEV = _HERE.parent / "clm" / "corpus" / "dev.txt"
_VAL = _HERE.parent / "clm" / "corpus" / "val.txt"


# ---------------------------------------------------------------------------
# Core: distribution validation + BPC computation
# ---------------------------------------------------------------------------

def _validate_distribution(logprobs: list[float]) -> None:
    if len(logprobs) != VOCAB_SIZE:
        raise RuntimeError(
            f"predict returned {len(logprobs)} entries, expected {VOCAB_SIZE}"
        )
    max_lp = max(logprobs)
    if max_lp > _LOGP_CEILING:
        raise RuntimeError(
            f"invalid distribution: max log-prob = {max_lp:.6f} "
            f"(exp = {math.exp(max_lp):.6e}); probabilities must be <= 1"
        )
    total = sum(math.exp(lp) for lp in logprobs)
    if not (_DIST_SUM_MIN <= total <= _DIST_SUM_MAX):
        raise RuntimeError(
            f"invalid distribution: sum(exp(logp)) = {total:.6f} (expected ~1.0)"
        )


def compute_bpc(text: str) -> float:
    state = ModelState()
    total_nll_bits = 0.0
    for ch in text:
        logprobs = predict(state)
        _validate_distribution(logprobs)
        idx = VOCAB_INDEX.get(ch)
        if idx is None:
            total_nll_bits += -math.log2(_PROB_FLOOR)
            continue
        logp_nat = max(logprobs[idx], math.log(_PROB_FLOOR))
        total_nll_bits += -logp_nat / _LN2
        state = advance(state, idx)
    return total_nll_bits / max(len(text), 1)


def sample_text(
    length: int = 500,
    temperature: float = 1.0,
    seed: int = 0,
    prefix: str = "",
) -> str:
    rng = random.Random(seed)
    state = ModelState()
    out: list[str] = []
    for ch in prefix:
        idx = VOCAB_INDEX.get(ch)
        if idx is None:
            continue
        state = advance(state, idx)
        out.append(ch)
    for _ in range(length):
        logprobs = predict(state)
        _validate_distribution(logprobs)
        if temperature != 1.0:
            logprobs = [lp / temperature for lp in logprobs]
        max_lp = max(logprobs)
        exps = [math.exp(lp - max_lp) for lp in logprobs]
        total = sum(exps)
        probs = [e / total for e in exps]
        r = rng.random()
        cum = 0.0
        chosen = VOCAB_SIZE - 1
        for i, p in enumerate(probs):
            cum += p
            if r < cum:
                chosen = i
                break
        out.append(VOCAB[chosen])
        state = advance(state, chosen)
    return "".join(out)


def _load_train() -> str:
    if not _TRAIN.exists():
        raise FileNotFoundError(f"train corpus not found at {_TRAIN}")
    return _TRAIN.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Agent-accessible subcommands
# ---------------------------------------------------------------------------

def cmd_train_eval_batch(args: argparse.Namespace) -> int:
    train = _load_train()
    batch_size = args.batch_size
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if batch_size >= len(train):
        print(compute_bpc(train))
        return 0
    rng = random.Random(args.seed)
    start = rng.randrange(0, len(train) - batch_size + 1)
    chunk = train[start : start + batch_size]
    bpc = compute_bpc(chunk)
    # Report the chunk bounds so the agent can verify it isn't over-fitting
    # to one region of the corpus.
    print(
        f"bpc={bpc:.4f}  seed={args.seed}  start={start}  length={batch_size}"
    )
    return 0


def cmd_train_eval_full(args: argparse.Namespace) -> int:
    train = _load_train()
    print(f"{compute_bpc(train):.4f}")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    out = sample_text(
        length=args.length,
        temperature=args.temperature,
        seed=args.seed,
        prefix=args.prefix or "",
    )
    sys.stdout.write(out)
    if not out.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


def cmd_view_train(args: argparse.Namespace) -> int:
    train = _load_train()
    offset = args.offset
    length = args.length
    if offset < 0 or offset >= len(train):
        raise ValueError(
            f"--offset must be in [0, {len(train)}); got {offset}"
        )
    chunk = train[offset : offset + length]
    sys.stdout.write(chunk)
    if not chunk.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


def cmd_check_distribution(args: argparse.Namespace) -> int:
    state = ModelState()
    lp = predict(state)
    _validate_distribution(lp)
    before = state.model_dump()
    _ = advance(state, 0)
    if state.model_dump() != before:
        raise RuntimeError("advance() mutated its input state")
    for tid in [0, 5, 17, 42]:
        if tid >= VOCAB_SIZE:
            continue
        state = advance(state, tid)
        _validate_distribution(predict(state))
    print("OK: predict() emits valid distributions and advance() is pure.")
    return 0


# ---------------------------------------------------------------------------
# Operator-only subcommands — DO NOT invoke from the optimizer
# ---------------------------------------------------------------------------

def cmd_dev_eval(args: argparse.Namespace) -> int:
    print(f"{compute_bpc(_DEV.read_text(encoding='utf-8')):.4f}")
    return 0


def cmd_val_eval(args: argparse.Namespace) -> int:
    print(f"{compute_bpc(_VAL.read_text(encoding='utf-8')):.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v2 harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Agent-accessible
    p = sub.add_parser("train-eval-batch", help="BPC on a random train chunk")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2000)
    p.set_defaults(func=cmd_train_eval_batch)

    sub.add_parser("train-eval-full", help="BPC on the full train split").set_defaults(
        func=cmd_train_eval_full
    )

    p = sub.add_parser("sample", help="Generate a completion")
    p.add_argument("--length", type=int, default=300)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", type=str, default="")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("view-train", help="Print a slice of the train corpus")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--length", type=int, default=1000)
    p.set_defaults(func=cmd_view_train)

    sub.add_parser("check-distribution").set_defaults(func=cmd_check_distribution)

    # Operator-only
    sub.add_parser("dev-eval", help="OPERATOR ONLY").set_defaults(func=cmd_dev_eval)
    sub.add_parser("val-eval", help="OPERATOR ONLY").set_defaults(func=cmd_val_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
