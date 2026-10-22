#!/usr/bin/env python3
"""Agreement between two annotators (second rater or retest).

Accepts flat scoreboards {task_id: score} or the canary format
{"tasks": {task_id: {"nota": n}}, "behavior": {...}}. (The "nota" key is
kept as-is: it is part of the file format.)

Metrics: n in common, % equal, mean absolute difference and Cohen's kappa
for the 3 categories (0, 1, 2). Kappa 1 is perfect agreement, 0 is chance
agreement, negative is worse than chance.

Usage:
    hod agree --a scores-rater1.json --b scores-rater2.json
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from .scores import flatten as _flatten
except ImportError:  # direct execution of the file, outside the package
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scores import flatten as _flatten  # noqa: E402


def flatten(scores):
    # validates 0/1/2: before, int() truncated 1.7 to 1, accepted 3 (outside
    # the kappa categories) and broke with TypeError on a None score
    return _flatten(scores, "annotator scoreboard")


def agree(a, b):
    a = flatten(a)
    b = flatten(b)
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("no tasks in common")
    equal = sum(1 for t in common if a[t] == b[t])
    abs_diff = sum(abs(a[t] - b[t]) for t in common) / len(common)
    po = equal / len(common)
    pe = 0.0
    for category in (0, 1, 2):
        pa = sum(1 for t in common if a[t] == category) / len(common)
        pb = sum(1 for t in common if b[t] == category) / len(common)
        pe += pa * pb
    kappa = (po - pe) / (1 - pe) if pe < 1 else (1.0 if po == 1 else 0.0)
    # kappa weighted quadratic: the scale 0-2 is ordinal, 0 vs 2 is worse that
    # 0 vs 1
    n = len(common)
    obs = sum((a[t] - b[t]) ** 2 for t in common) / n
    pa = [sum(1 for t in common if a[t] == c) / n for c in (0, 1, 2)]
    pb = [sum(1 for t in common if b[t] == c) / n for c in (0, 1, 2)]
    exp = sum(pa[i] * pb[j] * (i - j) ** 2 for i in range(3) for j in range(3))
    weighted = 1 - obs / exp if exp > 0 else (1.0 if obs == 0 else 0.0)
    return {
        "n": n,
        "only_a": len(set(a) - set(b)),
        "only_b": len(set(b) - set(a)),
        "equal": equal,
        "pct_equal": round(po, 3),
        "mean_abs_diff": round(abs_diff, 3),
        "kappa": round(kappa, 3),
        "kappa_weighted": round(weighted, 3),
        # without variation in the scores the kappa is undefined (0/0); 1.0 here is
        # convention, not evidence
        "kappa_defined": pe < 1,
    }


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = agree(
            json.loads(args.a.read_text()),
            json.loads(args.b.read_text()),
        )
    except ValueError as e:
        print("agree: %s" % e, file=sys.stderr)
        return 1
    print(
        "agreement: n=%d, equal=%d (%.1f%%), mean diff=%.3f, kappa=%.3f%s"
        ", weighted kappa=%.3f"
        % (
            result["n"],
            result["equal"],
            result["pct_equal"] * 100,
            result["mean_abs_diff"],
            result["kappa"],
            "" if result["kappa_defined"] else " (undefined: no variation)",
            result["kappa_weighted"],
        )
    )
    if result["only_a"] or result["only_b"]:
        print("warning: outside the comparison: %d only in A, %d only in B"
              % (result["only_a"], result["only_b"]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
