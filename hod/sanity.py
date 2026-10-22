#!/usr/bin/env python3
"""Evaluator sanity: does it separate a good harness from a bad one?

Idea from the vibe-coding-toolkit ("an evaluator that accepts both versions
is no use") and the V of Ornith-1.5 ("good solutions pass, bad ones fail").
Before believing a gap, check that your evaluator's scores separate a
known-good harness from a known-bad one on the SAME set.

Inputs: scores of the good and bad harnesses on the same tasks (JSON
{task_id: score}); optionally the manifest, to check per family.

Passes if the 95% CI (paired bootstrap) of good - bad stays above zero on
the set and, with --manifest, if no family has good <= bad on average.

Usage:
    hod sanity --good good-scores.json --bad bad-scores.json \
      [--manifest runs/heldout-42/manifest.json]
Exit code: 0 passed, 1 input error, 2 failed.
"""

import argparse
import json
import random
import sys
from pathlib import Path

try:
    from .scores import load_scores
    from . import stats as stats_mod
except ImportError:  # direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scores import load_scores  # noqa: E402
    import stats as stats_mod  # noqa: E402


def check(good, bad, family_of=None, n_boot=2000, seed=stats_mod.DEFAULT_SEED):
    ids = sorted(set(good) & set(bad))
    if not ids:
        raise ValueError("no tasks in common between good and bad")
    diffs = [good[t] - bad[t] for t in ids]
    rng = random.Random(seed)
    n = len(diffs)
    reps = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                  for _ in range(n_boot))
    lo = stats_mod._percentile(reps, 0.025)
    hi = stats_mod._percentile(reps, 0.975)
    mean = sum(diffs) / n
    out = {"n": n, "good_minus_bad": round(mean, 4),
           "ci": [round(lo, 4), round(hi, 4)], "ok": lo > 0,
           "families": {}}
    if family_of:
        fams = sorted({family_of[t] for t in ids if t in family_of})
        for fam in fams:
            fd = [good[t] - bad[t] for t in ids if family_of.get(t) == fam]
            m = sum(fd) / len(fd)
            out["families"][fam] = {"n": len(fd), "good_minus_bad": round(m, 4),
                                    "ok": m > 0}
        if any(not f["ok"] for f in out["families"].values()):
            out["ok"] = False
    return out


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--good", type=Path, required=True)
    parser.add_argument("--bad", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        good, bad = load_scores(args.good), load_scores(args.bad)
        family_of = None
        if args.manifest:
            m = json.loads(args.manifest.read_text())
            family_of = {t["id"]: t["family"] for t in m["tasks"]}
        r = check(good, bad, family_of)
    except (ValueError, OSError, KeyError) as e:
        print("sanity: %s" % e, file=sys.stderr)
        return 1
    print("good − bad: %+.3f CI [%+.3f, %+.3f] (n=%d)"
          % (r["good_minus_bad"], r["ci"][0], r["ci"][1], r["n"]))
    for fam, f in r["families"].items():
        print("  %-20s %+.3f%s" % (fam, f["good_minus_bad"],
                                   "" if f["ok"] else "  <- does not separate"))
    if r["ok"]:
        print("sanity: ok, the evaluator separates good from bad")
        return 0
    print("sanity: FAILED, the evaluator does not separate good from bad; "
          "do not trust a gap measured with it")
    return 2


if __name__ == "__main__":
    sys.exit(main())
