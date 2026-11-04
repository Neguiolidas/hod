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
It also requires at least stats.MIN_N_CI shared tasks (with fewer, the
bootstrap CI collapses: one task "passes" with a zero-width CI) and that
good and bad were scored on at least 90% of the same tasks (of the
manifest's tasks, with --manifest).

Usage:
    hod sanity --good good-scores.json --bad bad-scores.json \
      [--manifest runs/heldout-42/manifest.json]
Exit code: 0 passed, 1 input error, 2 failed.
"""

import argparse
import random
import sys
from pathlib import Path

try:
    from .scores import load_scores
    from . import manifest as manifest_mod
    from . import stats as stats_mod
except ImportError:  # direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scores import load_scores  # noqa: E402
    import manifest as manifest_mod  # noqa: E402
    import stats as stats_mod  # noqa: E402


def check(good, bad, family_of=None, n_boot=2000, seed=stats_mod.DEFAULT_SEED,
          expected=None):
    """`expected`: the task ids that should have been scored (coverage
    denominator); defaults to the manifest's ids when `family_of` is given,
    else to the union of both files."""
    stats_mod.check_n_boot(n_boot)
    ids = sorted(set(good) & set(bad))
    if not ids:
        raise ValueError("no tasks in common between good and bad")
    if expected is None:
        expected = set(family_of) if family_of else set(good) | set(bad)
    expected = set(expected)
    coverage = len(set(ids) & expected) / len(expected) if expected else 0.0
    diffs = [good[t] - bad[t] for t in ids]
    rng = random.Random(seed)
    n = len(diffs)
    reps = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                  for _ in range(n_boot))
    lo = stats_mod._percentile(reps, 0.025)
    hi = stats_mod._percentile(reps, 0.975)
    mean = sum(diffs) / n
    problems = []
    if n < stats_mod.MIN_N_CI:
        problems.append("only %d shared tasks (minimum %d): the CI does not "
                        "hold" % (n, stats_mod.MIN_N_CI))
    if coverage < stats_mod.MIN_COVERAGE - stats_mod.EPS:
        problems.append("good and bad share %.0f%% of the tasks (minimum %.0f%%)"
                        % (coverage * 100, stats_mod.MIN_COVERAGE * 100))
    if not lo > stats_mod.EPS:
        problems.append("CI of good - bad does not stay above zero")
    out = {"n": n, "good_minus_bad": round(mean, 4),
           "ci": [round(lo, 4), round(hi, 4)], "coverage": round(coverage, 4),
           "ok": not problems, "problems": problems, "families": {}}
    if family_of:
        fams = sorted({family_of[t] for t in ids if t in family_of})
        for fam in fams:
            fd = [good[t] - bad[t] for t in ids if family_of.get(t) == fam]
            m = sum(fd) / len(fd)
            out["families"][fam] = {"n": len(fd), "good_minus_bad": round(m, 4),
                                    "ok": m > 0}
        bad_fams = sorted(f for f, v in out["families"].items() if not v["ok"])
        if bad_fams:
            out["ok"] = False
            problems.append("families where good <= bad: %s" % ", ".join(bad_fams))
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
            m = manifest_mod.load(args.manifest)
            family_of = {t["id"]: t["family"] for t in m["tasks"]}
        r = check(good, bad, family_of)
    except (ValueError, OSError) as e:
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
    for problem in r["problems"]:
        print("  - %s" % problem)
    print("sanity: FAILED, the evaluator does not separate good from bad; "
          "do not trust a gap measured with it")
    return 2


if __name__ == "__main__":
    sys.exit(main())
