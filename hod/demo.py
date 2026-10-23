#!/usr/bin/env python3
"""Offline demo of `hod improve` with the toy harness (hod/toy.py).

Uses no network, no model, nothing outside the Python standard library.
Shows the loop rejecting overfit changes (memorizing the suite), placebos
and trades that sink a family, and accepting the genuine ones; then the
final exam on a sealed held-out. With --sweep N it repeats with N seeds
and counts what was accepted by change type.

Usage:
    hod demo --out runs/demo [--seed 7] [--sweep 30] [--via-commands]

--via-commands runs the same loop through the commands (`python -m hod.toy eval`
and `propose`), exactly as a harness of yours would be plugged in.
"""


import argparse
import json
import shlex
import sys
from collections import Counter
from pathlib import Path

from hod import improve as improve_mod
from hod import toy

KINDS = ["genuine", "overfit", "placebo", "trade", "mixed"]
KIND_LABEL = {
    "genuine": "genuine (+0.3 in all families)",
    "overfit": "overfit (memorize the suite)",
    "placebo": "placebo (cosmetic style)",
    "trade": "trade (+0.45 everywhere, zeroes one family)",
    "mixed": "mixed (memorize the suite + 0.08 real)",
}


def run_one(out, seed=7, rounds=6, k=5, n_held=96, per_family=8,
            via_commands=False, quiet=False):
    out = Path(out)
    families = toy.write_catalog(out / "catalog", per_family=per_family)
    log = (lambda *a: None) if quiet else print
    if via_commands:
        init = out / "configs"
        toy.main(["init", "--out", str(init)])
        py = shlex.quote(sys.executable)
        evaluator = improve_mod.CommandEvaluator(
            py + " -m hod.toy eval --config {config} --tasks {tasks} "
            "--seed {seed} --out {out}", out / "work")
        proposer = improve_mod.CommandProposer(
            py + " -m hod.toy propose --champion {champion} --history "
            "{history} --k {k} --round {round} --out {out}", out / "work")
        initial = json.loads((init / "inicial.json").read_text())
    else:
        evaluator, proposer, initial = toy.evaluate, toy.propose, toy.initial_config()
    return improve_mod.improve(
        initial, evaluator, proposer, out / "catalog", out / "improve",
        suite_families=families, rounds=rounds, k=k, n_held=n_held, seed=seed,
        sanity_good=toy.known_good(), sanity_bad=toy.known_bad(), log=log)


def tally(result):
    c = Counter()
    for e in result["entries"]:
        if e["type"] == "candidate":
            c[(e.get("kind"), "accepted" if e["decision"] == "accepted"
               else "rejected")] += 1
    return c


def render_sweep(n, counts, finals, reasons):
    lines = ["## Sweep with %d seeds" % n, "",
             "| Change type | Evaluated | Accepted |", "| --- | --- | --- |"]
    for kind in KINDS:
        a = counts[(kind, "accepted")]
        r = counts[(kind, "rejected")]
        lines.append("| %s | %d | %d |" % (KIND_LABEL[kind], a + r, a))
    lines += ["", "Final exam (initial -> champion, sealed held-out): %s"
              % ", ".join("%s %d" % (k or "none accepted", v)
                          for k, v in finals.most_common()),
              "Stop reasons: %s" % ", ".join(
                  "%s %d" % kv for kv in reasons.most_common())]
    return "\n".join(lines) + "\n"


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--n-held", type=int, default=96)
    parser.add_argument("--sweep", type=int, default=0,
                        help="repeats with N seeds and counts acceptances by type")
    parser.add_argument("--via-commands", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        print("demo: %s already exists and is not empty" % args.out, file=sys.stderr)
        return 1
    run_one(args.out / "principal", seed=args.seed,
                     rounds=args.rounds, n_held=args.n_held,
                     via_commands=args.via_commands)
    report = (args.out / "principal" / "improve" / "report.md").read_text()
    parts = ["# hod improve demo (toy harness, offline)", "",
             "Initial harness: skill 0.3 in all families. Each "
             "round the proposer brings 5 changes: genuine, overfit, placebo, "
             "trade and mixed. Seed %d, new held-out of %d tasks per round, "
             "suite of 64." % (args.seed, args.n_held), "", report]
    if args.sweep:
        counts, finals, reasons = Counter(), Counter(), Counter()
        for s in range(1, args.sweep + 1):
            r = run_one(args.out / "sweep" / ("seed-%02d" % s), seed=s,
                        rounds=args.rounds, n_held=args.n_held, quiet=True)
            counts.update(tally(r))
            finals[r["final"]["verdict"] if r["final"] else None] += 1
            reasons[r["summary"]["reason"].split(":")[0]] += 1
            print("seed %d: accepted %s" % (s, ", ".join(
                e["kind"] for e in r["entries"]
                if e["type"] == "candidate" and e["decision"] == "accepted")
                or "none"))
        sweep = render_sweep(args.sweep, counts, finals, reasons)
        parts += ["", sweep]
        print(sweep)
    (args.out / "DEMO.md").write_text("\n".join(parts))
    print("demo written to %s" % (args.out / "DEMO.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
