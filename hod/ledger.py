#!/usr/bin/env python3
"""Ledger of changes: each pipeline round becomes a JSONL line.

Input (written by the pipeline and by the canary hook):
    {"ts": "...", "change": "...", "placebo": false, "seeds": [42],
     "verdicts": ["OVERFIT"], "stable": true, "verdict": "OVERFIT",
     "gap": 1.0, "suite_delta": 1.0, "held_delta": 0.0,
     "hypothesis": "match", "wave_stopped": false,
     "by_category": {"instruction": "OVERFIT"}, "n_held": 8}

The report aggregates: totals, overfit rate and family weights for the
next generation. Weight = 1 + overfit / (seen + PRIOR) of the family (with
shrinkage: a single round does not double the weight). The weights come
from the `by_family` field (keys = generator families); `by_category` is
report only. Rounds stopped at wave 1 and rounds with a gap inside the
placebo noise (`gap_significant: false`) do not move the weights. This is
the feedback loop: the ledger teaches the generator where to look.

Usage:
    hod ledger --ledger runs/ledger.jsonl --weights-out weights.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: no lock, append stays atomic for a short line
    fcntl = None

PRIOR = 1.0


def append(path, entry):
    """Append of the line with lock exclusivo and fsync.

    The written only by line, with lock: pipeline and hook of the canary running
    to the same time not intercalam bytes; fsync avoids line truncada after
    drop of energia.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
    with open(path, "ab") as f:
        if fcntl:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        finally:
            if fcntl:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return entry


def load(path, strict=False):
    """Reads the ledger; lines corrompidas are puladas with warning (strict=True
    levanta). The line truncada not derruba more the report integer."""
    entries = []
    bad = 0
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            if strict:
                raise ValueError("ledger %s line %d: %s" % (path, lineno, e))
            bad += 1
            print("warning: ledger %s line %d corrupted, ignored"
                  % (path, lineno), file=sys.stderr)
            continue
        if isinstance(entry, dict):
            entries.append(entry)
        else:
            bad += 1
    return entries


def report(entries):
    real = [e for e in entries if not e.get("placebo")]
    placebo = [e for e in entries if e.get("placebo")]
    verdict_counts = {}
    for e in real:
        verdict_counts[e.get("verdict", "?")] = (
            verdict_counts.get(e.get("verdict", "?"), 0) + 1
        )
    overfit = verdict_counts.get("OVERFIT", 0)
    overfit_rate = round(overfit / len(real), 3) if real else 0.0
    family_seen = {}
    family_overfit = {}
    for e in real:
        if e.get("wave_stopped") or e.get("gap_significant") is False:
            continue
        for fam, verdict in (e.get("by_family") or {}).items():
            family_seen[fam] = family_seen.get(fam, 0) + 1
            if verdict == "OVERFIT":
                family_overfit[fam] = family_overfit.get(fam, 0) + 1
    weights = {
        fam: round(1 + family_overfit.get(fam, 0) / (seen + PRIOR), 2)
        for fam, seen in family_seen.items()
    }
    return {
        "n_total": len(entries),
        "n_real": len(real),
        "n_placebo": len(placebo),
        "verdict_counts": verdict_counts,
        "overfit_rate": overfit_rate,
        "family_seen": family_seen,
        "family_overfit": family_overfit,
        "weights": weights,
        "ready_for_gate_promotion": len(real) >= 3,
    }


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--weights-out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = report(load(args.ledger))
    print(
        "ledger: %d rounds (%d real, %d placebos), overfit %.1f%%"
        % (
            result["n_total"],
            result["n_real"],
            result["n_placebo"],
            result["overfit_rate"] * 100,
        )
    )
    print("verdicts: %s" % result["verdict_counts"])
    print("family weights: %s" % result["weights"])
    if result["ready_for_gate_promotion"]:
        print("ready to evaluate promotion of the blocking gate (3+ changes).")
    if args.weights_out:
        args.weights_out.write_text(json.dumps(result["weights"], indent=2))
        print("weights written: %s" % args.weights_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
