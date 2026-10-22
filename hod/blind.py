#!/usr/bin/env python3
"""Blind scoring: the annotator receives anonymous ids, not the real ones.

`hod blind` reads the set's manifest and emits an answer key with shuffled
t01..tNN ids plus the anon -> real key. `hod unblind` restores the real
ids after annotation.

Honest limit: in a single-agent setup, the key exists on disk and the
agent could open it; blinding here works by instruction (do not open the
key) and holds fully with a second annotator who has no access to the key.
The key stays OUTSIDE the run directory, and the report uses it only to
restore.

Usage:
    hod blind --heldout runs/heldout-42 --seed 42 \\
      --template run/heldout-blind.json --key keys/heldout-key.json
    hod unblind --scores run/heldout-blind.json --key keys/heldout-key.json \\
      --out run/heldout.json
"""

import argparse
import json
import random
import sys
from pathlib import Path


def make_blind(heldout_dir, seed=None):
    """Answer key with ids anonymous and the key anon -> real.

    Without seed, the key leaves of fonte cryptographic (not reproducible): with
    seed fixed and public (ex.: the same of the generation), any the refaz the
    key. The answer key lists the ids anonymous in order t01..tNN, that not
    carrega information (before the order of the answer key was the order alphabetical
    of the ids real and revelava the key).
    """
    manifest = json.loads(
        (Path(heldout_dir) / "manifest.json").read_text()
    )
    real_ids = sorted(t["id"] for t in manifest["tasks"])
    width = max(2, len(str(len(real_ids))))
    anon_ids = ["t%0*d" % (width, i + 1) for i in range(len(real_ids))]
    rng = random.SystemRandom() if seed is None else random.Random(seed)
    shuffled_real = real_ids[:]
    rng.shuffle(shuffled_real)
    key = {anon: real for anon, real in zip(anon_ids, shuffled_real)}
    template = {
        "tasks": {anon: {"nota": None} for anon in anon_ids},
        "behavior": {},
    }
    return template, key


def restore(filled, key, require_complete=True):
    out = {"tasks": {}, "behavior": {}}
    for section in ("tasks", "behavior"):
        for anon, value in filled.get(section, {}).items():
            if anon not in key:
                raise ValueError("anonymous id outside the key: %s" % anon)
            out[section][key[anon]] = value
    if require_complete:
        tasks = filled.get("tasks", {})
        missing = sorted(a for a in key if a not in tasks)
        if missing:
            raise ValueError("incomplete answer key, missing: %s" % missing)
        for anon, value in tasks.items():
            nota = value.get("nota") if isinstance(value, dict) else value
            if isinstance(nota, bool) or nota not in (0, 1, 2):
                raise ValueError("invalid score for %s: %r" % (anon, nota))
    return out


def main_blind(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="only for tests: with a seed the key is reproducible by anyone who "
        "knows the seed",
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.seed is not None:
        print("warning: --seed makes the key reproducible; blinding holds only if "
              "the seed is secret", file=sys.stderr)
    template, key = make_blind(args.heldout, args.seed)
    args.template.parent.mkdir(parents=True, exist_ok=True)
    args.template.write_text(json.dumps(template, indent=2))
    args.key.parent.mkdir(parents=True, exist_ok=True)
    args.key.write_text(json.dumps(key, indent=2))
    print(
        "blind answer key: %d tasks in %s (key in %s, do not open)"
        % (len(key), args.template, args.key)
    )
    return 0


def main_unblind(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    filled = json.loads(args.scores.read_text())
    key = json.loads(args.key.read_text())
    try:
        restored = restore(filled, key, not args.allow_incomplete)
    except ValueError as e:
        print("unblind: %s" % e, file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(restored, indent=2))
    print("ids restored: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main_blind())
