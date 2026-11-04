#!/usr/bin/env python3
"""Blind scoring: the annotator receives anonymous ids, not the real ones.

`hod blind` reads the set's manifest and emits an answer key with shuffled
t01..tNN ids plus the anon -> real key. `hod unblind` restores the real
ids after annotation.

Honest limit: in a single-agent setup, the key exists on disk and the
agent could open it; blinding here works by instruction (do not open the
key) and holds fully with a second annotator who has no access to the key.
The key stays OUTSIDE the set directory (enforced) and never in the
annotator's template (enforced); the report uses it only to restore.

Usage:
    hod blind --heldout runs/heldout-42 --seed 42 \\
      --template run/heldout-blind.json --key keys/heldout-key.json
    hod unblind --scores run/heldout-blind.json --key keys/heldout-key.json \\
      --out run/heldout.json
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

try:
    from .manifest import load as load_manifest
    from .scores import read_json, validate_score
except ImportError:  # direct execution of the file, outside the package
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from manifest import load as load_manifest  # noqa: E402
    from scores import read_json, validate_score  # noqa: E402


def make_blind(heldout_dir, seed=None):
    """Answer key with anonymous ids and the key anon -> real.

    Without a seed, the key comes from a cryptographic source (not
    reproducible); with a fixed, public seed (e.g. the generation seed),
    anyone can rebuild the key. The answer key lists the anonymous ids in
    order t01..tNN, which carries no information (the order used to be the
    alphabetical order of the real ids, which revealed the key).
    """
    manifest = load_manifest(Path(heldout_dir) / "manifest.json")
    real_ids = sorted(t["id"] for t in manifest["tasks"])
    if not real_ids:
        raise ValueError("the set has no tasks: %s" % heldout_dir)
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


def _score_of(value):
    """The score of an answer-key entry (None while it is unscored)."""
    return value.get("nota") if isinstance(value, dict) else value


def restore(filled, key, require_complete=True):
    """Maps the filled answer key back to the real ids.

    Every present score is validated (0, 1 or 2, not bool), with or without
    require_complete. Without it, unscored entries (None, the template's
    placeholder) are left out of the output, so that `hod.scores.load_scores`
    accepts it and the gap coverage gate counts them as missing."""
    if not isinstance(filled, dict):
        raise ValueError("scores: expected a JSON object")
    if not isinstance(key, dict) or not all(
            isinstance(a, str) and isinstance(r, str) for a, r in key.items()):
        raise ValueError("key: expected a JSON object {anonymous id: real id}")
    if len(set(key.values())) != len(key):
        raise ValueError("key: two anonymous ids map to the same task")
    out = {"tasks": {}, "behavior": {}}
    for section in ("tasks", "behavior"):
        entries = filled.get(section, {})
        if not isinstance(entries, dict):
            raise ValueError("scores: %r must be a JSON object" % section)
        for anon, value in entries.items():
            if anon not in key:
                raise ValueError("anonymous id outside the key: %s" % anon)
            if section == "tasks":
                nota = _score_of(value)
                if nota is None:
                    if require_complete:
                        raise ValueError("invalid score for %s: None" % anon)
                    continue
                try:
                    nota = validate_score(anon, nota, "answer key")
                except ValueError:
                    raise ValueError("invalid score for %s: %r" % (anon, nota))
                value = dict(value, nota=nota) if isinstance(value, dict) \
                    else nota
            out[section][key[anon]] = value
    if require_complete:
        tasks = filled.get("tasks", {})
        missing = sorted(a for a in key if a not in tasks)
        if missing:
            raise ValueError("incomplete answer key, missing: %s" % missing)
    return out


def _same_file(a, b):
    """Same path, also through aliases resolve() does not see."""
    a, b = Path(a), Path(b)
    if a.resolve() == b.resolve():
        return True
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def _inside(path, directory):
    """True if `path` is (or would be created) inside `directory`."""
    path, directory = Path(path).resolve(), Path(directory)
    try:
        path.relative_to(directory.resolve())
        return True
    except ValueError:
        pass
    if not directory.exists():
        return False
    for parent in path.parents:
        try:
            if parent.exists() and os.path.samefile(parent, directory):
                return True
        except OSError:
            continue
    return False


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
    try:
        if _same_file(args.template, args.key):
            raise ValueError("--template and --key are the same file: the key "
                             "would be written into the annotator's template")
        if _inside(args.key, args.heldout):
            raise ValueError("--key must be outside the --heldout set "
                             "directory: %s" % args.key)
        template, key = make_blind(args.heldout, args.seed)
    except ValueError as e:
        print("blind: %s" % e, file=sys.stderr)
        return 1
    if args.seed is not None:
        print("warning: --seed makes the key reproducible; blinding holds only if "
              "the seed is secret", file=sys.stderr)
    try:
        args.key.parent.mkdir(parents=True, exist_ok=True)
        args.key.write_text(json.dumps(key, indent=2), encoding="utf-8")
        args.template.parent.mkdir(parents=True, exist_ok=True)
        args.template.write_text(json.dumps(template, indent=2),
                                 encoding="utf-8")
    except OSError as e:
        print("blind: %s" % e, file=sys.stderr)
        return 1
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
    try:
        if _same_file(args.out, args.key) or _same_file(args.out, args.scores):
            raise ValueError("--out must differ from --scores and --key")
        filled = read_json(args.scores, "scores file", expect=dict)
        key = read_json(args.key, "key file", expect=dict)
        restored = restore(filled, key, not args.allow_incomplete)
    except ValueError as e:
        print("unblind: %s" % e, file=sys.stderr)
        return 1
    unscored = len(key) - len(restored["tasks"])
    if unscored:
        print("unblind: %d of %d tasks unscored, left out (they count as "
              "missing)" % (unscored, len(key)), file=sys.stderr)
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(restored, indent=2), encoding="utf-8")
    except OSError as e:
        print("unblind: %s" % e, file=sys.stderr)
        return 1
    print("ids restored: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main_blind())
