#!/usr/bin/env python3
"""Preregistration of the hypothesis: the author declares what is expected
BEFORE seeing the held-out, and HOD matches predicted vs observed after.

Format (JSON):
    {"change": "...", "expect": {"suite": "up", "held": "up",
     "min_ratio": 0.75}, "created_at": "2026-10-22T03:00:00"}

Match rule:
- suite "up" requires delta > 0, "flat" requires delta == 0, "down" requires < 0.
- held "up" requires delta > 0 and held/suite ratio >= min_ratio.
- held "flat" requires delta == 0, "down" requires delta < 0.

The timing check (prereg before the set was generated) is done by the
pipeline, which knows the manifests; creation and matching live here.

Seed commitment (`--seeds`): the prereg can declare the generation salts in
advance. Sets bound to the prereg (`hod generate --prereg`) must then use
exactly those salts, all of them: generating 100 bound sets and keeping
the most favorable one ("seed shopping") no longer verifies. Without the
declaration, binding proves the set came from this prereg, not that it
was the only one generated.

Usage:
    hod prereg --change "new routing rule" --seeds 42,43,44 --out prereg.json
    hod prereg --change "..." --expect-held flat --min-ratio 0.5 --out prereg.json
"""

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from .scores import read_json
except ImportError:  # direct execution of the file, outside the package
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scores import read_json  # noqa: E402

DIRECTIONS = ("up", "flat", "down")


def _number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def parse_seeds(text):
    """'42,43,44' -> [42, 43, 44]; clean ValueError on junk or repeats."""
    try:
        seeds = [int(s) for s in text.split(",") if s.strip()]
    except ValueError:
        raise ValueError("seeds must be integers separated by commas: %r" % text)
    if not seeds:
        raise ValueError("no seeds in %r" % text)
    if len(set(seeds)) != len(seeds):
        raise ValueError("repeated seed in %r" % text)
    return seeds


def create(change, expect_suite="up", expect_held="up", min_ratio=0.75,
           seeds=None, flat_tol=0.0):
    if expect_suite not in DIRECTIONS or expect_held not in DIRECTIONS:
        raise ValueError("invalid direction: use up, flat or down")
    if not _number(min_ratio) or not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio outside [0, 1]: %s" % min_ratio)
    if not _number(flat_tol) or flat_tol < 0:
        raise ValueError("flat_tol must be >= 0: %s" % flat_tol)
    data = {
        "change": change,
        "expect": {
            "suite": expect_suite,
            "held": expect_held,
            "min_ratio": min_ratio,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if flat_tol:
        data["expect"]["flat_tol"] = flat_tol
    if seeds is not None:
        seeds = [int(x) for x in seeds]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be a non-empty list without repeats")
        data["seeds"] = sorted(seeds)
    return data


def load(path):
    data = read_json(path, "prereg file", expect=dict)
    expect = data.get("expect")
    if not isinstance(expect, dict):
        raise ValueError("prereg %s: missing 'expect' object" % path)
    if expect.get("suite") not in DIRECTIONS:
        raise ValueError("invalid suite direction in %s" % path)
    if expect.get("held") not in DIRECTIONS:
        raise ValueError("invalid held direction in %s" % path)
    if not _number(expect.get("min_ratio")) or not 0 <= expect["min_ratio"] <= 1:
        raise ValueError("prereg %s: min_ratio must be a number in [0, 1]" % path)
    if "flat_tol" in expect and (not _number(expect["flat_tol"])
                                 or expect["flat_tol"] < 0):
        raise ValueError("prereg %s: flat_tol must be a number >= 0" % path)
    seeds = data.get("seeds")
    if seeds is not None and (
            not isinstance(seeds, list) or not seeds
            or not all(isinstance(x, int) and not isinstance(x, bool)
                       for x in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("prereg %s: seeds must be a list of distinct integers"
                         % path)
    return data


def _direction_match(expect, delta, flat_tol=0.0):
    if expect == "up":
        return delta > flat_tol
    if expect == "down":
        return delta < -flat_tol
    return abs(delta) <= flat_tol


def check(prereg, result):
    """result: dict with suite_delta and held_delta (output of gap.analyze)."""
    suite_delta = result["suite_delta"]
    held_delta = result["held_delta"]
    ratio = (
        held_delta / suite_delta
        if suite_delta > 0
        else None
    )
    flat_tol = float(prereg["expect"].get("flat_tol", 0.0))
    suite_ok = _direction_match(prereg["expect"]["suite"], suite_delta, flat_tol)
    held_expect = prereg["expect"]["held"]
    if held_expect == "up":
        held_ok = held_delta > flat_tol
        # min_ratio only makes sense when the suite rose; with the suite flat
        # or falling, "held rises" is enough (before, it could never match)
        if ratio is not None:
            held_ok = held_ok and ratio >= prereg["expect"]["min_ratio"] - 1e-9
    else:
        held_ok = _direction_match(held_expect, held_delta, flat_tol)
    return {
        "suite_match": suite_ok,
        "held_match": held_ok,
        "match": suite_ok and held_ok,
        "ratio": round(ratio, 3) if ratio is not None else None,
        "min_ratio": prereg["expect"]["min_ratio"],
    }


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_seed(prereg_path, salt=0):
    """Generator seed derived from the prereg's content plus the salt.

    The held-out set then depends on the hypothesis: a prereg written after
    seeing results generates tasks different from the ones that were seen.
    """
    digest = hashlib.sha256(
        Path(prereg_path).read_bytes() + b":" + str(salt).encode()
    ).hexdigest()
    return int(digest[:8], 16)


def verify(prereg_path, manifest_paths):
    """Verifies that the sets were generated from this prereg.

    Strong: the manifest stores the prereg's sha256 and the seed derived from it
    (`hod generate --prereg`). Returns (ok, reason). Manifests without that
    binding do not pass: mtime is trivial to forge (touch, or regenerating the
    same seed recreates the same files with a new mtime).

    If the prereg declares `seeds`, the salts of the manifests must be
    exactly that set: no undeclared salt, none left out.
    """
    digest = sha256_file(prereg_path)
    declared = read_json(prereg_path, "prereg file", expect=dict).get("seeds")
    salts = []
    for m in manifest_paths:
        manifest = read_json(m, "manifest", expect=dict)
        if manifest.get("prereg_sha256") is None:
            return False, (
                "%s was not generated with --prereg (no binding to the prereg)" % m
            )
        if manifest["prereg_sha256"] != digest:
            return False, "%s was generated with a different prereg" % m
        salt = manifest.get("seed_salt", 0)
        expected = derive_seed(prereg_path, salt)
        if manifest.get("seed") != expected:
            return False, "%s: seed does not derive from this prereg" % m
        salts.append(salt)
    if declared is not None:
        undeclared = sorted(set(salts) - set(declared), key=str)
        missing = sorted(set(declared) - set(salts), key=str)
        if undeclared:
            return False, ("salt(s) %s not declared in the prereg (declared: %s)"
                           % (undeclared, declared))
        if missing:
            return False, ("declared salt(s) %s not measured: every declared "
                           "set must be reported" % missing)
    return True, "ok"


def older_than(prereg_path, manifest_paths):
    """True if the prereg predates all manifests (no peeking)."""
    prereg_mtime = Path(prereg_path).stat().st_mtime
    return all(
        prereg_mtime < Path(m).stat().st_mtime for m in manifest_paths
    )


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--change", required=True)
    parser.add_argument("--expect-suite", default="up", choices=DIRECTIONS)
    parser.add_argument("--expect-held", default="up", choices=DIRECTIONS)
    parser.add_argument("--min-ratio", type=float, default=0.75)
    parser.add_argument("--flat-tol", type=float, default=0.0,
                        help="tolerance for 'flat' (|delta| <= tol)")
    parser.add_argument("--seeds", default=None,
                        help="commit in advance to the generation salts "
                        "(e.g. 42,43,44): bound sets must use exactly these")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        seeds = parse_seeds(args.seeds) if args.seeds else None
        data = create(args.change, args.expect_suite, args.expect_held,
                      args.min_ratio, seeds=seeds, flat_tol=args.flat_tol)
    except ValueError as e:
        print("prereg: %s" % e, file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("prereg created: %s (%s)" % (args.out, data["created_at"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
