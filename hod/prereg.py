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

Usage:
    hod prereg --change "new routing rule" --out prereg.json
    hod prereg --change "..." --expect-held flat --min-ratio 0.5 --out prereg.json
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DIRECTIONS = ("up", "flat", "down")


def create(change, expect_suite="up", expect_held="up", min_ratio=0.75):
    if expect_suite not in DIRECTIONS or expect_held not in DIRECTIONS:
        raise ValueError("invalid direction: use up, flat or down")
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio outside [0, 1]: %s" % min_ratio)
    return {
        "change": change,
        "expect": {
            "suite": expect_suite,
            "held": expect_held,
            "min_ratio": min_ratio,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load(path):
    data = json.loads(Path(path).read_text())
    if data["expect"]["suite"] not in DIRECTIONS:
        raise ValueError("invalid suite direction in %s" % path)
    if data["expect"]["held"] not in DIRECTIONS:
        raise ValueError("invalid held direction in %s" % path)
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
        # min_ratio only does sentido when the suite rose; with suite stopped or
        # caindo, "held rises" basta (before that never dava match)
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
    """Seed of the generator derivada of the content of the prereg more the sal.

    The held-out set passes the depender of the hypothesis: the prereg written
    after see results generates tasks different of the that were vistas.
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
    """
    digest = sha256_file(prereg_path)
    for m in manifest_paths:
        manifest = json.loads(Path(m).read_text())
        if manifest.get("prereg_sha256") is None:
            return False, (
                "%s was not generated with --prereg (no binding to the prereg)" % m
            )
        if manifest["prereg_sha256"] != digest:
            return False, "%s was generated with a different prereg" % m
        expected = derive_seed(prereg_path, manifest.get("seed_salt", 0))
        if manifest.get("seed") != expected:
            return False, "%s: seed does not derive from this prereg" % m
    return True, "ok"


def older_than(prereg_path, manifest_paths):
    """True if the prereg is previous the all manifests (without peeking)."""
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    data = create(
        args.change, args.expect_suite, args.expect_held, args.min_ratio
    )
    if args.flat_tol:
        data["expect"]["flat_tol"] = args.flat_tol
    args.out.write_text(json.dumps(data, indent=2))
    print("prereg created: %s (%s)" % (args.out, data["created_at"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
