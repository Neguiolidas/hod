#!/usr/bin/env python3
"""Noise statistics from the ledger's placebo controls.

A placebo is a pipeline round with a no-effect change. The gap observed
in it is pure noise. With enough placebos, noise stops being a guess:
the significance of a real gap is measured against the largest placebo
gap observed.

Usage:
    hod placebo --ledger runs/ledger.jsonl --out calibration.json
"""

import argparse
import json
import math
import sys
from pathlib import Path



def load_ledger(path):
    try:
        from .ledger import load
    except ImportError:  # direct execution of the file
        from ledger import load  # noqa: E402
    return load(path)


def _finite(value):
    """A usable number: int/float, not bool, not NaN/inf."""
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def stats(entries):
    usable = [
        e for e in entries
        if isinstance(e, dict) and e.get("placebo") is True
        and not e.get("wave_stopped") and _finite(e.get("gap"))
    ]
    gaps = [e["gap"] for e in usable]
    suite = [abs(e["suite_delta"]) for e in usable
             if _finite(e.get("suite_delta"))]
    held = [abs(e["held_delta"]) for e in usable
            if _finite(e.get("held_delta"))]
    if not gaps:
        return {
            "n_placebo": 0,
            "mean_gap": 0.0,
            "sd_gap": 0.0,
            "max_abs_gap": 0.0,
            "gaps": [],
        }
    mean = sum(gaps) / len(gaps)
    if len(gaps) >= 2:
        variance = sum((g - mean) ** 2 for g in gaps) / (len(gaps) - 1)
        sd = variance ** 0.5
    else:
        sd = 0.0
    return {
        "n_placebo": len(gaps),
        "mean_gap": round(mean, 3),
        "sd_gap": round(sd, 3),
        "max_abs_gap": round(max(abs(g) for g in gaps), 3),
        "max_abs_suite_delta": round(max(suite), 3) if suite else None,
        "max_abs_held_delta": round(max(held), 3) if held else None,
        "gaps": gaps,
    }


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = stats(load_ledger(args.ledger))
    except (ValueError, OSError) as e:
        print("placebo: %s" % e, file=sys.stderr)
        return 1
    print(
        "placebos: n=%d, mean gap=%+.3f, sd=%.3f, largest |gap|=%.3f"
        % (
            result["n_placebo"],
            result["mean_gap"],
            result["sd_gap"],
            result["max_abs_gap"],
        )
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("calibration written: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
