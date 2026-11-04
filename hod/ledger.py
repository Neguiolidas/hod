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
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None
try:
    import msvcrt
except ImportError:  # POSIX
    msvcrt = None

PRIOR = 1.0


@contextmanager
def _locked(f, path):
    """Exclusive lock while appending. POSIX: flock on the ledger itself.
    Windows (best effort, untested in CI): msvcrt lock on a side file
    `<ledger>.lock`, because append mode there is seek + write."""
    if fcntl:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    elif msvcrt:
        with open(str(path) + ".lock", "a+b") as lock:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        yield


def append(path, entry):
    """Appends one line with an exclusive lock and fsync.

    One write per line, under the lock: the pipeline and the canary hook
    running at the same time do not interleave bytes; fsync avoids a
    truncated line after a power loss. If a previous writer died mid-line
    (the file does not end in a newline), the torn line is closed first, so
    it costs only itself and never swallows the entry written after it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
    with open(path, "a+b") as f:
        with _locked(f, path):
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    data = b"\n" + data
                f.seek(0, os.SEEK_END)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    return entry


def load(path, strict=False):
    """Reads the ledger; corrupted lines (invalid JSON, or UTF-8 cut in the
    middle of a character) are skipped with a warning (strict=True raises).
    A truncated line never takes the whole report down."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("ledger missing: %s" % path)
    entries = []
    for lineno, raw in enumerate(path.read_bytes().split(b"\n"), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw.decode("utf-8"))
        except ValueError as e:  # JSONDecodeError and UnicodeDecodeError
            if strict:
                raise ValueError("ledger %s line %d: %s" % (path, lineno, e))
            print("warning: ledger %s line %d corrupted, ignored"
                  % (path, lineno), file=sys.stderr)
            continue
        if isinstance(entry, dict):
            entries.append(entry)
        elif strict:
            raise ValueError("ledger %s line %d: not a JSON object"
                             % (path, lineno))
    return entries


def is_placebo(entry):
    """Only a literal `true` marks a placebo ("false" as a string does not)."""
    return entry.get("placebo") is True


def report(entries):
    real = [e for e in entries if not is_placebo(e)]
    placebo = [e for e in entries if is_placebo(e)]
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
    try:
        result = report(load(args.ledger))
    except (ValueError, OSError) as e:
        print("ledger: %s" % e, file=sys.stderr)
        return 1
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
        args.weights_out.parent.mkdir(parents=True, exist_ok=True)
        args.weights_out.write_text(json.dumps(result["weights"], indent=2),
                                    encoding="utf-8")
        print("weights written: %s" % args.weights_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
