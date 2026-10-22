#!/usr/bin/env python3
"""Audit of contamination of a held-out set.

Checks:
1. Manifest matches the directory (files and hashes).
2. No held-out task is a near-duplicate of another (pairwise, 5-grams).
3. No held-out task exceeds the similarity ceiling against the suite.

Usage: hod audit --heldout runs/heldout-42 [--catalog ~/workspace/harness-eval/catalog]
    (from the clone, without installing: python3 -m hod audit ...)
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    from .generate import THRESHOLD, jaccard, normalize, shingles, similarity
except ImportError:  # direct execution of the file, outside the package
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate import (  # noqa: E402
        THRESHOLD, jaccard, normalize, shingles, similarity,
    )


def _section(text, title):
    if title not in text:
        return ""
    return text.split(title, 1)[1].split("###", 1)[0].strip()


def main(argv=None, prog=None):
    """Checks explicit (without assert: `python -O` removia all and the
    audit passava any coisa). Returns 1 in any failure."""
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path.home() / "workspace" / "harness-eval" / "catalog",
    )
    args = parser.parse_args(argv)
    failures = []

    def fail(msg):
        failures.append(msg)
        print("FAIL: %s" % msg, file=sys.stderr)

    manifest_path = args.heldout / "manifest.json"
    if not manifest_path.exists():
        fail("manifest missing: %s" % manifest_path)
        return 1
    manifest = json.loads(manifest_path.read_text())
    files = sorted(args.heldout.glob("heldout-*.md"))
    manifest_files = sorted(t["file"] for t in manifest["tasks"])
    if [f.name for f in files] != manifest_files:
        fail("manifest differs from directory: %s vs %s"
             % ([f.name for f in files], manifest_files))
    if not manifest["tasks"]:
        fail("empty set")
    texts = {}
    for task in manifest["tasks"]:
        path = args.heldout / task["file"]
        if not path.exists():
            fail("file missing: %s" % task["file"])
            continue
        text = path.read_text()
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest != task["sha256"]:
            fail("hash differs in %s" % task["file"])
        texts[task["id"]] = text
    print("manifest and hashes: %s (%d tasks)"
          % ("ok" if not failures else "with failures", len(texts)))

    ids = sorted(texts)
    family_of = {t["id"]: t["family"] for t in manifest["tasks"]}
    sh = {t: shingles(normalize(texts[t])) for t in ids}
    worst_cross = 0.0
    worst_cross_pair = None
    worst_same = 0.0
    worst_same_pair = None
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            sim = jaccard(sh[a], sh[b])
            if family_of[a] == family_of[b]:
                if sim > worst_same:
                    worst_same, worst_same_pair = sim, (a, b)
            else:
                if sim > worst_cross:
                    worst_cross, worst_cross_pair = sim, (a, b)
    print(
        "pairwise across different families: worst pair %s with %.3f (ceiling %.2f)"
        % (worst_cross_pair, worst_cross, THRESHOLD)
    )
    print(
        "pairwise same family (variants share the template, informational only): "
        "worst pair %s with %.3f" % (worst_same_pair, worst_same)
    )
    if worst_cross > THRESHOLD:
        fail("different families collapsing: %s" % (worst_cross_pair,))
    contents = {}
    dup = False
    for task_id in ids:
        text = texts[task_id]
        key = " ".join(normalize(
            _section(text, "### Setup") + " " + _section(text, "### Instruction")
        ))
        if key in contents:
            dup = True
            fail("duplicated task (same setup and instruction): %s and %s"
                 % (contents[key], task_id))
        contents[key] = task_id
    if not dup:
        print("duplicated tasks (setup + instruction): none")

    catalog_files = sorted(args.catalog.glob("*.md"))
    if not catalog_files:
        fail("catalog empty or missing: %s" % args.catalog)
        return 1
    catalog_shingles = [shingles(normalize(p.read_text())) for p in catalog_files]
    worst_suite = 0.0
    worst_task = ids[0] if ids else None
    for task_id in ids:
        sim = max(similarity(sh[task_id], other) for other in catalog_shingles)
        if sim > worst_suite:
            worst_suite, worst_task = sim, task_id
    print(
        "held-out vs suite (max of Jaccard and containment): worst %.3f in %s "
        "(ceiling %.2f)" % (worst_suite, worst_task, THRESHOLD)
    )
    if worst_suite > THRESHOLD:
        fail("held-out task above the ceiling: %s" % worst_task)
    if failures:
        print("audit: FAILED (%d problems)" % len(failures))
        return 1
    print("audit: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
