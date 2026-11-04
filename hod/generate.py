#!/usr/bin/env python3
"""Generates held-out tasks in catalog format, with a disjointness checker.

Usage:
    hod generate --seed 42 --n 8 --out runs/heldout-42
    (from the clone, without installing: python3 -m hod generate --seed 42 --n 8 --out runs/heldout-42)

Rules:
- The seed reproduces the exact set (same families, variants and hashes).
- No emitted task may exceed the similarity ceiling against the fixed
  suite: if no variant passes after the attempts, the generator fails
  instead of emitting a contaminated task (fail-closed).
- Validity (hard gate, the V of Ornith-1.5): the rendered task may not
  have an empty field, a leftover placeholder or an incomplete rubric.
- Novelty (the N of Ornith-1.5): the setup + instruction of each new task
  is compared with the tasks already emitted in the set and with a buffer
  of "burned" tasks (`--exclude`, sets already used in earlier decisions).
  Above NOVELTY_THRESHOLD (Jaccard on 3-grams of the content) the candidate
  is discarded. This is what makes held-out rotation work: a set used to
  decide does not come back disguised in the next one.
- The manifest stores id, family, file and hash (sha256 of the exact bytes
  written) for auditing.
- Writing is fail-closed: a set is written to a temporary sibling directory
  and moved into place only when complete, and an existing --out is
  replaced only if it is empty or already a HOD set.
"""


import argparse
import functools
import hashlib
import json
import os
import random
import re
import secrets
import shutil
import sys
import unicodedata
from pathlib import Path

try:
    from .families import FAMILIES, VARIANTS
    from .manifest import load as load_manifest
    from .scores import read_json
except ImportError:  # direct execution of the file, outside the package
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from families import FAMILIES, VARIANTS  # noqa: E402
    from manifest import load as load_manifest  # noqa: E402
    from scores import read_json  # noqa: E402

THRESHOLD = 0.30
NOVELTY_THRESHOLD = 0.60
CONTENT_SHINGLE = 3
MAX_ATTEMPTS = 50
SHINGLE = 5
MAX_WEIGHT = 1e6
DEFAULT_CATALOG = Path.home() / "hod" / "catalog"
# names a set directory may hold besides its own files that are safe to drop
DISPOSABLE = {".DS_Store", "Thumbs.db", "desktop.ini"}
_SET_FILE = re.compile(r"heldout-.*\.md\Z", re.S)


def _mark_ranges():
    """Code point ranges of the Unicode combining marks (Mn, Mc, Me).

    Marks only exist in planes 0-1 and in the variation selectors
    supplement (U+E0100-U+E01EF), so only those are scanned."""
    ranges = []
    for lo, hi in ((0, 0x20000), (0xE0100, 0xE01F0)):
        start = None
        for cp in range(lo, hi):
            if unicodedata.category(chr(cp))[0] == "M":
                if start is None:
                    start = cp
            elif start is not None:
                ranges.append((start, cp - 1))
                start = None
        if start is not None:
            ranges.append((start, hi - 1))
    return ranges


@functools.lru_cache(maxsize=None)
def _word_pattern():
    marks = "".join("\\U%08x-\\U%08x" % r for r in _mark_ranges())
    # a word starts with a letter or digit and may carry combining marks
    # (e.g. the dot of "i̇" from casefolding "İ" has no precomposed form)
    return re.compile(r"\w[\w%s]*" % marks)


def normalize(text):
    # NFC + casefold + NFC: the same text in NFC or NFD (e.g. a file saved on
    # macOS) becomes the same tokens; any Unicode letter counts (à, ü, ñ).
    # Casefolding can produce sequences that are not NFC (İ -> i + U+0307),
    # so NFC is applied again, and combining marks stay inside the word.
    text = unicodedata.normalize(
        "NFC", unicodedata.normalize("NFC", text).casefold())
    return _word_pattern().findall(text.replace("_", " "))


def shingles(words, k=SHINGLE):
    if not words:
        return set()
    if len(words) < k:
        # short text: a single shingle with the whole text, instead of an
        # empty set (which gave similarity 0 and passed the gate unchecked)
        return {tuple(words)}
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a, b):
    """Fraction of the shingles of `a` that appear in `b` (|A∩B| / |A|).

    Jaccard dilutes when the catalog file is much larger than the task:
    a full copy of the task inside a 4x larger file gives Jaccard
    ~0.25 and would pass the ceiling. Containment catches that case.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)


def similarity(a, b):
    """Similarity used in the gate: the larger of Jaccard and containment."""
    return max(jaccard(a, b), containment(a, b))


def read_text(path, what="file"):
    """Reads `path` as UTF-8 (never the locale encoding), with clean errors."""
    try:
        return Path(path).read_bytes().decode("utf-8")
    except OSError as e:
        raise ValueError("cannot read %s %s: %s" % (what, path, e))
    except UnicodeDecodeError as e:
        raise ValueError("%s is not UTF-8: %s (%s)" % (what, path, e))


def render(task_id, task):
    lines = [
        "# ID: %s" % task_id,
        "Category: %s. Target time: up to 10 min." % task["category"],
        "",
        "### Setup",
        task["setup"],
        "",
        "### Instruction",
        task["instruction"],
        "",
        "### Expected behavior",
    ]
    lines += ["- %s" % e for e in task["expected"]]
    lines += ["", "### Auto-fail", task["auto_fail"], "", "### Rubric"]
    for score in (2, 1, 0):
        lines.append("- %d: %s" % (score, task["rubric"][score]))
    return "\n".join(lines) + "\n"


def task_content(text):
    """Setup + instruction of the rendered task (what changes between tasks
    of a family; the rest is the family's fixed grading contract)."""
    def section(title):
        if title not in text:
            return ""
        return text.split(title, 1)[1].split("###", 1)[0]
    return section("### Setup") + " " + section("### Instruction")


def content_shingles(text):
    return shingles(normalize(task_content(text)), k=CONTENT_SHINGLE)


def validate_task(task, text):
    """Validity gate: list of problems (empty = valid)."""
    problems = []
    for key in ("setup", "instruction", "auto_fail"):
        if not str(task.get(key, "")).strip():
            problems.append("empty field: %s" % key)
    if not task.get("expected") or not all(str(e).strip()
                                           for e in task["expected"]):
        problems.append("empty expected behavior")
    rubric = task.get("rubric") or {}
    if set(rubric) != {0, 1, 2} or not all(str(v).strip()
                                           for v in rubric.values()):
        problems.append("rubric needs 0, 1 and 2 filled")
    if re.search(r"\{[A-Za-z_][A-Za-z_0-9]*\}", text):
        problems.append("unfilled placeholder")
    return problems


def load_exclude(dirs):
    """Content shingles of the tasks of sets already used (novelty buffer).

    Each entry is a set directory or a parent of set directories (one level
    down, e.g. the --out of a --seeds batch). Fail-closed: a directory that
    does not exist or holds no task files is an error, not an empty buffer,
    so a typo in --exclude cannot silently let a burned set come back."""
    buffer = []
    for d in dirs or []:
        d = Path(d)
        if not d.is_dir():
            raise ValueError("excluded set is not a directory: %s" % d)
        files = sorted(d.glob("heldout-*.md")) + sorted(d.glob("*/heldout-*.md"))
        if not files:
            raise ValueError("excluded set has no task files (heldout-*.md, "
                             "directly or in its seed-N sets): %s" % d)
        for path in files:
            buffer.append(content_shingles(read_text(path, "excluded task")))
    return buffer


def max_similarity(text, catalog_shingles):
    return similarity_detail(text, catalog_shingles)[0]


def similarity_detail(text, catalog_shingles):
    """(gate similarity, largest Jaccard, largest containment) against the catalog."""
    mine = shingles(normalize(text))
    best_j = best_c = 0.0
    for other in catalog_shingles:
        best_j = max(best_j, jaccard(mine, other))
        best_c = max(best_c, containment(mine, other))
    return max(best_j, best_c), best_j, best_c


FRONTIER_TARGET = 0.35
FRONTIER_SIGMA = 0.25
FRONTIER_FLOOR = 0.25


def frontier_weights(family_success, target=FRONTIER_TARGET,
                     sigma=FRONTIER_SIGMA, floor=FRONTIER_FLOOR):
    """Weights per family from the Ornith-1.5 frontier: families where the
    harness succeeds near `target` (neither saturated nor impossible) yield
    more tasks. `floor` keeps every family present, so that regressions in
    saturated families stay visible. A family without a known rate gets
    weight 1. (CLI input is checked with validate_frontier first.)"""
    import math
    weights = {}
    for name in FAMILIES:
        p = family_success.get(name)
        if p is None:
            weights[name] = 1.0
            continue
        d = math.exp(-((float(p) - target) ** 2) / (2 * sigma ** 2))
        weights[name] = round(max(floor, d), 4)
    return weights


def _check_family_map(data, what, low, high):
    """Keys must be families, values numbers in [low, high] (finite)."""
    if not isinstance(data, dict):
        raise ValueError("%s must be a JSON object {family: number}" % what)
    unknown = sorted(str(k) for k in set(data) - set(FAMILIES))
    if unknown:
        raise ValueError(
            "%s with keys that are not families: %s (families: %s)"
            % (what, unknown, sorted(FAMILIES))
        )
    for name, w in data.items():
        # the range check also rejects NaN, inf and ints too large for a float
        if isinstance(w, bool) or not isinstance(w, (int, float)) \
                or not low <= w <= high:
            raise ValueError("invalid %s value for %s: %r (expected a finite "
                             "number between %g and %g)"
                             % (what, name, w, low, high))


def validate_weights(family_weights):
    _check_family_map(family_weights, "weights", 0.0, MAX_WEIGHT)
    if not any(family_weights.get(f, 1.0) > 0 for f in FAMILIES):
        raise ValueError("all weights are zero")


def validate_frontier(family_success):
    """--frontier input: {family: baseline success rate in [0, 1]}."""
    _check_family_map(family_success, "frontier", 0.0, 1.0)


# --- writing sets (fail-closed) ----------------------------------------------

def check_set_dir(out, what="--out"):
    """Cleanup ownership: `out` may be replaced only if it does not exist,
    is an empty directory, or already holds a HOD set (a manifest.json that
    manifest.load accepts and that has HOD content) with nothing else in it.
    Anything else (a project directory, a browser extension's manifest.json,
    a file) raises ValueError instead of being overwritten. Returns the
    existing set's manifest, or None."""
    out = Path(out)
    if not os.path.lexists(out):
        return None
    if not out.is_dir():
        raise ValueError("%s exists and is not a directory: %s" % (what, out))
    entries = [e.name for e in out.iterdir()]
    if not entries:
        return None
    if "manifest.json" not in entries:
        raise ValueError(
            "%s is a non-empty directory that is not a HOD set (no "
            "manifest.json): %s; refusing to overwrite it, use a new or "
            "empty directory" % (what, out))
    try:
        manifest = load_manifest(out / "manifest.json")
    except ValueError as e:
        raise ValueError(
            "%s holds a manifest.json that is not a HOD set manifest (%s); "
            "refusing to overwrite it, use a new or empty directory"
            % (what, e))
    if not (manifest["tasks"] or any(k in manifest for k in
                                     ("seed", "threshold", "calibration"))):
        raise ValueError(
            "%s holds a manifest.json without HOD set content: %s; refusing "
            "to overwrite it, use a new or empty directory" % (what, out))
    owned = {"manifest.json"} | {t["file"] for t in manifest["tasks"]}
    foreign = sorted(n for n in entries if n not in owned
                     and n not in DISPOSABLE and not _SET_FILE.match(n))
    if foreign:
        raise ValueError(
            "%s holds entries that are not part of its HOD set (%s): %s; "
            "refusing to replace it, move them away or use a new directory"
            % (what, ", ".join(foreign[:5]) + (", ..." if len(foreign) > 5
                                               else ""), out))
    return manifest


def _fresh_sibling(path, tag):
    """Creates an empty hidden sibling directory of `path` (umask applies)."""
    while True:
        candidate = path.parent / (".%s.hod-%s-%s"
                                   % (path.name, tag, secrets.token_hex(4)))
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue


def install_set(out, files):
    """Writes `files` ({name: bytes}, manifest.json last) as the set `out`.

    The set is written to a temporary sibling directory and renamed into
    place only once complete: a failure never leaves a partial set nor
    wipes the previous one. Returns (undo, commit): `commit()` drops the
    previous set; `undo()` puts the previous state back (used to roll back
    a --seeds batch). Raises ValueError if `out` is not ours to replace
    (see check_set_dir), OSError if writing fails."""
    check_set_dir(out)
    out = Path(os.path.abspath(out))
    if out.is_symlink():  # replace the real directory, keep the link
        out = Path(os.path.realpath(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = _fresh_sibling(out, "new")
    try:
        for name, data in files.items():
            (tmp / name).write_bytes(data)
        if not os.path.lexists(out):
            os.rename(tmp, out)
            return (lambda: shutil.rmtree(out, ignore_errors=True),
                    lambda: None)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    backup = tmp.with_name(tmp.name.replace(".hod-new-", ".hod-old-"))
    try:
        cwd = Path.cwd().resolve()
        if cwd == out.resolve() or out.resolve() in cwd.parents:
            # moving the working directory away would strand the process
            raise OSError("--out holds the working directory")
        os.rename(out, backup)
    except OSError:
        # `out` cannot be moved (e.g. a mount point): replace its files in
        # place, manifest removed first and written last, so an interruption
        # leaves no manifest (not a set) rather than a mixed set
        try:
            (out / "manifest.json").unlink(missing_ok=True)
            for entry in out.iterdir():  # only set files (check_set_dir)
                if entry.is_file() or entry.is_symlink():
                    entry.unlink()
            for name in files:
                shutil.move(str(tmp / name), str(out / name))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        def undo_in_place():
            for name in files:
                (out / name).unlink(missing_ok=True)
        return undo_in_place, lambda: None
    try:
        os.rename(tmp, out)
    except BaseException:
        os.rename(backup, out)
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    def undo():
        shutil.rmtree(out, ignore_errors=True)
        os.rename(backup, out)
    return undo, lambda: shutil.rmtree(backup, ignore_errors=True)


# --- generation ----------------------------------------------------------

def build_set(seed, n, catalog_shingles, family_weights=None,
              extra_manifest=None, exclude=None,
              novelty_threshold=NOVELTY_THRESHOLD, quiet=False):
    """Builds a set in memory, without touching the disk.

    Returns (manifest, files, shingles): files is {name: bytes} with
    manifest.json last, shingles the content shingles of the emitted tasks
    (to exclude them from the next set of a batch). None if fail-closed."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        print("fail-closed: --n must be a positive integer, got %r" % (n,),
              file=sys.stderr)
        return None
    rng = random.Random(seed)
    family_names = sorted(FAMILIES)
    deck = family_names[:]
    rng.shuffle(deck)
    weights = None
    if family_weights:
        validate_weights(family_weights)
        weights = [family_weights.get(f, 1.0) for f in family_names]
    # Sampling without replacement: each variant (family, index) appears at
    # most once per set. With replacement, --n 8 repeated a task in ~55% of
    # the seeds.
    remaining = {
        f: rng.sample(range(len(VARIANTS[f])), len(VARIANTS[f]))
        for f in family_names
    }
    total = sum(len(v) for v in VARIANTS.values())
    if n > total:
        print(
            "fail-closed: --n %d larger than the number of distinct tasks "
            "available (%d)" % (n, total),
            file=sys.stderr,
        )
        return None
    manifest = {"seed": seed, "threshold": THRESHOLD,
                "similarity": "max(jaccard, containment) on 5-grams",
                "novelty_threshold": novelty_threshold,
                "novelty": "1 - max jaccard of setup+instruction (3-grams) "
                           "against the set and the excluded buffer",
                "n_excluded": len(exclude or []),
                "space_size": total,
                "tasks": []}
    if extra_manifest:
        manifest.update(extra_manifest)
    plan = []
    if weights:
        # Per-family quotas proportional to the weight (largest remainder),
        # then shuffled: stratified like the deck, without the variance of
        # drawing family by family (a family that gets 5 tasks by chance
        # leaves the per-family regression gate without power).
        positive = [(f, w) for f, w in zip(family_names, weights) if w > 0]
        total_w = sum(w for _, w in positive)
        raw = [(f, n * w / total_w) for f, w in positive]
        quota = {f: int(x) for f, x in raw}
        left = n - sum(quota.values())
        # Ties in the remainder are broken by a seeded order, not by name:
        # with equal weights the alphabetically first families would get
        # the extra tasks in every seed. A separate stream keeps the main
        # rng (and so every set without ties) unchanged.
        tie_order = family_names[:]
        random.Random("%s:quota-ties" % (seed,)).shuffle(tie_order)
        rank = {f: i for i, f in enumerate(tie_order)}
        for f, x in sorted(raw, key=lambda fx: (-(fx[1] - int(fx[1])),
                                                rank[fx[0]]))[:left]:
            quota[f] += 1
        plan = [f for f in family_names for _ in range(quota.get(f, 0))]
        rng.shuffle(plan)
    rejected = 0
    rejected_novelty = 0
    rejected_invalid = 0
    emitted_all = []
    emitted_shingles = []
    buffer = list(exclude or [])

    for i in range(n):
        available = [f for f in family_names if remaining[f]]
        if not available:
            print(
                "fail-closed: variants exhausted for task %d (rejected "
                "for similarity: %d)" % (i, rejected),
                file=sys.stderr,
            )
            return None
        if weights:
            family = plan[i] if i < len(plan) else None
            if family not in available:
                # quota exhausted: next family with weight that still has a variant
                pool = [f for f in plan if f in available]
                if not pool:
                    print("fail-closed: no family with weight and variant "
                          "available for task %d" % i, file=sys.stderr)
                    return None
                family = pool[0]
        else:
            family = deck[i % len(deck)]
            if family not in available:
                # family exhausted: next one in the deck that still has a variant
                start = i % len(deck)
                order = deck[start:] + deck[:start]
                family = next(f for f in order if f in available)
        emitted = None
        attempts = 0
        while remaining[family] and attempts < MAX_ATTEMPTS:
            attempts += 1
            idx = remaining[family].pop()
            task = FAMILIES[family](VARIANTS[family][idx])
            task_id = "heldout-%s-%02d" % (family, i)
            text = render(task_id, task)
            if validate_task(task, text):
                rejected_invalid += 1
                continue
            mine = content_shingles(text)
            closest = max((jaccard(mine, other) for other in buffer),
                          default=0.0)
            if closest > novelty_threshold:
                rejected_novelty += 1
                continue
            sim, sim_j, sim_c = similarity_detail(text, catalog_shingles)
            if sim <= THRESHOLD:
                emitted = (task_id, family, text, sim, sim_j, sim_c, idx,
                           1.0 - closest)
                buffer.append(mine)
                emitted_shingles.append(mine)
                break
            rejected += 1
        if emitted is None:
            if not quiet:
                print(
                    "fail-closed: no variant for task %d (family %s) "
                    "passed the gates (similarity with the suite: %d, "
                    "novelty: %d, validity: %d)"
                    % (i, family, rejected, rejected_novelty, rejected_invalid),
                    file=sys.stderr,
                )
            return None
        emitted_all.append(emitted)

    files = {}
    for task_id, family, text, sim, sim_j, sim_c, idx, nov in emitted_all:
        filename = "%s.md" % task_id
        # the hash covers exactly the bytes written (no newline translation,
        # no locale encoding): `hod audit` hashes the raw bytes back
        data = text.encode("utf-8")
        files[filename] = data
        manifest["tasks"].append(
            {
                "id": task_id,
                "family": family,
                "variant": idx,
                "file": filename,
                "sha256": hashlib.sha256(data).hexdigest(),
                "max_similarity": round(sim, 4),
                "max_jaccard": round(sim_j, 4),
                "max_containment": round(sim_c, 4),
                "novelty": round(nov, 4),
            }
        )

    manifest["rejected"] = {"suite_similarity": rejected,
                            "novelty": rejected_novelty,
                            "validity": rejected_invalid}
    files["manifest.json"] = json.dumps(manifest, indent=2).encode("utf-8")
    return manifest, files, emitted_shingles


def _summary(manifest, out):
    print(
        "emitted %d tasks in %s (seed %d, rejected for similarity: %d, "
        "highest similarity: %.3f)"
        % (
            len(manifest["tasks"]),
            out,
            manifest["seed"],
            manifest["rejected"]["suite_similarity"],
            max(t["max_similarity"] for t in manifest["tasks"]),
        )
    )


def generate_one(seed, n, out, catalog_shingles, family_weights=None,
                 extra_manifest=None, exclude=None,
                 novelty_threshold=NOVELTY_THRESHOLD, quiet=False):
    """Generates one set at `out`; returns its manifest, or None (fail-closed,
    in which case `out` is left as it was)."""
    out = Path(out)
    try:
        check_set_dir(out, "output directory")
    except ValueError as e:
        print("fail-closed: %s" % e, file=sys.stderr)
        return None
    built = build_set(seed, n, catalog_shingles, family_weights,
                      extra_manifest, exclude, novelty_threshold, quiet)
    if built is None:
        return None
    manifest, files, _ = built
    try:
        _, commit = install_set(out, files)
    except (OSError, ValueError) as e:
        print("fail-closed: could not write the set in %s: %s" % (out, e),
              file=sys.stderr)
        return None
    commit()
    if not quiet:
        _summary(manifest, out)
    return manifest


def parse_seeds(text):
    seeds = []
    for item in text.split(","):
        try:
            seeds.append(int(item.strip()))
        except ValueError:
            raise ValueError("--seeds: %r is not an integer (expected a comma-"
                             "separated list, e.g. 1,2,3)" % item.strip())
    repeated = sorted({s for s in seeds if seeds.count(s) > 1})
    if repeated:
        raise ValueError("--seeds: repeated seeds %s (each seed is one seed-N "
                         "set)" % repeated)
    return seeds


def check_batch_out(out, seeds):
    """A --seeds batch owns the seed-N sets of its --out: an --out already
    holding other seed-* sets (or a single set) is refused, because `hod
    pipeline` would discover them all and decide on sets that were not
    generated disjoint from this batch."""
    out = Path(out)
    if not os.path.lexists(out):
        return
    if not out.is_dir():
        raise ValueError("--out exists and is not a directory: %s" % out)
    if (out / "manifest.json").exists():
        raise ValueError("--out %s is a single set (has manifest.json); a "
                         "--seeds batch writes seed-N sets: use a new --out"
                         % out)
    wanted = {"seed-%d" % s for s in seeds}
    foreign = sorted(e.name for e in out.iterdir()
                     if e.name.startswith("seed-") and e.name not in wanted)
    if foreign:
        raise ValueError(
            "--out %s already holds sets that are not in this batch (%s); "
            "`hod pipeline` would discover them all and they would overlap "
            "with the new ones: use a new --out (with --exclude %s to keep "
            "the new batch disjoint from them), or rerun the whole batch"
            % (out, ", ".join(foreign), out))
    for name in sorted(wanted):
        check_set_dir(out / name, "--out set %s" % name)


def _positive_int(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected an integer, got %r" % text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1, got %d" % value)
    return value


def _unit_float(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a number, got %r" % text)
    if not 0.0 <= value <= 1.0:  # also rejects nan and inf
        raise argparse.ArgumentTypeError("must be between 0 and 1, got %r"
                                         % text)
    return value


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seeds: writes one disjoint set "
                        "per seed in OUT/seed-N")
    parser.add_argument("--n", type=_positive_int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--family-weights", type=Path, default=None)
    parser.add_argument(
        "--frontier", type=Path, default=None,
        help="JSON {family: baseline success rate} (e.g. output of "
        "`hod calibrate --frontier-out`); families near the target yield more "
        "tasks",
    )
    parser.add_argument("--frontier-target", type=_unit_float,
                        default=FRONTIER_TARGET)
    parser.add_argument(
        "--exclude", type=Path, action="append", default=[],
        help="already used set (directory); its tasks become a novelty buffer "
        "and do not return in the new set. Can repeat.",
    )
    parser.add_argument(
        "--novelty", type=_unit_float, default=NOVELTY_THRESHOLD,
        help="content similarity ceiling (setup+instruction) against the "
        "set and the buffer (default %.2f)" % NOVELTY_THRESHOLD,
    )
    parser.add_argument(
        "--prereg", type=Path, default=None,
        help="derives the seed from the prereg hash (the seed becomes just a salt): "
        "a prereg written after seeing results generates another set",
    )
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except ValueError as e:
        print("generate: %s" % e, file=sys.stderr)
        return 1


def _run(args):
    seeds = parse_seeds(args.seeds) if args.seeds else None
    if seeds is not None:
        check_batch_out(args.out, seeds)
    else:
        check_set_dir(args.out)
    if args.prereg is not None and not args.prereg.is_file():
        raise ValueError("--prereg file missing: %s" % args.prereg)

    catalog_files = sorted(args.catalog.glob("*.md"))
    if not catalog_files:
        raise ValueError("catalog empty or missing: %s" % args.catalog)
    catalog_shingles = [shingles(normalize(read_text(p, "catalog file")))
                        for p in catalog_files]

    family_weights = None
    if args.family_weights:
        family_weights = read_json(args.family_weights, "--family-weights file",
                                   expect=dict)
        validate_weights(family_weights)
    if args.frontier:
        success = read_json(args.frontier, "--frontier file", expect=dict)
        validate_frontier(success)
        fw = frontier_weights(success, target=args.frontier_target)
        family_weights = {f: fw[f] * (family_weights or {}).get(f, 1.0)
                          for f in FAMILIES}
        validate_weights(family_weights)
    exclude = load_exclude(args.exclude)

    def seed_and_extra(salt):
        if not args.prereg:
            return salt, None
        try:
            from .prereg import derive_seed, sha256_file
        except ImportError:  # direct execution of the file
            from prereg import derive_seed, sha256_file  # noqa: E402
        try:
            return derive_seed(args.prereg, salt), {
                "prereg_sha256": sha256_file(args.prereg),
                "seed_salt": salt,
            }
        except OSError as e:
            raise ValueError("cannot read --prereg %s: %s" % (args.prereg, e))

    if seeds is None:
        seed, extra = seed_and_extra(args.seed)
        manifest = generate_one(
            seed, args.n, args.out, catalog_shingles, family_weights, extra,
            exclude=exclude, novelty_threshold=args.novelty,
        )
        return 0 if manifest is not None else 1

    # A batch is all or nothing: every set is built in memory first (each
    # excluding the tasks of the previous ones), then all are written; if a
    # write fails, the sets already written by this run are rolled back.
    built = []
    for salt in seeds:
        seed, extra = seed_and_extra(salt)
        result = build_set(seed, args.n, catalog_shingles, family_weights,
                           extra, exclude=exclude,
                           novelty_threshold=args.novelty)
        if result is None:
            print("generate: seed %d failed; nothing was written (a --seeds "
                  "batch is all or nothing)" % salt, file=sys.stderr)
            return 1
        manifest, files, new_shingles = result
        built.append((args.out / ("seed-%d" % salt), manifest, files))
        # seeds of the same batch do not repeat tasks between themselves
        exclude = exclude + new_shingles
    installed = []
    try:
        for out, _, files in built:
            installed.append(install_set(out, files))
    except (OSError, ValueError) as e:
        for undo, _ in reversed(installed):
            undo()
        print("generate: could not write the batch (%s); the sets written by "
              "this run were rolled back" % e, file=sys.stderr)
        return 1
    for _, commit in installed:
        commit()
    for out, manifest, _ in built:
        _summary(manifest, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
