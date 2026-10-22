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
- The manifest stores id, family, file and hash for auditing.
"""


import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

try:
    from .families import FAMILIES, VARIANTS
except ImportError:  # direct execution of the file, outside the package
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from families import FAMILIES, VARIANTS  # noqa: E402

THRESHOLD = 0.30
NOVELTY_THRESHOLD = 0.60
CONTENT_SHINGLE = 3
MAX_ATTEMPTS = 50
SHINGLE = 5
DEFAULT_CATALOG = Path.home() / "workspace" / "harness-eval" / "catalog"


_WORD = re.compile(r"[^\W_]+")


def normalize(text):
    # NFC + casefold: the same text in NFC or NFD (e.g. file saved on
    # macOS) becomes the same tokens; any Unicode letter counts (à, ü, ñ).
    text = unicodedata.normalize("NFC", text).casefold()
    return _WORD.findall(text)


def shingles(words, k=SHINGLE):
    if not words:
        return set()
    if len(words) < k:
        # text short: the shingle only with the text integer, instead of set
        # empty (that dava similarity 0 and passava in the gate without compare nothing)
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
    """Similarity used in the gate: the larger between Jaccard and containment."""
    return max(jaccard(a, b), containment(a, b))


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
    """Setup + instruction of the task renderizada (the that changes between tasks
    of the family; the rest is the contrato of fix of the family)."""
    def section(title):
        if title not in text:
            return ""
        return text.split(title, 1)[1].split("###", 1)[0]
    return section("### Setup") + " " + section("### Instruction")


def content_shingles(text):
    return shingles(normalize(task_content(text)), k=CONTENT_SHINGLE)


def validate_task(task, text):
    """Gate of validity: lists of problemas (empty = valid)."""
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
        problems.append("placeholder sem preencher")
    return problems


def load_exclude(dirs):
    """Shingles of content of the tasks of sets already used."""
    buffer = []
    for d in dirs or []:
        d = Path(d)
        files = sorted(d.glob("heldout-*.md")) + sorted(d.glob("*/heldout-*.md"))
        for path in files:
            buffer.append(content_shingles(path.read_text()))
    return buffer


def max_similarity(text, catalog_shingles):
    return similarity_detail(text, catalog_shingles)[0]


def similarity_detail(text, catalog_shingles):
    """(similarity of the gate, larger Jaccard, larger containment) against the catalog."""
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
    """Weights per family by the frontier of the Ornith-1.5: families in that the
    harness acerta perto of `target` (nem saturated, nem impossible)
    rendem more tasks. `floor` keeps all family present, for that
    regressions in families saturated continuem visible. Family without rate
    known stays with weight 1."""
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


def validate_weights(family_weights):
    unknown = sorted(set(family_weights) - set(FAMILIES))
    if unknown:
        raise ValueError(
            "weights with keys that are not families: %s (families: %s)"
            % (unknown, sorted(FAMILIES))
        )
    for name, w in family_weights.items():
        if isinstance(w, bool) or not isinstance(w, (int, float)) or w < 0:
            raise ValueError("invalid weight for %s: %r" % (name, w))
    if not any(family_weights.get(f, 1.0) > 0 for f in FAMILIES):
        raise ValueError("all weights are zero")


def generate_one(seed, n, out, catalog_shingles, family_weights=None,
                 extra_manifest=None, exclude=None,
                 novelty_threshold=NOVELTY_THRESHOLD, quiet=False):
    rng = random.Random(seed)
    family_names = sorted(FAMILIES)
    deck = family_names[:]
    rng.shuffle(deck)
    weights = None
    if family_weights:
        validate_weights(family_weights)
        weights = [family_weights.get(f, 1.0) for f in family_names]
    # Sorteio Without replacement: each variant (family, index) leaves in the maximum the
    # time by set. With replacement, --n 8 repetia task in ~55% of the seeds.
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
        # Cotas per family proporcionais to the weight (larger rest), after
        # embaralhadas: estratificado the the deck, without the variance of sortear
        # family the family (the family with 5 tasks by azar deixa the gate
        # of regression per family without power).
        positive = [(f, w) for f, w in zip(family_names, weights) if w > 0]
        total_w = sum(w for _, w in positive)
        raw = [(f, n * w / total_w) for f, w in positive]
        quota = {f: int(x) for f, x in raw}
        left = n - sum(quota.values())
        for f, x in sorted(raw, key=lambda fx: (-(fx[1] - int(fx[1])), fx[0]))[:left]:
            quota[f] += 1
        plan = [f for f in family_names for _ in range(quota.get(f, 0))]
        rng.shuffle(plan)
    rejected = 0
    rejected_novelty = 0
    rejected_invalid = 0
    emitted_all = []
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
                # quota esgotada: next family with weight that still has variant
                pool = [f for f in plan if f in available]
                if not pool:
                    print("fail-closed: no family with weight and variant "
                          "available for task %d" % i, file=sys.stderr)
                    return None
                family = pool[0]
        else:
            family = deck[i % len(deck)]
            if family not in available:
                # family esgotada: next of the deck that still has variant
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

    # Only writes after that the set integer passed: fail-closed not deixa
    # files partial without manifest in the directory.
    out.mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob("heldout-*.md")) + [out / "manifest.json"]:
        stale.unlink(missing_ok=True)
    for task_id, family, text, sim, sim_j, sim_c, idx, nov in emitted_all:
        filename = "%s.md" % task_id
        (out / filename).write_text(text)
        manifest["tasks"].append(
            {
                "id": task_id,
                "family": family,
                "variant": idx,
                "file": filename,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "max_similarity": round(sim, 4),
                "max_jaccard": round(sim_j, 4),
                "max_containment": round(sim_c, 4),
                "novelty": round(nov, 4),
            }
        )

    manifest["rejected"] = {"suite_similarity": rejected,
                            "novelty": rejected_novelty,
                            "validity": rejected_invalid}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if quiet:
        return manifest
    print(
        "emitted %d tasks in %s (seed %d, rejected for similarity: %d, "
        "highest similarity: %.3f)"
        % (
            len(manifest["tasks"]),
            out,
            seed,
            rejected,
            max(t["max_similarity"] for t in manifest["tasks"]),
        )
    )
    return manifest


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--family-weights", type=Path, default=None)
    parser.add_argument(
        "--frontier", type=Path, default=None,
        help="JSON {family: baseline success rate} (e.g. output of "
        "`hod calibrate --frontier-out`); families near the target yield more "
        "tasks",
    )
    parser.add_argument("--frontier-target", type=float, default=FRONTIER_TARGET)
    parser.add_argument(
        "--exclude", type=Path, action="append", default=[],
        help="already used set (directory); its tasks become a novelty buffer "
        "and do not return in the new set. Can repeat.",
    )
    parser.add_argument(
        "--novelty", type=float, default=NOVELTY_THRESHOLD,
        help="content similarity ceiling (setup+instruction) against the "
        "set and the buffer (default %.2f)" % NOVELTY_THRESHOLD,
    )
    parser.add_argument(
        "--prereg", type=Path, default=None,
        help="derives the seed from the prereg hash (the seed becomes just a salt): "
        "a prereg written after seeing results generates another set",
    )
    args = parser.parse_args(argv)

    catalog_files = sorted(args.catalog.glob("*.md"))
    if not catalog_files:
        print("catalog empty or missing: %s" % args.catalog, file=sys.stderr)
        return 1
    catalog_shingles = []
    for path in catalog_files:
        catalog_shingles.append(shingles(normalize(path.read_text())))

    family_weights = None
    if args.family_weights:
        family_weights = json.loads(args.family_weights.read_text())
        try:
            validate_weights(family_weights)
        except ValueError as e:
            print("generate: %s" % e, file=sys.stderr)
            return 1

    if args.frontier:
        fw = frontier_weights(json.loads(args.frontier.read_text()),
                              target=args.frontier_target)
        family_weights = {f: fw[f] * (family_weights or {}).get(f, 1.0)
                          for f in FAMILIES}
    exclude = load_exclude(args.exclude)

    def seed_and_extra(salt):
        if not args.prereg:
            return salt, None
        try:
            from .prereg import derive_seed, sha256_file
        except ImportError:  # direct execution of the file
            from prereg import derive_seed, sha256_file  # noqa: E402
        return derive_seed(args.prereg, salt), {
            "prereg_sha256": sha256_file(args.prereg),
            "seed_salt": salt,
        }

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
        for salt in seeds:
            seed, extra = seed_and_extra(salt)
            manifest = generate_one(
                seed, args.n, args.out / ("seed-%d" % salt),
                catalog_shingles, family_weights, extra,
                exclude=exclude, novelty_threshold=args.novelty,
            )
            if manifest is None:
                return 1
            # seeds of the same lote also not repeat task between si
            exclude = exclude + load_exclude([args.out / ("seed-%d" % salt)])
        return 0
    seed, extra = seed_and_extra(args.seed)
    manifest = generate_one(
        seed, args.n, args.out, catalog_shingles, family_weights, extra,
        exclude=exclude, novelty_threshold=args.novelty,
    )
    return 0 if manifest is not None else 1


if __name__ == "__main__":
    sys.exit(main())
