"""Toy harness, deterministic and offline, to prove `hod improve`.

There is no model here. The "harness" is JSON with knobs:

- `skills`: {family: 0..1}. Genuine skill: counts on any task of the
  family, in the suite or in the held-out.
- `memorized`: lists of ids of SUITE tasks whose answers the harness
  memorized. A memorized task scores 2; on a new task it helps nothing.
  This is the overfit knob.
- `style`: cosmetic, does not change the score (placebo).

Task score: hit probability p = 0.15 + 0.8*skill of the family
(clamped to [0, 1]); two attempts drawn with hash(task, round); the
score is the number of hits (0, 1 or 2). Same input and same round =
same score; different rounds have noise, like a real agent.

It also provides the toy catalog (suite), the scripted proposer that
mixes genuine, overfit, placebo and regression-trade changes, and the
CLI to use all of that via commands (`python -m hod.toy eval|propose|...`).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    from .families import FAMILIES
except ImportError:  # direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from families import FAMILIES  # noqa: E402

BASE_P = 0.15
SKILL_P = 0.80

SUITE_WORDS = [
    "warehouse", "invoice", "schedule", "dispatch", "stock", "bill",
    "crane", "hangar", "inventory", "gardening", "laundry",
    "maintenance", "notification", "workshop", "assets", "kiosk",
]


def _u(*parts):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def score_task(config, task, run_seed):
    if task["id"] in set(config.get("memorized", [])):
        return 2
    skill = float(config.get("skills", {}).get(task.get("family"), 0.0))
    p = min(1.0, max(0.0, BASE_P + SKILL_P * skill))
    return sum(1 for j in (0, 1) if _u(task["id"], run_seed, j) < p)


def evaluate(config, tasks, run_seed):
    return {t["id"]: score_task(config, t, run_seed) for t in tasks}


def write_catalog(out_dir, per_family=4):
    """Toy suite: texts with their own vocabulary (so the disjointness
    checker does not confuse them with the generator families) and the
    id -> family map in suite_families.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    families = {}
    k = 0
    for fam in sorted(FAMILIES):
        for j in range(per_family):
            word = SUITE_WORDS[k % len(SUITE_WORDS)]
            task_id = "suite-%02d" % k
            text = (
                "# %s\n\nFixed regression case number %d, area %s.\n\n"
                "Catalog context: spreadsheet of %s with %d rows, operator "
                "fictitious %s, shift %s.\nControl question: does the recorded "
                "behavior match internal norm %d-%s?\n"
                % (task_id, k, fam, word, 10 + 7 * k, word.upper(),
                   "AB"[j % 2], 100 + k, word[:3])
            )
            (out_dir / ("%s.md" % task_id)).write_text(text)
            families[task_id] = fam
            k += 1
    (out_dir / "suite_families.json").write_text(json.dumps(families, indent=2))
    return families


def initial_config():
    return {"skills": {f: 0.3 for f in sorted(FAMILIES)}, "memorized": [],
            "estilo": "neutro"}


def known_good():
    return {"skills": {f: 0.95 for f in sorted(FAMILIES)}, "memorized": []}


def known_bad():
    return {"skills": {f: 0.0 for f in sorted(FAMILIES)}, "memorized": []}


def _clone(cfg):
    return json.loads(json.dumps(cfg))


def propose(champion, history, k, round_index):
    """Scripted proposer. Uses only what an honest proposer can see:
    the champion, the champion's scores on the SUITE (public) and the
    aggregated history. Never sees per-task held-out scores."""
    suite_scores = {}
    for entry in reversed(history):
        if entry.get("champion_suite_scores"):
            suite_scores = entry["champion_suite_scores"]
            break
    fams = sorted(FAMILIES)
    skills = champion.get("skills", {})
    out = []

    # 1. overfit: memorize the tasks of the suite in that the champion not tirou 2
    c = _clone(champion)
    weak = sorted(t for t, s in suite_scores.items() if s < 2)
    c["memorized"] = sorted(set(c.get("memorized", [])) | set(weak))
    c["change"] = "memorize %d suite tasks that still fail" % len(weak)
    c["kind"] = "overfit"
    out.append(c)

    # 2. genuine: rule geral new, +0,3 in all families
    c = _clone(champion)
    c["skills"] = {f: min(1.0, round(skills.get(f, 0) + 0.3, 3)) for f in fams}
    c["change"] = "general verification rule (+0.3 in all families)"
    c["kind"] = "genuine"
    out.append(c)

    # 3. placebo: changes only the style
    c = _clone(champion)
    c["estilo"] = "formal" if champion.get("estilo") != "formal" else "neutro"
    c["change"] = "change the prompt style (cosmetic)"
    c["kind"] = "placebo"
    out.append(c)

    # 4. swaps with regression: the mean rises, but the family despenca
    c = _clone(champion)
    victim = fams[round_index % len(fams)]
    c["skills"] = {f: (0.0 if f == victim
                       else min(1.0, round(skills.get(f, 0) + 0.45, 3)))
                   for f in fams}
    c["change"] = "aggressive shortcut (+0.45 everywhere, but zeroes %s)" % victim
    c["kind"] = "trade"
    out.append(c)

    # 5. mixed: memorizes the suite and improvement the little of verdade
    c = _clone(champion)
    c["memorized"] = sorted(set(c.get("memorized", [])) | set(weak))
    c["skills"] = {f: min(1.0, round(skills.get(f, 0) + 0.08, 3)) for f in fams}
    c["change"] = "memorize the suite + small adjustment (+0.08)"
    c["kind"] = "mixed"
    out.append(c)

    # order rotativa: the genuine not arrives always in the same position
    shift = round_index % len(out)
    out = out[shift:] + out[:shift]
    return out[:k]


# --- CLI (for use the toy via --eval / --propose of command) -------

def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m hod.toy")
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="scores a list of tasks")
    e.add_argument("--config", type=Path, required=True)
    e.add_argument("--tasks", type=Path, required=True)
    e.add_argument("--seed", type=int, required=True)
    e.add_argument("--out", type=Path, required=True)
    pr = sub.add_parser("propose", help="proposes k candidates")
    pr.add_argument("--champion", type=Path, required=True)
    pr.add_argument("--history", type=Path, required=True)
    pr.add_argument("--k", type=int, required=True)
    pr.add_argument("--round", type=int, required=True)
    pr.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("catalog", help="writes the toy suite")
    c.add_argument("--out", type=Path, required=True)
    i = sub.add_parser("init", help="escreve harness inicial, bom e ruim")
    i.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "eval":
        cfg = json.loads(args.config.read_text())
        tasks = json.loads(args.tasks.read_text())
        args.out.write_text(json.dumps(evaluate(cfg, tasks, args.seed)))
    elif args.cmd == "propose":
        champion = json.loads(args.champion.read_text())
        history = []
        if args.history.exists():
            for line in args.history.read_text().splitlines():
                if line.strip():
                    history.append(json.loads(line))
        args.out.write_text(json.dumps(
            propose(champion, history, args.k, args.round), indent=2))
    elif args.cmd == "catalog":
        write_catalog(args.out)
    elif args.cmd == "init":
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "inicial.json").write_text(json.dumps(initial_config(), indent=2))
        (args.out / "bom.json").write_text(json.dumps(known_good(), indent=2))
        (args.out / "ruim.json").write_text(json.dumps(known_bad(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
