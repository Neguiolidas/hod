#!/usr/bin/env python3
"""Calibration of difficulty and quality Q of a held-out set.

Inputs: the set directory (tasks + manifest.json) and the BASELINE
scoreboards on those tasks (JSON {task_id: 0|1|2}, scored by the agent
with the harness without the change).

Per task:
- difficulty = (2 - score_baseline) / 2, from 0 (saturated, easy) to 1 (hard).
- novelty = 1 - max_similarity (from the manifest).
- validity = 1: emitted tasks passed the similarity gate.
- Q = validity x difficulty x novelty.
- v0.2, frontier (the D of Ornith-1.5): p = score_baseline / 2 is the
  baseline success rate on the task; frontier = exp(-(p - target)^2 / 2sigma^2).
  Ornith uses target 0.2 for RL training (it wants rare but existing
  success); to MEASURE a change the default target here is 0.5, where the
  variance of a binary task is maximal and it discriminates most.
  `q_frontier` = validity x frontier x novelty; `--target` changes the target.

Outputs: a report with the per-task table and the set statistics,
and, with --out, a filtered set without the saturated tasks (score 2).
If all saturate, the report marks NO SIGNAL: the set discriminates
nothing, and measuring a gap on it is theater.

Usage:
    hod calibrate --heldout runs/heldout-42 --baseline held-base.json \
      --out runs/heldout-42-cal --report runs/cal-42.md
"""


import argparse
import json
import math
import shutil
import sys
from pathlib import Path

try:
    from .scores import load_scores
except ImportError:  # direct execution of the file, outside the package
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scores import load_scores  # noqa: E402


FRONTIER_TARGET = 0.5
FRONTIER_SIGMA = 0.25


def frontier(score, target=FRONTIER_TARGET, sigma=FRONTIER_SIGMA):
    p = score / 2.0
    return math.exp(-((p - target) ** 2) / (2 * sigma ** 2))


def calibrate(heldout_dir, baseline, target=FRONTIER_TARGET,
              sigma=FRONTIER_SIGMA):
    heldout_dir = Path(heldout_dir)
    manifest = json.loads((heldout_dir / "manifest.json").read_text())
    tasks = []
    for task in manifest["tasks"]:
        task_id = task["id"]
        if task_id not in baseline:
            raise ValueError("no baseline score for %s" % task_id)
        score = baseline[task_id]
        if score not in (0, 1, 2):
            raise ValueError("invalid score for %s: %s" % (task_id, score))
        difficulty = (2 - score) / 2.0
        novelty = 1.0 - task.get("max_similarity", 0.0)
        tasks.append(
            {
                "id": task_id,
                "family": task["family"],
                "file": task["file"],
                "baseline": score,
                "difficulty": round(difficulty, 3),
                "novelty": round(novelty, 3),
                "q": round(difficulty * novelty, 3),
                "frontier": round(frontier(score, target, sigma), 3),
                "q_frontier": round(frontier(score, target, sigma) * novelty, 3),
            }
        )
    kept = [t for t in tasks if t["baseline"] < 2]
    mean_q = (
        round(sum(t["q"] for t in tasks) / len(tasks), 3) if tasks else 0.0
    )
    mean_difficulty = (
        round(sum(t["difficulty"] for t in tasks) / len(tasks), 3)
        if tasks
        else 0.0
    )
    mean_qf = (
        round(sum(t["q_frontier"] for t in tasks) / len(tasks), 3)
        if tasks else 0.0
    )
    by_family = {}
    for t in tasks:
        by_family.setdefault(t["family"], []).append(t["baseline"] / 2.0)
    family_rate = {f: round(sum(v) / len(v), 3) for f, v in by_family.items()}
    return {
        "tasks": tasks,
        "target": target,
        "mean_q_frontier": mean_qf,
        "family_success": family_rate,
        "kept": kept,
        "dropped": [t for t in tasks if t["baseline"] == 2],
        "mean_q": mean_q,
        "mean_difficulty": mean_difficulty,
        "all_saturated": len(kept) == 0,
    }


def render(result):
    lines = [
        "# Held-out set calibration",
        "",
        "Tasks: %d, kept: %d, dropped (saturated): %d"
        % (
            len(result["tasks"]),
            len(result["kept"]),
            len(result["dropped"]),
        ),
        "Mean difficulty: %.3f | Mean Q: %.3f | Mean frontier Q "
        "(target %.2f): %.3f"
        % (result["mean_difficulty"], result["mean_q"],
           result.get("target", FRONTIER_TARGET),
           result.get("mean_q_frontier", 0.0)),
        "Baseline success by family: %s"
        % ", ".join("%s %.2f" % kv
                    for kv in sorted(result.get("family_success", {}).items())),
    ]
    if result["all_saturated"]:
        lines.append("")
        lines.append("NO SIGNAL: all tasks saturated at baseline.")
    lines += [
        "",
        "| Task | Family | Baseline | Difficulty | Novelty | Q | Frontier | Frontier Q |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in result["tasks"]:
        lines.append(
            "| %s | %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f |"
            % (
                t["id"],
                t["family"],
                t["baseline"],
                t["difficulty"],
                t["novelty"],
                t["q"],
                t["frontier"],
                t["q_frontier"],
            )
        )
    return "\n".join(lines) + "\n"


def write_filtered(heldout_dir, out_dir, result, keep_saturated=False):
    heldout_dir = Path(heldout_dir)
    out_dir = Path(out_dir)
    if out_dir.resolve() == heldout_dir.resolve():
        raise ValueError("--out cannot be the source set itself")
    out_dir.mkdir(parents=True, exist_ok=True)
    # restos of the calibration previous fariam the audit fail
    for stale in list(out_dir.glob("heldout-*.md")) + [out_dir / "manifest.json"]:
        stale.unlink(missing_ok=True)
    chosen = result["tasks"] if keep_saturated else result["kept"]
    manifest = json.loads((heldout_dir / "manifest.json").read_text())
    by_id = {t["id"]: t for t in result["tasks"]}
    chosen_ids = {t["id"] for t in chosen}
    new_tasks = []
    for task in manifest["tasks"]:
        if task["id"] not in chosen_ids:
            continue
        merged = dict(task)
        merged.update(
            {
                "difficulty": by_id[task["id"]]["difficulty"],
                "novelty": by_id[task["id"]]["novelty"],
                "q": by_id[task["id"]]["q"],
            }
        )
        new_tasks.append(merged)
        shutil.copyfile(heldout_dir / task["file"], out_dir / task["file"])
    new_manifest = dict(manifest)
    new_manifest["tasks"] = new_tasks
    new_manifest["calibration"] = {
        "mean_q": result["mean_q"],
        "mean_difficulty": result["mean_difficulty"],
        "n_dropped": len(result["dropped"]),
        "all_saturated": result["all_saturated"],
        # scores used in the filter: the pipeline recusa measure the delta with these
        # same scores the baseline (bias of selection / regression to the mean)
        "baseline_scores": {t["id"]: t["baseline"] for t in result["tasks"]},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(new_manifest, indent=2)
    )
    return len(new_tasks)


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--keep-saturated", action="store_true")
    parser.add_argument(
        "--frontier-out", type=Path, default=None,
        help="writes {family: baseline success} for `hod generate "
        "--frontier`",
    )
    parser.add_argument(
        "--target", type=float, default=FRONTIER_TARGET,
        help="frontier target success rate (default %(default)s; "
        "Ornith-1.5 uses 0.2 for training)",
    )
    args = parser.parse_args(argv)
    try:
        baseline = load_scores(args.baseline)
        result = calibrate(args.heldout, baseline, target=args.target)
    except ValueError as e:
        print("calibrate: %s" % e, file=sys.stderr)
        return 1
    report = render(result)
    if args.frontier_out:
        args.frontier_out.write_text(json.dumps(result["family_success"], indent=2))
    if args.report:
        args.report.write_text(report)
    else:
        print(report, end="")
    if args.out:
        n = write_filtered(
            args.heldout, args.out, result, args.keep_saturated
        )
        print("filtered set: %d tasks in %s" % (n, args.out))
    print("attention: measure the delta with a NEW baseline on the kept tasks; "
          "reusing the filter scores inflates the held-out (regression to the mean).")
    if result["all_saturated"]:
        print("NO SIGNAL: all tasks saturated at baseline.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
