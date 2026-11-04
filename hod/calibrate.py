#!/usr/bin/env python3
"""Calibration of difficulty and quality Q of a held-out set.

Inputs: the set directory (tasks + manifest.json) and the BASELINE
scoreboards on those tasks (JSON {task_id: 0|1|2}, scored by the agent
with the harness without the change).

Per task:
- difficulty = (2 - score_baseline) / 2, from 0 (saturated, easy) to 1 (hard).
- q_novelty = 1 - max_similarity (from the manifest: novelty against the
  suite). Stored as `q_novelty`, so the generator's own `novelty` field
  (against the set and the excluded buffer) is kept as it was.
- validity = 1: emitted tasks passed the similarity gate.
- Q = validity x difficulty x q_novelty.
- v0.2, frontier (the D of Ornith-1.5): p = score_baseline / 2 is the
  baseline success rate on the task; frontier = exp(-(p - target)^2 / 2sigma^2).
  Ornith uses target 0.2 for RL training (it wants rare but existing
  success); to MEASURE a change the default target here is 0.5, where the
  variance of a binary task is maximal and it discriminates most.
  `q_frontier` = validity x frontier x q_novelty; `--target` changes the target.

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
import os
import sys
from pathlib import Path

try:
    from .generate import check_set_dir, install_set
    from .manifest import load as load_manifest
    from .scores import load_scores
except ImportError:  # direct execution of the file, outside the package
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate import check_set_dir, install_set  # noqa: E402
    from manifest import load as load_manifest  # noqa: E402
    from scores import load_scores  # noqa: E402


FRONTIER_TARGET = 0.5
FRONTIER_SIGMA = 0.25


def frontier(score, target=FRONTIER_TARGET, sigma=FRONTIER_SIGMA):
    p = score / 2.0
    return math.exp(-((p - target) ** 2) / (2 * sigma ** 2))


def calibrate(heldout_dir, baseline, target=FRONTIER_TARGET,
              sigma=FRONTIER_SIGMA):
    heldout_dir = Path(heldout_dir)
    manifest = load_manifest(heldout_dir / "manifest.json")
    tasks = []
    for task in manifest["tasks"]:
        task_id = task["id"]
        if task_id not in baseline:
            raise ValueError("no baseline score for %s" % task_id)
        score = baseline[task_id]
        if isinstance(score, bool) or score not in (0, 1, 2):
            raise ValueError("invalid score for %s: %r" % (task_id, score))
        sim = task.get("max_similarity", 0.0)
        if isinstance(sim, bool) or not isinstance(sim, (int, float)) \
                or not 0.0 <= sim <= 1.0:
            raise ValueError("manifest: invalid max_similarity for %s: %r"
                             % (task_id, sim))
        difficulty = (2 - score) / 2.0
        q_novelty = 1.0 - sim
        tasks.append(
            {
                "id": task_id,
                "family": task["family"],
                "file": task["file"],
                "baseline": score,
                "difficulty": round(difficulty, 3),
                "q_novelty": round(q_novelty, 3),
                "q": round(difficulty * q_novelty, 3),
                "frontier": round(frontier(score, target, sigma), 3),
                "q_frontier": round(frontier(score, target, sigma) * q_novelty,
                                    3),
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
        "| Task | Family | Baseline | Difficulty | Novelty (vs suite) | Q "
        "| Frontier | Frontier Q |",
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
                t["q_novelty"],
                t["q"],
                t["frontier"],
                t["q_frontier"],
            )
        )
    return "\n".join(lines) + "\n"


def _same_dir(a, b):
    """True if `a` and `b` are the same directory, also through aliases that
    resolve() does not see (bind mounts, case-insensitive file systems)."""
    a, b = Path(a), Path(b)
    if a.resolve() == b.resolve():
        return True
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def prepare_filtered(heldout_dir, out_dir, result, keep_saturated=False):
    """Everything write_filtered needs, read and validated in memory before
    anything is written or deleted: returns {name: bytes} (manifest last).

    Raises ValueError if the source is invalid, if `out_dir` is the source
    itself (also through an alias) or if `out_dir` is not ours to replace."""
    heldout_dir = Path(heldout_dir)
    out_dir = Path(out_dir)
    manifest = load_manifest(heldout_dir / "manifest.json")
    if _same_dir(out_dir, heldout_dir):
        raise ValueError("--out cannot be the source set itself: %s" % out_dir)
    check_set_dir(out_dir)
    chosen = result["tasks"] if keep_saturated else result["kept"]
    by_id = {t["id"]: t for t in result["tasks"]}
    chosen_ids = {t["id"] for t in chosen}
    unknown = sorted(chosen_ids - {t["id"] for t in manifest["tasks"]})
    if unknown:
        raise ValueError("calibration result has tasks outside %s: %s"
                         % (heldout_dir, unknown))
    files = {}
    new_tasks = []
    for task in manifest["tasks"]:
        if task["id"] not in chosen_ids:
            continue
        merged = dict(task)
        merged.update(
            {
                "difficulty": by_id[task["id"]]["difficulty"],
                "q_novelty": by_id[task["id"]]["q_novelty"],
                "q": by_id[task["id"]]["q"],
            }
        )
        new_tasks.append(merged)
        path = heldout_dir / task["file"]  # a plain name (manifest.load)
        try:
            files[task["file"]] = path.read_bytes()
        except OSError as e:
            raise ValueError("cannot read task file %s: %s" % (path, e))
    new_manifest = dict(manifest)
    new_manifest["tasks"] = new_tasks
    new_manifest["calibration"] = {
        "mean_q": result["mean_q"],
        "mean_difficulty": result["mean_difficulty"],
        "n_dropped": len(result["dropped"]),
        "all_saturated": result["all_saturated"],
        # scores used in the filter: the pipeline refuses to measure the delta
        # with these same scores as the baseline (selection bias / regression
        # to the mean)
        "baseline_scores": {t["id"]: t["baseline"] for t in result["tasks"]},
    }
    files["manifest.json"] = json.dumps(new_manifest, indent=2).encode("utf-8")
    return files


def install_filtered(heldout_dir, out_dir, files):
    """Writes the prepared set (temporary sibling + rename: a failure leaves
    neither a partial set nor a wiped one). Returns the number of tasks."""
    if _same_dir(out_dir, heldout_dir):  # re-checked right before writing
        raise ValueError("--out cannot be the source set itself: %s" % out_dir)
    _, commit = install_set(out_dir, files)
    commit()
    return len(files) - 1


def write_filtered(heldout_dir, out_dir, result, keep_saturated=False):
    files = prepare_filtered(heldout_dir, out_dir, result, keep_saturated)
    return install_filtered(heldout_dir, out_dir, files)


def _target(text):
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
        "--target", type=_target, default=FRONTIER_TARGET,
        help="frontier target success rate (default %(default)s; "
        "Ornith-1.5 uses 0.2 for training)",
    )
    args = parser.parse_args(argv)
    try:
        # everything is read and validated before anything is written
        baseline = load_scores(args.baseline)
        result = calibrate(args.heldout, baseline, target=args.target)
        files = None
        if args.out:
            files = prepare_filtered(args.heldout, args.out, result,
                                     args.keep_saturated)
    except ValueError as e:
        print("calibrate: %s" % e, file=sys.stderr)
        return 1
    report = render(result)
    try:
        if args.frontier_out:
            args.frontier_out.write_text(
                json.dumps(result["family_success"], indent=2),
                encoding="utf-8")
        if args.report:
            args.report.write_text(report, encoding="utf-8")
        else:
            print(report, end="")
        if files is not None:
            n = install_filtered(args.heldout, args.out, files)
            print("filtered set: %d tasks in %s" % (n, args.out))
    except (OSError, ValueError) as e:
        print("calibrate: %s" % e, file=sys.stderr)
        return 1
    print("attention: measure the delta with a NEW baseline on the kept tasks; "
          "reusing the filter scores inflates the held-out (regression to the mean).")
    if result["all_saturated"]:
        print("NO SIGNAL: all tasks saturated at baseline.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
