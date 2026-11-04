#!/usr/bin/env python3
"""Measures the gap between the fixed suite and the held-out, with a verdict.

Inputs: 4 JSON scoreboards, format {"task_id": score} with scores 0 to 2,
for suite-base, suite-candidate, heldout-base and heldout-candidate.

Verdicts:
- NO IMPROVEMENT: the suite did not improve, so there is nothing to transfer.
- OVERFIT: the suite improved and the held-out did not, or captured less than half
  of the suite gain.
- PARTIAL: the held-out captured between half and three quarters of the gain.
- GENUINE: the held-out captured at least three quarters of the gain.
- REGRESSION: the suite did not improve and the held-out dropped (with the suite
  at the ceiling, this is the only signal left).

With small n in the held-out (fewer than 8), the verdict is marked as
indicative: noise dominates the gap.

v0.2: the default verdict (`--decision ci`) uses paired-bootstrap confidence
intervals, the placebo noise floor and coverage (a task without a score is
"not run"). It can give INCONCLUSIVE, which the point rule could not.
The point rule above stays available with `--decision point` and always
leaves `verdict_point` in the report. Details in stats.py.

Deltas are paired: only tasks present in both base and candidate count
(missing ones appear in `unpaired`).

Coverage (minimum 90%, both decisions): the fraction of expected tasks with a
score on both sides. Expected tasks are the ones in `--manifest` when given;
otherwise the union of the base and candidate files, which cannot see a task
missing from both.
"""


import argparse
import sys
from pathlib import Path

try:
    from .scores import load_scores, read_json
    from . import stats as stats_mod
except ImportError:  # direct execution of the file, outside the package
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scores import load_scores, read_json  # noqa: E402
    import stats as stats_mod  # noqa: E402

SMALL_N = 8
MIN_COVERAGE = stats_mod.MIN_COVERAGE  # fraction of tasks scored on both sides
DECISIONS = ("ci", "point")
GENUINE_RATIO = 0.75
PARTIAL_RATIO = 0.50
EPS = 1e-9  # floating-point tolerance at the boundaries


def mean(scores):
    values = list(scores.values())
    return sum(values) / len(values) if values else 0.0


def paired(base, cand):
    common = sorted(set(base) & set(cand))
    return (
        {t: base[t] for t in common},
        {t: cand[t] for t in common},
        sorted(set(base) ^ set(cand)),
    )


def verdict(suite_delta, held_delta):
    if suite_delta <= EPS:
        if held_delta < -EPS:
            return "REGRESSION"
        return "NO IMPROVEMENT"
    if held_delta <= EPS:
        return "OVERFIT"
    ratio = held_delta / suite_delta
    if ratio >= GENUINE_RATIO - EPS:
        return "GENUINE"
    if ratio >= PARTIAL_RATIO - EPS:
        return "PARTIAL"
    return "OVERFIT"


def _significance(result, suite_delta, held_delta, gap, calibration):
    """Compares the observed values with the noise measured in placebo rounds.

    - effect (suite): |suite_delta| against the largest placebo |suite_delta|
      (falls back to the largest |gap| in old calibrations);
    - gap: |gap| against the largest placebo |gap|, plus the empirical
      p-value (1 + #placebos with |gap| >= |observed gap|) / (n + 1).
    Placebo values are stored rounded to 3 decimals, so a tie within half of
    that unit counts as a tie (not significant), never as "above the noise".
    Without placebos there is no measured noise: nothing is declared significant.
    """
    floor = stats_mod.noise_floor(calibration)
    n_placebo = floor["n_placebo"]
    result["n_placebo"] = n_placebo
    if n_placebo < 1:
        result["significant"] = None
        result["gap_significant"] = None
        result["noise_max"] = None
        result["noise_note"] = "no placebos: noise not measured"
        return
    noise_gap = abs(stats_mod._num(calibration.get("max_abs_gap")) or 0.0)
    noise_suite = floor["suite"]
    result["noise_max"] = noise_gap
    result["noise_suite_max"] = noise_suite
    result["significant"] = stats_mod.beyond_floor(abs(suite_delta), noise_suite)
    result["gap_significant"] = stats_mod.beyond_floor(abs(gap), noise_gap)
    gaps = calibration.get("gaps")
    gaps = [g for g in (gaps if isinstance(gaps, list) else [])
            if stats_mod._num(g) is not None]
    if gaps:
        hits = sum(1 for g in gaps
                   if abs(g) >= abs(gap) - stats_mod.FLOOR_TOL)
        result["p_gap"] = round((1 + hits) / (len(gaps) + 1), 3)
    if result["verdict"] in ("OVERFIT", "PARTIAL") and not result["gap_significant"]:
        result["noise_note"] = "gap within placebo noise"


def analyze(suite_base, suite_cand, held_base, held_cand,
            categories=None, calibration=None, decision="ci",
            n_boot=stats_mod.DEFAULT_BOOT, boot_seed=stats_mod.DEFAULT_SEED,
            strict=False, suite_ids=None, held_ids=None):
    """Gap with two verdicts:

    - `verdict_point`: the v0.1 rule (held/suite ratio at the points);
    - `verdict_ci`: the v0.2 rule (paired bootstrap + placebo noise floor
      + coverage), see stats.py.

    `verdict` is the one of the chosen `decision` ("ci", default since v0.2,
    or "point" to reproduce v0.1). Under either decision, coverage below
    MIN_COVERAGE makes `verdict` INCONCLUSIVE (`verdict_point` keeps the raw
    rule). `suite_ids` / `held_ids` are the expected task ids (e.g. the
    manifest): without them a task missing from both files is invisible."""
    if decision not in DECISIONS:
        raise ValueError("decision must be one of %s" % (DECISIONS,))
    stats_mod.check_n_boot(n_boot)
    held_base_all, held_cand_all = held_base, held_cand

    def _coverage(base, cand, expected):
        both = set(base) & set(cand)
        if expected is not None:
            expected = set(expected)
            return round(len(both & expected) / len(expected), 4) if expected else 0.0
        union = set(base) | set(cand)
        return round(len(both) / len(union), 4) if union else 0.0

    coverage = {"suite": _coverage(suite_base, suite_cand, suite_ids),
                "held": _coverage(held_base, held_cand, held_ids)}
    suite_base, suite_cand, suite_unpaired = paired(suite_base, suite_cand)
    held_base, held_cand, held_unpaired = paired(held_base, held_cand)
    if not suite_base:
        raise ValueError("suite with no tasks in common between base and candidate")
    if not held_base:
        raise ValueError("held-out with no tasks in common between base and candidate")
    suite_base_m = mean(suite_base)
    suite_cand_m = mean(suite_cand)
    held_base_m = mean(held_base)
    held_cand_m = mean(held_cand)
    suite_delta = suite_cand_m - suite_base_m
    held_delta = held_cand_m - held_base_m
    gap = suite_delta - held_delta
    v = verdict(suite_delta, held_delta)
    indicative = len(held_base) < SMALL_N
    per_task = []
    for task_id in sorted(set(held_base_all) | set(held_cand_all)):
        before = held_base_all.get(task_id)
        after = held_cand_all.get(task_id)
        delta = None if before is None or after is None else after - before
        per_task.append((task_id, before, after, delta))
    result = {
        "suite_base": round(suite_base_m, 3),
        "suite_cand": round(suite_cand_m, 3),
        "suite_delta": round(suite_delta, 3),
        "held_base": round(held_base_m, 3),
        "held_cand": round(held_cand_m, 3),
        "held_delta": round(held_delta, 3),
        "gap": round(gap, 3),
        "n_suite": len(suite_base),
        "n_held": len(held_base),
        "verdict": v,
        "indicative": indicative,
        "per_task": per_task,
        "unpaired": {"suite": suite_unpaired, "held": held_unpaired},
    }
    if categories:
        by_category = {}
        cats = sorted(
            {categories.get(t, "uncategorized")
             for t in set(suite_base) | set(suite_cand)
             | set(held_base) | set(held_cand)}
        )
        for cat in cats:
            sb = {t: s for t, s in suite_base.items()
                  if categories.get(t, "uncategorized") == cat}
            sc = {t: s for t, s in suite_cand.items()
                  if categories.get(t, "uncategorized") == cat}
            hb = {t: s for t, s in held_base.items()
                  if categories.get(t, "uncategorized") == cat}
            hc = {t: s for t, s in held_cand.items()
                  if categories.get(t, "uncategorized") == cat}
            if not sb or not sc or not hb or not hc:
                continue
            sb_m, sc_m = mean(sb), mean(sc)
            hb_m, hc_m = mean(hb), mean(hc)
            sd, hd = sc_m - sb_m, hc_m - hb_m
            by_category[cat] = {
                "suite_delta": round(sd, 3),
                "held_delta": round(hd, 3),
                "gap": round(sd - hd, 3),
                "verdict": verdict(sd, hd),
                "n_suite": len(sb),
                "n_held": len(hb),
            }
        result["by_category"] = by_category
    # --- v0.2: CI, noise floor, power, coverage -------------------------
    result["verdict_point"] = v
    summary = stats_mod.summarize(suite_base, suite_cand, held_base,
                                  held_cand, calibration=calibration,
                                  n_boot=n_boot, seed=boot_seed, strict=strict)
    summary["coverage"] = coverage
    v_ci, why = summary["verdict"], list(summary["why"])
    coverage_ok = min(coverage.values()) >= MIN_COVERAGE - EPS
    if not coverage_ok:
        # a task without a score is "not run", never "passed": with many
        # missing, no verdict holds, whatever the numbers say
        why.append("coverage below %.0f%% (suite %.0f%%, held-out %.0f%%)"
                   % (MIN_COVERAGE * 100, coverage["suite"] * 100,
                      coverage["held"] * 100))
        v_ci = "INCONCLUSIVE"
    summary["verdict"], summary["why"] = v_ci, why
    result["verdict_ci"] = v_ci
    result["coverage_ok"] = coverage_ok
    result["stats"] = summary
    result["decision"] = decision
    if decision == "ci":
        result["verdict"] = v_ci
    else:
        result["verdict"] = v if coverage_ok else "INCONCLUSIVE"
    if calibration is not None:
        _significance(result, suite_delta, held_delta, gap, calibration)
    return result


def analyze_multi(results):
    """Aggregates verdicts of several seeds: stable only if all equal."""
    if not results:
        raise ValueError("no seeds to aggregate")
    verdicts = [r["verdict"] for r in results]
    gaps = [r["gap"] for r in results]
    suite_deltas = [r["suite_delta"] for r in results]
    held_deltas = [r["held_delta"] for r in results]
    mean_suite = sum(suite_deltas) / len(suite_deltas)
    mean_held = sum(held_deltas) / len(held_deltas)
    return {
        "n_seeds": len(results),
        "verdicts": verdicts,
        "stable": len(set(verdicts)) == 1,
        "verdict": verdicts[0] if len(set(verdicts)) == 1 else "UNSTABLE",
        "mean_gap": round(sum(gaps) / len(gaps), 3),
        "mean_suite_delta": round(mean_suite, 3),
        "mean_held_delta": round(mean_held, 3),
        "mean_ratio": (
            round(mean_held / mean_suite, 3) if mean_suite > 0 else None
        ),
        "indicative": any(r.get("indicative") for r in results),
    }


def render(result):
    lines = [
        "# Gap suite x held-out",
        "",
        "Suite: %.3f -> %.3f (delta %+.3f, n=%d)"
        % (
            result["suite_base"],
            result["suite_cand"],
            result["suite_delta"],
            result["n_suite"],
        ),
        "Held-out: %.3f -> %.3f (delta %+.3f, n=%d)"
        % (
            result["held_base"],
            result["held_cand"],
            result["held_delta"],
            result["n_held"],
        ),
        "Gap (suite minus held-out): %+.3f" % result["gap"],
        "",
        "Verdict: **%s**%s"
        % (
            result["verdict"],
            " (indicative: held-out n below 8)"
            if result["indicative"]
            else "",
        ),
        "",
        "## Per task (held-out)",
        "",
        "| Task | Before | After | Delta |",
        "| --- | --- | --- | --- |",
    ]
    for task_id, before, after, delta in result["per_task"]:
        lines.append(
            "| %s | %s | %s | %s |"
            % (
                task_id,
                before if before is not None else "-",
                after if after is not None else "-",
                ("%+.1f" % delta) if delta is not None else "-",
            )
        )
    if result.get("by_category"):
        lines += ["", "## By category", ""]
        lines.append("| Category | Suite delta | Held delta | Gap | Verdict |")
        lines.append("| --- | --- | --- | --- | --- |")
        for cat, c in sorted(result["by_category"].items()):
            lines.append(
                "| %s | %+.3f | %+.3f | %+.3f | %s |"
                % (cat, c["suite_delta"], c["held_delta"], c["gap"], c["verdict"])
            )
    if result.get("unpaired", {}).get("held") or result.get("unpaired", {}).get("suite"):
        lines += ["", "Unpaired tasks (outside the deltas): %s"
                  % (result["unpaired"]["suite"] + result["unpaired"]["held"])]
    if result.get("significant") is not None:
        lines += [
            "",
            "Effect significance: %s (|suite delta| %.3f vs placebo noise "
            "placebo %.3f, n_placebo=%d)"
            % (
                "yes" if result["significant"] else "no",
                abs(result["suite_delta"]),
                result["noise_suite_max"],
                result["n_placebo"],
            ),
            "Gap significance: %s (|gap| %.3f vs placebo noise %.3f%s)"
            % (
                "yes" if result["gap_significant"] else "no",
                abs(result["gap"]),
                result["noise_max"],
                (", p=%.3f" % result["p_gap"]) if "p_gap" in result else "",
            ),
        ]
    elif result.get("noise_note"):
        lines += ["", "Significance: not evaluated (%s)" % result["noise_note"]]
    st = result.get("stats")
    if st:
        b = st["bootstrap"]
        lines += [
            "",
            "## Confidence intervals (paired bootstrap, %d replicas, CI %d%%)"
            % (b["n_boot"], round((1 - b["alpha"]) * 100)),
            "",
            "| Measure | Point | CI |",
            "| --- | --- | --- |",
        ]
        for key, label in (("suite_delta", "Suite delta"),
                           ("held_delta", "Held-out delta"),
                           ("gap", "Gap"),
                           ("t50", "held − ½·suite"),
                           ("t75", "held − ¾·suite")):
            lo, hi = b["ci"][key]
            lines.append("| %s | %+.3f | [%+.3f, %+.3f] |"
                         % (label, b["point"][key], lo, hi))
        nf = st["noise_floor"]
        lines += [
            "",
            "Noise floor: suite %.3f, held-out %.3f (%s)"
            % (nf["suite"], nf["held"], nf["source"]),
            "Coverage: suite %.0f%%, held-out %.0f%%"
            % (st["coverage"]["suite"] * 100, st["coverage"]["held"] * 100),
            "Minimum detectable effect in the held-out (80%% power): %.3f"
            % st["power"]["mde_held_80"],
            "",
            "Verdict by CI: **%s** | point verdict (v0.1 rule): %s"
            % (result["verdict_ci"], result["verdict_point"]),
        ]
        lines += ["- %s" % w for w in st["why"]]
        if st["power"]["warnings"]:
            lines += ["", "Warnings:"] + ["- %s" % w for w in st["power"]["warnings"]]
    return "\n".join(lines) + "\n"


def load(path):
    return load_scores(path)


def boot_arg(value):
    """argparse type for --boot: an integer >= stats.MIN_BOOT."""
    try:
        return stats_mod.check_n_boot(int(value))
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def manifest_ids(path):
    manifest = read_json(path, "manifest", expect=dict)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not all(
            isinstance(t, dict) and isinstance(t.get("id"), str) for t in tasks):
        raise ValueError("manifest %s: expected tasks with an id" % path)
    return [t["id"] for t in tasks]


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--suite-base", type=Path, required=True)
    parser.add_argument("--suite-cand", type=Path, required=True)
    parser.add_argument("--held-base", type=Path, required=True)
    parser.add_argument("--held-cand", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--categories", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--prereg", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="held-out manifest: coverage counts its tasks "
                        "(a task missing from both score files is caught) and "
                        "a score id outside it is an error")
    parser.add_argument(
        "--decision", choices=DECISIONS, default="ci",
        help="ci (default): verdict by confidence interval and noise floor; "
        "point: v0.1 point rule",
    )
    parser.add_argument("--boot", type=boot_arg, default=stats_mod.DEFAULT_BOOT,
                        help="bootstrap replicas, >= %d (default %%(default)s)"
                        % stats_mod.MIN_BOOT)
    parser.add_argument("--strict", action="store_true",
                        help="GENUINE requires transfer ≥ ½ with 95%% "
                        "confidence (asks for more tasks)")
    args = parser.parse_args(argv)

    try:
        categories = None
        if args.categories:
            categories = read_json(args.categories, "categories file", expect=dict)
        calibration = None
        if args.calibration:
            calibration = read_json(args.calibration, "calibration file",
                                    expect=dict)
        held_base, held_cand = load(args.held_base), load(args.held_cand)
        held_ids = None
        if args.manifest:
            held_ids = manifest_ids(args.manifest)
            extra = sorted((set(held_base) | set(held_cand)) - set(held_ids))
            if extra:
                raise ValueError("held-out scores have tasks outside the "
                                 "manifest (scores from another set?): %s"
                                 % extra[:10])
        result = analyze(
            load(args.suite_base),
            load(args.suite_cand),
            held_base,
            held_cand,
            categories=categories,
            calibration=calibration,
            decision=args.decision,
            n_boot=args.boot,
            strict=args.strict,
            held_ids=held_ids,
        )
        prereg = None
        if args.prereg:
            try:
                from .prereg import load as prereg_load
            except ImportError:  # direct execution of the file
                from prereg import load as prereg_load  # noqa: E402
            prereg = prereg_load(args.prereg)
    except (ValueError, OSError, TypeError, KeyError) as e:
        print("gap: %s" % e, file=sys.stderr)
        return 1
    if prereg is not None:
        try:
            from .prereg import check as prereg_check
        except ImportError:  # direct execution of the file
            from prereg import check as prereg_check  # noqa: E402
        print(
            "warning: `hod gap` does not check peeking; use `hod pipeline "
            "--prereg` with sets generated by `hod generate --prereg`",
            file=sys.stderr,
        )

        hypothesis = prereg_check(prereg, result)
        result["hypothesis"] = hypothesis
        print(
            "hypothesis: %s (suite %s, held %s)"
            % (
                "match" if hypothesis["match"] else "MISMATCH",
                "ok" if hypothesis["suite_match"] else "no",
                "ok" if hypothesis["held_match"] else "no",
            )
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(result), encoding="utf-8")
    ci = result["stats"]["bootstrap"]["ci"]
    print(
        "verdict: %s (gap %+.3f CI [%+.3f, %+.3f], suite %+.3f, held-out "
        "%+.3f CI [%+.3f, %+.3f]; point v0.1: %s)"
        % (
            result["verdict"],
            result["gap"], ci["gap"][0], ci["gap"][1],
            result["suite_delta"],
            result["held_delta"], ci["held_delta"][0], ci["held_delta"][1],
            result["verdict_point"],
        )
    )
    for w in result["stats"]["power"]["warnings"]:
        print("warning: %s" % w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
