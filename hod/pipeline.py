#!/usr/bin/env python3
"""Evaluation pipeline: runs all robustness layers at once.

Order:
1. Preregistration: the hypothesis file must predate all manifests
   (no peeking), otherwise it fails.
2. Wave 1 (if --wave1): gap on a subset; if it gives OVERFIT, stop here
   and do not expand (code 3).
3. Multi-seed: gap per seed + aggregate + stability.
4. Hypothesis: predicted vs observed, on the means.
5. Significance: against the placebo calibration, if given.
6. Ledger: automatic append (if --ledger), the log hook.

Exit codes: 0 complete, 3 stopped at wave 1, 1 validation failure.

Scoreboard files in --scores-dir:
- suite-base.json, suite-cand.json (always).
- Per-seed sets: held-base-NN.json, held-cand-NN.json.
- Single set: held-base.json, held-cand.json.

Categories: suite via --suite-categories (JSON task -> category);
held via the "Category:" line of each task, with fallback to the family.

Usage:
    hod pipeline --change "new rule" --prereg prereg.json \
      --heldout-dir runs/heldout --scores-dir runs/scores \
      --seeds 42,43,44 --ledger runs/ledger.jsonl --out runs/pipe-01
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from hod import gap as gap_mod
from hod import ledger as ledger_mod
from hod import prereg as prereg_mod


def _seed_sort_key(path):
    name = path.name[5:]
    return (0, int(name), name) if name.isdigit() else (1, 0, name)


def discover_sets(heldout_dir, seeds_arg):
    heldout_dir = Path(heldout_dir)
    if seeds_arg:
        names = [s.strip() for s in seeds_arg.split(",")]
        sets = []
        for name in names:
            set_dir = heldout_dir / ("seed-%s" % name)
            if not (set_dir / "manifest.json").exists():
                raise ValueError("set missing: %s" % set_dir)
            sets.append((name, set_dir))
        return sets
    seed_dirs = sorted(
        (d for d in heldout_dir.iterdir()
         if d.is_dir() and d.name.startswith("seed-")
         and (d / "manifest.json").exists()),
        key=_seed_sort_key,  # seed-9 before seed-10
    )
    if seed_dirs:
        return [(d.name[5:], d) for d in seed_dirs]
    if (heldout_dir / "manifest.json").exists():
        return [("single", heldout_dir)]
    raise ValueError("no set in %s" % heldout_dir)


def held_score_file(scores_dir, seed_name, kind):
    if seed_name == "single":
        return Path(scores_dir) / ("held-%s.json" % kind)
    return Path(scores_dir) / ("held-%s-%s.json" % (kind, seed_name))


def task_category(text, fallback):
    match = re.search(r"^Category:\s*([^.\n]+)", text, re.M)
    if match:
        return match.group(1).strip()
    return fallback


def build_categories(sets, suite_categories):
    mapping = {}
    if suite_categories:
        mapping.update(suite_categories)
    for _, set_dir in sets:
        manifest = json.loads((set_dir / "manifest.json").read_text())
        for task in manifest["tasks"]:
            text = (set_dir / task["file"]).read_text()
            mapping[task["id"]] = task_category(text, task["family"])
    return mapping


def same_baseline_problem(manifest, held_base):
    """Message of error if the held-base reusa the scores of the filter of
    calibration (selection by the same noise that enters in the delta), otherwise None."""
    cal = manifest.get("calibration") or {}
    used = cal.get("baseline_scores")
    if not used:
        return None
    ids = [t["id"] for t in manifest["tasks"]]
    if ids and all(t in held_base and t in used and held_base[t] == used[t]
                   for t in ids):
        return (
            "held-base equals the scores used to filter the saturated tasks: the "
            "delta is inflated by regression to the mean; run a new baseline "
            "on the calibrated set (or --allow-same-baseline)"
        )
    return None


def check_ids(manifest, scores, label):
    """Scoreboard must be for the right set: an id outside the manifest is an error."""
    ids = {t["id"] for t in manifest["tasks"]}
    extra = sorted(set(scores) - ids)
    if extra:
        raise ValueError(
            "%s has tasks outside the manifest (scoreboard from another set?): %s"
            % (label, extra[:5])
        )
    missing = sorted(ids - set(scores))
    if missing:
        print("warning: %s without a score for %s" % (label, missing), file=sys.stderr)


def stratified_wave(ids, family_of, k):
    """Wave-1 subset with one task per family in rotation (before
    it was the first k in alphabetical order: only the first families)."""
    by_fam = {}
    for t in sorted(ids):
        by_fam.setdefault(family_of.get(t, "?"), []).append(t)
    chosen = []
    while len(chosen) < k and any(by_fam.values()):
        for fam in sorted(by_fam):
            if by_fam[fam] and len(chosen) < k:
                chosen.append(by_fam[fam].pop(0))
    return sorted(chosen)


def by_family(result_seed, suite_base, suite_cand, held_base, held_cand,
              family_of, categories):
    """Verdict per family of the generator (is the key that the weights use).
    Suite of the family = tasks of the suite in the same category; if not houver,
    the suite integer."""
    out = {}
    fams = sorted({family_of[t] for t in held_base if t in family_of})
    for fam in fams:
        hb = {t: s for t, s in held_base.items() if family_of.get(t) == fam}
        hc = {t: s for t, s in held_cand.items() if family_of.get(t) == fam}
        cat = categories.get(next(iter(hb)), None) if hb else None
        sb = {t: s for t, s in suite_base.items() if categories.get(t) == cat}
        sc = {t: s for t, s in suite_cand.items() if categories.get(t) == cat}
        if not sb or not sc:
            sb, sc = suite_base, suite_cand
        try:
            # per family the n is small too much for CI: the rule point only
            # alimenta the weights of the ledger (where olhar), not decides nothing
            r = gap_mod.analyze(sb, sc, hb, hc, decision="point", n_boot=200)
        except ValueError:
            continue
        out[fam] = {"verdict": r["verdict"], "gap": r["gap"],
                    "n_held": r["n_held"]}
    return out


def merge_by_category(per_seed, key="by_category"):
    cats = {}
    for result in per_seed:
        for cat, data in (result.get(key) or {}).items():
            cats.setdefault(cat, []).append(data)
    merged = {}
    for cat, entries in cats.items():
        verdicts = [e["verdict"] for e in entries]
        merged[cat] = {
            "verdict": verdicts[0] if len(set(verdicts)) == 1 else "UNSTABLE",
            "verdicts": verdicts,
            "mean_gap": round(
                sum(e["gap"] for e in entries) / len(entries), 3
            ),
        }
    return merged


def render(data):
    lines = [
        "# Pipeline HOD: %s" % data["change"],
        "",
        "Seeds: %s | Placebo: %s | Onda 1: %s"
        % (
            ", ".join(data["seeds"]),
            "yes" if data["placebo"] else "no",
            "stopped" if data["wave_stopped"] else "not used",
        ),
        "",
        "## Agregado",
        "",
        "Verdict: **%s** (stable: %s)"
        % (
            data["verdict"],
            "yes" if data["stable"] else "no",
        ),
        "Mean gap: %+.3f | Suite: %+.3f | Held-out: %+.3f"
        % (
            data["mean_gap"],
            data["mean_suite_delta"],
            data["mean_held_delta"],
        ),
    ]
    if data.get("pooled"):
        pl = data["pooled"]
        lines.append(
            "Todas as seeds juntas (n=%d no held-out): **%s**; delta held-out "
            "CI [%+.3f, %+.3f], gap CI [%+.3f, %+.3f]"
            % (pl["n_held"], pl["verdict_ci"],
               pl["ci"]["held_delta"][0], pl["ci"]["held_delta"][1],
               pl["ci"]["gap"][0], pl["ci"]["gap"][1]))
        for w in pl["power"]["warnings"]:
            lines.append("Warning: %s" % w)
    if data["hypothesis"]:
        lines.append(
            "Hypothesis: %s"
            % ("match" if data["hypothesis"]["match"] else "MISMATCH")
        )
    if data["significant"] is not None:
        lines.append(
            "Effect significance: %s"
            % ("yes" if data["significant"] else "no")
        )
    if data.get("gap_significant") is not None:
        lines.append(
            "Gap significance: %s"
            % ("yes" if data["gap_significant"] else "no")
        )
    lines += [
        "",
        "## Por seed",
        "",
        "| Seed | Suite delta | Held delta | Gap | Verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    for seed, result in zip(data["seeds"], data["per_seed"]):
        lines.append(
            "| %s | %+.3f | %+.3f | %+.3f | %s |"
            % (
                seed,
                result["suite_delta"],
                result["held_delta"],
                result["gap"],
                result["verdict"],
            )
        )
    if data["by_category"]:
        lines += [
            "",
            "## By category",
            "",
            "| Category | Mean gap | Verdict |",
            "| --- | --- | --- |",
        ]
        for cat, c in sorted(data["by_category"].items()):
            lines.append(
                "| %s | %+.3f | %s |" % (cat, c["mean_gap"], c["verdict"])
            )
    return "\n".join(lines) + "\n"


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--change", required=True)
    parser.add_argument("--prereg", type=Path, default=None)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--suite-categories", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--wave1", type=int, default=None)
    parser.add_argument("--placebo", action="store_true")
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--allow-same-baseline", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--decision", choices=gap_mod.DECISIONS, default="ci",
        help="ci (v0.2 default): verdict by CI and noise floor; point: "
        "point rule da v0.1",
    )
    args = parser.parse_args(argv)

    try:
        sets = discover_sets(args.heldout_dir, args.seeds)
    except ValueError as e:
        print("pipeline: %s" % e, file=sys.stderr)
        return 1
    seed_names = [name for name, _ in sets]

    prereg_data = None
    if args.prereg:
        prereg_data = prereg_mod.load(args.prereg)
        manifests = [set_dir / "manifest.json" for _, set_dir in sets]
        bound = all(
            json.loads(m.read_text()).get("prereg_sha256") for m in manifests
        )
        if bound:
            ok, why = prereg_mod.verify(args.prereg, manifests)
            if not ok:
                print("pipeline: prereg does not match the set: %s" % why,
                      file=sys.stderr)
                return 1
        else:
            print(
                "warning: set generated without --prereg; peeking check "
                "only by mtime (weak: touch or regenerating the same seed bypasses it)",
                file=sys.stderr,
            )
            if not prereg_mod.older_than(args.prereg, manifests):
                print(
                    "pipeline: prereg after the set was generated "
                    "(peeking?); abort and register the hypothesis before generating",
                    file=sys.stderr,
                )
                return 1

    try:
        suite_base = gap_mod.load(args.scores_dir / "suite-base.json")
        suite_cand = gap_mod.load(args.scores_dir / "suite-cand.json")
    except ValueError as e:
        print("pipeline: %s" % e, file=sys.stderr)
        return 1
    suite_categories = None
    if args.suite_categories:
        suite_categories = json.loads(args.suite_categories.read_text())
    categories = build_categories(sets, suite_categories)
    calibration = None
    if args.calibration:
        calibration = json.loads(args.calibration.read_text())

    manifests_by_seed = {
        name: json.loads((set_dir / "manifest.json").read_text())
        for name, set_dir in sets
    }
    family_of = {}
    for m in manifests_by_seed.values():
        for t in m["tasks"]:
            family_of[t["id"]] = t["family"]

    def held_scores(seed_name, kind):
        scores = gap_mod.load(
            held_score_file(args.scores_dir, seed_name, kind)
        )
        check_ids(manifests_by_seed[seed_name], scores,
                  "held-%s (seed %s)" % (kind, seed_name))
        return scores

    try:
        for seed_name, _ in sets:
            hb = held_scores(seed_name, "base")
            held_scores(seed_name, "cand")
            problem = same_baseline_problem(manifests_by_seed[seed_name], hb)
            if problem and not args.allow_same_baseline:
                raise ValueError("seed %s: %s" % (seed_name, problem))
    except ValueError as e:
        print("pipeline: %s" % e, file=sys.stderr)
        return 1

    data = {
        "change": args.change,
        "placebo": args.placebo,
        "seeds": seed_names,
        "wave_stopped": False,
        "hypothesis": None,
        "significant": None,
        "per_seed": [],
        "by_category": {},
    }

    if args.wave1:
        first_name, _ = sets[0]
        hb = held_scores(first_name, "base")
        hc = held_scores(first_name, "cand")
        wave_ids = stratified_wave(set(hb) & set(hc), family_of, args.wave1)
        wave_result = gap_mod.analyze(
            suite_base,
            suite_cand,
            {t: hb[t] for t in wave_ids},
            {t: hc[t] for t in wave_ids},
            decision=args.decision,
        )
        if wave_result["verdict"] == "OVERFIT":
            data["wave_stopped"] = True
            data["per_seed"] = [wave_result]
            data["verdict"] = "OVERFIT"
            data["stable"] = True
            data["mean_gap"] = wave_result["gap"]
            data["mean_suite_delta"] = wave_result["suite_delta"]
            data["mean_held_delta"] = wave_result["held_delta"]
            data["mean_ratio"] = None
            data["verdicts"] = ["OVERFIT"]
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "report.md").write_text(render(data))
            (args.out / "summary.json").write_text(
                json.dumps(data, indent=2)
            )
            if args.ledger:
                ledger_mod.append(
                    args.ledger,
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "change": args.change,
                        "placebo": args.placebo,
                        "seeds": seed_names,
                        "verdicts": ["OVERFIT"],
                        "stable": True,
                        "verdict": "OVERFIT",
                        "gap": wave_result["gap"],
                        "suite_delta": wave_result["suite_delta"],
                        "held_delta": wave_result["held_delta"],
                        "hypothesis": None,
                        "wave_stopped": True,
                        "by_category": {},
                        "n_held": len(wave_ids),
                    },
                )
            print("pipeline: STOPPED AT WAVE 1 (OVERFIT, without expanding)")
            return 3

    per_seed = []
    pooled_hb, pooled_hc = {}, {}
    for seed_name, _ in sets:
        hb = held_scores(seed_name, "base")
        hc = held_scores(seed_name, "cand")
        r = gap_mod.analyze(
            suite_base,
            suite_cand,
            hb,
            hc,
            categories=categories,
            calibration=calibration,
            decision=args.decision,
        )
        r["by_family"] = by_family(r, suite_base, suite_cand, hb, hc,
                                   family_of, categories)
        per_seed.append(r)
        for t in hb:
            pooled_hb["%s/%s" % (seed_name, t)] = hb[t]
        for t in hc:
            pooled_hc["%s/%s" % (seed_name, t)] = hc[t]
    aggregate = gap_mod.analyze_multi(per_seed)
    data["per_seed"] = per_seed
    data["verdicts"] = aggregate["verdicts"]
    data["stable"] = aggregate["stable"]
    data["verdict"] = aggregate["verdict"]
    data["mean_gap"] = aggregate["mean_gap"]
    # v0.2: the sets of the seeds are disjuntos between si (generate --seeds
    # excludes the tasks of the seeds previous), then gives for join all
    # held-out tasks num bootstrap only: n larger, CI more narrow.
    pooled = gap_mod.analyze(suite_base, suite_cand, pooled_hb, pooled_hc,
                             calibration=calibration, decision="ci")
    data["pooled"] = {
        "n_held": pooled["n_held"],
        "verdict_ci": pooled["verdict_ci"],
        "verdict_point": pooled["verdict_point"],
        "ci": pooled["stats"]["bootstrap"]["ci"],
        "why": pooled["stats"]["why"],
        "power": pooled["stats"]["power"],
    }
    data["decision"] = args.decision
    data["mean_suite_delta"] = aggregate["mean_suite_delta"]
    data["mean_held_delta"] = aggregate["mean_held_delta"]
    data["mean_ratio"] = aggregate["mean_ratio"]
    data["by_category"] = merge_by_category(per_seed)
    data["by_family"] = merge_by_category(per_seed, "by_family")
    data["gap_significant"] = None
    if calibration is not None and per_seed[0].get("significant") is not None:
        # suite is the same in all seeds: the effect significance is the
        # only one; gap significance is evaluated per seed and requires all
        data["significant"] = all(r.get("significant") for r in per_seed)
        data["gap_significant"] = all(
            r.get("gap_significant") for r in per_seed
        )
    if prereg_data:
        data["hypothesis"] = prereg_mod.check(
            prereg_data,
            {
                "suite_delta": aggregate["mean_suite_delta"],
                "held_delta": aggregate["mean_held_delta"],
            },
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(render(data))
    (args.out / "summary.json").write_text(json.dumps(data, indent=2))
    if args.ledger:
        ledger_mod.append(
            args.ledger,
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "change": args.change,
                "placebo": args.placebo,
                "seeds": seed_names,
                "verdicts": aggregate["verdicts"],
                "stable": aggregate["stable"],
                "verdict": aggregate["verdict"],
                "gap": aggregate["mean_gap"],
                "suite_delta": aggregate["mean_suite_delta"],
                "held_delta": aggregate["mean_held_delta"],
                "decision": args.decision,
                "pooled_verdict_ci": data["pooled"]["verdict_ci"],
                "pooled_held_ci": data["pooled"]["ci"]["held_delta"],
                "hypothesis": (
                    "match"
                    if data["hypothesis"] and data["hypothesis"]["match"]
                    else (
                        "mismatch"
                        if data["hypothesis"]
                        else None
                    )
                ),
                "wave_stopped": False,
                "by_category": {
                    cat: c["verdict"]
                    for cat, c in data["by_category"].items()
                },
                "by_family": {
                    fam: c["verdict"]
                    for fam, c in data["by_family"].items()
                },
                "significant": data["significant"],
                "gap_significant": data["gap_significant"],
                "n_held": [r["n_held"] for r in per_seed],
            },
        )
    print(
        "pipeline: verdict %s (stable: %s, mean gap %+.3f)"
        % (
            aggregate["verdict"],
            "yes" if aggregate["stable"] else "no",
            aggregate["mean_gap"],
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
