#!/usr/bin/env python3
"""`hod improve`: self-improvement loop with HOD as judge.

Idea (adapted from Harness-squared, the vibe-coding-toolkit, Amazon Transform
and Ornith-1.5): the proposer suggests changes to the harness, HOD measures
each one on the suite and on a NEW held-out, and only a change with a
GENUINE verdict that passes all gates becomes the new champion.

Round by round:
1. Checks that the evaluator has not changed (hash recorded at the start).
   Changed evaluator = invalid comparison: the loop stops.
2. Asks the proposer for k candidates. It sees the champion, the champion's
   scores on the suite (public) and the aggregated history. It never sees
   per-task held-out scores, and the round's held-out is only generated
   AFTER the proposal.
3. Generates the new held-out, excluding all tasks already used (rotation):
   no set decides twice, so the loop does not learn the held-out.
   If the generator can no longer find new tasks, the loop stops.
4. Measures the champion (round baseline) and the PLACEBO (the same champion
   again, another evaluation round). The placebo calibrates the noise floor.
5. Evaluator sanity (if `--sanity-good/--sanity-bad`): the good harness
   must beat the bad one with CI above zero on the suite and on the round's
   held-out. If the evaluator does not separate the two, the loop stops.
6. For each candidate, gates in order (verify-then-commit):
   validation (`--validate`, a command of yours that must exit 0) -> coverage
   (a task without a score counts 0 and coverage < 90% fails) -> verdict by CI
   = GENUINE -> no held-out family dropped with CI below zero.
7. Among the approved, the one with the largest lower bound of the held-out
   CI stays. The champion is only swapped after all of that (written
   atomically); otherwise the previous one stays intact.
8. Everything goes to the lineage ledger (JSONL): each candidate, the gate
   that blocked it, numbers, and the lesson with the invalidation trigger.

Stop: `--rounds`, `--budget` (candidate evaluations), `--patience`
(consecutive rounds without acceptance), empty proposer, exhausted
held-out, failed sanity or changed evaluator.

Final exam: the sealed held-out is generated before the first round and
excluded from all of them. With something accepted, the final champion is
compared with the initial harness on that set. This protects against the
"winner's curse": picking the best of several candidates over several
rounds inflates the measured gain; the final exam measures again, on
tasks nobody saw.

Command interface (placeholders replaced by already-quoted paths):
    --eval     "CMD {config} {tasks} {seed} {out}"   writes {task_id: 0|1|2}
    --propose  "CMD {champion} {history} {k} {round} {out}"  writes lists
               of JSON configs (each may have "change": description)
    --validate "CMD {config}"   exits 0 if the candidate is valid
The {tasks} file is the list [{"id", "family", "text"}].
"""

import argparse
import hashlib
import inspect
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from hod import gap as gap_mod
from hod import generate as gen_mod
from hod import ledger as ledger_mod
from hod import placebo as placebo_mod
from hod import stats as stats_mod
from hod.scores import validate_score

DEFAULT_ROUNDS = 5
DEFAULT_K = 4
DEFAULT_PATIENCE = 2
DEFAULT_N_HELD = 64
DEFAULT_TIMEOUT = 600


class ImproveError(Exception):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def config_hash(cfg):
    clean = {k: v for k, v in cfg.items() if k not in ("change", "kind")}
    return _sha(json.dumps(clean, sort_keys=True).encode())[:12]


def callable_identity(fn):
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = repr(fn)
    return _sha(src.encode())


def command_identity(cmd):
    """Hash of the command and of the content of each file cited in it: swapping the
    evaluator script mid-loop changes the hash."""
    h = hashlib.sha256(cmd.encode())
    for token in shlex.split(cmd):
        path = Path(token)
        try:
            if path.is_file():
                h.update(path.read_bytes())
        except OSError:
            pass
    return h.hexdigest()


def _run(cmd, log_path, timeout):
    """Runs the command its with output for file (without pipe preso)."""
    with open(log_path, "ab") as log:
        log.write(("$ %s\n" % cmd).encode())
        log.flush()
        try:
            proc = subprocess.run(cmd, shell=True, stdout=log, stderr=log,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124
    return proc.returncode


def _fill(template, **paths):
    return template.format(**{k: shlex.quote(str(v)) for k, v in paths.items()})


class CommandEvaluator:
    def __init__(self, cmd, workdir, timeout=DEFAULT_TIMEOUT):
        self.cmd, self.workdir, self.timeout = cmd, Path(workdir), timeout
        self.calls = 0

    def identity(self):
        return command_identity(self.cmd)

    def __call__(self, config, tasks, run_seed):
        self.calls += 1
        d = self.workdir / ("eval-%04d" % self.calls)
        d.mkdir(parents=True, exist_ok=True)
        cfg, tk, out = d / "config.json", d / "tasks.json", d / "scores.json"
        cfg.write_text(json.dumps(config, indent=2))
        tk.write_text(json.dumps(tasks, ensure_ascii=False))
        rc = _run(_fill(self.cmd, config=cfg, tasks=tk, seed=run_seed, out=out),
                  d / "log.txt", self.timeout)
        if rc != 0 or not out.exists():
            raise ImproveError("evaluator failed (code %s), see %s"
                               % (rc, d / "log.txt"))
        return json.loads(out.read_text())


class CommandProposer:
    def __init__(self, cmd, workdir, timeout=DEFAULT_TIMEOUT):
        self.cmd, self.workdir, self.timeout = cmd, Path(workdir), timeout

    def identity(self):
        return command_identity(self.cmd)

    def __call__(self, champion, history, k, round_index):
        d = self.workdir / ("propose-%02d" % round_index)
        d.mkdir(parents=True, exist_ok=True)
        ch, hist, out = d / "champion.json", d / "history.jsonl", d / "candidates.json"
        ch.write_text(json.dumps(champion, indent=2))
        hist.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n"
                                for e in history))
        rc = _run(_fill(self.cmd, champion=ch, history=hist, k=k,
                        round=round_index, out=out), d / "log.txt", self.timeout)
        if rc != 0 or not out.exists():
            raise ImproveError("proposer failed (code %s), see %s"
                               % (rc, d / "log.txt"))
        data = json.loads(out.read_text())
        if not isinstance(data, list):
            raise ImproveError("proposer must write a JSON list")
        return data


def command_validator(cmd, workdir, timeout=DEFAULT_TIMEOUT):
    state = {"n": 0}

    def validate(config):
        state["n"] += 1
        d = Path(workdir) / ("validate-%04d" % state["n"])
        d.mkdir(parents=True, exist_ok=True)
        cfg = d / "config.json"
        cfg.write_text(json.dumps(config, indent=2))
        rc = _run(_fill(cmd, config=cfg), d / "log.txt", timeout)
        return rc == 0, "code %d (log in %s)" % (rc, d / "log.txt")
    return validate


def load_suite(catalog_dir, suite_families=None):
    catalog_dir = Path(catalog_dir)
    files = sorted(catalog_dir.glob("*.md"))
    if not files:
        raise ImproveError("catalog empty or missing: %s" % catalog_dir)
    tasks = []
    for path in files:
        text = path.read_text()
        fam = (suite_families or {}).get(path.stem)
        tasks.append({"id": path.stem, "family": fam, "text": text})
    shingles = [gen_mod.shingles(gen_mod.normalize(t["text"])) for t in tasks]
    return tasks, shingles


def _clean_scores(raw, tasks, origin):
    """Scores validated; task without the score becomes 0 ("not run" counts the
    failure, never the hit). Returns (scores, coverage)."""
    if not isinstance(raw, dict):
        raise ImproveError("%s: evaluator returned something that is not an object" % origin)
    out, have = {}, 0
    for t in tasks:
        if t["id"] in raw and raw[t["id"]] is not None:
            out[t["id"]] = validate_score(t["id"], raw[t["id"]], origin)
            have += 1
        else:
            out[t["id"]] = 0
    return out, (have / len(tasks) if tasks else 0.0)


def _lesson(entry):
    v = entry.get("verdict")
    s, h = entry.get("suite_delta"), entry.get("held_delta")
    trig = "invalid if the suite, evaluator or generator change"
    if entry.get("gate") == "validation":
        return "failed validation before being measured; %s" % trig
    if entry.get("gate") == "family_regression":
        return ("mean rose but family %s dropped; a trade is not an improvement; %s"
                % (", ".join(entry.get("regressed", [])), trig))
    if v == "OVERFIT":
        return ("suite %+.2f without transfer (held %+.2f): fitted to the suite; "
                "do not repeat variations without evidence in the held-out; %s" % (s, h, trig))
    if v == "GENUINE":
        return "transferred (held %+.2f, suite %+.2f); %s" % (h, s, trig)
    if v == "NO IMPROVEMENT":
        return "no effect above noise; %s" % trig
    if v == "INCONCLUSIVE":
        return "measure did not decide; re-evaluate with more tasks; %s" % trig
    return "%s (suite %+.2f, held %+.2f); %s" % (v, s or 0, h or 0, trig)


def _history_view(entries):
    """The that the proposer can see: nothing per task of the held-out."""
    view = []
    for e in entries:
        e = {k: v for k, v in e.items() if k not in ("held_scores",)}
        view.append(e)
    return view


def _write_atomic(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def improve(initial, evaluator, proposer, catalog_dir, out_dir,
            suite_families=None, rounds=DEFAULT_ROUNDS, k=DEFAULT_K,
            budget=None, patience=DEFAULT_PATIENCE, n_held=DEFAULT_N_HELD,
            seed=1, sanity_good=None, sanity_bad=None, validate=None,
            final_exam=True, n_boot=1000, evaluator_id=None, proposer_id=None,
            family_margin=stats_mod.FAMILY_MARGIN, frontier_target=None,
            log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lineage = out_dir / "lineage.jsonl"
    if lineage.exists() and lineage.stat().st_size:
        raise ImproveError("a ledger already exists in %s; use another --out" % lineage)
    suite_tasks, catalog_shingles = load_suite(catalog_dir, suite_families)
    if evaluator_id is None:
        evaluator_id = (evaluator.identity() if hasattr(evaluator, "identity")
                        else callable_identity(evaluator))
    if proposer_id is None:
        proposer_id = (proposer.identity() if hasattr(proposer, "identity")
                       else callable_identity(proposer))

    entries = []

    def record(entry):
        entry.setdefault("ts", _now())
        ledger_mod.append(lineage, entry)
        entries.append(entry)
        return entry

    champion = dict(initial)
    champion_id = "h0"
    _write_atomic(out_dir / "champion.json", champion)
    # scores of the initial champion in the suite: the suite is public, the proposer
    # can use them already in the first round (held-out never)
    start_suite, _ = _clean_scores(evaluator(champion, suite_tasks, seed * 100000),
                                   suite_tasks, "inicial/suite")
    record({"type": "start", "champion": champion_id,
            "champion_suite_scores": start_suite,
            "config_hash": config_hash(champion), "evaluator_sha": evaluator_id,
            "proposer_sha": proposer_id, "rounds": rounds, "k": k,
            "budget": budget, "patience": patience, "n_held": n_held,
            "family_margin": family_margin, "frontier_target": frontier_target,
            "seed": seed, "n_suite": len(suite_tasks)})
    log("improve: suite with %d tasks, up to %d rounds of %d candidates, "
        "held-out of %d per round" % (len(suite_tasks), rounds, k, n_held))

    last_suite = start_suite
    burned = []
    # Proof final sealed: generated Before of the first round and excluded of all
    # the rounds, for existir always and never ter sido vista by nobody.
    final_dir = out_dir / "final" / "heldout"
    final_manifest = None
    if final_exam:
        final_manifest = gen_mod.generate_one(seed * 1000 + 999, n_held,
                                              final_dir, catalog_shingles,
                                              quiet=True)
        if final_manifest is None:
            raise ImproveError("could not generate the sealed final exam with "
                               "%d tasks" % n_held)
        burned += gen_mod.load_exclude([final_dir])
    placebos = []
    evals_used = 0
    stall = 0
    accepted = []
    stop_reason = "rounds exhausted"

    def current_evaluator_id():
        if hasattr(evaluator, "identity"):
            return evaluator.identity()
        return callable_identity(evaluator)

    for r in range(1, rounds + 1):
        if current_evaluator_id() != evaluator_id:
            stop_reason = "evaluator changed mid-loop (different hash)"
            break
        if budget is not None and evals_used >= budget:
            stop_reason = "evaluation budget exhausted"
            break
        rdir = out_dir / "rounds" / ("r%02d" % r)
        rdir.mkdir(parents=True, exist_ok=True)
        want = k if budget is None else min(k, budget - evals_used)
        candidates = proposer(champion, _history_view(entries), want, r) or []
        candidates = [c for c in candidates if isinstance(c, dict)][:want]
        if not candidates:
            stop_reason = "proposer brought no candidates"
            break

        # held-out new, generated after the proposal, without tasks already used.
        # With frontier_target, the families perto of the frontier of the champion
        # (measurement in the suite, that is public) rendem more tasks (Ornith-1.5).
        weights = None
        if frontier_target is not None:
            fam_scores = {}
            for t in suite_tasks:
                if t.get("family") and t["id"] in last_suite:
                    fam_scores.setdefault(t["family"], []).append(
                        last_suite[t["id"]] / 2.0)
            success = {f: sum(v) / len(v) for f, v in fam_scores.items()}
            weights = gen_mod.frontier_weights(success, target=frontier_target)
        hdir = rdir / "heldout"
        manifest = gen_mod.generate_one(seed * 1000 + r, n_held, hdir,
                                        catalog_shingles, family_weights=weights,
                                        exclude=burned, quiet=True)
        if manifest is None:
            stop_reason = ("held-out exhausted: the generator found no %d tasks "
                           "new and distinct from those already used" % n_held)
            break
        burned += gen_mod.load_exclude([hdir])
        held_tasks = [{"id": t["id"], "family": t["family"],
                       "text": (hdir / t["file"]).read_text()}
                      for t in manifest["tasks"]]
        family_of = {t["id"]: t["family"] for t in held_tasks}
        base_seed = seed * 100000 + r * 100

        def measure(cfg, run_seed, label):
            s_raw = evaluator(cfg, suite_tasks, run_seed)
            h_raw = evaluator(cfg, held_tasks, run_seed)
            s, cs = _clean_scores(s_raw, suite_tasks, "%s/suite" % label)
            h, ch = _clean_scores(h_raw, held_tasks, "%s/held" % label)
            return s, h, min(cs, ch)

        suite_b, held_b, cov_b = measure(champion, base_seed, "champion")
        last_suite = suite_b
        if cov_b < gap_mod.MIN_COVERAGE:
            stop_reason = "the champion had no score in %.0f%% of the tasks" % (
                (1 - cov_b) * 100)
            break

        # placebo: the own champion of new
        suite_p, held_p, _ = measure(champion, base_seed + 1, "placebo")
        pr = gap_mod.analyze(suite_b, suite_p, held_b, held_p, n_boot=200)
        placebos.append(record({
            "type": "placebo", "placebo": True, "round": r,
            "suite_delta": pr["suite_delta"], "held_delta": pr["held_delta"],
            "gap": pr["gap"]}))
        calibration = placebo_mod.stats(placebos)
        floor = stats_mod.noise_floor(calibration)

        # sanity of the evaluator neste held-out
        if sanity_good is not None and sanity_bad is not None:
            sg, hg, _ = measure(sanity_good, base_seed + 2, "sanity-good")
            sb_, hb_, _ = measure(sanity_bad, base_seed + 3, "sanity-bad")
            boot = stats_mod.bootstrap(sb_, sg, hb_, hg, n_boot=500)
            ok = (boot["ci"]["suite_delta"][0] > 0
                  and boot["ci"]["held_delta"][0] > 0)
            record({"type": "sanity", "round": r, "ok": ok,
                    "suite_good_minus_bad": boot["point"]["suite_delta"],
                    "held_good_minus_bad": boot["point"]["held_delta"],
                    "ci_suite": boot["ci"]["suite_delta"],
                    "ci_held": boot["ci"]["held_delta"]})
            if not ok:
                stop_reason = ("sanity failed: the evaluator does not separate the "
                               "good harness from the bad one in this set")
                break

        results = []
        for i, cand in enumerate(candidates):
            cid = "h%d.%d" % (r, i + 1)
            change = cand.get("change", "(no description)")
            cfg = {kk: vv for kk, vv in cand.items() if kk not in ("change", "kind")}
            (rdir / ("%s.json" % cid)).write_text(json.dumps(cand, indent=2,
                                                             ensure_ascii=False))
            entry = {"type": "candidate", "round": r, "id": cid,
                     "parent": champion_id, "change": change,
                     "config_hash": config_hash(cfg)}
            if "kind" in cand:
                entry["kind"] = cand["kind"]
            if validate is not None:
                ok, msg = validate(cfg)
                if not ok:
                    entry.update({"decision": "rejected", "gate": "validation",
                                  "detail": msg})
                    entry["lesson"] = _lesson(entry)
                    results.append((entry, None, cfg))
                    continue
            evals_used += 1
            suite_c, held_c, cov_c = measure(cfg, base_seed + 10 + i, cid)
            res = gap_mod.analyze(suite_b, suite_c, held_b, held_c,
                                  calibration=calibration, n_boot=n_boot)
            fam = stats_mod.family_regressions(held_b, held_c, family_of,
                                               floor_held=floor["held"],
                                               margin=family_margin,
                                               n_boot=500)
            regressed = sorted(f for f, v in fam.items() if v["regressed"])
            # the suite also is stores-corpo of regression (blog of the Google):
            # the family that despenca in the suite counts, same that the held-out
            # not tenha n for see
            suite_family_of = {t["id"]: t["family"] for t in suite_tasks
                               if t.get("family")}
            if suite_family_of:
                fam_s = stats_mod.family_regressions(
                    suite_b, suite_c, suite_family_of,
                    floor_held=floor["suite"], margin=family_margin, n_boot=500)
                regressed = sorted(set(regressed) | {
                    "%s (suite)" % f for f, v in fam_s.items() if v["regressed"]})
            entry.update({
                "verdict": res["verdict"], "verdict_point": res["verdict_point"],
                "suite_delta": res["suite_delta"], "held_delta": res["held_delta"],
                "gap": res["gap"],
                "ci_held": res["stats"]["bootstrap"]["ci"]["held_delta"],
                "ci_gap": res["stats"]["bootstrap"]["ci"]["gap"],
                "why": res["stats"]["why"], "coverage": round(cov_c, 4),
                "regressed": regressed,
            })
            if cov_c < gap_mod.MIN_COVERAGE:
                entry.update({"decision": "rejected", "gate": "coverage"})
            elif res["verdict"] != "GENUINE":
                entry.update({"decision": "rejected", "gate": "verdict"})
            elif regressed:
                entry.update({"decision": "rejected",
                              "gate": "family_regression"})
            else:
                entry.update({"decision": "approved", "gate": None})
            entry["lesson"] = _lesson(entry)
            results.append((entry, res, cfg))

        passing = [(e, res, cfg) for e, res, cfg in results
                   if e.get("decision") == "approved"]
        winner = None
        if passing:
            winner = max(passing, key=lambda x: (x[0]["ci_held"][0],
                                                 x[0]["held_delta"]))
        for e, res, cfg in results:
            if winner is not None and e is winner[0]:
                e["decision"] = "accepted"
            elif e.get("decision") == "approved":
                e["decision"] = "rejected"
                e["gate"] = "another_better"
                e["lesson"] = ("GENUINE, but another candidate in the round had a CI "
                               "held-out higher; may be proposed again")
            record(e)
        before = champion_id
        if winner is not None:
            champion = winner[2]
            champion_id = winner[0]["id"]
            _write_atomic(out_dir / "champion.json", champion)
            accepted.append(winner[0])
            stall = 0
        else:
            stall += 1
        record({"type": "round", "round": r, "champion_before": before,
                "champion_after": champion_id,
                "accepted": winner[0]["id"] if winner else None,
                "n_held": len(held_tasks), "noise_floor": floor,
                "family_weights": weights,
                "champion_suite_scores": suite_b,
                "champion_held_mean": round(sum(held_b.values()) / len(held_b), 3)})
        log("round %d: %s" % (r, "; ".join(
            "%s %s (%s)" % (e["id"], e.get("verdict", "-"), e["decision"])
            for e, _, _ in results)))
        if stall >= patience:
            stop_reason = "%d consecutive round(s) without acceptance" % stall
            break

    final = None
    if accepted and final_exam:
        fdir, manifest = final_dir, final_manifest
        if manifest is None:
            final = {"type": "final_exam", "skipped": "no unseen tasks"}
        else:
            held_tasks = [{"id": t["id"], "family": t["family"],
                           "text": (fdir / t["file"]).read_text()}
                          for t in manifest["tasks"]]
            fs = seed * 100000 + 99900
            s0, _ = _clean_scores(evaluator(initial, suite_tasks, fs),
                                  suite_tasks, "final/inicial")
            h0, _ = _clean_scores(evaluator(initial, held_tasks, fs),
                                  held_tasks, "final/inicial")
            s1, _ = _clean_scores(evaluator(champion, suite_tasks, fs + 1),
                                  suite_tasks, "final/champion")
            h1, _ = _clean_scores(evaluator(champion, held_tasks, fs + 1),
                                  held_tasks, "final/champion")
            res = gap_mod.analyze(s0, s1, h0, h1,
                                  calibration=placebo_mod.stats(placebos),
                                  n_boot=n_boot)
            final = {"type": "final_exam", "from": "h0", "to": champion_id,
                     "verdict": res["verdict"], "suite_delta": res["suite_delta"],
                     "held_delta": res["held_delta"], "gap": res["gap"],
                     "ci_held": res["stats"]["bootstrap"]["ci"]["held_delta"],
                     "ci_gap": res["stats"]["bootstrap"]["ci"]["gap"],
                     "n_held": len(held_tasks)}
        record(final)
    summary = {"type": "stop", "reason": stop_reason, "champion": champion_id,
               "accepted": [e["id"] for e in accepted],
               "candidate_evals": evals_used, "rounds_run": len(
                   [e for e in entries if e["type"] == "round"])}
    record(summary)
    (out_dir / "report.md").write_text(render(entries))
    log("improve: stopped (%s); champion %s; accepted: %s"
        % (stop_reason, champion_id, ", ".join(summary["accepted"]) or "none"))
    return {"champion": champion, "champion_id": champion_id,
            "entries": entries, "summary": summary, "final": final}


def render(entries):
    lines = ["# hod improve: lineage", ""]
    start = next((e for e in entries if e["type"] == "start"), {})
    lines.append("Suite: %s tasks | held-out per round: %s | k=%s | "
                 "budget: %s | patience: %s"
                 % (start.get("n_suite"), start.get("n_held"), start.get("k"),
                    start.get("budget") or "no limit", start.get("patience")))
    lines.append("Evaluator (sha256): `%s`" % str(start.get("evaluator_sha"))[:16])
    lines += ["", "| Round | Candidate | Change | Suite Δ | Held Δ (CI 95%) | "
              "Verdict | Decision | Gate |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in entries:
        if e["type"] != "candidate":
            continue
        ci = e.get("ci_held")
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            e["round"], e["id"], e["change"],
            ("%+.3f" % e["suite_delta"]) if "suite_delta" in e else "-",
            ("%+.3f [%+.3f, %+.3f]" % (e["held_delta"], ci[0], ci[1]))
            if ci else "-",
            e.get("verdict", "-"), e["decision"], e.get("gate") or "-"))
    lines += ["", "## Rounds", ""]
    for e in entries:
        if e["type"] == "round":
            lines.append("- Round %d: champion %s -> %s (noise floor in the "
                         "held-out %.3f, n=%d)"
                         % (e["round"], e["champion_before"],
                            e["champion_after"], e["noise_floor"]["held"],
                            e["n_held"]))
        elif e["type"] == "sanity":
            lines.append("- Sanity round %d: %s (good − bad in the held-out "
                         "%+.3f)" % (e["round"], "ok" if e["ok"] else "FAILED",
                                     e["held_good_minus_bad"]))
    final = next((e for e in entries if e["type"] == "final_exam"), None)
    if final:
        lines += ["", "## Final exam (unseen held-out)", ""]
        if final.get("skipped"):
            lines.append("Skipped: %s" % final["skipped"])
        else:
            lines.append(
                "%s -> %s: **%s** | suite %+.3f | held-out %+.3f CI [%+.3f, "
                "%+.3f] | gap %+.3f (n=%d)"
                % (final["from"], final["to"], final["verdict"],
                   final["suite_delta"], final["held_delta"],
                   final["ci_held"][0], final["ci_held"][1], final["gap"],
                   final["n_held"]))
    stop = next((e for e in entries if e["type"] == "stop"), None)
    if stop:
        lines += ["", "Stopped: %s. Final champion: %s. Accepted: %s."
                  % (stop["reason"], stop["champion"],
                     ", ".join(stop["accepted"]) or "none")]
    lines += ["", "## Lessons", ""]
    for e in entries:
        if e["type"] == "candidate":
            lines.append("- %s (%s): %s" % (e["id"], e["change"], e.get("lesson")))
    return "\n".join(lines) + "\n"


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(
        prog=prog, description="self-improvement loop: only accepts "
        "GENUINE changes measured on a new held-out each round")
    parser.add_argument("--initial", type=Path, required=True,
                        help="config JSON do harness inicial")
    parser.add_argument("--eval", dest="eval_cmd", required=True,
                        help='evaluator command with {config} {tasks} {seed} {out}')
    parser.add_argument("--propose", dest="propose_cmd", required=True,
                        help='proposer command with {champion} {history} {k} '
                        '{round} {out}')
    parser.add_argument("--catalog", type=Path, required=True,
                        help="fixed suite directory (*.md)")
    parser.add_argument("--suite-families", type=Path, default=None,
                        help="JSON {suite_id: family} (optional)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--budget", type=int, default=None,
                        help="maximum candidate evaluations")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--n-held", type=int, default=DEFAULT_N_HELD)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sanity-good", type=Path, default=None)
    parser.add_argument("--sanity-bad", type=Path, default=None)
    parser.add_argument("--validate", default=None,
                        help="command with {config}; must exit 0")
    parser.add_argument("--no-final-exam", action="store_true")
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--family-margin", type=float,
                        default=stats_mod.FAMILY_MARGIN,
                        help="maximum tolerated drop per family in the held-out "
                        "(points, default %(default)s)")
    parser.add_argument("--frontier-target", type=float, default=None,
                        help="enables the frontier held-out: families with "
                        "champion success near this target (e.g. 0.35) "
                        "rendem mais tasks")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="seconds per command call")
    args = parser.parse_args(argv)
    if (args.sanity_good is None) != (args.sanity_bad is None):
        print("improve: --sanity-good and --sanity-bad go together", file=sys.stderr)
        return 1
    work = args.out / "work"
    try:
        initial = json.loads(args.initial.read_text())
        families = (json.loads(args.suite_families.read_text())
                    if args.suite_families else None)
        evaluator = CommandEvaluator(args.eval_cmd, work, args.timeout)
        proposer = CommandProposer(args.propose_cmd, work, args.timeout)
        validate = (command_validator(args.validate, work, args.timeout)
                    if args.validate else None)
        good = json.loads(args.sanity_good.read_text()) if args.sanity_good else None
        bad = json.loads(args.sanity_bad.read_text()) if args.sanity_bad else None
        result = improve(initial, evaluator, proposer, args.catalog, args.out,
                         suite_families=families, rounds=args.rounds, k=args.k,
                         budget=args.budget, patience=args.patience,
                         n_held=args.n_held, seed=args.seed, sanity_good=good,
                         sanity_bad=bad, validate=validate,
                         final_exam=not args.no_final_exam, n_boot=args.boot,
                         family_margin=args.family_margin,
                         frontier_target=args.frontier_target)
    except (ImproveError, ValueError, OSError) as e:
        print("improve: %s" % e, file=sys.stderr)
        return 1
    print("report: %s" % (args.out / "report.md"))
    return 0 if result["summary"]["accepted"] else 2


if __name__ == "__main__":
    sys.exit(main())
