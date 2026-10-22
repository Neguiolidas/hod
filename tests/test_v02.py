"""Tests of the v0.2: families parametric, novelty and rotation, statistics
by IC, sanity of the evaluator and the loop `hod improve` with the toy.

Usage: python3 -m pytest tests/test_v02.py
"""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hod import cli  # noqa: E402
from hod import demo as demo_mod  # noqa: E402
from hod import families as fam  # noqa: E402
from hod import gap as gap_mod  # noqa: E402
from hod import generate as gen  # noqa: E402
from hod import improve as imp  # noqa: E402
from hod import sanity as sanity_mod  # noqa: E402
from hod import stats as st  # noqa: E402
from hod import toy  # noqa: E402

FILLER = ("um texto de catálogo qualquer sobre roteamento de pedidos e logs "
          "de deploy que não tem relação nenhuma com o conjunto held-out ") * 3


def _cat():
    return [gen.shingles(gen.normalize(FILLER))]


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **kw)


# --- families and generator ----------------------------------------------------

def test_eight_families_and_large_space():
    assert len(fam.FAMILIES) == 8
    for name in fam.FAMILIES:
        assert len(fam.VARIANTS[name]) >= 300, name
    assert fam.space_size() > 5000


def test_param_space_indexing_is_stable():
    space = fam.VARIANTS["quota-first"]
    assert space[0] == space[0]
    assert space[-1] == space[len(space) - 1]
    with pytest.raises(IndexError):
        space[len(space)]


def test_register_refuses_axis_missing_from_text():
    with pytest.raises(ValueError):
        fam.register("x-teste", "c", ["setup {a}"], ["instr"],
                     [("a", ["1", "2"]), ("b", ["3", "4"])],
                     ["e"], "f", {0: "z", 1: "u", 2: "d"})
    fam.FAMILIES.pop("x-teste", None)
    fam.VARIANTS.pop("x-teste", None)


def test_every_family_renders_valid_tasks():
    for name in fam.FAMILIES:
        for i in (0, len(fam.VARIANTS[name]) // 2, len(fam.VARIANTS[name]) - 1):
            task = fam.FAMILIES[name](fam.VARIANTS[name][i])
            text = gen.render("heldout-%s-00" % name, task)
            assert gen.validate_task(task, text) == [], (name, i)


def test_validity_gate_catches_placeholder():
    task = fam.FAMILIES["destructive-order"](fam.VARIANTS["destructive-order"][0])
    task["setup"] = "Criar {pasta} sem preencher"
    assert gen.validate_task(task, gen.render("x", task))


def test_large_set_has_no_near_duplicates():
    """The ceiling of the v0.1 was 18 tasks; now 80 by set without none par
    of content above of the ceiling of novelty."""
    with tempfile.TemporaryDirectory() as tmp:
        m = gen.generate_one(3, 80, Path(tmp) / "s", _cat(), quiet=True)
        assert m is not None and len(m["tasks"]) == 80
        sh = [gen.content_shingles((Path(tmp) / "s" / t["file"]).read_text())
              for t in m["tasks"]]
        worst = max(gen.jaccard(sh[i], sh[j]) for i in range(len(sh))
                    for j in range(i + 1, len(sh)))
        assert worst <= gen.NOVELTY_THRESHOLD
        assert all(t["novelty"] >= 1 - gen.NOVELTY_THRESHOLD for t in m["tasks"])


def test_exclude_makes_rotation_disjoint():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a = gen.generate_one(1, 40, tmp / "a", _cat(), quiet=True)
        b = gen.generate_one(1, 40, tmp / "b", _cat(),
                             exclude=gen.load_exclude([tmp / "a"]), quiet=True)
        assert a and b
        sa = [gen.content_shingles((tmp / "a" / t["file"]).read_text())
              for t in a["tasks"]]
        for t in b["tasks"]:
            mine = gen.content_shingles((tmp / "b" / t["file"]).read_text())
            assert max(gen.jaccard(mine, o) for o in sa) <= gen.NOVELTY_THRESHOLD


def test_generator_fails_closed_when_novelty_exhausted():
    with tempfile.TemporaryDirectory() as tmp:
        # ceiling of novelty tight: the space "really distinct" runs out
        m = gen.generate_one(1, 400, Path(tmp) / "x", _cat(),
                             novelty_threshold=0.15, quiet=True)
        assert m is None
        assert not list((Path(tmp) / "x").glob("heldout-*.md")) \
            if (Path(tmp) / "x").exists() else True


def test_cli_seeds_batch_is_disjoint_between_seeds():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cat = tmp / "cat"
        cat.mkdir()
        (cat / "a.md").write_text(FILLER)
        rc = _quiet(gen.main, ["--seeds", "1,2,3", "--n", "24", "--out",
                               str(tmp / "o"), "--catalog", str(cat)])
        assert rc == 0
        texts = []
        for s in (1, 2, 3):
            for p in sorted((tmp / "o" / ("seed-%d" % s)).glob("heldout-*.md")):
                texts.append(gen.task_content(p.read_text()))
        assert len(set(texts)) == len(texts) == 72


def test_frontier_weights_prefer_mid_difficulty_and_keep_floor():
    w = gen.frontier_weights({"quota-first": 1.0, "ambiguity": 0.35,
                              "gate-precedent": 0.0})
    assert w["ambiguity"] > w["gate-precedent"] > w["quota-first"]
    assert w["quota-first"] >= gen.FRONTIER_FLOOR   # saturated stays present
    assert w["destructive-order"] == 1.0            # sem taxa conhecida


def test_weighted_generation_uses_stratified_quotas():
    weights = {f: 1.0 for f in fam.FAMILIES}
    weights["ambiguity"] = 3.0
    with tempfile.TemporaryDirectory() as tmp:
        m = gen.generate_one(5, 50, Path(tmp) / "w", _cat(),
                             family_weights=weights, quiet=True)
        counts = {}
        for t in m["tasks"]:
            counts[t["family"]] = counts.get(t["family"], 0) + 1
        assert counts["ambiguity"] == 15          # 50 * 3/10
        assert min(counts.values()) >= 5


# --- statistics --------------------------------------------------------------

def _scores(prefix, values):
    return {"%s%d" % (prefix, i): v for i, v in enumerate(values)}


def test_bootstrap_is_deterministic():
    sb, sc = _scores("s", [0, 1] * 20), _scores("s", [1, 2] * 20)
    hb, hc = _scores("h", [0, 1, 1] * 15), _scores("h", [1, 1, 2] * 15)
    a = st.bootstrap(sb, sc, hb, hc, n_boot=500)
    b = st.bootstrap(sb, sc, hb, hc, n_boot=500)
    assert a == b
    lo, hi = a["ci"]["held_delta"]
    assert lo <= a["point"]["held_delta"] <= hi


def test_ci_verdicts_on_clear_cases():
    sb = _scores("s", [0, 1, 0, 1] * 10)
    sc_up = _scores("s", [1, 2, 1, 2] * 10)
    hb = _scores("h", [0, 1, 1, 0] * 10)
    hc_up = _scores("h", [1, 2, 2, 1] * 10)
    assert gap_mod.analyze(sb, sc_up, hb, hc_up)["verdict"] == "GENUINE"
    assert gap_mod.analyze(sb, sc_up, hb, dict(hb))["verdict"] == "OVERFIT"
    assert gap_mod.analyze(sb, dict(sb), hb, dict(hb))["verdict"] == "NO IMPROVEMENT"
    hc_down = _scores("h", [0, 0, 0, 0] * 10)
    assert gap_mod.analyze(sb, dict(sb), hb, hc_down)["verdict"] == "REGRESSION"


def test_ci_says_inconclusive_where_point_rule_overclaims():
    """n small and noisy: the rule point nails the verdict, the of IC not."""
    sb, sc = {"s0": 0, "s1": 1, "s2": 1}, {"s0": 2, "s1": 1, "s2": 1}
    hb, hc = {"h0": 0, "h1": 2, "h2": 1}, {"h0": 2, "h1": 1, "h2": 1}
    r = gap_mod.analyze(sb, sc, hb, hc)
    assert r["verdict_point"] in ("GENUINE", "PARTIAL", "OVERFIT")
    assert r["verdict"] == "INCONCLUSIVE"
    assert r["stats"]["power"]["warnings"]


def test_placebo_noise_floor_blocks_small_effects():
    sb, sc = _scores("s", [0, 1] * 30), _scores("s", [1, 1] * 30)  # +0.5
    hb, hc = _scores("h", [0, 1] * 30), _scores("h", [1, 1] * 30)
    assert gap_mod.analyze(sb, sc, hb, hc)["verdict"] == "GENUINE"
    noisy = {"n_placebo": 3, "max_abs_suite_delta": 0.6,
             "max_abs_held_delta": 0.6, "max_abs_gap": 0.6, "gaps": [0.6]}
    r = gap_mod.analyze(sb, sc, hb, hc, calibration=noisy)
    assert r["verdict"] == "NO IMPROVEMENT"


def test_missing_scores_reduce_coverage_and_block_verdict():
    sb = _scores("s", [0, 1, 0, 1] * 10)
    sc = _scores("s", [1, 2, 1, 2] * 10)
    hb = _scores("h", [0, 1, 1, 0] * 10)
    hc = {k: v + 1 for k, v in list(hb.items())[:30]}  # 10 sem nota
    r = gap_mod.analyze(sb, sc, hb, hc)
    assert r["stats"]["coverage"]["held"] < gap_mod.MIN_COVERAGE
    assert r["verdict"] == "INCONCLUSIVE"


def test_strict_mode_needs_more_evidence():
    sb, sc = _scores("s", [0, 1] * 15), _scores("s", [1, 2] * 15)
    hb = _scores("h", [0, 1, 1] * 10)
    hc = _scores("h", [1, 1, 2] * 10)
    lenient = gap_mod.analyze(sb, sc, hb, hc)
    strict = gap_mod.analyze(sb, sc, hb, hc, strict=True)
    order = ["OVERFIT", "INCONCLUSIVE", "PARTIAL", "GENUINE"]
    assert order.index(strict["verdict"]) <= order.index(lenient["verdict"])


def test_family_regression_guard():
    base = _scores("h", [1] * 24)
    fam_of = {t: ("a" if i < 12 else "b") for i, t in enumerate(sorted(base))}
    cand = {t: (2 if fam_of[t] == "a" else 0) for t in base}
    r = st.family_regressions(base, cand, fam_of)
    assert r["b"]["regressed"] and not r["a"]["regressed"]


def test_gap_cli_reports_ci(tmp_path):
    files = {}
    data = {"sb": _scores("s", [0, 1] * 10), "sc": _scores("s", [1, 2] * 10),
            "hb": _scores("h", [0, 1] * 10), "hc": _scores("h", [0, 1] * 10)}
    for k, v in data.items():
        files[k] = tmp_path / ("%s.json" % k)
        files[k].write_text(json.dumps(v))
    out = tmp_path / "r.md"
    rc = _quiet(cli.main, ["gap", "--suite-base", str(files["sb"]),
                           "--suite-cand", str(files["sc"]),
                           "--held-base", str(files["hb"]),
                           "--held-cand", str(files["hc"]), "--out", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "Confidence intervals" in text and "OVERFIT" in text


# --- sanity --------------------------------------------------------------

def test_sanity_detects_evaluator_that_cannot_separate():
    good = _scores("h", [2, 2, 1, 2] * 8)
    bad = _scores("h", [0, 1, 0, 0] * 8)
    assert sanity_mod.check(good, bad)["ok"]
    assert not sanity_mod.check(good, dict(good))["ok"]


# --- loop improve ----------------------------------------------------------

@pytest.fixture(scope="module")
def demo_run():
    tmp = tempfile.mkdtemp()
    result = demo_mod.run_one(Path(tmp) / "d", seed=7, quiet=True)
    return Path(tmp) / "d", result


def test_improve_rejects_overfit_and_placebo_accepts_genuine(demo_run):
    _, result = demo_run
    cands = [e for e in result["entries"] if e["type"] == "candidate"]
    accepted = [e for e in cands if e["decision"] == "accepted"]
    assert accepted and all(e["kind"] == "genuine" for e in accepted)
    for e in cands:
        if e["kind"] in ("overfit", "placebo"):
            assert e["decision"] == "rejected"
    assert any(e["kind"] == "overfit" and e["verdict"] == "OVERFIT"
               for e in cands)
    assert result["final"]["verdict"] == "GENUINE"


def test_improve_family_guard_fires(demo_run):
    _, result = demo_run
    assert any(e.get("gate") == "family_regression" for e in result["entries"]
               if e["type"] == "candidate")


def test_improve_rotates_heldout_and_seals_final(demo_run):
    root, _ = demo_run
    sets = sorted((root / "improve" / "rounds").glob("r*/heldout"))
    sets.append(root / "improve" / "final" / "heldout")
    assert len(sets) >= 3
    seen = []
    for d in sets:
        for p in sorted(d.glob("heldout-*.md")):
            mine = gen.content_shingles(p.read_text())
            assert all(gen.jaccard(mine, o) <= gen.NOVELTY_THRESHOLD
                       for o in seen), p
        seen += gen.load_exclude([d])


def test_improve_lineage_and_champion_files(demo_run):
    root, result = demo_run
    lines = (root / "improve" / "lineage.jsonl").read_text().splitlines()
    types = [json.loads(line)["type"] for line in lines]
    assert types[0] == "start" and types[-1] == "stop"
    assert "placebo" in types and "final_exam" in types
    champ = json.loads((root / "improve" / "champion.json").read_text())
    assert champ == result["champion"]
    for line in lines:
        e = json.loads(line)
        if e["type"] == "candidate":
            assert e.get("lesson") and "invalid" in e["lesson"] \
                or e["gate"] == "another_better"


def test_proposer_never_sees_heldout_tasks(tmp_path):
    seen = []

    def spy(champion, history, k, r):
        seen.append(json.dumps(history))
        return toy.propose(champion, history, k, r)

    families = toy.write_catalog(tmp_path / "cat")
    imp.improve(toy.initial_config(), toy.evaluate, spy, tmp_path / "cat",
                tmp_path / "out", suite_families=families, rounds=2, k=3,
                n_held=40, seed=3, log=lambda *a: None)
    assert seen and all("heldout-" not in h for h in seen)


def test_improve_stops_when_evaluator_changes(tmp_path):
    families = toy.write_catalog(tmp_path / "cat")
    state = {"n": 0}

    class Ev:
        def identity(self):
            state["n"] += 1
            return "a" if state["n"] < 3 else "b"

        def __call__(self, cfg, tasks, seed):
            return toy.evaluate(cfg, tasks, seed)

    r = imp.improve(toy.initial_config(), Ev(), toy.propose, tmp_path / "cat",
                    tmp_path / "out", suite_families=families, rounds=5, k=2,
                    n_held=32, seed=1, log=lambda *a: None)
    assert "evaluator changed" in r["summary"]["reason"]


def test_improve_budget_and_validation_gate(tmp_path):
    families = toy.write_catalog(tmp_path / "cat")
    r = imp.improve(toy.initial_config(), toy.evaluate, toy.propose,
                    tmp_path / "cat", tmp_path / "out", suite_families=families,
                    rounds=10, k=3, budget=4, n_held=32, seed=2,
                    validate=lambda cfg: (False, "always fails"),
                    log=lambda *a: None)
    cands = [e for e in r["entries"] if e["type"] == "candidate"]
    assert cands and all(e["gate"] == "validation" for e in cands)
    assert not r["summary"]["accepted"]


def test_improve_stops_on_failed_sanity(tmp_path):
    families = toy.write_catalog(tmp_path / "cat")
    r = imp.improve(toy.initial_config(), toy.evaluate, toy.propose,
                    tmp_path / "cat", tmp_path / "out", suite_families=families,
                    rounds=3, k=2, n_held=32, seed=2,
                    sanity_good=toy.known_bad(), sanity_bad=toy.known_bad(),
                    log=lambda *a: None)
    assert "sanity" in r["summary"]["reason"]


def test_improve_frontier_mode_records_weights(tmp_path):
    families = toy.write_catalog(tmp_path / "cat")
    cfg = toy.initial_config()
    cfg["skills"]["quota-first"] = 1.0       # saturated family
    r = imp.improve(cfg, toy.evaluate, toy.propose, tmp_path / "cat",
                    tmp_path / "out", suite_families=families, rounds=1, k=2,
                    n_held=48, seed=4, frontier_target=0.35,
                    log=lambda *a: None)
    rounds = [e for e in r["entries"] if e["type"] == "round"]
    w = rounds[0]["family_weights"]
    assert w and w["quota-first"] < max(w.values())


def test_improve_refuses_existing_ledger(tmp_path):
    families = toy.write_catalog(tmp_path / "cat")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "lineage.jsonl").write_text('{"type": "start"}\n')
    with pytest.raises(imp.ImproveError):
        imp.improve(toy.initial_config(), toy.evaluate, toy.propose,
                    tmp_path / "cat", tmp_path / "out",
                    suite_families=families, log=lambda *a: None)


def test_improve_cli_via_commands(tmp_path):
    """The same loop by the CLI, with evaluator and proposer the commands."""
    families = toy.write_catalog(tmp_path / "cat")
    (tmp_path / "fam.json").write_text(json.dumps(families))
    toy.main(["init", "--out", str(tmp_path / "cfg")])
    py = sys.executable
    rc = _quiet(cli.main, [
        "improve", "--initial", str(tmp_path / "cfg" / "inicial.json"),
        "--eval", py + " -m hod.toy eval --config {config} --tasks {tasks} "
        "--seed {seed} --out {out}",
        "--propose", py + " -m hod.toy propose --champion {champion} "
        "--history {history} --k {k} --round {round} --out {out}",
        "--catalog", str(tmp_path / "cat"),
        "--suite-families", str(tmp_path / "fam.json"),
        "--out", str(tmp_path / "out"), "--rounds", "2", "--k", "3",
        "--n-held", "48", "--seed", "5",
        "--sanity-good", str(tmp_path / "cfg" / "bom.json"),
        "--sanity-bad", str(tmp_path / "cfg" / "ruim.json"),
    ])
    assert rc in (0, 2)
    assert (tmp_path / "out" / "report.md").exists()
    lines = (tmp_path / "out" / "lineage.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["type"] == "stop"


def test_cli_help_lists_new_sections():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli.main(["--help"])
    for name in ("improve", "demo", "sanity"):
        assert name in buf.getvalue()
