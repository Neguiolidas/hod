#!/usr/bin/env python3
"""Tests of regression of the audit of the v0.1.0.

Each test reproduz the bug encontrado in the audit and afirma the
behavior correct: failure in the v0.1.0 original and passes with hod_fixes.patch.
"""

import json
import os
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hod import blind as blind_mod  # noqa: E402
from hod import calibrate as calibrate_mod  # noqa: E402
from hod import gap as gap_mod  # noqa: E402
from hod import generate as gen  # noqa: E402
from hod import ledger as ledger_mod  # noqa: E402
from hod import placebo as placebo_mod  # noqa: E402
from hod import prereg as prereg_mod  # noqa: E402
from hod.families import FAMILIES, VARIANTS  # noqa: E402

FILLER = (
    "um texto de catálogo qualquer sobre roteamento de pedidos e logs "
    "de deploy que não tem relação nenhuma com o conjunto held-out "
)


def _catalog(tmp, texts=None):
    cat = Path(tmp) / "catalog"
    cat.mkdir(exist_ok=True)
    for i, text in enumerate(texts or [FILLER * 3]):
        (cat / ("t%02d.md" % i)).write_text(text)
    return cat


def _cat_shingles(cat):
    return [gen.shingles(gen.normalize(p.read_text()))
            for p in sorted(cat.glob("*.md"))]


def _quiet(fn, *a, **kw):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **kw)


def _setup_instruction(text):
    setup = text.split("### Setup", 1)[1].split("### Comportamento", 1)[0]
    return " ".join(setup.split())


# --- generation -------------------------------------------------------------

def test_generate_default_n8_has_no_duplicate_tasks():
    """B2: with --n 8 (default of the README) the v0.1.0 repeats task in ~55% of the
    seeds, and the set fails in the own audit."""
    with tempfile.TemporaryDirectory() as tmp:
        cat = _catalog(tmp)
        shingles = _cat_shingles(cat)
        for seed in range(30):
            out = Path(tmp) / ("s%d" % seed)
            manifest = _quiet(gen.generate_one, seed, 8, out, shingles)
            assert manifest is not None
            texts = [_setup_instruction((out / t["file"]).read_text())
                     for t in manifest["tasks"]]
            assert len(set(texts)) == len(texts), "seed %d repetiu" % seed


def test_generate_fails_closed_when_n_exceeds_distinct_tasks():
    total = sum(len(v) for v in VARIANTS.values())
    with tempfile.TemporaryDirectory() as tmp:
        cat = _catalog(tmp)
        out = Path(tmp) / "big"
        manifest = _quiet(gen.generate_one, 1, total + 1, out,
                          _cat_shingles(cat))
        assert manifest is None
        assert not list(out.glob("heldout-*.md")), "left partial files"


def test_similarity_gate_catches_contained_copy():
    """B3: task held-out copied integer inside the file long of the
    catalog. Jaccard dilui (|The|/|B|) and the copy passes in the ceiling of 0.30."""
    task = FAMILIES["destructive-order"](VARIANTS["destructive-order"][0])
    text = gen.render("heldout-destructive-order-00", task)
    filler = " ".join("termo%d" % i for i in range(400))
    long_doc = text + "\n" + filler
    with tempfile.TemporaryDirectory() as tmp:
        cat = _catalog(tmp, [long_doc])
        sim = gen.max_similarity(text, _cat_shingles(cat))
        assert sim > gen.THRESHOLD, sim


def test_normalize_is_unicode_stable():
    """B16: the same text in NFC and NFD must give similarity 1; the
    letters à, ü, ñ must not break words."""
    text = ("A configuração não deve ser apagada à noite sem aprovação "
            "explícita do usuário responsável pela operação")
    nfd = unicodedata.normalize("NFD", text)
    a = gen.shingles(gen.normalize(text))
    b = gen.shingles(gen.normalize(nfd))
    assert gen.jaccard(a, b) == 1.0
    assert "à" in gen.normalize(text)
    assert gen.normalize("pingüim niño") == ["pingüim", "niño"]


def test_short_texts_are_not_invisible():
    """B16: text with less of 5 words becomes set empty and
    similarity 0 (fail-open)."""
    a = gen.shingles(gen.normalize("apaga os arquivos velhos"))
    assert a, "short text ended up without shingles"
    assert gen.jaccard(a, a) == 1.0


def test_family_weights_reject_unknown_keys():
    """B1: keys that not are families (ex.: categories of the ledger) were
    ignoradas in silence."""
    with tempfile.TemporaryDirectory() as tmp:
        cat = _catalog(tmp)
        wfile = Path(tmp) / "w.json"
        wfile.write_text(json.dumps({"memory": 2.0}))
        rc = _quiet(gen.main, ["--seed", "1", "--n", "4", "--out",
                               str(Path(tmp) / "o"), "--catalog", str(cat),
                               "--family-weights", str(wfile)])
        assert rc != 0


# --- ledger and loop of feedback ------------------------------------------

def test_ledger_weights_are_keyed_by_family():
    """B1: the weights leave by category (vindos of by_category), but the
    generator reads per family: the loop of feedback not does nothing."""
    entry = {"placebo": False, "verdict": "OVERFIT",
             "by_category": {"memory": "OVERFIT", "instruction": "OVERFIT"},
             "by_family": {"memory-conflict": "OVERFIT",
                           "destructive-order": "GENUINE"}}
    rep = ledger_mod.report([entry])
    assert set(rep["weights"]) <= set(FAMILIES), rep["weights"]
    assert rep["weights"]["memory-conflict"] > \
        rep["weights"]["destructive-order"]


def test_ledger_survives_corrupted_line():
    """B14: the line truncada (crash in the meio of the append) derruba
    `hod ledger` and `hod placebo` integers."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ledger.jsonl"
        ledger_mod.append(path, {"placebo": True, "gap": 0.25})
        with open(path, "a") as f:
            f.write('{"placebo": true, "gap": 0.')
        entries = _quiet(ledger_mod.load, path)
        assert len(entries) == 1
        assert placebo_mod.stats(entries)["n_placebo"] == 1


# --- gap and verdict ------------------------------------------------------

def test_verdict_boundary_is_not_float_dependent():
    """B9: held captured exactly 3/4 of the gain; in float the reason becomes
    0.7499999999999999 and the verdict drops for Partial."""
    suite_base = {"s0": 1, "s1": 0, "s2": 0}
    suite_cand = {"s0": 1, "s1": 1, "s2": 1}
    held_base = {"h%d" % i: 0 for i in range(8)}
    held_cand = {"h%d" % i: (1 if i < 4 else 0) for i in range(8)}
    # rule point (v0.1); the rule by IC has tests own
    r = gap_mod.analyze(suite_base, suite_cand, held_base, held_cand,
                        decision="point")
    assert r["verdict"] == "GENUINE", r["verdict"]
    assert r["verdict_point"] == "GENUINE"


def test_heldout_regression_is_reported_when_suite_flat():
    """B5: suite in the ceiling (not can rise) and held-out despenca: the v0.1.0
    diz Without Improvement and esconde the regression."""
    suite = {"s%d" % i: 2 for i in range(4)}
    held_base = {"h%d" % i: 2 for i in range(8)}
    held_cand = {"h%d" % i: 0 for i in range(8)}
    r = gap_mod.analyze(suite, suite, held_base, held_cand)
    assert r["verdict"] == "REGRESSION", r["verdict"]


def test_deltas_are_paired():
    """B8: task missing in the candidate changes the mean and the delta compares
    populations different. h0 rose 1 point; the delta must be +1."""
    r = gap_mod.analyze({"s0": 0}, {"s0": 2},
                        {"h0": 0, "h1": 2}, {"h0": 1})
    assert r["held_delta"] == 1.0, r["held_delta"]


def test_empty_scores_raise():
    with pytest.raises(ValueError):
        gap_mod.analyze({"s0": 0}, {"s0": 2}, {}, {})


def test_significance_without_placebos_is_not_claimed():
    """B6: calibration with n_placebo=0 has max_abs_gap=0 and any effect
    becomes significant."""
    r = gap_mod.analyze({"s0": 1}, {"s0": 2}, {"h0": 1}, {"h0": 1},
                        calibration=placebo_mod.stats([]))
    assert r.get("significant") is not True


def test_overfit_gap_inside_placebo_noise_is_flagged():
    """B6: the v0.1.0 compares |suite_delta| with the larger |gap| placebo; the gap
    in si never is tested against the noise."""
    calibration = {"n_placebo": 5, "max_abs_gap": 0.5,
                   "max_abs_suite_delta": 0.1,
                   "gaps": [0.5, -0.5, 0.25, -0.25, 0.0]}
    r = gap_mod.analyze({"s%d" % i: 1 for i in range(4)},
                        {"s0": 2, "s1": 2, "s2": 1, "s3": 1},
                        {"h0": 1, "h1": 1}, {"h0": 1, "h1": 1},
                        calibration=calibration, decision="point")
    assert r["verdict"] == "OVERFIT"
    assert r["gap_significant"] is False


def test_load_rejects_bool_and_float_scores():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "s.json"
        p.write_text(json.dumps({"t0": True}))
        with pytest.raises(ValueError):
            gap_mod.load(p)
        p.write_text(json.dumps({"t0": 1.5}))
        with pytest.raises(ValueError):
            gap_mod.load(p)


def test_unblinded_scores_feed_gap():
    """B13: the output of `hod unblind` (format of the canary) was recusada by the
    `hod gap`, `hod calibrate` and `hod pipeline`."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "held.json"
        p.write_text(json.dumps({"tasks": {"h0": {"nota": 2}},
                                 "behavior": {}}))
        assert gap_mod.load(p) == {"h0": 2}


# --- prereg ---------------------------------------------------------------

def test_prereg_held_up_with_suite_flat_can_match():
    """B11: hypothesis "suite stopped, held-out rises" never dava match."""
    pre = prereg_mod.create("generaliza", "flat", "up", 0.75)
    res = prereg_mod.check(pre, {"suite_delta": 0.0, "held_delta": 0.5})
    assert res["match"], res


def test_prereg_regeneration_does_not_bypass_peeking():
    """B10: after see the result, regenerar the same seed recria the
    same files with mtime new and the prereg late passes."""
    with tempfile.TemporaryDirectory() as tmp:
        cat = _catalog(tmp)
        held = Path(tmp) / "held"
        _quiet(gen.main, ["--seed", "42", "--n", "6", "--out", str(held),
                          "--catalog", str(cat)])
        old = json.loads((held / "manifest.json").read_text())
        pre = Path(tmp) / "prereg.json"
        pre.write_text(json.dumps(prereg_mod.create("tardio")))
        os.utime(held / "manifest.json", (1, 1))
        os.utime(pre, (2, 2))
        _quiet(gen.main, ["--seed", "42", "--n", "6", "--out", str(held),
                          "--catalog", str(cat)])
        new = json.loads((held / "manifest.json").read_text())
        same = [t["sha256"] for t in old["tasks"]] == \
            [t["sha256"] for t in new["tasks"]]
        assert not (same and prereg_mod.verify(pre, [held / "manifest.json"])[0])


# --- calibration -----------------------------------------------------------

def test_calibration_selection_bias_is_refused():
    """B7: filter saturated with the Same baseline that enters in the delta generates
    regression to the mean: with scores puro noise, the held-out "improvement" +0.5."""
    import random
    rng = random.Random(0)
    ids = ["h%02d" % i for i in range(400)]
    base = {t: rng.choice((0, 1, 2)) for t in ids}
    cand = {t: rng.choice((0, 1, 2)) for t in ids}
    kept = [t for t in ids if base[t] < 2]
    delta = (sum(cand[t] for t in kept) - sum(base[t] for t in kept)) \
        / len(kept)
    assert delta > 0.3  # the bias exists
    with tempfile.TemporaryDirectory() as tmp:
        held = Path(tmp) / "held"
        held.mkdir()
        tasks = []
        for t in ids[:6]:
            (held / (t + ".md")).write_text("# ID: %s\nCategory: x.\n" % t)
            tasks.append({"id": t, "family": "f", "file": t + ".md",
                          "sha256": "x", "max_similarity": 0.1})
        (held / "manifest.json").write_text(json.dumps({"tasks": tasks}))
        b6 = {t: base[t] for t in ids[:6]}
        result = calibrate_mod.calibrate(held, b6)
        out = Path(tmp) / "cal"
        calibrate_mod.write_filtered(held, out, result)
        kept6 = {t["id"] for t in result["kept"]}
        from hod import pipeline as pipe
        problem = pipe.same_baseline_problem(
            json.loads((out / "manifest.json").read_text()),
            {t: b6[t] for t in kept6})
        assert problem


# --- blind ----------------------------------------------------------------

def test_blind_template_order_does_not_leak_key():
    """B12: the order of the keys in the answer key is the order alphabetical of the ids
    real: the position i of the answer key revela the task real i."""
    with tempfile.TemporaryDirectory() as tmp:
        held = Path(tmp) / "held"
        held.mkdir()
        real = ["heldout-%s-%02d" % (f, i) for i, f in
                enumerate(sorted(FAMILIES))]
        (held / "manifest.json").write_text(json.dumps(
            {"tasks": [{"id": r, "family": "f", "file": r + ".md"}
                       for r in real]}))
        leaks = 0
        for seed in range(20):
            template, key = blind_mod.make_blind(held, seed=seed)
            order = list(template["tasks"])
            if [key[a] for a in order] == sorted(real):
                leaks += 1
        assert leaks < 5, "answer key order leaked the key %d/20" % leaks


def test_blind_default_key_is_not_reproducible():
    with tempfile.TemporaryDirectory() as tmp:
        held = Path(tmp) / "held"
        held.mkdir()
        real = ["heldout-x-%02d" % i for i in range(10)]
        (held / "manifest.json").write_text(json.dumps(
            {"tasks": [{"id": r, "family": "f", "file": r + ".md"}
                       for r in real]}))
        keys = {json.dumps(blind_mod.make_blind(held)[1], sort_keys=True)
                for _ in range(5)}
        assert len(keys) > 1


def test_unblind_refuses_incomplete_scores():
    key = {"t01": "a", "t02": "b"}
    with pytest.raises(ValueError):
        blind_mod.restore({"tasks": {"t01": {"nota": 2}}}, key)
    with pytest.raises(ValueError):
        blind_mod.restore({"tasks": {"t01": {"nota": 2},
                                     "t02": {"nota": None}}}, key)


# --- audit --------------------------------------------------------------

def test_audit_still_fails_under_python_O():
    """B4: the audit uses assert; with `python -O` everything passes."""
    with tempfile.TemporaryDirectory() as tmp:
        cat = _catalog(tmp)
        held = Path(tmp) / "held"
        _quiet(gen.main, ["--seed", "3", "--n", "4", "--out", str(held),
                          "--catalog", str(cat)])
        first = sorted(held.glob("heldout-*.md"))[0]
        first.write_text(first.read_text() + "adulterado\n")
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run(
            [sys.executable, "-O", "-m", "hod", "audit", "--heldout",
             str(held), "--catalog", str(cat)],
            capture_output=True, text=True, env=env, timeout=60)
        assert proc.returncode != 0, proc.stdout


def test_audit_empty_catalog_is_clean_error():
    with tempfile.TemporaryDirectory() as tmp:
        cat = _catalog(tmp)
        held = Path(tmp) / "held"
        _quiet(gen.main, ["--seed", "3", "--n", "4", "--out", str(held),
                          "--catalog", str(cat)])
        empty = Path(tmp) / "empty"
        empty.mkdir()
        from hod import audit as audit_mod
        rc = _quiet(audit_mod.main, ["--heldout", str(held), "--catalog",
                                     str(empty)])
        assert rc != 0


# --- multi-seed -------------------------------------------------------------

def test_analyze_multi_empty_raises_valueerror():
    with pytest.raises(ValueError):
        gap_mod.analyze_multi([])


# --- pipeline: strong binding to the prereg, ids and by_family ------------------

def _pipeline_fixture(tmp, prereg_args=True):
    from hod import cli
    tmp = Path(tmp)
    cat = _catalog(tmp)
    pre = tmp / "prereg.json"
    pre.write_text(json.dumps(prereg_mod.create("x", "up", "up", 0.75)))
    held = tmp / "held"
    args = ["generate", "--seeds", "1,2", "--n", "6", "--out", str(held),
            "--catalog", str(cat)]
    if prereg_args:
        args += ["--prereg", str(pre)]
    assert _quiet(cli.main, args) == 0
    scores = tmp / "scores"
    scores.mkdir()
    (scores / "suite-base.json").write_text(json.dumps({"s0": 1, "s1": 1}))
    (scores / "suite-cand.json").write_text(json.dumps({"s0": 2, "s1": 2}))
    for seed in ("1", "2"):
        m = json.loads((held / ("seed-" + seed) / "manifest.json").read_text())
        ids = [t["id"] for t in m["tasks"]]
        (scores / ("held-base-%s.json" % seed)).write_text(
            json.dumps({t: 1 for t in ids}))
        (scores / ("held-cand-%s.json" % seed)).write_text(
            json.dumps({t: 2 for t in ids}))
    return cli, pre, held, scores


def test_pipeline_strong_prereg_and_by_family_in_ledger():
    with tempfile.TemporaryDirectory() as tmp:
        cli, pre, held, scores = _pipeline_fixture(tmp)
        m = json.loads((held / "seed-1" / "manifest.json").read_text())
        assert m["prereg_sha256"] and m["seed"] == prereg_mod.derive_seed(pre, 1)
        ledger = Path(tmp) / "ledger.jsonl"
        rc = _quiet(cli.main, [
            "pipeline", "--change", "x", "--prereg", str(pre),
            "--heldout-dir", str(held), "--scores-dir", str(scores),
            "--ledger", str(ledger), "--out", str(Path(tmp) / "out")])
        assert rc == 0
        entry = ledger_mod.load(ledger)[0]
        assert set(entry["by_family"]) <= set(FAMILIES)
        assert entry["by_family"]
        # prereg editado after the generation: recusado same with mtime old
        pre.write_text(pre.read_text().replace('"x"', '"y"'))
        os.utime(pre, (1, 1))
        rc = _quiet(cli.main, [
            "pipeline", "--change", "x", "--prereg", str(pre),
            "--heldout-dir", str(held), "--scores-dir", str(scores),
            "--out", str(Path(tmp) / "out2")])
        assert rc == 1


def test_pipeline_rejects_scores_from_another_set():
    with tempfile.TemporaryDirectory() as tmp:
        cli, pre, held, scores = _pipeline_fixture(tmp, prereg_args=False)
        (scores / "held-base-2.json").write_text(
            json.dumps({"heldout-something-else-00": 1}))
        rc = _quiet(cli.main, [
            "pipeline", "--change", "x", "--heldout-dir", str(held),
            "--scores-dir", str(scores), "--out", str(Path(tmp) / "out")])
        assert rc == 1


def test_wave_is_stratified_by_family():
    from hod import pipeline as pipe
    ids = ["heldout-a-00", "heldout-a-01", "heldout-b-02", "heldout-c-03"]
    fam = {"heldout-a-00": "a", "heldout-a-01": "a", "heldout-b-02": "b",
           "heldout-c-03": "c"}
    assert pipe.stratified_wave(ids, fam, 3) == [
        "heldout-a-00", "heldout-b-02", "heldout-c-03"]


def test_agree_validates_scores():
    from hod import agree as agree_mod
    with pytest.raises(ValueError):
        agree_mod.agree({"t0": None}, {"t0": 1})
    with pytest.raises(ValueError):
        agree_mod.agree({"t0": 3}, {"t0": 3})
    r = agree_mod.agree({"t0": 2, "t1": 2}, {"t0": 2, "t1": 2})
    assert r["kappa_defined"] is False
    w = agree_mod.agree({"t0": 0, "t1": 2, "t2": 1}, {"t0": 1, "t1": 2, "t2": 1})
    far = agree_mod.agree({"t0": 0, "t1": 2, "t2": 1}, {"t0": 2, "t1": 2, "t2": 1})
    assert w["kappa"] == far["kappa"]  # plain kappa does not see distance
    assert w["kappa_weighted"] > far["kappa_weighted"]
