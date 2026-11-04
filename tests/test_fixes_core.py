"""Regression tests for the v0.1.2 audit of the measurement core:
stats, gap, scores, sanity, agree, placebo, ledger, prereg and pipeline.
One test per defect; each failed on v0.1.1."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hod import cli  # noqa: E402
from hod import gap as gap_mod  # noqa: E402
from hod import ledger as ledger_mod  # noqa: E402
from hod import manifest as manifest_mod  # noqa: E402
from hod import placebo as placebo_mod  # noqa: E402
from hod import prereg as prereg_mod  # noqa: E402
from hod import sanity as sanity_mod  # noqa: E402
from hod import scores as scores_mod  # noqa: E402
from hod import stats as st  # noqa: E402


def S(prefix, values):
    return {"%s%d" % (prefix, i): v for i, v in enumerate(values)}


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if not isinstance(data, str) else data)
    return path


# --- stats / gap ---------------------------------------------------------

def test_single_task_is_not_genuine():
    """n=1 per side gave GENUINE with a zero-width "CI" [2.0, 2.0]."""
    r = gap_mod.analyze({"s0": 0}, {"s0": 2}, {"h0": 0}, {"h0": 2})
    assert r["verdict"] == "INCONCLUSIVE"
    assert any("too few paired tasks" in w for w in r["stats"]["why"])
    r = gap_mod.analyze(S("s", [0, 0]), S("s", [2, 2]), S("h", [0, 0]),
                        S("h", [2, 2]), strict=True)
    assert r["verdict"] == "INCONCLUSIVE"


def test_small_held_out_cannot_assert_overfit_or_genuine():
    sb, sc = S("s", [0] * 10), S("s", [2] * 10)
    r = gap_mod.analyze(sb, sc, S("h", [1] * 3), S("h", [1] * 3))
    assert r["verdict_ci"] == "INCONCLUSIVE"
    # a drop on a held-out with enough tasks is still a REGRESSION
    r = gap_mod.analyze(S("s", [2] * 4), S("s", [2] * 4),
                        S("h", [2] * 8), S("h", [0] * 8))
    assert r["verdict"] == "REGRESSION"


def test_effect_equal_to_placebo_floor_is_a_tie():
    """Placebo stored 0.333 (rounded); the identical effect 1/3 counted as
    above the floor (GENUINE, significant) and the identical gap as
    significant with p=0.5."""
    sb, sp = S("s", [0] * 30), S("s", [1] * 10 + [0] * 20)
    hb, hp = S("h", [0] * 30), S("h", [1] * 10 + [0] * 20)
    pr = gap_mod.analyze(sb, sp, hb, hp, n_boot=200)
    cal = placebo_mod.stats([{"placebo": True, "suite_delta": pr["suite_delta"],
                              "held_delta": pr["held_delta"], "gap": pr["gap"]}])
    r = gap_mod.analyze(sb, sp, hb, hp, calibration=cal)
    assert r["verdict_ci"] != "GENUINE"
    assert r["significant"] is False
    hc = dict(hb)
    g = gap_mod.analyze(sb, sp, hb, hc, n_boot=200)
    cal2 = placebo_mod.stats([{"placebo": True, "suite_delta": 0.0,
                               "held_delta": 0.0, "gap": g["gap"]}])
    r2 = gap_mod.analyze(sb, sp, hb, hc, calibration=cal2)
    assert r2["gap_significant"] is False
    assert r2["p_gap"] == 1.0


def test_coverage_counts_tasks_missing_from_both_files():
    sb, sc = S("s", [0] * 10), S("s", [2] * 10)
    hb, hc = S("h", [0] * 10), S("h", [2] * 10)
    expected = ["h%d" % i for i in range(40)]
    r = gap_mod.analyze(sb, sc, hb, hc, held_ids=expected)
    assert r["stats"]["coverage"]["held"] == 0.25
    assert r["verdict"] == "INCONCLUSIVE"
    # and the point decision obeys the same gate
    r = gap_mod.analyze(sb, sc, hb, hc, held_ids=expected, decision="point")
    assert r["verdict"] == "INCONCLUSIVE"
    assert r["verdict_point"] == "GENUINE"
    assert r["coverage_ok"] is False


def test_point_decision_obeys_coverage_without_manifest():
    sb, sc = S("s", [0] * 10), S("s", [2] * 10)
    hb = S("h", [0] * 10)
    hc = {t: 2 for t in list(hb)[:4]}
    r = gap_mod.analyze(sb, sc, hb, hc, decision="point")
    assert r["verdict"] == "INCONCLUSIVE"


def test_n_boot_is_validated():
    sb, sc = S("s", [0] * 10), S("s", [2] * 10)
    for bad in (0, -3, 1, 199):
        with pytest.raises(ValueError):
            st.bootstrap(sb, sc, sb, sc, n_boot=bad)
        with pytest.raises(ValueError):
            gap_mod.analyze(sb, sc, sb, sc, n_boot=bad)


def test_gap_cli_rejects_tiny_boot(tmp_path):
    paths = [write(tmp_path / n, S("t", [1] * 8)) for n in "abcd"]
    with pytest.raises(SystemExit) as e:
        cli.main(["gap", "--suite-base", str(paths[0]), "--suite-cand",
                  str(paths[1]), "--held-base", str(paths[2]), "--held-cand",
                  str(paths[3]), "--out", str(tmp_path / "r.md"), "--boot", "0"])
    assert e.value.code == 2


def test_old_calibration_floor_falls_back_to_gap():
    cal = {"n_placebo": 3, "max_abs_gap": 0.5, "gaps": [0.5, -0.2, 0.1]}
    floor = st.noise_floor(cal)
    assert floor["suite"] == 0.5 and floor["held"] == 0.5
    assert st.noise_floor({"n_placebo": 2, "max_abs_gap": None})["suite"] == 0.0


def test_family_regressions_do_not_depend_on_key_order():
    base = {"t%02d" % i: 2 for i in range(8)}
    cand = {"t%02d" % i: (0 if i < 3 else 2) for i in range(8)}
    fam = {t: "f" for t in base}
    a = st.family_regressions(base, cand, fam, n_boot=500)
    rev = dict(reversed(list(base.items())))
    b = st.family_regressions(rev, cand, fam, n_boot=500)
    assert a == b


def test_verdict_uses_unrounded_transfer():
    """held captured 74.994% of the suite gain: t75 = -5e-5 rounds to -0.0;
    the CI verdict said GENUINE while the point rule said PARTIAL."""
    ns, nh, b, a = 20, 251, 17, 160
    sb = S("s", [0] * ns)
    sc = S("s", [1 if i < b else 0 for i in range(ns)])
    hb = S("h", [0] * nh)
    hc = S("h", [1 if i < a else 0 for i in range(nh)])
    r = gap_mod.analyze(sb, sc, hb, hc, n_boot=300)
    assert r["verdict_point"] == "PARTIAL"
    assert r["verdict_ci"] == "PARTIAL"
    assert r["stats"]["bootstrap"]["exact"]["point"]["t75"] < 0


def test_n_held_needed_is_a_true_ceiling():
    boot = {"n_held": 10, "n_suite": 40, "se": {"held_delta": 0.1304},
            "point": {"suite_delta": 0.1}}
    pw = st.power(boot)
    # 10 * (2.8 * 0.1304 / 0.075) ** 2 = 237.0002...
    assert pw["n_held_needed"] == 238


def test_genuine_reason_has_no_doubled_percent():
    sb, sc = S("s", [0] * 40), S("s", [2] * 40)
    r = gap_mod.analyze(sb, sc, S("h", [0] * 40), S("h", [2] * 40))
    assert r["verdict"] == "GENUINE"
    assert not any("%%" in w for w in r["stats"]["why"])


def test_gap_cli_errors_are_clean(tmp_path, capsys):
    ok = write(tmp_path / "s.json", S("t", [1] * 8))
    base = ["gap", "--suite-base", str(ok), "--suite-cand", str(ok),
            "--held-base", str(ok), "--held-cand", str(ok)]
    out = str(tmp_path / "r.md")
    cases = [
        ["--calibration", str(tmp_path / "missing.json")],
        ["--categories", str(write(tmp_path / "list.json", [1, 2]))],
        ["--calibration", str(write(tmp_path / "bad.json", "{nope"))],
        ["--prereg", str(write(tmp_path / "p.json", {"change": "x"}))],
    ]
    for extra in cases:
        assert cli.main(base + ["--out", out] + extra) == 1, extra
        assert "Traceback" not in capsys.readouterr().err
    # a score path that is a directory
    rc = cli.main(["gap", "--suite-base", str(tmp_path), "--suite-cand", str(ok),
                   "--held-base", str(ok), "--held-cand", str(ok), "--out", out])
    assert rc == 1
    # --out in a directory that does not exist yet is created
    assert cli.main(base + ["--out", str(tmp_path / "new" / "r.md")]) == 0


def test_gap_cli_manifest_sets_coverage_and_rejects_foreign_ids(tmp_path):
    m = {"tasks": [{"id": "h%d" % i, "family": "f", "file": "heldout-%d.md" % i}
                   for i in range(20)]}
    man = write(tmp_path / "manifest.json", m)
    s = write(tmp_path / "s.json", S("s", [0] * 10))
    s2 = write(tmp_path / "s2.json", S("s", [2] * 10))
    h = write(tmp_path / "h.json", S("h", [0] * 10))
    h2 = write(tmp_path / "h2.json", S("h", [2] * 10))
    out = tmp_path / "r.md"
    args = ["gap", "--suite-base", str(s), "--suite-cand", str(s2),
            "--held-base", str(h), "--held-cand", str(h2), "--out", str(out),
            "--manifest", str(man)]
    assert cli.main(args) == 0
    assert "Verdict: **INCONCLUSIVE**" in out.read_text()
    foreign = write(tmp_path / "f.json", {"zz": 2, **S("h", [2] * 10)})
    args[8] = str(foreign)
    assert cli.main(args) == 1


# --- scores / manifest ---------------------------------------------------

def test_duplicate_score_ids_are_rejected(tmp_path):
    p = write(tmp_path / "dup.json", '{"t0": 0, "t0": 2}')
    with pytest.raises(ValueError, match="duplicate"):
        scores_mod.load_scores(p)


def test_manifest_rejects_traversal_and_duplicates():
    good = {"id": "a", "family": "f", "file": "heldout-a.md"}
    manifest_mod.validate({"tasks": [good]})
    for bad_file in ("../x.md", "/etc/passwd", "sub/x.md", "..", ".hidden.md",
                     "a\\b.md"):
        with pytest.raises(ValueError):
            manifest_mod.validate({"tasks": [dict(good, file=bad_file)]})
    with pytest.raises(ValueError, match="duplicate task id"):
        manifest_mod.validate({"tasks": [good, dict(good, file="other.md")]})
    with pytest.raises(ValueError):
        manifest_mod.validate({"tasks": [{"id": "a", "file": "x.md"}]})


# --- sanity / agree / placebo --------------------------------------------

def test_sanity_needs_enough_shared_tasks():
    r = sanity_mod.check({"t0": 2}, {"t0": 0})
    assert r["ok"] is False
    good = S("t", [2] * 40)
    bad = {"t0": 0, "t1": 0}
    bad.update({"x%d" % i: 0 for i in range(38)})
    assert sanity_mod.check(good, bad)["ok"] is False
    assert sanity_mod.check(S("t", [2] * 10), S("t", [0] * 10))["ok"] is True


def test_sanity_and_agree_cli_errors_are_clean(tmp_path, capsys):
    s = write(tmp_path / "s.json", S("t", [1] * 8))
    lst = write(tmp_path / "m.json", [1, 2])
    rc = cli.main(["sanity", "--good", str(s), "--bad", str(s),
                   "--manifest", str(lst)])
    assert rc == 1
    rc = cli.main(["agree", "--a", str(tmp_path / "missing.json"), "--b", str(s)])
    assert rc == 1
    rc = cli.main(["agree", "--a", str(tmp_path), "--b", str(s)])
    assert rc == 1
    assert "Traceback" not in capsys.readouterr().err


def test_placebo_ignores_malformed_entries():
    entries = [
        {"placebo": True, "gap": True},
        {"placebo": True, "gap": float("nan")},
        {"placebo": "false", "gap": 0.9},
        {"placebo": True, "gap": 0.2, "suite_delta": True, "held_delta": 0.1},
    ]
    r = placebo_mod.stats(entries)
    assert r["n_placebo"] == 1
    assert r["max_abs_gap"] == 0.2
    assert r["max_abs_suite_delta"] is None
    assert ledger_mod.report(entries)["n_placebo"] == 3


def test_placebo_and_ledger_cli_missing_file(tmp_path, capsys):
    missing = str(tmp_path / "none.jsonl")
    assert cli.main(["placebo", "--ledger", missing]) == 1
    assert cli.main(["ledger", "--ledger", missing]) == 1
    assert "Traceback" not in capsys.readouterr().err


# --- ledger --------------------------------------------------------------

def test_torn_ledger_line_does_not_swallow_next_entry(tmp_path):
    p = tmp_path / "ledger.jsonl"
    ledger_mod.append(p, {"change": "c1"})
    with open(p, "ab") as f:
        f.write(b'{"change": "c2", "ga')  # writer died mid-line
    ledger_mod.append(p, {"change": "c3"})
    assert [e["change"] for e in ledger_mod.load(p)] == ["c1", "c3"]


def test_ledger_survives_utf8_cut_mid_character(tmp_path):
    p = tmp_path / "ledger.jsonl"
    ledger_mod.append(p, {"change": "ação"})
    data = json.dumps({"change": "ação 2"}, ensure_ascii=False).encode("utf-8")
    cut = data[:data.index("ç".encode("utf-8")) + 1]  # half of a 2-byte char
    with open(p, "ab") as f:
        f.write(cut)
    ledger_mod.append(p, {"change": "c3"})
    assert [e["change"] for e in ledger_mod.load(p)] == ["ação", "c3"]
    with pytest.raises(ValueError):
        ledger_mod.load(p, strict=True)


# --- prereg --------------------------------------------------------------

def test_prereg_load_rejects_malformed_files(tmp_path):
    for bad in ({"change": "x"}, {"expect": "up"}, [1],
                {"expect": {"suite": "up", "held": "up", "min_ratio": 2}},
                {"expect": {"suite": "up", "held": "up", "min_ratio": 0.5},
                 "seeds": [1, 1]}):
        with pytest.raises(ValueError):
            prereg_mod.load(write(tmp_path / "p.json", bad))


def test_prereg_cli_validates_ratio_and_tolerance(tmp_path, capsys):
    out = str(tmp_path / "p.json")
    assert cli.main(["prereg", "--change", "x", "--min-ratio", "2",
                     "--out", out]) == 1
    assert cli.main(["prereg", "--change", "x", "--flat-tol", "-1",
                     "--out", out]) == 1
    assert cli.main(["prereg", "--change", "x", "--seeds", "1,x",
                     "--out", out]) == 1
    assert "Traceback" not in capsys.readouterr().err
    assert cli.main(["prereg", "--change", "x", "--seeds", "43,42",
                     "--out", out]) == 0
    assert prereg_mod.load(out)["seeds"] == [42, 43]


def _bound_manifest(path, prereg_path, salt):
    write(path, {"tasks": [], "seed": prereg_mod.derive_seed(prereg_path, salt),
                 "seed_salt": salt,
                 "prereg_sha256": prereg_mod.sha256_file(prereg_path)})
    return path


def test_prereg_seed_commitment_blocks_seed_shopping(tmp_path):
    pre = write(tmp_path / "pre.json", prereg_mod.create("c", seeds=[1, 2]))
    m1 = _bound_manifest(tmp_path / "m1.json", pre, 1)
    m2 = _bound_manifest(tmp_path / "m2.json", pre, 2)
    m57 = _bound_manifest(tmp_path / "m57.json", pre, 57)
    assert prereg_mod.verify(pre, [m1, m2])[0] is True
    ok, why = prereg_mod.verify(pre, [m57])
    assert not ok and "not declared" in why
    ok, why = prereg_mod.verify(pre, [m1])
    assert not ok and "not measured" in why
    free = write(tmp_path / "free.json", prereg_mod.create("c"))
    assert prereg_mod.verify(free, [_bound_manifest(tmp_path / "f.json",
                                                    free, 57)])[0] is True


# --- pipeline ------------------------------------------------------------

def _catalog(tmp):
    catalog = Path(tmp) / "catalog"
    catalog.mkdir()
    (catalog / "a.md").write_text("# A\n\n### Setup\nCreate a file.\n\n"
                                  "### Instruction\nWrite hi.\n")
    (catalog / "b.md").write_text("# B\n\n### Setup\nOpen the editor.\n\n"
                                  "### Instruction\nWrite bye.\n")
    return catalog


def _ids(set_dir):
    return [t["id"] for t in manifest_mod.load(set_dir)["tasks"]]


def _scores(scores_dir, held):
    suite = ["s%d" % i for i in range(8)]
    write(scores_dir / "suite-base.json", {t: 1 for t in suite})
    write(scores_dir / "suite-cand.json", {t: 2 for t in suite})
    for name, data in held.items():
        write(scores_dir / (name + ".json"), data)


def test_pipeline_refuses_mixed_and_renamed_bound_sets(tmp_path, capsys):
    catalog = _catalog(tmp_path)
    pre = write(tmp_path / "pre.json", prereg_mod.create("c"))
    held = tmp_path / "held"
    assert cli.main(["generate", "--seeds", "1,2", "--n", "8", "--prereg",
                     str(pre), "--out", str(held), "--catalog",
                     str(catalog)]) == 0
    assert cli.main(["generate", "--seed", "3", "--n", "8", "--out",
                     str(held / "seed-3"), "--catalog", str(catalog),
                     "--exclude", str(held / "seed-1"),
                     "--exclude", str(held / "seed-2")]) == 0
    sc = tmp_path / "scores"
    hs = {}
    for name in ("1", "2", "3"):
        ids = _ids(held / ("seed-" + name))
        hs["held-base-" + name] = {t: 1 for t in ids}
        hs["held-cand-" + name] = {t: 2 for t in ids}
    _scores(sc, hs)
    base = ["pipeline", "--change", "c", "--prereg", str(pre), "--heldout-dir",
            str(held), "--scores-dir", str(sc), "--out", str(tmp_path / "o")]
    assert cli.main(base + ["--seeds", "1,2,3"]) == 1
    assert "mixed sets" in capsys.readouterr().err
    assert cli.main(base + ["--seeds", "1,2"]) == 0
    os.rename(held / "seed-2", held / "seed-9")
    for kind in ("base", "cand"):
        os.rename(sc / ("held-%s-2.json" % kind), sc / ("held-%s-9.json" % kind))
    assert cli.main(base + ["--seeds", "1,9"]) == 1
    assert "renamed directory" in capsys.readouterr().err


def test_pipeline_bad_inputs_are_clean_errors(tmp_path, capsys):
    catalog = _catalog(tmp_path)
    held = tmp_path / "held"
    assert cli.main(["generate", "--seed", "5", "--n", "8", "--out", str(held),
                     "--catalog", str(catalog)]) == 0
    ids = _ids(held)
    sc = tmp_path / "scores"
    _scores(sc, {"held-base": {t: 1 for t in ids}, "held-cand": {}})
    base = ["pipeline", "--change", "c", "--heldout-dir", str(held),
            "--scores-dir", str(sc), "--out", str(tmp_path / "o")]
    assert cli.main(base) == 1  # held-cand {}: no paired tasks
    write(sc / "held-cand.json", {t: 2 for t in ids})
    assert cli.main(base + ["--calibration", str(tmp_path / "none.json")]) == 1
    assert cli.main(base + ["--prereg", str(tmp_path / "none.json")]) == 1
    assert cli.main(base + ["--suite-categories",
                            str(write(tmp_path / "l.json", [1]))]) == 1
    assert "Traceback" not in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(base + ["--wave1", "0"])
    # a manifest pointing outside the set directory is refused
    m = json.loads((held / "manifest.json").read_text())
    m["tasks"][0]["file"] = "../../etc/passwd"
    (held / "manifest.json").write_text(json.dumps(m))
    assert cli.main(base) == 1


def test_pipeline_coverage_uses_the_manifest(tmp_path):
    catalog = _catalog(tmp_path)
    held = tmp_path / "held"
    assert cli.main(["generate", "--seed", "6", "--n", "20", "--out", str(held),
                     "--catalog", str(catalog)]) == 0
    ids = _ids(held)[:8]  # 12 of 20 tasks never scored on either side
    sc = tmp_path / "scores"
    _scores(sc, {"held-base": {t: 1 for t in ids},
                 "held-cand": {t: 2 for t in ids}})
    out = tmp_path / "o"
    assert cli.main(["pipeline", "--change", "c", "--heldout-dir", str(held),
                     "--scores-dir", str(sc), "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "INCONCLUSIVE"
    assert summary["pooled"]["verdict_ci"] == "INCONCLUSIVE"
