#!/usr/bin/env python3
"""Tests of the camadas of robustness (fase 5): prereg, calibration, categories,
multi-seed, placebo, ledger, blind and agreement. Everything synthetic.

Usage: python3 tests/test_robustness.py (code 0 = everything passes).
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hod import agree as agree_mod  # noqa: E402
from hod import blind as blind_mod  # noqa: E402
from hod import calibrate as calibrate_mod  # noqa: E402
from hod import gap as gap_mod  # noqa: E402
from hod import ledger as ledger_mod  # noqa: E402
from hod import placebo as placebo_mod  # noqa: E402
from hod import prereg as prereg_mod  # noqa: E402


def test_prereg():
    pre = prereg_mod.create("change X", "up", "up", 0.75)
    ok = prereg_mod.check(pre, {"suite_delta": 1.0, "held_delta": 1.0})
    assert ok["match"] and ok["ratio"] == 1.0, ok
    bad = prereg_mod.check(pre, {"suite_delta": 1.0, "held_delta": 0.5})
    assert not bad["match"] and bad["ratio"] == 0.5, bad
    flat = prereg_mod.check(
        prereg_mod.create("Y", "up", "flat", 0.75),
        {"suite_delta": 1.0, "held_delta": 0.0},
    )
    assert flat["match"], flat
    print("ok: prereg match/mismatch")


def _heldout_dir(tmp, tasks):
    held = Path(tmp) / "held"
    held.mkdir()
    manifest_tasks = []
    for task_id, family, sim in tasks:
        (held / (task_id + ".md")).write_text(
            "# ID: %s\nCategory: instruction. Target time: up to 10 min.\n"
            % task_id
        )
        manifest_tasks.append(
            {
                "id": task_id,
                "family": family,
                "file": task_id + ".md",
                "sha256": "x",
                "max_similarity": sim,
            }
        )
    (held / "manifest.json").write_text(
        json.dumps({"seed": 7, "threshold": 0.30, "tasks": manifest_tasks})
    )
    return held


def test_calibrate():
    with tempfile.TemporaryDirectory() as tmp:
        held = _heldout_dir(
            tmp,
            [("heldout-a-00", "fam-a", 0.1),
             ("heldout-b-01", "fam-b", 0.2),
             ("heldout-c-02", "fam-c", 0.05)],
        )
        result = calibrate_mod.calibrate(
            held, {"heldout-a-00": 2, "heldout-b-01": 1, "heldout-c-02": 0}
        )
        by_id = {t["id"]: t for t in result["tasks"]}
        assert by_id["heldout-a-00"]["difficulty"] == 0.0
        assert by_id["heldout-b-01"]["difficulty"] == 0.5
        assert by_id["heldout-c-02"]["difficulty"] == 1.0
        assert by_id["heldout-c-02"]["q"] == 0.95
        assert len(result["kept"]) == 2 and len(result["dropped"]) == 1
        assert not result["all_saturated"]
        out = Path(tmp) / "filtrado"
        n = calibrate_mod.write_filtered(held, out, result)
        assert n == 2 and (out / "manifest.json").exists()
        sat = calibrate_mod.calibrate(
            held, {"heldout-a-00": 2, "heldout-b-01": 2, "heldout-c-02": 2}
        )
        assert sat["all_saturated"] and not sat["kept"]
    print("ok: calibrate difficulty, Q, filter and NO SIGNAL")


def test_categories_and_multi():
    cats = {"s0": "catA", "s1": "catB", "h0": "catA", "h1": "catB"}
    r = gap_mod.analyze(
        {"s0": 1, "s1": 1}, {"s0": 2, "s1": 1},
        {"h0": 1, "h1": 1}, {"h0": 2, "h1": 1},
        categories=cats, decision="point",
    )
    assert r["by_category"]["catA"]["verdict"] == "GENUINE"
    assert r["by_category"]["catB"]["verdict"] == "NO IMPROVEMENT"
    agg = gap_mod.analyze_multi([r, r])
    assert agg["stable"] and agg["verdict"] == "GENUINE"
    other = gap_mod.analyze(
        {"s0": 1}, {"s0": 2}, {"h0": 1}, {"h0": 1}, decision="point"
    )
    assert other["verdict"] == "OVERFIT"
    agg2 = gap_mod.analyze_multi([r, other])
    assert not agg2["stable"] and agg2["verdict"] == "UNSTABLE"
    print("ok: categories and stable/unstable multi-seed")


def test_placebo_ledger_agree():
    entries = [
        {"gap": 0.1, "placebo": True},
        {"gap": -0.2, "placebo": True},
        {"gap": 0.05, "placebo": True},
        {"gap": 1.0, "placebo": False},
    ]
    stats = placebo_mod.stats(entries)
    assert stats["n_placebo"] == 3 and stats["max_abs_gap"] == 0.2
    r = gap_mod.analyze(
        {"s0": 1}, {"s0": 2}, {"h0": 1}, {"h0": 1},
        calibration={"max_abs_gap": 0.2, "n_placebo": 3},
    )
    assert r["significant"] is True
    r2 = gap_mod.analyze(
        {"s0": 1, "s1": 1, "s2": 1, "s3": 1},
        {"s0": 1, "s1": 1, "s2": 1, "s3": 2},
        {"h0": 1}, {"h0": 1},
        calibration={"max_abs_gap": 0.5, "n_placebo": 3},
    )
    assert r2["significant"] is False
    with tempfile.TemporaryDirectory() as tmp:
        ledger_path = Path(tmp) / "ledger.jsonl"
        ledger_mod.append(ledger_path, {
            "change": "X", "placebo": False, "verdict": "OVERFIT",
            "by_family": {"fam-a": "OVERFIT", "fam-b": "GENUINE"},
        })
        ledger_mod.append(ledger_path, {
            "change": "Y", "placebo": False, "verdict": "GENUINE",
            "by_family": {"fam-a": "GENUINE", "fam-b": "GENUINE"},
        })
        rep = ledger_mod.report(ledger_mod.load(ledger_path))
        assert rep["n_real"] == 2 and rep["overfit_rate"] == 0.5
        # 1 + overfit / (vistas + PRIOR): 1 + 1/3
        assert rep["weights"]["fam-a"] == 1.33
        assert rep["weights"]["fam-b"] == 1.0
    a = agree_mod.agree({"t0": 1, "t1": 2}, {"t0": 1, "t1": 1})
    assert a["n"] == 2 and a["pct_equal"] == 0.5 and a["kappa"] == 0.0
    b = agree_mod.agree({"t0": 1, "t1": 2}, {"t0": 1, "t1": 2})
    assert b["kappa"] == 1.0
    print("ok: placebo, ledger/weights, agree/kappa, significance")


def test_blind():
    with tempfile.TemporaryDirectory() as tmp:
        held = _heldout_dir(
            tmp,
            [("heldout-a-00", "fam-a", 0.1),
             ("heldout-b-01", "fam-b", 0.1),
             ("heldout-c-02", "fam-c", 0.1)],
        )
        template, key = blind_mod.make_blind(held, seed=42)
        assert len(key) == 3 and set(template["tasks"]) == set(key)
        filled = {"tasks": {}, "behavior": {}}
        for anon in template["tasks"]:
            filled["tasks"][anon] = {"nota": 2}
        restored = blind_mod.restore(filled, key)
        assert set(restored["tasks"]) == {
            "heldout-a-00", "heldout-b-01", "heldout-c-02"
        }
        try:
            blind_mod.restore({"tasks": {"t99": {"nota": 1}}}, key)
            raise AssertionError("anon outside the key should fail")
        except ValueError:
            pass
    print("ok: blind/unblind roundtrip and invalid key")


def main():
    test_prereg()
    test_calibrate()
    test_categories_and_multi()
    test_placebo_ledger_agree()
    test_blind()
    print("robustness: all pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
