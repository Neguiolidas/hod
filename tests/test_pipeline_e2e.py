#!/usr/bin/env python3
"""Pipeline ponta the ponta, synthetic: prereg -> generation multi-seed ->
scoreboards -> pipeline (genuine stable + match) -> wave 1 stop ->
prereg late recusado -> placebo with significance.

Usage: python3 tests/test_pipeline_and2and.py (code 0 = everything passes).
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hod import cli  # noqa: E402
from hod import ledger as ledger_mod  # noqa: E402
from hod import prereg as prereg_mod  # noqa: E402


def _catalog(tmp):
    catalog = Path(tmp) / "catalog"
    catalog.mkdir()
    (catalog / "a.md").write_text(
        "# Example task\n\n### Setup\nCreate a file.\n\n"
        "### Instruction\nEscreva oi.\n"
    )
    (catalog / "b.md").write_text(
        "# Another task\n\n### Setup\nOpen the editor.\n\n"
        "### Instruction\nEscreva tchau.\n"
    )
    return catalog


def _held_ids(held_dir, seed):
    manifest = json.loads(
        (Path(held_dir) / ("seed-%s" % seed) / "manifest.json").read_text()
    )
    return [t["id"] for t in manifest["tasks"]]


def _write_scores(scores_dir, suite_base, suite_cand, held):
    scores_dir = Path(scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    (scores_dir / "suite-base.json").write_text(json.dumps(suite_base))
    (scores_dir / "suite-cand.json").write_text(json.dumps(suite_cand))
    for name, data in held.items():
        (scores_dir / (name + ".json")).write_text(json.dumps(data))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        catalog = _catalog(tmp)
        held_dir = tmp / "held"
        # prereg Before of the generation (mtime forced for the passado).
        prereg_path = tmp / "prereg.json"
        prereg_path.write_text(
            json.dumps(prereg_mod.create("change e2e", "up", "up", 0.75))
        )
        old = time.time() - 60
        os.utime(prereg_path, (old, old))
        rc = cli.main([
            "generate", "--seeds", "7,8", "--n", "4",
            "--out", str(held_dir), "--catalog", str(catalog),
        ])
        assert rc == 0, "generate multi-seed failed"
        assert (held_dir / "seed-7" / "manifest.json").exists()
        assert (held_dir / "seed-8" / "manifest.json").exists()
        ids7 = _held_ids(held_dir, "7")
        ids8 = _held_ids(held_dir, "8")
        scores = tmp / "scores"
        _write_scores(
            scores,
            {"s0": 1, "s1": 1}, {"s0": 2, "s1": 2},
            {
                "held-base-7": {t: 1 for t in ids7},
                "held-cand-7": {t: 2 for t in ids7},
                "held-base-8": {t: 1 for t in ids8},
                "held-cand-8": {t: 2 for t in ids8},
            },
        )
        ledger_path = tmp / "ledger.jsonl"
        out = tmp / "pipe"
        rc = cli.main([
            "pipeline", "--change", "change e2e",
            "--prereg", str(prereg_path),
            "--heldout-dir", str(held_dir),
            "--scores-dir", str(scores),
            "--seeds", "7,8",
            "--ledger", str(ledger_path),
            "--out", str(out),
        ])
        assert rc == 0, "pipeline failed"
        summary = json.loads((out / "summary.json").read_text())
        assert summary["verdict"] == "GENUINE", summary
        assert summary["stable"] is True
        assert summary["hypothesis"]["match"] is True
        assert (out / "report.md").exists()
        entries = ledger_mod.load(ledger_path)
        assert len(entries) == 1 and entries[0]["verdict"] == "GENUINE"
        print("ok: stable genuine pipeline + hypothesis match + ledger")

        # wave 1 stop: primeiros 2 of the seed 7 without improvement in the held-out.
        wave_scores = tmp / "scores-wave"
        shutil.copytree(scores, wave_scores)
        wave_ids = sorted(ids7)[:2]
        cand7 = {t: 2 for t in ids7}
        for t in wave_ids:
            cand7[t] = 1
        (wave_scores / "held-cand-7.json").write_text(json.dumps(cand7))
        out_wave = tmp / "pipe-wave"
        rc = cli.main([
            "pipeline", "--change", "change wave",
            "--prereg", str(prereg_path),
            "--heldout-dir", str(held_dir),
            "--scores-dir", str(wave_scores),
            "--seeds", "7,8", "--wave1", "2",
            "--out", str(out_wave),
        ])
        assert rc == 3, "wave 1 should stop with code 3, got %d" % rc
        wave_summary = json.loads((out_wave / "summary.json").read_text())
        assert wave_summary["wave_stopped"] is True
        print("ok: wave 1 stopped (OVERFIT, without expanding)")

        # prereg late recusado.
        held2 = tmp / "held2"
        rc = cli.main([
            "generate", "--seed", "9", "--n", "2",
            "--out", str(held2), "--catalog", str(catalog),
        ])
        assert rc == 0
        late = tmp / "prereg-tardio.json"
        late.write_text(
            json.dumps(prereg_mod.create("tardia", "up", "up", 0.75))
        )
        scores2 = tmp / "scores2"
        manifest = json.loads((held2 / "manifest.json").read_text())
        ids9 = [t["id"] for t in manifest["tasks"]]
        _write_scores(
            scores2,
            {"s0": 1}, {"s0": 2},
            {"held-base": {t: 1 for t in ids9},
             "held-cand": {t: 2 for t in ids9}},
        )
        rc = cli.main([
            "pipeline", "--change", "tardia",
            "--prereg", str(late),
            "--heldout-dir", str(held2),
            "--scores-dir", str(scores2),
            "--out", str(tmp / "pipe-tardia"),
        ])
        assert rc == 1, "late prereg should fail with code 1"
        print("ok: prereg after generation refused")

        # placebo + significance.
        ledger_mod.append(ledger_path, {
            "change": "placebo", "placebo": True, "gap": 0.25,
        })
        rc = cli.main([
            "placebo", "--ledger", str(ledger_path),
            "--out", str(tmp / "calibration.json"),
        ])
        assert rc == 0
        out_cal = tmp / "pipe-cal"
        rc = cli.main([
            "pipeline", "--change", "change e2e",
            "--prereg", str(prereg_path),
            "--heldout-dir", str(held_dir),
            "--scores-dir", str(scores),
            "--seeds", "7,8",
            "--calibration", str(tmp / "calibration.json"),
            "--out", str(out_cal),
        ])
        assert rc == 0
        cal_summary = json.loads((out_cal / "summary.json").read_text())
        assert cal_summary["significant"] is True
        print("ok: placebo calibrates significance")
    print("pipeline e2e: all pass")
    return 0


def test_main():
    """Wrapper for the pytest coletar the test in format of script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
