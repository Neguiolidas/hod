"""Regression tests for the generator / audit / calibrate / blind fixes.

Each test reproduces a defect found in the audit of the generator side
(fail-open paths, unsafe writes, crashes) and asserts the fixed behavior.
"""

import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hod import audit as audit_mod  # noqa: E402
from hod import blind as blind_mod  # noqa: E402
from hod import calibrate as calibrate_mod  # noqa: E402
from hod import generate as gen  # noqa: E402
from hod import scores as scores_mod  # noqa: E402
from hod.families import FAMILIES  # noqa: E402

FILLER = ("um texto de catálogo qualquer sobre roteamento de pedidos e logs "
          "de deploy que não tem relação nenhuma com o conjunto held-out ") * 3


def _run(fn, argv):
    """Runs a CLI entry point; returns (rc, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = fn(argv)
    return rc, out.getvalue(), err.getvalue()


def _catalog(tmp_path):
    cat = tmp_path / "cat"
    cat.mkdir(exist_ok=True)
    (cat / "t00.md").write_bytes(FILLER.encode("utf-8"))
    return cat


def _generate(tmp_path, out, seed=1, n=4, extra=()):
    argv = ["--seed", str(seed), "--n", str(n), "--out", str(out),
            "--catalog", str(_catalog(tmp_path))] + list(extra)
    return _run(gen.main, argv)


def _audit(tmp_path, held):
    return _run(audit_mod.main, ["--heldout", str(held), "--catalog",
                                 str(_catalog(tmp_path))])


def _snapshot(d):
    return {p.name: p.read_bytes() for p in sorted(Path(d).iterdir())
            if p.is_file()}


def _manifest(d):
    return json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8"))


def _hand_set(tmp_path, name="held", files=None):
    """A tiny calibratable set: {id: (family, file)}."""
    held = tmp_path / name
    held.mkdir()
    tasks = []
    for task_id, (family, filename) in (files or {
            "heldout-a-00": ("fam-a", "heldout-a-00.md"),
            "heldout-b-01": ("fam-b", "heldout-b-01.md")}).items():
        if "/" not in filename:
            (held / filename).write_bytes(b"# ID: %s\n" % task_id.encode())
        tasks.append({"id": task_id, "family": family, "file": filename,
                      "sha256": "x", "max_similarity": 0.1, "novelty": 0.9})
    (held / "manifest.json").write_text(json.dumps({"seed": 1, "threshold": 0.3,
                                                    "tasks": tasks}))
    return held


def _baseline(tmp_path, ids, score=1):
    p = tmp_path / "base.json"
    p.write_text(json.dumps({t: score for t in ids}))
    return p


def _no_tmp_dirs(parent):
    return not [p.name for p in Path(parent).iterdir() if ".hod-" in p.name]


# --- 1. calibrate --out aliasing the source ----------------------------------

def test_calibrate_out_alias_unseen_by_resolve_keeps_source(tmp_path,
                                                            monkeypatch):
    """A bind mount / case-insensitive alias passes the resolve() comparison;
    the old cleanup deleted the source before reading it."""
    held = _hand_set(tmp_path)
    before = _snapshot(held)
    alias = tmp_path / "alias"
    alias.symlink_to(held, target_is_directory=True)
    base = _baseline(tmp_path, ["heldout-a-00", "heldout-b-01"])
    # simulate an alias that resolve() cannot see (bind mount, case folding)
    monkeypatch.setattr(Path, "resolve",
                        lambda self, strict=False: self.absolute())
    try:
        rc, _, err = _run(calibrate_mod.main, [
            "--heldout", str(held), "--baseline", str(base),
            "--out", str(alias)])
    except Exception:  # the old code crashed after wiping the source
        rc, err = None, ""
    monkeypatch.undo()
    assert _snapshot(held) == before, "source set was modified"
    assert rc == 1 and "source set itself" in err


def test_calibrate_out_equal_source_is_clean_error_before_report(tmp_path):
    held = _hand_set(tmp_path)
    base = _baseline(tmp_path, ["heldout-a-00", "heldout-b-01"])
    report = tmp_path / "r.md"
    rc, _, err = _run(calibrate_mod.main, [
        "--heldout", str(held), "--baseline", str(base), "--out", str(held),
        "--report", str(report)])
    assert rc == 1 and err.startswith("calibrate: ")
    assert not report.exists(), "report written before the refusal"


# --- 2. --exclude fails closed ----------------------------------------------

def test_exclude_missing_or_empty_dir_is_an_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    for bad in (tmp_path / "typo", empty):
        rc, _, err = _generate(tmp_path, tmp_path / "o", extra=[
            "--exclude", str(bad)])
        assert rc == 1 and err.startswith("generate: "), err
        assert not (tmp_path / "o").exists()
        with pytest.raises(ValueError):
            gen.load_exclude([bad])


def test_exclude_accepts_set_and_batch_parent(tmp_path):
    assert _generate(tmp_path, tmp_path / "a")[0] == 0
    assert len(gen.load_exclude([tmp_path / "a"])) == 4
    rc = _run(gen.main, ["--seeds", "1,2", "--n", "3", "--out",
                         str(tmp_path / "b"), "--catalog",
                         str(_catalog(tmp_path))])[0]
    assert rc == 0 and len(gen.load_exclude([tmp_path / "b"])) == 6


# --- 3 / 14. audit loads the manifest through manifest.load -----------------

def test_audit_duplicate_id_cannot_hide_a_task(tmp_path):
    held = tmp_path / "held"
    assert _generate(tmp_path, held, n=2)[0] == 0
    m = _manifest(held)
    first, second = m["tasks"]
    # task 0 becomes a verbatim copy of the suite, hidden behind task 1's id
    data = FILLER.encode("utf-8")
    (held / first["file"]).write_bytes(data)
    first["sha256"] = hashlib.sha256(data).hexdigest()
    first["id"] = second["id"]
    (held / "manifest.json").write_text(json.dumps(m))
    rc, _, err = _audit(tmp_path, held)
    assert rc == 1 and "duplicate task id" in err


@pytest.mark.parametrize("breakage", ["bad-json", "no-family", "list",
                                      "traversal"])
def test_audit_bad_manifest_is_clean_failure(tmp_path, breakage):
    held = tmp_path / "held"
    assert _generate(tmp_path, held, n=2)[0] == 0
    m = _manifest(held)
    if breakage == "bad-json":
        text = "{"
    elif breakage == "list":
        text = json.dumps(m["tasks"])
    else:
        if breakage == "no-family":
            del m["tasks"][0]["family"]
        else:
            m["tasks"][0]["file"] = "../PWNED.md"
        text = json.dumps(m)
    (held / "manifest.json").write_text(text)
    rc, _, err = _audit(tmp_path, held)
    assert rc == 1 and err.startswith("FAIL: ")


def test_audit_missing_sha_is_failure_not_traceback(tmp_path):
    held = tmp_path / "held"
    assert _generate(tmp_path, held, n=2)[0] == 0
    m = _manifest(held)
    del m["tasks"][0]["sha256"]
    (held / "manifest.json").write_text(json.dumps(m))
    rc, _, err = _audit(tmp_path, held)
    assert rc == 1 and "no sha256" in err


# --- 4. path traversal through the manifest `file` --------------------------

def test_calibrate_refuses_traversal_in_manifest_file(tmp_path):
    (tmp_path / "PWNED.md").write_text("outside the set\n")
    held = _hand_set(tmp_path, files={
        "heldout-a-00": ("fam-a", "heldout-a-00.md"),
        "heldout-b-01": ("fam-b", "../PWNED.md")})
    base = _baseline(tmp_path, ["heldout-a-00", "heldout-b-01"])
    outs = tmp_path / "outs"
    outs.mkdir()
    rc, _, err = _run(calibrate_mod.main, [
        "--heldout", str(held), "--baseline", str(base),
        "--out", str(outs / "cal")])
    assert rc == 1 and "unsafe task file name" in err
    assert not (outs / "PWNED.md").exists(), "wrote outside --out"
    assert not (outs / "cal").exists()


def test_blind_validates_manifest(tmp_path):
    for name, files in (("trav", {"heldout-a-00": ("f", "../x.md")}),
                        ("dup", {"heldout-a-00": ("f", "heldout-a-00.md")})):
        held = _hand_set(tmp_path, name, files)
        if name == "dup":
            m = _manifest(held)
            m["tasks"].append(dict(m["tasks"][0], file="heldout-a-01.md"))
            (held / "manifest.json").write_text(json.dumps(m))
        rc, _, err = _run(blind_mod.main_blind, [
            "--heldout", str(held), "--template", str(tmp_path / "t.json"),
            "--key", str(tmp_path / "keys" / "k.json")])
        assert rc == 1 and err.startswith("blind: "), err
        assert not (tmp_path / "t.json").exists()


# --- 5. weighted quota ties ----------------------------------------------------

def test_equal_weight_quota_ties_are_seeded_not_alphabetical(tmp_path):
    weights = {f: 1.0 for f in FAMILIES}
    cat = [gen.shingles(gen.normalize(FILLER))]
    picks = []
    for seed in range(40):
        m = gen.generate_one(seed, 3, tmp_path / ("s%d" % seed), cat,
                             family_weights=weights, quiet=True)
        picks.append(frozenset(t["family"] for t in m["tasks"]))
    assert len(set(picks)) > 5, "the same families win every tie"
    assert set().union(*picks) == set(FAMILIES)
    again = gen.generate_one(7, 3, tmp_path / "again", cat,
                             family_weights=weights, quiet=True)
    assert frozenset(t["family"] for t in again["tasks"]) == picks[7]


# --- 6. --seeds batches -----------------------------------------------------

def _batch(tmp_path, out, seeds, n=3, extra=()):
    return _run(gen.main, ["--seeds", seeds, "--n", str(n), "--out", str(out),
                           "--catalog", str(_catalog(tmp_path))] + list(extra))


def test_batch_failing_part_way_writes_nothing(tmp_path):
    # seed 1 alone passes; seed 2, excluding seed 1's tasks, runs out
    out = tmp_path / "o"
    rc, _, err = _batch(tmp_path, out, "1,2", n=20,
                        extra=["--novelty", "0.15"])
    assert rc == 1 and "nothing was written" in err
    assert not (out / "seed-1").exists(), "earlier set left on disk"


def test_batch_write_failure_rolls_back_this_run(tmp_path, monkeypatch):
    out = tmp_path / "o"
    assert _batch(tmp_path, out, "1,2")[0] == 0
    before = {s: _snapshot(out / s) for s in ("seed-1", "seed-2")}
    real = gen.install_set
    calls = []

    def flaky(target, files):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("disk full (simulated)")
        return real(target, files)
    monkeypatch.setattr(gen, "install_set", flaky)
    rc, _, err = _batch(tmp_path, out, "1,2", n=5)
    assert rc == 1 and "rolled back" in err
    assert {s: _snapshot(out / s) for s in before} == before
    assert _no_tmp_dirs(out)


def test_batch_refuses_repeated_and_bad_seeds(tmp_path):
    for seeds in ("7,7", "1,x", "1,,2"):
        rc, _, err = _batch(tmp_path, tmp_path / "o", seeds)
        assert rc == 1 and err.startswith("generate: --seeds"), (seeds, err)
        assert not (tmp_path / "o").exists()


def test_batch_refuses_out_with_foreign_seed_sets(tmp_path):
    out = tmp_path / "o"
    assert _batch(tmp_path, out, "1,2")[0] == 0
    rc, _, err = _batch(tmp_path, out, "3,4")
    assert rc == 1 and "--exclude" in err and "seed-1" in err
    assert not (out / "seed-3").exists()
    # rerunning the same batch (same seed-N sets) is allowed
    assert _batch(tmp_path, out, "1,2")[0] == 0
    # a batch into a single set's directory is refused too
    assert _generate(tmp_path, tmp_path / "single")[0] == 0
    assert _batch(tmp_path, tmp_path / "single", "1")[0] == 1


# --- 7. fail-closed writes and cleanup ownership ------------------------------

def test_n_zero_or_negative_is_refused_without_touching_set(tmp_path):
    held = tmp_path / "held"
    assert _generate(tmp_path, held)[0] == 0
    before = _snapshot(held)
    for n in ("0", "-3"):
        with pytest.raises(SystemExit) as exc:
            _generate(tmp_path, held, n=n)
        assert exc.value.code == 2
    assert gen.generate_one(1, 0, held, [], quiet=True) is None
    assert _snapshot(held) == before


def test_write_failure_keeps_previous_set(tmp_path, monkeypatch):
    held = tmp_path / "held"
    assert _generate(tmp_path, held, seed=1)[0] == 0
    before = _snapshot(held)
    state = {"n": 0}
    real_bytes, real_text = Path.write_bytes, Path.write_text

    def counted(real):
        def write(self, *a, **kw):
            state["n"] += 1
            if state["n"] == 3:
                raise OSError("disk full (simulated)")
            return real(self, *a, **kw)
        return write
    monkeypatch.setattr(Path, "write_bytes", counted(real_bytes))
    monkeypatch.setattr(Path, "write_text", counted(real_text))
    cat = [gen.shingles(gen.normalize(FILLER))]
    try:
        result = gen.generate_one(2, 4, held, cat, quiet=True)
    except OSError:  # the old code crashed mid-write
        result = "crashed"
    monkeypatch.undo()
    assert result is None
    assert _snapshot(held) == before, "previous set wiped or partial"
    assert _no_tmp_dirs(tmp_path)


def test_out_ownership_refuses_foreign_directories(tmp_path):
    ext = tmp_path / "extension"
    ext.mkdir()
    (ext / "manifest.json").write_text(json.dumps(
        {"manifest_version": 3, "name": "my extension", "version": "1.0"}))
    (ext / "background.js").write_text("// code\n")
    project = tmp_path / "project"
    project.mkdir()
    (project / "notes.txt").write_text("keep me\n")
    a_file = tmp_path / "a-file"
    a_file.write_text("x")
    for target in (ext, project, a_file):
        before = _snapshot(target) if target.is_dir() else target.read_bytes()
        rc, _, err = _generate(tmp_path, target)
        assert rc == 1 and err.startswith("generate: "), err
        after = _snapshot(target) if target.is_dir() else target.read_bytes()
        assert after == before
    # calibrate --out follows the same rule
    held = _hand_set(tmp_path)
    base = _baseline(tmp_path, ["heldout-a-00", "heldout-b-01"])
    rc, _, err = _run(calibrate_mod.main, [
        "--heldout", str(held), "--baseline", str(base), "--out", str(ext)])
    assert rc == 1 and (ext / "background.js").exists()


def test_existing_set_is_replaced_atomically(tmp_path):
    held = tmp_path / "held"
    assert _generate(tmp_path, held, seed=1, n=6)[0] == 0
    (held / ".DS_Store").write_bytes(b"\0")
    assert _generate(tmp_path, held, seed=2, n=3)[0] == 0
    names = sorted(p.name for p in held.iterdir())
    assert names == sorted(["manifest.json"] +
                           [t["file"] for t in _manifest(held)["tasks"]])
    assert _manifest(held)["seed"] == 2 and _no_tmp_dirs(tmp_path)
    assert _audit(tmp_path, held)[0] == 0
    # a set with foreign files in it is not replaced (nor are they deleted)
    (held / "notes.txt").write_text("mine\n")
    rc, _, err = _generate(tmp_path, held, seed=3)
    assert rc == 1 and "not part of its HOD set" in err
    assert (held / "notes.txt").exists() and _manifest(held)["seed"] == 2


def test_replace_in_place_when_out_cannot_be_moved(tmp_path, monkeypatch):
    """--out as a mount point: renaming it fails, files are swapped inside."""
    held = tmp_path / "held"
    assert _generate(tmp_path, held, seed=1, n=6)[0] == 0
    real = os.rename

    def no_move_out(src, dst, *a, **kw):
        if Path(src) == held:
            raise OSError(16, "Device or resource busy (simulated)")
        return real(src, dst, *a, **kw)
    monkeypatch.setattr(os, "rename", no_move_out)
    assert _generate(tmp_path, held, seed=2, n=3)[0] == 0
    monkeypatch.undo()
    assert _manifest(held)["seed"] == 2 and len(list(held.iterdir())) == 4
    assert _no_tmp_dirs(tmp_path) and _audit(tmp_path, held)[0] == 0


def test_symlinked_out_keeps_the_link(tmp_path):
    real = tmp_path / "real"
    assert _generate(tmp_path, real, seed=1)[0] == 0
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert _generate(tmp_path, link, seed=2)[0] == 0
    assert link.is_symlink() and _manifest(real)["seed"] == 2


def test_generate_and_audit_under_ascii_locale(tmp_path):
    """read_text()/write_text() without encoding crash under a non-UTF-8
    locale (catalog and tasks have non-ASCII letters)."""
    cat = _catalog(tmp_path)
    env = dict(os.environ, PYTHONPATH=str(ROOT), LC_ALL="C", LANG="C",
               PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
    held = tmp_path / "held"
    for args in (["generate", "--seed", "3", "--n", "8", "--out", str(held)],
                 ["audit", "--heldout", str(held)]):
        proc = subprocess.run([sys.executable, "-m", "hod"] + args +
                              ["--catalog", str(cat)], capture_output=True,
                              text=True, env=env, timeout=120)
        assert proc.returncode == 0, proc.stderr[-500:]


# --- 8. unblind --allow-incomplete ---------------------------------------------

def test_unblind_incomplete_validates_and_drops_unscored(tmp_path):
    key = {"t01": "real-a", "t02": "real-b", "t03": "real-c"}
    with pytest.raises(ValueError):
        blind_mod.restore({"tasks": {"t01": {"nota": 7}}}, key, False)
    with pytest.raises(ValueError):
        blind_mod.restore({"tasks": {"t01": {"nota": True}}}, key, False)
    (tmp_path / "k.json").write_text(json.dumps(key))
    (tmp_path / "s.json").write_text(json.dumps({"tasks": {
        "t01": {"nota": 2}, "t02": {"nota": None}}, "behavior": {}}))
    rc, _, err = _run(blind_mod.main_unblind, [
        "--scores", str(tmp_path / "s.json"), "--key", str(tmp_path / "k.json"),
        "--out", str(tmp_path / "out.json"), "--allow-incomplete"])
    assert rc == 0 and "2 of 3 tasks unscored" in err
    assert scores_mod.load_scores(tmp_path / "out.json") == {"real-a": 2}


@pytest.mark.parametrize("content", ["{", "[1, 2]", '{"tasks": [1]}'])
def test_unblind_bad_scores_file_is_clean_error(tmp_path, content):
    (tmp_path / "k.json").write_text(json.dumps({"t01": "a"}))
    (tmp_path / "s.json").write_text(content)
    rc, _, err = _run(blind_mod.main_unblind, [
        "--scores", str(tmp_path / "s.json"), "--key", str(tmp_path / "k.json"),
        "--out", str(tmp_path / "out.json")])
    assert rc == 1 and err.startswith("unblind: ")


# --- 9. blind key placement --------------------------------------------------

def test_blind_refuses_key_in_template_or_set(tmp_path):
    held = _hand_set(tmp_path)
    same = tmp_path / "same.json"
    for template, key in ((same, same), (same, held / "key.json"),
                          (same, tmp_path / "held" / "sub" / "key.json")):
        rc, _, err = _run(blind_mod.main_blind, [
            "--heldout", str(held), "--template", str(template),
            "--key", str(key)])
        assert rc == 1 and err.startswith("blind: "), err
        assert not same.exists() and not (held / "key.json").exists()
    rc, _, _ = _run(blind_mod.main_blind, [
        "--heldout", str(held), "--template", str(tmp_path / "t.json"),
        "--key", str(tmp_path / "keys" / "k.json")])
    assert rc == 0


# --- 10. calibrate keeps the generator's novelty ---------------------------------

def test_calibrate_keeps_generator_novelty(tmp_path):
    held = tmp_path / "held"
    assert _generate(tmp_path, held, n=4)[0] == 0
    src = {t["id"]: t for t in _manifest(held)["tasks"]}
    base = _baseline(tmp_path, list(src))
    rc, _, _ = _run(calibrate_mod.main, [
        "--heldout", str(held), "--baseline", str(base),
        "--out", str(tmp_path / "cal")])
    assert rc == 0
    for t in _manifest(tmp_path / "cal")["tasks"]:
        assert t["novelty"] == src[t["id"]]["novelty"]
        assert t["q_novelty"] == round(1 - src[t["id"]]["max_similarity"], 3)
    assert _audit(tmp_path, tmp_path / "cal")[0] == 0


# --- 11. sha256 over raw bytes ---------------------------------------------------

def test_audit_hashes_raw_bytes(tmp_path):
    held = tmp_path / "held"
    assert _generate(tmp_path, held, n=3)[0] == 0
    for t in _manifest(held)["tasks"]:
        data = (held / t["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == t["sha256"]
    first = held / _manifest(held)["tasks"][0]["file"]
    first.write_bytes(first.read_bytes().replace(b"\n", b"\r\n"))
    rc, _, err = _audit(tmp_path, held)
    assert rc == 1 and "hash differs" in err


# --- 12. normalize ---------------------------------------------------------------

def test_normalize_keeps_casefold_results_whole():
    assert gen.normalize("İSTANBUL") == gen.normalize("i\u0307stanbul")
    assert len(gen.normalize("İSTANBUL limanı")) == 2
    assert gen.normalize("हिन्दी भाषा") == ["हिन्दी", "भाषा"]
    text = "Ação À NOITE pingüim x_y 42"
    assert gen.normalize(unicodedata.normalize("NFD", text)) == \
        gen.normalize(text) == ["ação", "à", "noite", "pingüim", "x", "y", "42"]


# --- 13. weights and frontier validation ------------------------------------------

@pytest.mark.parametrize("payload", [
    '{"ambiguity": Infinity}', '{"ambiguity": NaN}', '{"ambiguity": 1e308}',
    '{"ambiguity": -1}', '{"ambiguity": true}', '{"memory": 2.0}', "[1]", "{",
])
def test_family_weights_reject_bad_values(tmp_path, payload):
    w = tmp_path / "w.json"
    w.write_text(payload)
    rc, _, err = _generate(tmp_path, tmp_path / "o", extra=[
        "--family-weights", str(w)])
    assert rc == 1 and err.startswith("generate: "), err
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("payload", [
    '{"memory": 0.5}', '{"ambiguity": NaN}', '{"ambiguity": 1.5}',
    '{"ambiguity": "high"}', "[0.5]", "{",
])
def test_frontier_rejects_bad_input(tmp_path, payload):
    f = tmp_path / "f.json"
    f.write_text(payload)
    rc, _, err = _generate(tmp_path, tmp_path / "o", extra=[
        "--frontier", str(f)])
    assert rc == 1 and err.startswith("generate: "), err


def test_frontier_with_valid_rates_still_works(tmp_path):
    f = tmp_path / "f.json"
    f.write_text(json.dumps({"ambiguity": 0.35, "quota-first": 1.0}))
    assert _generate(tmp_path, tmp_path / "o", n=8,
                     extra=["--frontier", str(f)])[0] == 0


# --- 14. clean CLI errors ---------------------------------------------------------

def test_missing_inputs_are_clean_errors(tmp_path):
    missing = tmp_path / "nowhere"
    rc, _, err = _generate(tmp_path, tmp_path / "o", extra=[
        "--prereg", str(missing / "prereg.json")])
    assert rc == 1 and "--prereg" in err and not (tmp_path / "o").exists()
    base = _baseline(tmp_path, ["x"])
    rc, _, err = _run(calibrate_mod.main, ["--heldout", str(missing),
                                           "--baseline", str(base)])
    assert rc == 1 and err.startswith("calibrate: ")
    rc, _, err = _run(blind_mod.main_blind, [
        "--heldout", str(missing), "--template", str(tmp_path / "t.json"),
        "--key", str(tmp_path / "k.json")])
    assert rc == 1 and err.startswith("blind: ")
    rc, _, err = _run(audit_mod.main, ["--heldout", str(missing)])
    assert rc == 1


# --- 15. user-facing English ------------------------------------------------------

def test_no_leftover_portuguese_in_generator_modules():
    task = {"setup": "Criar {pasta}", "instruction": "i", "auto_fail": "f",
            "expected": ["e"], "rubric": {0: "z", 1: "u", 2: "d"}}
    assert gen.validate_task(task, gen.render("x", dict(task, category="c"))) \
        == ["unfilled placeholder"]
    words = re.compile(r"\b(sem|preencher|esgotada|embaralhadas|estratificado|"
                       r"lote|fariam|recusa|removia|passava|renderizada|"
                       r"contrato|acerta|perto|rendem|continuem|revelava|"
                       r"refaz|carrega|Sorteio|repetia|Cotas|restos)\b")
    for name in ("generate", "audit", "calibrate", "blind"):
        src = (ROOT / "hod" / ("%s.py" % name)).read_text(encoding="utf-8")
        assert not words.findall(src), (name, words.findall(src))
