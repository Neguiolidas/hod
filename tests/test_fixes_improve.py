"""Regression tests for the `hod improve` / demo / toy audit fixes.

Usage: python3 -m pytest tests/test_fixes_improve.py
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hod import demo as demo_mod  # noqa: E402
from hod import generate as gen  # noqa: E402
from hod import improve as imp  # noqa: E402
from hod import toy  # noqa: E402
from hod.families import FAMILIES  # noqa: E402

PY = sys.executable
TOY_EVAL = (PY + " -m hod.toy eval --config {config} --tasks {tasks} "
            "--seed {seed} --out {out}")
TOY_PROPOSE = (PY + " -m hod.toy propose --champion {champion} --history "
               "{history} --k {k} --round {round} --out {out}")


def quiet(*a):
    pass


def _big(champion, history, k, r):
    """One candidate with a huge genuine effect: accepted for sure."""
    return [{"skills": {f: 0.95 for f in sorted(FAMILIES)}, "memorized": [],
             "change": "big skill jump", "kind": "genuine"}][:k]


def _lineage(out):
    return [json.loads(line) for line in
            (Path(out) / "lineage.jsonl").read_text().splitlines()]


def _run_main(argv):
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        rc = imp.main(argv)
    return rc, err.getvalue()


@pytest.fixture
def pypath(monkeypatch):
    """Commands run `python -m hod.toy`: make hod importable from anywhere."""
    old = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", str(REPO) + (os.pathsep + old if old else ""))


# --- 1. seeds: not predictable by the proposer ---------------------------------

def test_master_seed_and_weights_never_reach_the_proposer(tmp_path):
    fams = toy.write_catalog(tmp_path / "cat")
    out = tmp_path / "out"
    views, ledgers = [], []

    def spy(champion, history, k, r):
        views.append(history)
        ledgers.append((out / "lineage.jsonl").read_text())
        return toy.propose(champion, history, k, r)

    res = imp.improve(toy.initial_config(), toy.evaluate, spy, tmp_path / "cat",
                      out, suite_families=fams, rounds=2, k=2, n_held=24,
                      patience=5, frontier_target=0.35, log=quiet)
    master = res["seed"]
    assert len(views) == 2
    for view in views:
        for e in view:
            assert not set(e) & {"seed", "n_held", "family_weights"}, e
    for text in ledgers:  # what a command proposer could read in --out
        assert all("seed" not in json.loads(line)
                   for line in text.splitlines())
        assert str(master) not in text
    entries = _lineage(out)
    assert "seed" not in entries[0] and entries[0]["seed_source"] == "random"
    assert entries[-1]["type"] == "stop" and entries[-1]["seed"] == master
    # the weights are still recorded (they derive from the public suite)
    assert any(e.get("family_weights") for e in entries if e["type"] == "round")


def _final_texts(tmp_path, name, seed):
    fams = toy.write_catalog(tmp_path / "cat")
    out = tmp_path / name
    res = imp.improve(toy.initial_config(), toy.evaluate, toy.propose,
                      tmp_path / "cat", out, suite_families=fams, rounds=1,
                      k=1, n_held=24, seed=seed, keep_work=True, log=quiet)
    final = sorted(p.read_text() for p in (out / "final" / "heldout")
                   .glob("heldout-*.md"))
    rnd = sorted(p.read_text() for p in (out / "rounds" / "r01" / "heldout")
                 .glob("heldout-*.md"))
    return res, final, rnd


def test_default_seed_is_random_and_explicit_seed_reproduces(tmp_path):
    ra, fa, _ = _final_texts(tmp_path, "a", None)
    rb, fb, _ = _final_texts(tmp_path, "b", None)
    assert ra["seed"] != rb["seed"] and fa != fb
    rc, fc, rndc = _final_texts(tmp_path, "c", 11)
    _, fd, rndd = _final_texts(tmp_path, "d", 11)
    assert rc["seed"] == 11 and fc == fd and rndc == rndd and fc != rndc
    # the old, guessable derivation (seed * 1000 + 999) no longer gives the exam
    _, sh = imp.load_suite(tmp_path / "cat")
    m = gen.generate_one(11 * 1000 + 999, 24, tmp_path / "guess", sh, quiet=True)
    guess = sorted((tmp_path / "guess" / t["file"]).read_text() for t in m["tasks"])
    assert guess != fc


def test_seed_help_warns_about_predictability(capsys):
    with pytest.raises(SystemExit):
        imp.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "predict" in text and "--keep-work" in text
    assert "--evaluator-file" in text


# --- 2. no held-out data on disk where the proposer looks -------------------------

SPY = r'''
import json, os, sys
sys.path.insert(0, %(repo)r)
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
leaks, n = [], 0
for root in os.environ["HOD_SPY_ROOTS"].split(os.pathsep):
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            try:
                data = open(path, "rb").read()
            except OSError:
                continue
            n += 1
            if b"heldout-" in data or b"### Setup" in data:
                leaks.append(path)
with open(os.environ["HOD_SPY_LOG"], "a") as f:
    f.write(json.dumps({"round": args["--round"], "files": n,
                        "leaks": leaks}) + "\n")
from hod import toy
sys.exit(toy.main(["propose"] + sys.argv[1:]))
'''


def _spy_run(tmp_path, monkeypatch, keep_work, scan_tmp):
    run_root, spy_dir, tmp_root = (tmp_path / "run", tmp_path / "spy",
                                   tmp_path / "systmp")
    for d in (run_root, spy_dir, tmp_root):
        d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_root))
    (spy_dir / "spy.py").write_text(SPY % {"repo": str(REPO)})
    roots = [str(run_root)] + ([str(tmp_root)] if scan_tmp else [])
    monkeypatch.setenv("HOD_SPY_ROOTS", os.pathsep.join(roots))
    monkeypatch.setenv("HOD_SPY_LOG", str(spy_dir / "log.jsonl"))
    fams = toy.write_catalog(run_root / "cat")
    out = run_root / "out"
    ev = imp.CommandEvaluator(TOY_EVAL, out / "work", keep=keep_work)
    prop = imp.CommandProposer(
        PY + " " + str(spy_dir / "spy.py") + " --champion {champion} "
        "--history {history} --k {k} --round {round} --out {out}", out / "work")
    try:
        res = imp.improve(toy.initial_config(), ev, prop, run_root / "cat", out,
                          suite_families=fams, rounds=2, k=2, n_held=24,
                          seed=3, patience=5, keep_work=keep_work, log=quiet)
    finally:
        ev.close()
    log = [json.loads(line) for line in
           (spy_dir / "log.jsonl").read_text().splitlines()]
    return res, out, tmp_root, log


def test_spy_command_proposer_finds_no_heldout_on_disk(tmp_path, monkeypatch,
                                                       pypath):
    res, out, tmp_root, log = _spy_run(tmp_path, monkeypatch, keep_work=False,
                                       scan_tmp=True)
    assert [e["round"] for e in log] == ["1", "2"]
    assert log[-1]["files"] > 5          # it did read --out (lineage, champion...)
    assert all(e["leaks"] == [] for e in log), log
    # nothing left behind: no held-out in --out, no private temp dirs
    assert not list(out.rglob("heldout-*.md")) and not list(out.rglob("tasks.json"))
    assert list(tmp_root.iterdir()) == []
    assert res["entries"][-1]["type"] == "stop"


def test_keep_work_writes_sets_into_out_only_after_the_run(tmp_path, monkeypatch,
                                                           pypath):
    res, out, tmp_root, log = _spy_run(tmp_path, monkeypatch, keep_work=True,
                                       scan_tmp=False)
    assert all(e["leaks"] == [] for e in log), log   # nothing in --out mid-run
    # ... but everything is there once the run ended (the spy would see it)
    assert len(list((out / "rounds" / "r01" / "heldout").glob("heldout-*.md"))) == 24
    assert len(list((out / "final" / "heldout").glob("heldout-*.md"))) == 24
    # call 1 = start (suite), 2 = round-1 champion on the suite, 3 = held-out
    tasks = json.loads((out / "work" / "eval-0003" / "tasks.json").read_text())
    assert tasks[0]["id"].startswith("heldout-")
    assert list(tmp_root.iterdir()) == []


def test_failed_eval_dir_is_kept_and_error_hides_private_path(tmp_path,
                                                              monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    ev = imp.CommandEvaluator("exit 3 # {config} {tasks} {out}",
                              tmp_path / "work")
    with pytest.raises(imp.ImproveError) as err:
        ev({"a": 1}, [{"id": "heldout-x-00", "family": "x", "text": "t"}], 1)
    private = next(tmp_path.glob("hod-eval-*"))
    assert "exit code 3" in str(err.value) and str(private) not in str(err.value)
    ev.close()
    assert (tmp_path / "work" / "eval-0001" / "log.txt").exists()
    assert not private.exists()


# --- 3. evaluator change: no final exam, exit code 1 ----------------------------

class FlipEvaluator:
    """In-process toy evaluator whose identity turns "b" from call `at` on,
    or when `state["swapped"]` is set."""

    def __init__(self, at=None):
        self.at, self.n, self.swapped = at, 0, False

    def identity(self):
        self.n += 1
        return "b" if self.swapped or (self.at and self.n >= self.at) else "a"

    def __call__(self, cfg, tasks, seed):
        return toy.evaluate(cfg, tasks, seed)


def test_evaluator_changed_during_last_round_skips_final_exam(tmp_path):
    fams = toy.write_catalog(tmp_path / "cat")
    ev = FlipEvaluator()

    def prop(champion, history, k, r):
        if r == 2:  # swapped after the round-start check
            ev.swapped = True
        return _big(champion, history, k, r)

    res = imp.improve(toy.initial_config(), ev, prop, tmp_path / "cat",
                      tmp_path / "out", suite_families=fams, rounds=2, k=1,
                      n_held=32, seed=7, patience=5, log=quiet)
    assert res["summary"]["accepted"] == ["h1.1"]
    assert "evaluator changed during round 2" in res["summary"]["reason"]
    assert res["status"] == "error" and res["summary"]["status"] == "error"
    assert res["final"]["skipped"].startswith("not run: evaluator changed")
    assert "verdict" not in res["final"]


def test_evaluator_changed_before_final_exam_skips_it(tmp_path):
    fams = toy.write_catalog(tmp_path / "cat")
    # identity calls: start, round-1 start, round-1 end, final exam check
    res = imp.improve(toy.initial_config(), FlipEvaluator(at=4), _big,
                      tmp_path / "cat", tmp_path / "out", suite_families=fams,
                      rounds=1, k=1, n_held=32, seed=7, log=quiet)
    assert res["summary"]["accepted"] == ["h1.1"]
    assert "before the final exam" in res["summary"]["reason"]
    assert res["status"] == "error" and "skipped" in res["final"]
    assert "Skipped: not run" in (tmp_path / "out" / "report.md").read_text()


def test_evaluator_changed_mid_round_rejects_winner(tmp_path):
    fams = toy.write_catalog(tmp_path / "cat")
    res = imp.improve(toy.initial_config(), FlipEvaluator(at=3), _big,
                      tmp_path / "cat", tmp_path / "out", suite_families=fams,
                      rounds=3, k=1, n_held=32, seed=7, log=quiet)
    cand = [e for e in res["entries"] if e["type"] == "candidate"]
    assert len(cand) == 1 and cand[0]["gate"] == "evaluator_changed"
    assert res["summary"]["accepted"] == [] and res["status"] == "error"
    assert json.loads((tmp_path / "out" / "champion.json").read_text()) \
        == toy.initial_config()


def test_cli_evaluator_file_change_exits_1(tmp_path, pypath):
    fams = toy.write_catalog(tmp_path / "cat")
    (tmp_path / "fam.json").write_text(json.dumps(fams))
    toy.main(["init", "--out", str(tmp_path / "cfg")])
    evdir = tmp_path / "evaluator"
    (evdir / "sub").mkdir(parents=True)
    (evdir / "sub" / "rubric.txt").write_text("pass = 2\n")
    tamper = tmp_path / "tamper.py"
    tamper.write_text(
        "import subprocess, sys\n"
        "with open(%r, 'a') as f:\n    f.write('pass = 0\\n')\n"
        "sys.exit(subprocess.call([sys.executable, '-m', 'hod.toy', 'propose']"
        " + sys.argv[1:]))\n" % str(evdir / "sub" / "rubric.txt"))
    rc, err = _run_main([
        "--initial", str(tmp_path / "cfg" / "initial.json"),
        "--eval", TOY_EVAL,
        "--propose", PY + " " + str(tamper) + " --champion {champion} "
        "--history {history} --k {k} --round {round} --out {out}",
        "--catalog", str(tmp_path / "cat"),
        "--suite-families", str(tmp_path / "fam.json"),
        "--out", str(tmp_path / "out"), "--rounds", "3", "--k", "1",
        "--n-held", "24", "--seed", "5", "--evaluator-file", str(evdir)])
    assert rc == 1, err
    assert "evaluator changed" in err and "Traceback" not in err
    stop = _lineage(tmp_path / "out")[-1]
    assert stop["type"] == "stop" and stop["status"] == "error"
    assert (tmp_path / "out" / "report.md").exists()


# --- 4. evaluator identity ---------------------------------------------------

def test_identity_covers_programs_on_path(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("POSIX executable bit")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    prog = bindir / "my-eval"
    prog.write_text("#!/bin/sh\necho v1\n")
    prog.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    cmd = "my-eval --config {config} --tasks {tasks} --out {out}"
    a = imp.command_identity(cmd)
    assert imp.command_identity(cmd) == a          # deterministic
    prog.write_text("#!/bin/sh\necho v2-cheating\n")
    assert imp.command_identity(cmd) != a


def test_identity_covers_evaluator_files_and_dirs(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "ev.py").write_text("import rubric\n")
    (pkg / "sub" / "rubric.py").write_text("PASS = 2\n")
    single = tmp_path / "weights.bin"
    single.write_bytes(b"\x00\x01")
    ev = imp.CommandEvaluator("python3 -m ev {config} {tasks} {out}",
                              tmp_path / "w", files=[pkg, single])
    a = ev.identity()
    assert ev.identity() == a
    # Python's own bytecode cache must not count as a change
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "ev.cpython-313.pyc").write_bytes(b"junk")
    (pkg / "sub" / "rubric.pyc").write_bytes(b"junk")
    assert ev.identity() == a
    (pkg / "sub" / "rubric.py").write_text("PASS = 0\n")
    b = ev.identity()
    assert b != a
    single.write_bytes(b"\x00\x02")
    assert ev.identity() != b


def test_validate_command_is_part_of_the_identity(tmp_path):
    v1 = imp.command_validator("check-a {config}", tmp_path)
    v2 = imp.command_validator("check-b {config}", tmp_path)
    assert v1.identity() == imp.command_identity("check-a {config}")
    assert v1.identity() != v2.identity()

    fams = toy.write_catalog(tmp_path / "cat")
    state = {"n": 0}

    class Validator:
        def identity(self):
            state["n"] += 1
            return "a" if state["n"] < 3 else "b"

        def __call__(self, cfg):
            return True, "ok"

    res = imp.improve(toy.initial_config(), toy.evaluate, toy.propose,
                      tmp_path / "cat", tmp_path / "out", suite_families=fams,
                      rounds=3, k=1, n_held=24, seed=2, validate=Validator(),
                      log=quiet)
    assert "evaluator changed" in res["summary"]["reason"]
    assert res["entries"][0]["validate_sha"] == "a"


# --- 5. one failing candidate does not kill the run ------------------------------

def test_failing_candidate_is_rejected_with_evaluation_error(tmp_path):
    fams = toy.write_catalog(tmp_path / "cat", per_family=8)

    def ev(cfg, tasks, seed):
        if cfg.get("style") == "formal":  # the toy's placebo candidate
            raise RuntimeError("cannot load this candidate " + "x" * 2000)
        return toy.evaluate(cfg, tasks, seed)

    res = imp.improve(toy.initial_config(), ev, toy.propose, tmp_path / "cat",
                      tmp_path / "out", suite_families=fams, rounds=1, k=5,
                      n_held=48, seed=7, log=quiet)
    cands = {e["kind"]: e for e in _lineage(tmp_path / "out")
             if e["type"] == "candidate"}
    assert len(cands) == 5
    bad = cands["placebo"]
    assert bad["decision"] == "rejected" and bad["gate"] == "evaluation_error"
    assert "cannot load" in bad["error"] and len(bad["error"]) <= imp.MAX_ERROR
    assert all(e["gate"] != "evaluation_error" for k, e in cands.items()
               if k != "placebo")
    assert res["status"] != "error" and res["summary"]["candidate_evals"] == 5
    assert "evaluation_error" in (tmp_path / "out" / "report.md").read_text()


def test_infrastructure_failure_still_writes_stop_and_report(tmp_path):
    fams = toy.write_catalog(tmp_path / "cat")
    calls = {"n": 0}

    def ev(cfg, tasks, seed):
        calls["n"] += 1
        # start 1 + round 1 (champion 2, placebo 2, one candidate 2) = 7:
        # call 8 is the round-2 champion
        if calls["n"] == 8:
            raise RuntimeError("evaluator crashed")
        return toy.evaluate(cfg, tasks, seed)

    with pytest.raises(RuntimeError):
        imp.improve(toy.initial_config(), ev, _big, tmp_path / "cat",
                    tmp_path / "out", suite_families=fams, rounds=3, k=1,
                    n_held=32, seed=7, patience=5, log=quiet)
    entries = _lineage(tmp_path / "out")
    assert [e["id"] for e in entries if e["type"] == "candidate"] == ["h1.1"]
    stop = entries[-1]
    assert stop["type"] == "stop" and stop["status"] == "error"
    assert "evaluator crashed" in stop["reason"] and stop["seed"] == 7
    assert any(e["type"] == "final_exam" and "skipped" in e for e in entries)
    assert "aborted" in (tmp_path / "out" / "report.md").read_text()


class Interrupt(BaseException):
    pass


def test_interrupt_mid_round_records_finished_candidates(tmp_path):
    fams = toy.write_catalog(tmp_path / "cat")

    def prop(champion, history, k, r):
        c = _big(champion, history, k, r)[0]
        return [c, dict(c, change="second")]

    def ev(cfg, tasks, seed):
        if seed == imp.run_seed(7, "r1/h1.2"):
            raise Interrupt()
        return toy.evaluate(cfg, tasks, seed)

    with pytest.raises(Interrupt):
        imp.improve(toy.initial_config(), ev, prop, tmp_path / "cat",
                    tmp_path / "out", suite_families=fams, rounds=1, k=2,
                    n_held=32, seed=7, log=quiet)
    entries = _lineage(tmp_path / "out")
    first = [e for e in entries if e["type"] == "candidate"]
    assert [e["id"] for e in first] == ["h1.1"]
    assert first[0]["gate"] == "round_aborted"    # approved, never committed
    assert entries[-1]["type"] == "stop" and entries[-1]["status"] == "error"
    assert json.loads((tmp_path / "out" / "champion.json").read_text()) \
        == toy.initial_config()


# --- 6. missing scores are "not run", never 0 ---------------------------------------

def test_clean_scores_never_fills_zero():
    tasks = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
    scores, cov = imp._clean_scores({"a": 2, "b": None, "zz": 1}, tasks, "x")
    assert scores == {"a": 2} and cov == 0.25


def _drop_half(scores):
    return {t: v for i, (t, v) in enumerate(sorted(scores.items())) if i % 2}


def test_final_exam_without_coverage_is_inconclusive(tmp_path):
    fams = toy.write_catalog(tmp_path / "cat")
    drop_seed = imp.run_seed(7, "final/initial")

    def ev(cfg, tasks, seed):
        s = toy.evaluate(cfg, tasks, seed)
        if seed == drop_seed and tasks[0]["id"].startswith("heldout-"):
            s = _drop_half(s)
        return s

    res = imp.improve(toy.initial_config(), ev, _big, tmp_path / "cat",
                      tmp_path / "out", suite_families=fams, rounds=1, k=1,
                      n_held=32, seed=7, log=quiet)
    assert res["summary"]["accepted"] == ["h1.1"]
    assert res["final"]["verdict"] == "INCONCLUSIVE"
    assert res["final"]["coverage"] == 0.5


def test_sanity_requires_coverage(tmp_path):
    fams = toy.write_catalog(tmp_path / "cat")

    def ev(cfg, tasks, seed):
        s = toy.evaluate(cfg, tasks, seed)
        return _drop_half(s) if cfg == toy.known_good() else s

    res = imp.improve(toy.initial_config(), ev, toy.propose, tmp_path / "cat",
                      tmp_path / "out", suite_families=fams, rounds=2, k=1,
                      n_held=32, seed=7, sanity_good=toy.known_good(),
                      sanity_bad=toy.known_bad(), log=quiet)
    san = [e for e in res["entries"] if e["type"] == "sanity"]
    assert len(san) == 1 and not san[0]["ok"]
    assert san[0]["coverage"]["held"] == 0.5
    assert "sanity failed" in res["summary"]["reason"]


def test_placebo_without_coverage_stops_with_error(tmp_path):
    fams = toy.write_catalog(tmp_path / "cat")
    drop_seed = imp.run_seed(7, "r1/placebo")

    def ev(cfg, tasks, seed):
        s = toy.evaluate(cfg, tasks, seed)
        return _drop_half(s) if seed == drop_seed else s

    res = imp.improve(toy.initial_config(), ev, toy.propose, tmp_path / "cat",
                      tmp_path / "out", suite_families=fams, rounds=2, k=1,
                      n_held=32, seed=7, log=quiet)
    assert "placebo" in res["summary"]["reason"] and res["status"] == "error"
    assert not [e for e in res["entries"] if e["type"] == "candidate"]


# --- 7. timeout kills the whole process group --------------------------------------

def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie (killed, not yet reaped by init) is dead
        return Path("/proc/%d/stat" % pid).read_text().split(")")[1].split()[0] != "Z"
    except OSError:
        return True


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
def test_timeout_kills_children(tmp_path):
    pidfile = tmp_path / "pid"
    t0 = time.time()
    rc = imp._run("sleep 30 & echo $! > %s; wait" % pidfile, tmp_path / "log",
                  timeout=1)
    assert rc == 124 and time.time() - t0 < 10
    pid = int(pidfile.read_text())
    deadline = time.time() + 5
    while _alive(pid) and time.time() < deadline:
        time.sleep(0.05)
    if _alive(pid):  # do not leak it, then fail
        os.kill(pid, 9)
        pytest.fail("child of the timed-out command survived")
    assert "timeout" in (tmp_path / "log").read_text()


# --- 8. bad templates: clean errors, before any work ----------------------------------

def test_bad_templates_raise_clean_errors():
    for bad, needle in (("ev {task} {tasks} {config} {out}", "{task}"),
                        ("ev {} {tasks} {config} {out}", "positional"),
                        ("awk '{print $1}' {config} {tasks} {out}", "{{"),
                        ("ev {seed:05d} {config} {tasks} {out}", "format"),
                        ("ev {config} {tasks}", "missing {out}")):
        with pytest.raises(imp.ImproveError) as err:
            imp.check_template(bad, "eval")
        msg = str(err.value)
        assert needle in msg and "allowed placeholders" in msg, msg
    with pytest.raises(imp.ImproveError):
        imp._fill("x {nope}", "validate", config="c")
    assert imp.check_template("awk '{{print $1}}' {config} {tasks} {out}", "eval")


def test_cli_bad_template_fails_before_any_work(tmp_path):
    toy.main(["init", "--out", str(tmp_path / "cfg")])
    toy.write_catalog(tmp_path / "cat")
    rc, err = _run_main([
        "--initial", str(tmp_path / "cfg" / "initial.json"),
        "--eval", TOY_EVAL, "--propose", TOY_PROPOSE,
        "--validate", "check {cfg}",
        "--catalog", str(tmp_path / "cat"), "--out", str(tmp_path / "out")])
    assert rc == 1 and "--validate" in err and "{cfg}" in err
    assert "Traceback" not in err and not (tmp_path / "out").exists()


# --- 9. CLI input errors: no tracebacks -------------------------------------------

def test_cli_input_errors_are_clean(tmp_path):
    toy.write_catalog(tmp_path / "cat")
    (tmp_path / "list.json").write_text("[1, 2]")
    (tmp_path / "ok.json").write_text(json.dumps(toy.initial_config()))
    base = ["--eval", TOY_EVAL, "--propose", TOY_PROPOSE,
            "--catalog", str(tmp_path / "cat")]
    cases = [
        (["--initial", str(tmp_path / "list.json")], "expected a JSON object"),
        (["--initial", str(tmp_path / "missing.json")], "missing"),
        (["--initial", str(tmp_path / "ok.json"), "--catalog",
          str(tmp_path / "nope")], "catalog"),
        (["--initial", str(tmp_path / "ok.json"), "--evaluator-file",
          str(tmp_path / "nope.py")], "evaluator file missing"),
        (["--initial", str(tmp_path / "ok.json"), "--boot", "10"], "--boot"),
        (["--initial", str(tmp_path / "ok.json"), "--sanity-good",
          str(tmp_path / "list.json"), "--sanity-bad", str(tmp_path / "ok.json")],
         "expected a JSON object"),
    ]
    for i, (extra, needle) in enumerate(cases):
        rc, err = _run_main(base + ["--out", str(tmp_path / ("o%d" % i))] + extra)
        assert rc == 1 and "Traceback" not in err, (extra, err)
        assert err.startswith("improve: ") and needle in err, (extra, err)


def test_cli_usage_error_exits_1_not_2(tmp_path):
    rc, err = _run_main(["--initial", str(tmp_path / "x.json")])  # no --eval...
    assert rc == 1 and "required" in err
    rc, _ = _run_main(["--initial", "x", "--eval", "e", "--propose", "p",
                       "--catalog", "c", "--out", "o", "--rounds", "many"])
    assert rc == 1


def test_improve_rejects_non_object_initial(tmp_path):
    with pytest.raises(imp.ImproveError):
        imp.improve([1, 2], toy.evaluate, toy.propose, tmp_path / "cat",
                    tmp_path / "out", log=quiet)


# --- 10. durable champion write ---------------------------------------------------

def test_write_atomic_fsyncs_file_and_dir(tmp_path, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd)))
    imp._write_atomic(tmp_path / "champion.json", {"a": "ação"})
    assert json.loads((tmp_path / "champion.json").read_text(encoding="utf-8")) \
        == {"a": "ação"}
    assert not list(tmp_path.glob("*.tmp"))
    assert len(calls) == (2 if os.name == "posix" else 1)


# --- 11. demo --sweep honors --via-commands ------------------------------------------

def test_demo_sweep_honors_via_commands(tmp_path, monkeypatch):
    calls = []

    def fake(out, **kw):
        calls.append(kw)
        (Path(out) / "improve").mkdir(parents=True)
        (Path(out) / "improve" / "report.md").write_text("report\n")
        return {"entries": [], "final": None, "summary": {"reason": "x"}}

    monkeypatch.setattr(demo_mod, "run_one", fake)
    with contextlib.redirect_stdout(io.StringIO()):
        rc = demo_mod.main(["--out", str(tmp_path / "d"), "--sweep", "2",
                            "--via-commands"])
    assert rc == 0 and len(calls) == 3
    assert all(c["via_commands"] for c in calls)
    assert [c.get("keep_work", True) for c in calls] == [True, False, False]


# --- 12. English user-facing strings ---------------------------------------------

def test_toy_init_and_help_are_english(tmp_path, capsys):
    toy.main(["init", "--out", str(tmp_path / "cfg")])
    assert sorted(p.name for p in (tmp_path / "cfg").iterdir()) \
        == ["bad.json", "good.json", "initial.json"]
    assert toy.initial_config()["style"] == "neutral"
    for fn, argv in ((imp.main, ["--help"]), (toy.main, ["init", "--help"]),
                     (demo_mod.main, ["--help"])):
        with pytest.raises(SystemExit):
            fn(argv)
    text = capsys.readouterr().out
    for word in ("inicial", "rendem", "mais ", "escreve", "estilo", " bom ",
                 "ruim"):
        assert word not in text, word


def test_toy_commands_round_trip_utf8(tmp_path, pypath):
    """The command path reads what improve writes (UTF-8, any locale)."""
    tasks = [{"id": "t-ação", "family": "ambiguity", "text": "ção"}]
    (tmp_path / "c.json").write_text(json.dumps(toy.initial_config()),
                                     encoding="utf-8")
    (tmp_path / "t.json").write_text(json.dumps(tasks, ensure_ascii=False),
                                     encoding="utf-8")
    # plain C locale, no UTF-8 mode, no locale coercion: ASCII by default
    env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONUTF8="0",
               PYTHONCOERCECLOCALE="0")
    subprocess.run([PY, "-m", "hod.toy", "eval", "--config",
                    str(tmp_path / "c.json"), "--tasks", str(tmp_path / "t.json"),
                    "--seed", "1", "--out", str(tmp_path / "s.json")],
                   check=True, env=env)
    out = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert list(out) == ["t-ação"]
