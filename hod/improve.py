#!/usr/bin/env python3
"""`hod improve`: self-improvement loop with HOD as judge.

Idea (adapted from Harness-squared, the vibe-coding-toolkit, Amazon Transform
and Ornith-1.5): the proposer suggests changes to the harness, HOD measures
each one on the suite and on a NEW held-out, and only a change with a
GENUINE verdict that passes all gates becomes the new champion.

Round by round:
1. Checks that the evaluator has not changed (hash recorded at the start).
   Changed evaluator = invalid comparison: the loop stops with an error.
2. Asks the proposer for k candidates. It sees the champion, the champion's
   scores on the suite (public) and the aggregated history. It never sees
   per-task held-out scores, nor what would let it rebuild the generator
   calls (master seed, set size, family weights), and the round's held-out
   is only generated AFTER the proposal.
3. Generates the new held-out, excluding all tasks already used (rotation):
   no set decides twice, so the loop does not learn the held-out.
   If the generator can no longer find new tasks, the loop stops.
4. Measures the champion (round baseline) and the PLACEBO (the same champion
   again, another evaluation round). The placebo calibrates the noise floor.
5. Evaluator sanity (if `--sanity-good/--sanity-bad`): the good harness
   must beat the bad one with CI above zero on the suite and on the round's
   held-out, with coverage. If the evaluator does not separate the two, the
   loop stops.
6. For each candidate, gates in order (verify-then-commit):
   validation (`--validate`, a command of yours that must exit 0) -> a
   failed evaluation rejects only that candidate (`evaluation_error`) ->
   coverage (a task without a score is "not run", never a 0 or a pass;
   below 90% the candidate fails) -> verdict by CI = GENUINE -> no family
   dropped (held-out or suite).
7. Among the approved, the one with the largest lower bound of the held-out
   CI stays. The evaluator hash is checked again before the swap; the
   champion is only replaced after all of that (written atomically and
   durably); otherwise the previous one stays intact.
8. Everything goes to the lineage ledger (JSONL): each candidate (written as
   soon as its decision is final), the gate that blocked it, numbers, and
   the lesson with the invalidation trigger.

Stop: `--rounds`, `--budget` (candidate evaluations), `--patience`
(consecutive rounds without acceptance), empty proposer, exhausted
held-out, failed sanity, changed evaluator. An error (changed evaluator,
an evaluator that cannot score the champion or the placebo, a failing
proposer) still writes the `stop` record and report.md.

Final exam: the sealed held-out is generated before the first round and
excluded from all of them. With something accepted, the final champion is
compared with the initial harness on that set (skipped if the evaluator
changed). This protects against the "winner's curse": picking the best of
several candidates over several rounds inflates the measured gain; the final
exam measures again, on tasks nobody saw.

Secrecy of the held-out:
- Seeds. Every set (each round's and the final exam's) and every evaluator
  run seed is derived from a master seed with sha256, so knowing one set
  or one run seed says nothing about the others. Without `--seed` the
  master seed comes from `secrets` and is written only to the final `stop`
  record (reproducibility after the fact). With an explicit `--seed` the
  run is reproducible, and a proposer who knows it can predict every set.
- Disk. The held-out sets are generated in a private temporary directory
  (mode 0700, outside --out), loaded into memory and deleted. Each command
  evaluator call works in its own private directory, deleted as soon as its
  scores are read. During a proposer call no held-out task text or per-task
  held-out score is left in --out (nor, by default, anywhere on disk).
  `--keep-work` keeps them, but only moves them into --out when the run
  ends. Limit: a proposer that runs as the same OS user can read whatever
  that user can (other processes, the temporary directory); real isolation
  needs a sandbox or a separate user.

Command interface (placeholders replaced by already-quoted paths):
    --eval     "CMD {config} {tasks} {seed} {out}"   writes {task_id: 0|1|2}
    --propose  "CMD {champion} {history} {k} {round} {out}"  writes a list
               of JSON configs (each may have "change": description)
    --validate "CMD {config}"   exits 0 if the candidate is valid
The {tasks} file is the list [{"id", "family", "text"}]. Literal braces in a
command are written {{ and }}.
"""

import argparse
import hashlib
import inspect
import json
import os
import secrets
import shlex
import shutil
import signal
import string
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from hod import gap as gap_mod
from hod import generate as gen_mod
from hod import ledger as ledger_mod
from hod import placebo as placebo_mod
from hod import sanity as sanity_mod
from hod import stats as stats_mod
from hod.scores import read_json, validate_score

DEFAULT_ROUNDS = 5
DEFAULT_K = 4
DEFAULT_PATIENCE = 2
DEFAULT_N_HELD = 64
DEFAULT_TIMEOUT = 600
MAX_ERROR = 500  # characters of an error message kept in the lineage

# placeholders each command template may use, and those it must use
PLACEHOLDERS = {
    "eval": ("config", "tasks", "seed", "out"),
    "propose": ("champion", "history", "k", "round", "out"),
    "validate": ("config",),
}
REQUIRED = {
    "eval": ("config", "tasks", "out"),
    "propose": ("out",),
    "validate": ("config",),
}
# lineage keys the proposer never sees: per-task held-out scores, and what
# would let it rebuild the generator calls (master seed, set size, weights)
HIDDEN_FROM_PROPOSER = ("held_scores", "seed", "n_held", "family_weights")
# rewritten by Python itself at import: hashing them would report a changed
# evaluator right after its first run
SKIP_DIRS = ("__pycache__",)
SKIP_SUFFIXES = (".pyc", ".pyo")


class ImproveError(Exception):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _short(exc):
    text = str(exc) or type(exc).__name__
    if not isinstance(exc, ImproveError):
        text = "%s: %s" % (type(exc).__name__, text) if str(exc) else text
    return text if len(text) <= MAX_ERROR else text[:MAX_ERROR - 3] + "..."


def config_hash(cfg):
    clean = {k: v for k, v in cfg.items() if k not in ("change", "kind")}
    return _sha(json.dumps(clean, sort_keys=True).encode())[:12]


# --- seeds -------------------------------------------------------------------

def derive_seed(master, label, bits=63):
    """Seed for one use (`label`) within a run, derived from the master seed
    with sha256: reproducible with the master seed, unpredictable without it
    (one set's seed says nothing about another's)."""
    digest = hashlib.sha256(("hod-improve\0%d\0%s" % (master, label))
                            .encode()).digest()
    return int.from_bytes(digest[:8], "big") >> (64 - bits)


def run_seed(master, label):
    """Evaluator run seed ({seed}); 31 bits, safe for any RNG."""
    return derive_seed(master, "run/" + label, bits=31)


# --- identity (hash) of the evaluator -----------------------------------------

def callable_identity(fn):
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = repr(fn)
    return _sha(src.encode())


def _hash_file(h, path, label):
    h.update(b"\0file\0" + label.encode("utf-8", "surrogateescape") + b"\0")
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError as e:
        h.update(b"\0unreadable\0" + str(e.errno).encode())


def _hash_tree(h, path):
    """A file's content, or the content of every file under a directory
    (recursive, sorted by path; __pycache__ and .pyc/.pyo skipped)."""
    path = Path(path)
    if path.is_file():
        _hash_file(h, path, str(path))
    elif path.is_dir():
        h.update(b"\0dir\0" + str(path).encode("utf-8", "surrogateescape"))
        for p in sorted(path.rglob("*")):
            rel = p.relative_to(path)
            if any(part in SKIP_DIRS for part in rel.parts) \
                    or p.suffix in SKIP_SUFFIXES or not p.is_file():
                continue
            _hash_file(h, p, rel.as_posix())
    else:
        h.update(b"\0missing\0" + str(path).encode("utf-8", "surrogateescape"))


def _tokens(cmd):
    try:
        return shlex.split(cmd)
    except ValueError:  # unbalanced quotes: the shell will complain, not us
        return cmd.split()


def command_identity(cmd, files=()):
    """sha256 of the command and of the content of what it runs: each token
    that is a file (relative to the current directory), each token that is a
    program found on PATH (shutil.which), and `files` (files or directories,
    recursive). Swapping the evaluator script mid-loop changes the hash.
    Not covered unless listed in `files`: modules a script imports, data it
    reads, the interpreter's libraries."""
    h = hashlib.sha256(b"cmd\0" + cmd.encode("utf-8", "surrogateescape"))
    for token in _tokens(cmd):
        try:
            if Path(token).is_file():
                _hash_file(h, Path(token), token)
                continue
            if os.sep in token or (os.altsep and os.altsep in token):
                continue  # a path that is not a file: nothing to look up
            found = shutil.which(token)
        except (OSError, ValueError):
            continue
        if found:
            _hash_file(h, Path(found), "which:%s=%s"
                       % (token, os.path.realpath(found)))
    for f in files:
        _hash_tree(h, f)
    return h.hexdigest()


def _identity_of(obj):
    if hasattr(obj, "identity"):
        return obj.identity()
    return callable_identity(obj)


# --- commands ----------------------------------------------------------------

def _kill_tree(proc):
    """Kills the command and everything it started (its process group)."""
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    else:  # Windows: taskkill /T takes the children too (best effort)
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _run(cmd, log_path, timeout):
    """Runs the command through the shell with its output going to a file
    (never a pipe that could fill up). It runs in its own session: on timeout
    or interrupt the whole process group is killed, not only /bin/sh.
    Returns the exit code (124 on timeout)."""
    with open(log_path, "ab") as log:
        log.write(("$ %s\n" % cmd).encode("utf-8", "replace"))
        log.flush()
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=log,
                                start_new_session=(os.name == "posix"))
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            log.write(("\n[hod] timeout after %ss: command and its children "
                       "killed\n" % timeout).encode())
            return 124
        except BaseException:
            _kill_tree(proc)
            raise


def _template_error(template, what, problem):
    allowed = " ".join("{%s}" % p for p in PLACEHOLDERS[what])
    return ("bad --%s command %r: %s; allowed placeholders: %s (write a "
            "literal brace as {{ or }})" % (what, template, problem, allowed))


def _fill(template, what="eval", **values):
    try:
        return template.format(**{k: shlex.quote(str(v))
                                  for k, v in values.items()})
    except KeyError as e:
        problem = "unknown placeholder {%s}" % e.args[0]
    except IndexError:
        problem = "positional placeholder ({} or {0})"
    except (ValueError, AttributeError, TypeError) as e:
        problem = str(e)
    raise ImproveError(_template_error(template, what, problem))


def check_template(template, what):
    """Validates a command template up front: known placeholders only, the
    required ones present, braces balanced."""
    try:
        fields = [f for _, f, _, _ in string.Formatter().parse(template)
                  if f is not None]
    except ValueError as e:
        raise ImproveError(_template_error(template, what, str(e))) from None
    names = set()
    for field in fields:
        name = field.split(".")[0].split("[")[0]
        if name == "" or name.isdigit():
            raise ImproveError(_template_error(
                template, what, "positional placeholder ({} or {0})"))
        if name not in PLACEHOLDERS[what]:
            raise ImproveError(_template_error(
                template, what, "unknown placeholder {%s}" % name))
        names.add(name)
    missing = [p for p in REQUIRED[what] if p not in names]
    if missing:
        raise ImproveError(_template_error(
            template, what, "missing %s" % " ".join("{%s}" % p for p in missing)))
    _fill(template, what, **{p: "x" for p in PLACEHOLDERS[what]})
    return template


class CommandEvaluator:
    """Runs the --eval command. Each call works in its own private directory
    (mode 0700, outside --out), deleted as soon as the scores are read: the
    held-out tasks.json and the per-task scores.json never sit where a
    proposer looking around --out can read them. Calls kept (all of them with
    keep=True, a failed one always) stay in the private directory during the
    run; close() moves them to `workdir` when the run ends.

    `files`: files or directories (recursive) that are part of the evaluator;
    their content enters identity()."""

    def __init__(self, cmd, workdir, timeout=DEFAULT_TIMEOUT, keep=False,
                 files=()):
        check_template(cmd, "eval")
        self.cmd, self.workdir, self.timeout = cmd, Path(workdir), timeout
        self.keep = keep
        self.files = [Path(f) for f in files]
        self.calls = 0
        self._private = None
        self._retained = []

    def identity(self):
        return command_identity(self.cmd, self.files)

    def __call__(self, config, tasks, run_seed):
        self.calls += 1
        if self._private is None:
            self._private = Path(tempfile.mkdtemp(prefix="hod-eval-"))
        name = "eval-%04d" % self.calls
        d = self._private / name
        d.mkdir()
        cfg, tk, out = d / "config.json", d / "tasks.json", d / "scores.json"
        cfg.write_text(json.dumps(config, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tk.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
        # the error message names where the log WILL be: the private path
        # is never handed out (it ends up in the history the proposer reads)
        later = "log in %s when the run ends" % (self.workdir / name / "log.txt")
        ok = False
        try:
            rc = _run(_fill(self.cmd, "eval", config=cfg, tasks=tk,
                            seed=run_seed, out=out), d / "log.txt", self.timeout)
            if rc != 0:
                raise ImproveError("evaluator failed (exit code %s); %s"
                                   % (rc, later))
            if not out.exists():
                raise ImproveError("evaluator exited 0 without writing the "
                                   "scores file; %s" % later)
            try:
                data = read_json(out, "evaluator scores", expect=dict)
            except ValueError:
                raise ImproveError("evaluator scores are not a valid JSON "
                                   "object; %s" % later) from None
            ok = True
            return data
        finally:
            if ok and not self.keep:
                shutil.rmtree(d, ignore_errors=True)
            else:
                self._retained.append(name)

    def close(self):
        """Moves the kept call directories to `workdir` and deletes the
        private directory. Idempotent; call it when the run ends."""
        private, self._private = self._private, None
        retained, self._retained = self._retained, []
        if private is None:
            return
        try:
            for name in retained:
                src = private / name
                if src.exists():
                    self.workdir.mkdir(parents=True, exist_ok=True)
                    dst = self.workdir / name
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.move(str(src), str(dst))
        finally:
            shutil.rmtree(private, ignore_errors=True)


class CommandProposer:
    def __init__(self, cmd, workdir, timeout=DEFAULT_TIMEOUT):
        check_template(cmd, "propose")
        self.cmd, self.workdir, self.timeout = cmd, Path(workdir), timeout

    def identity(self):
        return command_identity(self.cmd)

    def __call__(self, champion, history, k, round_index):
        d = self.workdir / ("propose-%02d" % round_index)
        d.mkdir(parents=True, exist_ok=True)
        ch, hist, out = d / "champion.json", d / "history.jsonl", d / "candidates.json"
        ch.write_text(json.dumps(champion, indent=2, ensure_ascii=False),
                      encoding="utf-8")
        hist.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n"
                                for e in history), encoding="utf-8")
        if out.exists():
            out.unlink()
        rc = _run(_fill(self.cmd, "propose", champion=ch, history=hist, k=k,
                        round=round_index, out=out), d / "log.txt", self.timeout)
        if rc != 0:
            raise ImproveError("proposer failed (exit code %s), see %s"
                               % (rc, d / "log.txt"))
        if not out.exists():
            raise ImproveError("proposer exited 0 without writing %s, see %s"
                               % (out, d / "log.txt"))
        try:
            return read_json(out, "proposer output", expect=list)
        except ValueError as e:
            raise ImproveError("%s (the proposer must write a JSON list)"
                               % e) from None


def command_validator(cmd, workdir, timeout=DEFAULT_TIMEOUT):
    check_template(cmd, "validate")
    state = {"n": 0}

    def validate(config):
        state["n"] += 1
        d = Path(workdir) / ("validate-%04d" % state["n"])
        d.mkdir(parents=True, exist_ok=True)
        cfg = d / "config.json"
        cfg.write_text(json.dumps(config, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        rc = _run(_fill(cmd, "validate", config=cfg), d / "log.txt", timeout)
        return rc == 0, "exit code %d (log in %s)" % (rc, d / "log.txt")

    # the --validate command is part of the evaluator's identity
    validate.identity = lambda: command_identity(cmd)
    return validate


# --- data --------------------------------------------------------------------

def load_suite(catalog_dir, suite_families=None):
    catalog_dir = Path(catalog_dir)
    if suite_families is not None and not isinstance(suite_families, dict):
        raise ImproveError("suite families must be a JSON object "
                           "{suite_id: family}")
    for task_id, fam in (suite_families or {}).items():
        if fam is not None and not isinstance(fam, str):
            raise ImproveError("suite family of %s must be a string, got %r"
                               % (task_id, fam))
    files = sorted(catalog_dir.glob("*.md")) if catalog_dir.is_dir() else []
    if not files:
        raise ImproveError("catalog empty or missing: %s" % catalog_dir)
    tasks = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ImproveError("catalog file %s is not UTF-8: %s"
                               % (path, e)) from None
        fam = (suite_families or {}).get(path.stem)
        tasks.append({"id": path.stem, "family": fam, "text": text})
    shingles = [gen_mod.shingles(gen_mod.normalize(t["text"])) for t in tasks]
    return tasks, shingles


def _generate_set(private, seed, n, catalog_shingles, weights=None,
                  exclude=None):
    """Generates a held-out set inside the private directory, loads it into
    memory and deletes it from disk. Returns {"tasks", "files", "shingles"}
    (files: {relative path: bytes}, for --keep-work), or None if the
    generator could not fill the set."""
    tmp = Path(tempfile.mkdtemp(dir=private))
    try:
        target = tmp / "heldout"
        manifest = gen_mod.generate_one(seed, n, target, catalog_shingles,
                                        family_weights=weights,
                                        exclude=exclude, quiet=True)
        if manifest is None:
            return None
        files = {p.relative_to(target).as_posix(): p.read_bytes()
                 for p in sorted(target.rglob("*")) if p.is_file()}
        tasks = [{"id": t["id"], "family": t["family"],
                  "text": files[t["file"]].decode("utf-8")}
                 for t in manifest["tasks"]]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # the exclusion buffer stays in memory: same shingles load_exclude reads
    return {"tasks": tasks, "files": files,
            "shingles": [gen_mod.content_shingles(t["text"]) for t in tasks]}


def _write_files(dest, files):
    for rel, data in files.items():
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _clean_scores(raw, tasks, origin):
    """Validated scores of the expected tasks. A task without a score stays
    out (it is "not run": never a 0, never a pass); coverage says how many
    are missing. Returns (scores, coverage)."""
    if not isinstance(raw, dict):
        raise ImproveError("%s: evaluator returned something that is not an "
                           "object" % origin)
    out = {}
    for t in tasks:
        if raw.get(t["id"]) is not None:
            out[t["id"]] = validate_score(t["id"], raw[t["id"]], origin)
    return out, (len(out) / len(tasks) if tasks else 0.0)


def _measure(evaluator, cfg, suite_tasks, held_tasks, seed, label):
    s, cs = _clean_scores(evaluator(cfg, suite_tasks, seed), suite_tasks,
                          "%s/suite" % label)
    h, ch = _clean_scores(evaluator(cfg, held_tasks, seed), held_tasks,
                          "%s/held-out" % label)
    return s, h, min(cs, ch)


def _analyze(sb, sc, hb, hc, suite_ids, held_ids, **kw):
    """gap.analyze with the expected ids (coverage counts tasks missing on
    both sides); None if a side has no task scored on both sides."""
    if not set(sb) & set(sc) or not set(hb) & set(hc):
        return None
    return gap_mod.analyze(sb, sc, hb, hc, suite_ids=suite_ids,
                           held_ids=held_ids, **kw)


def _coverage(res):
    return 0.0 if res is None else min(res["stats"]["coverage"].values())


def _mean(scores):
    return round(sum(scores.values()) / len(scores), 3) if scores else None


# --- lineage -----------------------------------------------------------------

TRIGGER = "invalid if the suite, evaluator or generator change"


def _lesson(entry):
    v = entry.get("verdict")
    s, h = entry.get("suite_delta"), entry.get("held_delta")
    gate = entry.get("gate")
    if gate == "validation":
        return "failed validation before being measured; %s" % TRIGGER
    if gate == "evaluation_error":
        return ("could not be measured (the evaluator failed on it); may be "
                "proposed again once the evaluator handles it; %s" % TRIGGER)
    if gate == "coverage":
        return ("the evaluator scored only %.0f%% of the tasks: nothing can be "
                "concluded; %s" % (entry.get("coverage", 0) * 100, TRIGGER))
    if gate == "family_regression":
        return ("mean rose but family %s dropped; a trade is not an "
                "improvement; %s" % (", ".join(entry.get("regressed", [])),
                                     TRIGGER))
    if v == "OVERFIT":
        return ("suite %+.2f without transfer (held %+.2f): fitted to the "
                "suite; do not repeat variations without evidence in the "
                "held-out; %s" % (s, h, TRIGGER))
    if v == "GENUINE":
        return "transferred (held %+.2f, suite %+.2f); %s" % (h, s, TRIGGER)
    if v == "NO IMPROVEMENT":
        return "no effect above noise; %s" % TRIGGER
    if v == "INCONCLUSIVE":
        return "measure did not decide; re-evaluate with more tasks; %s" % TRIGGER
    return "%s (suite %+.2f, held %+.2f); %s" % (v, s or 0, h or 0, TRIGGER)


def _history_view(entries):
    """What the proposer can see: nothing per task of the held-out, nothing
    that rebuilds the generator calls (seed, set size, family weights)."""
    return [{k: v for k, v in e.items() if k not in HIDDEN_FROM_PROPOSER}
            for e in entries]


def _fsync_dir(directory):
    if os.name != "posix":
        return
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_atomic(path, data):
    """Atomic (os.replace) and durable (fsync of the file, then of the
    directory, best effort): a crash leaves the old champion or the new one,
    never an empty or torn file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _judge(cand, cid, ctx):
    """Validates, measures one candidate and runs it through the gates.
    Returns (entry, config, measured). A failure while measuring rejects
    this candidate only (gate evaluation_error); the round goes on."""
    change = cand.get("change", "(no description)")
    cfg = {kk: vv for kk, vv in cand.items() if kk not in ("change", "kind")}
    entry = {"type": "candidate", "round": ctx["round"], "id": cid,
             "parent": ctx["champion_id"], "change": change,
             "config_hash": config_hash(cfg)}
    if "kind" in cand:
        entry["kind"] = cand["kind"]
    if ctx["validate"] is not None:
        try:
            ok, msg = ctx["validate"](cfg)
        except Exception as e:  # noqa: BLE001 - a crashing validator rejects
            ok, msg = False, "validator raised %s" % _short(e)
        if not ok:
            entry.update({"decision": "rejected", "gate": "validation",
                          "detail": msg})
            entry["lesson"] = _lesson(entry)
            return entry, cfg, False
    try:
        suite_c, held_c, _ = _measure(ctx["evaluator"], cfg, ctx["suite_tasks"],
                                      ctx["held_tasks"],
                                      run_seed(ctx["master"], "r%d/%s"
                                               % (ctx["round"], cid)), cid)
    except Exception as e:  # noqa: BLE001 - any failure of THIS candidate
        entry.update({"decision": "rejected", "gate": "evaluation_error",
                      "error": _short(e)})
        entry["lesson"] = _lesson(entry)
        return entry, cfg, True
    suite_b, held_b = ctx["suite_b"], ctx["held_b"]
    res = _analyze(suite_b, suite_c, held_b, held_c, ctx["suite_ids"],
                   ctx["held_ids"], calibration=ctx["calibration"],
                   n_boot=ctx["n_boot"])
    entry["coverage"] = round(_coverage(res), 4)
    if res is None or not res["coverage_ok"]:
        if res is not None:
            entry.update(_numbers(res))
        entry.update({"decision": "rejected", "gate": "coverage"})
        entry["lesson"] = _lesson(entry)
        return entry, cfg, True
    floor = ctx["floor"]
    fam = stats_mod.family_regressions(held_b, held_c, ctx["family_of"],
                                       floor_held=floor["held"],
                                       margin=ctx["family_margin"], n_boot=500)
    regressed = sorted(f for f, v in fam.items() if v["regressed"])
    # the suite is also a regression guard (Google blog): a family that
    # collapses on the suite counts, even if the held-out lacks n to see it
    if ctx["suite_family_of"]:
        fam_s = stats_mod.family_regressions(
            suite_b, suite_c, ctx["suite_family_of"], floor_held=floor["suite"],
            margin=ctx["family_margin"], n_boot=500)
        regressed = sorted(set(regressed) | {
            "%s (suite)" % f for f, v in fam_s.items() if v["regressed"]})
    entry.update(_numbers(res))
    entry["regressed"] = regressed
    if res["verdict"] != "GENUINE":
        entry.update({"decision": "rejected", "gate": "verdict"})
    elif regressed:
        entry.update({"decision": "rejected", "gate": "family_regression"})
    else:
        entry.update({"decision": "approved", "gate": None})
    entry["lesson"] = _lesson(entry)
    return entry, cfg, True


def _numbers(res):
    return {"verdict": res["verdict"], "verdict_point": res["verdict_point"],
            "suite_delta": res["suite_delta"], "held_delta": res["held_delta"],
            "gap": res["gap"],
            "ci_held": res["stats"]["bootstrap"]["ci"]["held_delta"],
            "ci_gap": res["stats"]["bootstrap"]["ci"]["gap"],
            "why": res["stats"]["why"]}


def _sanity(evaluator, good, bad, suite_tasks, held_tasks, master, r):
    """Good must beat bad (CI above zero, enough shared tasks, coverage) on
    the suite and on the round's held-out."""
    sg, hg, _ = _measure(evaluator, good, suite_tasks, held_tasks,
                         run_seed(master, "r%d/sanity-good" % r), "sanity-good")
    sb, hb, _ = _measure(evaluator, bad, suite_tasks, held_tasks,
                         run_seed(master, "r%d/sanity-bad" % r), "sanity-bad")
    sides, problems = {}, []
    for side, g, b, tasks in (("suite", sg, sb, suite_tasks),
                              ("held", hg, hb, held_tasks)):
        try:
            chk = sanity_mod.check(g, b, n_boot=500,
                                   expected=[t["id"] for t in tasks])
        except ValueError as e:
            chk = {"ok": False, "good_minus_bad": None, "ci": None,
                   "coverage": 0.0, "problems": [str(e)]}
        sides[side] = chk
        problems += ["%s: %s" % ("held-out" if side == "held" else side, p)
                     for p in chk["problems"]]
    return {"type": "sanity", "round": r,
            "ok": sides["suite"]["ok"] and sides["held"]["ok"],
            "suite_good_minus_bad": sides["suite"]["good_minus_bad"],
            "held_good_minus_bad": sides["held"]["good_minus_bad"],
            "ci_suite": sides["suite"]["ci"], "ci_held": sides["held"]["ci"],
            "coverage": {"suite": sides["suite"]["coverage"],
                         "held": sides["held"]["coverage"]},
            "problems": problems}


# --- the loop ----------------------------------------------------------------

def improve(initial, evaluator, proposer, catalog_dir, out_dir,
            suite_families=None, rounds=DEFAULT_ROUNDS, k=DEFAULT_K,
            budget=None, patience=DEFAULT_PATIENCE, n_held=DEFAULT_N_HELD,
            seed=None, sanity_good=None, sanity_bad=None, validate=None,
            final_exam=True, n_boot=1000, evaluator_id=None, proposer_id=None,
            family_margin=stats_mod.FAMILY_MARGIN, frontier_target=None,
            keep_work=False, log=print):
    """Runs the loop. `seed=None` draws the master seed with `secrets`.
    `keep_work=True` writes the held-out sets to <out>/rounds/rNN/heldout and
    <out>/final/heldout when the run ends (never during it).

    Returns {"champion", "champion_id", "entries", "summary", "final",
    "seed", "status"}; status is "accepted", "none_accepted" or "error"
    (changed evaluator, champion or placebo not scored). An exception (a
    failing proposer, an evaluator failing on the champion, ...) still
    writes the stop record and report.md before propagating."""
    for name, value in (("initial", initial), ("sanity_good", sanity_good),
                        ("sanity_bad", sanity_bad)):
        if (value is not None or name == "initial") and not isinstance(value, dict):
            raise ImproveError("%s harness must be a JSON object, got %s"
                               % (name, type(value).__name__))
    try:
        stats_mod.check_n_boot(n_boot)
    except ValueError as e:
        raise ImproveError(str(e)) from None
    if seed is None:
        master, seed_source = secrets.randbits(63), "random"
    elif isinstance(seed, bool) or not isinstance(seed, int):
        raise ImproveError("seed must be an integer, got %r" % (seed,))
    else:
        master, seed_source = seed, "given"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lineage = out_dir / "lineage.jsonl"
    if lineage.exists() and lineage.stat().st_size:
        raise ImproveError("a ledger already exists in %s; use another --out"
                           % lineage)
    suite_tasks, catalog_shingles = load_suite(catalog_dir, suite_families)
    suite_ids = [t["id"] for t in suite_tasks]
    suite_family_of = {t["id"]: t["family"] for t in suite_tasks
                       if t.get("family")}
    if evaluator_id is None:
        evaluator_id = _identity_of(evaluator)
    validate_id = _identity_of(validate) if validate is not None else None
    if proposer_id is None:
        proposer_id = _identity_of(proposer)

    def evaluator_changed():
        return (_identity_of(evaluator) != evaluator_id
                or (validate is not None
                    and _identity_of(validate) != validate_id))

    entries = []

    def record(entry):
        entry.setdefault("ts", _now())
        ledger_mod.append(lineage, entry)
        entries.append(entry)
        return entry

    champion = dict(initial)
    champion_id = "h0"
    accepted, pending = [], []
    evals_used = 0
    stop_reason, error = "rounds exhausted", False
    final = None
    kept_sets = {}  # destination under --out -> files (keep_work only)
    private = Path(tempfile.mkdtemp(prefix="hod-improve-"))  # mode 0700

    def skip_final(reason):
        if accepted and final_exam and final is None:
            return record({"type": "final_exam", "skipped": reason})
        return final

    try:
        _write_atomic(out_dir / "champion.json", champion)
        # scores of the initial champion on the suite: the suite is public,
        # the proposer can use them already in round 1 (held-out never)
        start_suite, _ = _clean_scores(
            evaluator(champion, suite_tasks, run_seed(master, "start")),
            suite_tasks, "initial/suite")
        record({"type": "start", "champion": champion_id,
                "champion_suite_scores": start_suite,
                "config_hash": config_hash(champion),
                "evaluator_sha": evaluator_id, "validate_sha": validate_id,
                "proposer_sha": proposer_id, "rounds": rounds, "k": k,
                "budget": budget, "patience": patience, "n_held": n_held,
                "family_margin": family_margin,
                "frontier_target": frontier_target,
                "seed_source": seed_source, "n_suite": len(suite_tasks)})
        log("improve: suite with %d tasks, up to %d rounds of %d candidates, "
            "held-out of %d per round" % (len(suite_tasks), rounds, k, n_held))

        last_suite = start_suite
        burned = []
        # Sealed final exam: generated before round 1 so that every round
        # excludes it, but held in memory only (its files are deleted at
        # once); the evaluator sees it only at final-exam time.
        final_set = None
        if final_exam:
            final_set = _generate_set(private, derive_seed(master, "heldout/final"),
                                      n_held, catalog_shingles)
            if final_set is None:
                raise ImproveError("could not generate the sealed final exam "
                                   "with %d tasks" % n_held)
            burned += final_set["shingles"]
            if keep_work:
                kept_sets[Path("final") / "heldout"] = final_set["files"]
        placebos = []
        stall = 0

        for r in range(1, rounds + 1):
            if evaluator_changed():
                stop_reason = "evaluator changed mid-loop (different hash)"
                error = True
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

            # new held-out, generated after the proposal, without tasks
            # already used. With frontier_target, the families near the
            # champion's frontier (measured on the public suite) get more
            # tasks (Ornith-1.5).
            weights = None
            if frontier_target is not None:
                fam_scores = {}
                for t in suite_tasks:
                    if t.get("family") and t["id"] in last_suite:
                        fam_scores.setdefault(t["family"], []).append(
                            last_suite[t["id"]] / 2.0)
                success = {f: sum(v) / len(v) for f, v in fam_scores.items()}
                weights = gen_mod.frontier_weights(success, target=frontier_target)
            hset = _generate_set(private, derive_seed(master, "heldout/r%d" % r),
                                 n_held, catalog_shingles, weights=weights,
                                 exclude=burned)
            if hset is None:
                stop_reason = ("held-out exhausted: the generator found no %d "
                               "tasks new and distinct from those already used"
                               % n_held)
                break
            burned += hset["shingles"]
            if keep_work:
                kept_sets[Path("rounds") / ("r%02d" % r) / "heldout"] = hset["files"]
            held_tasks = hset["tasks"]
            held_ids = [t["id"] for t in held_tasks]

            suite_b, held_b, cov_b = _measure(
                evaluator, champion, suite_tasks, held_tasks,
                run_seed(master, "r%d/champion" % r), "champion")
            last_suite = suite_b
            if cov_b < gap_mod.MIN_COVERAGE:
                stop_reason = ("the champion had no score in %.0f%% of the tasks"
                               % ((1 - cov_b) * 100))
                error = True
                break

            # placebo: the champion itself again
            suite_p, held_p, _ = _measure(
                evaluator, champion, suite_tasks, held_tasks,
                run_seed(master, "r%d/placebo" % r), "placebo")
            pr = _analyze(suite_b, suite_p, held_b, held_p, suite_ids, held_ids,
                          n_boot=200)
            if pr is None or not pr["coverage_ok"]:
                stop_reason = ("the placebo (champion measured again) was "
                               "scored on %.0f%% of the tasks"
                               % (_coverage(pr) * 100))
                error = True
                break
            placebos.append(record({
                "type": "placebo", "placebo": True, "round": r,
                "suite_delta": pr["suite_delta"], "held_delta": pr["held_delta"],
                "gap": pr["gap"]}))
            calibration = placebo_mod.stats(placebos)
            floor = stats_mod.noise_floor(calibration)

            # evaluator sanity on this held-out
            if sanity_good is not None and sanity_bad is not None:
                san = record(_sanity(evaluator, sanity_good, sanity_bad,
                                     suite_tasks, held_tasks, master, r))
                if not san["ok"]:
                    stop_reason = ("sanity failed: the evaluator does not "
                                   "separate the good harness from the bad one "
                                   "in this set (%s)" % "; ".join(san["problems"]))
                    break

            ctx = {"round": r, "champion_id": champion_id, "master": master,
                   "evaluator": evaluator, "validate": validate,
                   "suite_tasks": suite_tasks, "held_tasks": held_tasks,
                   "suite_ids": suite_ids, "held_ids": held_ids,
                   "suite_b": suite_b, "held_b": held_b,
                   "calibration": calibration, "floor": floor,
                   "family_of": {t["id"]: t["family"] for t in held_tasks},
                   "suite_family_of": suite_family_of,
                   "family_margin": family_margin, "n_boot": n_boot}
            results = []
            for i, cand in enumerate(candidates):
                cid = "h%d.%d" % (r, i + 1)
                (rdir / ("%s.json" % cid)).write_text(
                    json.dumps(cand, indent=2, ensure_ascii=False),
                    encoding="utf-8")
                entry, cfg, measured = _judge(cand, cid, ctx)
                evals_used += measured
                results.append(entry)
                # a final decision goes to the ledger at once; an approved
                # one waits for the others of the round
                if entry["decision"] == "approved":
                    pending.append((entry, cfg))
                else:
                    record(entry)

            # verify-then-commit: the evaluator that measured the round must
            # still be the one recorded at the start
            changed = evaluator_changed()
            winner = None
            if pending and not changed:
                winner = max(pending, key=lambda x: (x[0]["ci_held"][0],
                                                     x[0]["held_delta"]))
            for e, _cfg in pending:
                if changed:
                    e.update({"decision": "rejected", "gate": "evaluator_changed",
                              "lesson": "GENUINE, but the evaluator changed "
                              "during the round; may be proposed again"})
                elif e is winner[0]:
                    e["decision"] = "accepted"
                else:
                    e.update({"decision": "rejected", "gate": "another_better",
                              "lesson": "GENUINE, but another candidate in the "
                              "round had a higher held-out CI; may be proposed "
                              "again"})
                record(e)
            pending = []
            before = champion_id
            if winner is not None:
                champion = winner[1]
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
                    "champion_held_mean": _mean(held_b)})
            log("round %d: %s" % (r, "; ".join(
                "%s %s (%s)" % (e["id"], e.get("verdict", "-"), e["decision"])
                for e in results)))
            if changed:
                stop_reason = ("evaluator changed during round %d (different "
                               "hash); its candidates were not accepted" % r)
                error = True
                break
            if stall >= patience:
                stop_reason = "%d consecutive round(s) without acceptance" % stall
                break

        if accepted and final_exam:
            if not error and evaluator_changed():
                stop_reason = "evaluator changed before the final exam (different hash)"
                error = True
            if error:
                final = skip_final("not run: %s" % stop_reason)
            else:
                final_tasks = final_set["tasks"]
                final_ids = [t["id"] for t in final_tasks]
                s0, h0, _ = _measure(evaluator, initial, suite_tasks, final_tasks,
                                     run_seed(master, "final/initial"),
                                     "final/initial")
                s1, h1, _ = _measure(evaluator, champion, suite_tasks,
                                     final_tasks,
                                     run_seed(master, "final/champion"),
                                     "final/champion")
                res = _analyze(s0, s1, h0, h1, suite_ids, final_ids,
                               calibration=placebo_mod.stats(placebos),
                               n_boot=n_boot)
                final = {"type": "final_exam", "from": "h0", "to": champion_id,
                         "n_held": len(final_tasks),
                         "coverage": round(_coverage(res), 4)}
                if res is None:
                    final["verdict"] = "INCONCLUSIVE"
                else:
                    final.update({
                        "verdict": res["verdict"],
                        "suite_delta": res["suite_delta"],
                        "held_delta": res["held_delta"], "gap": res["gap"],
                        "ci_held": res["stats"]["bootstrap"]["ci"]["held_delta"],
                        "ci_gap": res["stats"]["bootstrap"]["ci"]["gap"]})
                record(final)
    except BaseException as e:
        error = True
        stop_reason = "aborted: %s" % _short(e)
        raise
    finally:
        try:
            for e, _cfg in pending:  # approved, but the round never closed
                e.update({"decision": "rejected", "gate": "round_aborted",
                          "lesson": "approved, but the run stopped before the "
                          "round chose its winner; may be proposed again"})
                record(e)
            if final is None and error:
                final = skip_final("not run: %s" % stop_reason)
            status = "error" if error else (
                "accepted" if accepted else "none_accepted")
            summary = {"type": "stop", "reason": stop_reason, "status": status,
                       "champion": champion_id,
                       "accepted": [e["id"] for e in accepted],
                       "candidate_evals": evals_used,
                       "rounds_run": len([e for e in entries
                                          if e["type"] == "round"]),
                       # the master seed appears only now, after the last
                       # proposer call: reproducible, never predictable
                       "seed": master, "seed_source": seed_source}
            record(summary)
            (out_dir / "report.md").write_text(render(entries), encoding="utf-8")
            for dest, files in kept_sets.items():
                _write_files(out_dir / dest, files)
        finally:
            shutil.rmtree(private, ignore_errors=True)
    log("improve: stopped (%s); champion %s; accepted: %s"
        % (stop_reason, champion_id, ", ".join(summary["accepted"]) or "none"))
    return {"champion": champion, "champion_id": champion_id,
            "entries": entries, "summary": summary, "final": final,
            "seed": master, "status": status}


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _candidate_order(e):
    try:
        return (e["round"], int(str(e["id"]).rsplit(".", 1)[1]))
    except (IndexError, ValueError):
        return (e.get("round", 0), 0)


def render(entries):
    lines = ["# hod improve: lineage", ""]
    start = next((e for e in entries if e["type"] == "start"), {})
    lines.append("Suite: %s tasks | held-out per round: %s | k=%s | "
                 "budget: %s | patience: %s"
                 % (start.get("n_suite"), start.get("n_held"), start.get("k"),
                    start.get("budget") or "no limit", start.get("patience")))
    lines.append("Evaluator (sha256): `%s`" % str(start.get("evaluator_sha"))[:16])
    cands = sorted((e for e in entries if e["type"] == "candidate"),
                   key=_candidate_order)
    lines += ["", "| Round | Candidate | Change | Suite Δ | Held Δ (CI 95%) | "
              "Verdict | Decision | Gate |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in cands:
        ci = e.get("ci_held")
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            e["round"], e["id"], _cell(e["change"]),
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
            gmb = e.get("held_good_minus_bad")
            lines.append("- Sanity round %d: %s (good − bad in the held-out %s)%s"
                         % (e["round"], "ok" if e["ok"] else "FAILED",
                            "%+.3f" % gmb if gmb is not None else "-",
                            "" if e["ok"] or not e.get("problems")
                            else ": " + "; ".join(e["problems"])))
    final = next((e for e in entries if e["type"] == "final_exam"), None)
    if final:
        lines += ["", "## Final exam (unseen held-out)", ""]
        if final.get("skipped"):
            lines.append("Skipped: %s" % final["skipped"])
        elif "held_delta" in final:
            lines.append(
                "%s -> %s: **%s** | suite %+.3f | held-out %+.3f CI [%+.3f, "
                "%+.3f] | gap %+.3f (n=%d, coverage %.0f%%)"
                % (final["from"], final["to"], final["verdict"],
                   final["suite_delta"], final["held_delta"],
                   final["ci_held"][0], final["ci_held"][1], final["gap"],
                   final["n_held"], final.get("coverage", 1.0) * 100))
        else:
            lines.append("%s -> %s: **%s** (coverage %.0f%%: no task scored "
                         "on both sides)" % (final["from"], final["to"],
                                             final["verdict"],
                                             final.get("coverage", 0) * 100))
    stop = next((e for e in entries if e["type"] == "stop"), None)
    if stop:
        lines += ["", "Stopped: %s. Final champion: %s. Accepted: %s."
                  % (stop["reason"], stop["champion"],
                     ", ".join(stop["accepted"]) or "none")]
        if stop.get("status") == "error":
            lines.append("Status: ERROR (exit code 1).")
        if "seed" in stop:
            lines.append("Master seed: %s (%s)." % (stop["seed"],
                                                    stop.get("seed_source")))
    lines += ["", "## Lessons", ""]
    for e in cands:
        lines.append("- %s (%s): %s%s" % (
            e["id"], _cell(e["change"]), e.get("lesson"),
            (" [error: %s]" % e["error"]) if e.get("error") else ""))
    return "\n".join(lines) + "\n"


def _check_args(args):
    problems = []
    for name in ("rounds", "k", "patience", "n_held"):
        if getattr(args, name) < 1:
            problems.append("--%s must be >= 1" % name.replace("_", "-"))
    if args.budget is not None and args.budget < 1:
        problems.append("--budget must be >= 1")
    if args.timeout <= 0:
        problems.append("--timeout must be > 0")
    if args.boot < stats_mod.MIN_BOOT:
        problems.append("--boot must be >= %d" % stats_mod.MIN_BOOT)
    if args.family_margin < 0:
        problems.append("--family-margin must be >= 0")
    if args.frontier_target is not None and not 0 <= args.frontier_target <= 1:
        problems.append("--frontier-target must be between 0 and 1")
    if (args.sanity_good is None) != (args.sanity_bad is None):
        problems.append("--sanity-good and --sanity-bad go together")
    return problems


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(
        prog=prog, description="self-improvement loop: only accepts "
        "GENUINE changes measured on a new held-out each round")
    parser.add_argument("--initial", type=Path, required=True,
                        help="initial harness config (JSON object)")
    parser.add_argument("--eval", dest="eval_cmd", required=True,
                        help="evaluator command with {config} {tasks} {out} "
                        "and optionally {seed}")
    parser.add_argument("--propose", dest="propose_cmd", required=True,
                        help="proposer command with {out} and optionally "
                        "{champion} {history} {k} {round}")
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
    parser.add_argument("--seed", type=int, default=None,
                        help="master seed of the held-out sets and evaluator "
                        "runs. Default: drawn at random (secrets) and written "
                        "only to the final stop record of lineage.jsonl. "
                        "Given explicitly, the run is reproducible, but a "
                        "proposer who knows it can predict every held-out "
                        "set, the sealed final exam included")
    parser.add_argument("--sanity-good", type=Path, default=None,
                        help="config JSON of a known-good harness")
    parser.add_argument("--sanity-bad", type=Path, default=None,
                        help="config JSON of a known-bad harness")
    parser.add_argument("--validate", default=None,
                        help="command with {config}; must exit 0 (part of "
                        "the evaluator hash)")
    parser.add_argument("--evaluator-file", type=Path, action="append",
                        default=[], metavar="PATH",
                        help="file or directory (recursive) that is part of "
                        "the evaluator: its content enters the evaluator hash "
                        "(repeatable). The hash already covers the --eval and "
                        "--validate commands, files named in them and "
                        "programs they call found on PATH")
    parser.add_argument("--keep-work", action="store_true",
                        help="keep the evaluator work directories and the "
                        "held-out sets, written to --out/work, "
                        "--out/rounds/rNN/heldout and --out/final/heldout "
                        "only when the run ends (during the run the sets stay "
                        "in memory and the work directories in a private "
                        "temporary directory outside --out, still readable by "
                        "your OS user)")
    parser.add_argument("--no-final-exam", action="store_true")
    parser.add_argument("--boot", type=int, default=1000,
                        help="bootstrap replicas, >= %d" % stats_mod.MIN_BOOT)
    parser.add_argument("--family-margin", type=float,
                        default=stats_mod.FAMILY_MARGIN,
                        help="maximum tolerated drop per family in the held-out "
                        "(points, default %(default)s)")
    parser.add_argument("--frontier-target", type=float, default=None,
                        help="enables the frontier held-out: families with "
                        "champion success near this target (e.g. 0.35) "
                        "get more tasks")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="seconds per command call (the command and its "
                        "children are killed past it)")
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # argparse exits 2 on a usage error; here 2 means "nothing accepted"
        if e.code == 2:
            return 1
        raise
    problems = _check_args(args)
    if problems:
        for p in problems:
            print("improve: %s" % p, file=sys.stderr)
        return 1
    work = args.out / "work"
    evaluator = None
    try:
        # every template is checked before any work is done
        evaluator = CommandEvaluator(args.eval_cmd, work, args.timeout,
                                     keep=args.keep_work,
                                     files=args.evaluator_file)
        proposer = CommandProposer(args.propose_cmd, work, args.timeout)
        validate = (command_validator(args.validate, work, args.timeout)
                    if args.validate else None)
        for f in args.evaluator_file:
            if not f.exists():
                raise ImproveError("evaluator file missing: %s" % f)
        initial = read_json(args.initial, "initial harness", expect=dict)
        families = (read_json(args.suite_families, "suite families file",
                              expect=dict) if args.suite_families else None)
        good = (read_json(args.sanity_good, "sanity-good harness", expect=dict)
                if args.sanity_good else None)
        bad = (read_json(args.sanity_bad, "sanity-bad harness", expect=dict)
               if args.sanity_bad else None)
        result = improve(initial, evaluator, proposer, args.catalog, args.out,
                         suite_families=families, rounds=args.rounds, k=args.k,
                         budget=args.budget, patience=args.patience,
                         n_held=args.n_held, seed=args.seed, sanity_good=good,
                         sanity_bad=bad, validate=validate,
                         final_exam=not args.no_final_exam, n_boot=args.boot,
                         family_margin=args.family_margin,
                         frontier_target=args.frontier_target,
                         keep_work=args.keep_work)
    except (ImproveError, ValueError, OSError) as e:
        print("improve: %s" % e, file=sys.stderr)
        if (args.out / "report.md").exists():
            print("report: %s" % (args.out / "report.md"), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("improve: interrupted", file=sys.stderr)
        return 130
    finally:
        if evaluator is not None:
            try:
                evaluator.close()
            except OSError as e:
                print("improve: could not move the evaluator work "
                      "directories: %s" % e, file=sys.stderr)
    print("report: %s" % (args.out / "report.md"))
    if result["status"] == "error":
        print("improve: stopped on an error: %s" % result["summary"]["reason"],
              file=sys.stderr)
        return 1
    return 0 if result["summary"]["accepted"] else 2


if __name__ == "__main__":
    sys.exit(main())
