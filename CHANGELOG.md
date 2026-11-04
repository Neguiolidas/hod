# Changelog

## Unreleased

Audit of v0.1.1: corrections and improvements. Some verdicts change on
purpose (marked **behavior change**); every fix has a regression test in
`tests/test_fixes_core.py`, `tests/test_fixes_generator.py` or
`tests/test_fixes_improve.py`.

### Measurement core (stats, gap, scores, sanity, agree, placebo)

- **Behavior change:** a side with fewer than 6 paired tasks (`stats.MIN_N_CI`)
  can no longer be "up" or "down". Before, one task per side gave GENUINE
  with a zero-width CI, and under a true null 2 tasks excluded zero ~47% of
  the time. Such cases are now INCONCLUSIVE; a significant drop on a side
  that has enough tasks is still REGRESSION.
- **Behavior change:** coverage below 90% now makes `verdict` INCONCLUSIVE
  under `--decision point` too (`verdict_point` keeps the raw rule).
- Coverage can count tasks missing from both score files: `gap --manifest`
  (and `analyze(held_ids=...)`, used by the pipeline) sets the denominator;
  before, 10 scored tasks out of 40 read as 100% coverage.
- An effect equal to the placebo floor at its stored precision (3 decimals)
  is a tie, not "above the noise": a placebo of 0.333 no longer lets an
  identical 1/3 effect through as GENUINE / significant (p_gap is 1.0).
- Verdicts are decided on unrounded values (`bootstrap["exact"]`); a transfer
  of 74.994% no longer rounds up to GENUINE.
- `--boot` / `n_boot` must be >= 200 (0 crashed, negative values gave an
  invented verdict, 1 built a "CI" from one replica).
- Old calibrations with only `max_abs_gap` no longer give a silent zero noise
  floor; the floor falls back to the gap, as the significance check did.
- Per-family regression guard resamples in sorted id order: the same data in
  another JSON key order no longer flips `regressed`. Its rule labels are now
  `ci` / `margin` (were `ic` / `margem`).
- `n_held_needed` is a true ceiling; the GENUINE reason no longer prints a
  literal `95%%`.
- Score files: a duplicate task id is an error (it kept one of two
  contradictory scores). All JSON inputs go through `scores.read_json`:
  missing files, directories, bad JSON and wrong top-level types give a
  one-line error instead of a traceback in `gap`, `sanity`, `agree`,
  `placebo`, `ledger`, `prereg` and `pipeline`.
- New `hod/manifest.py`: one validator for manifests (unique ids and files,
  task file names that cannot leave the set directory).
- `sanity` requires at least 6 shared tasks and 90% coverage (1 shared task
  used to pass), and lists why it failed.
- `placebo` ignores malformed ledger entries: bool or NaN gaps, and
  `"placebo": "false"` as a string (only a literal `true` is a placebo, here
  and in `hod ledger`).

### Ledger, prereg, pipeline

- Ledger: a line torn by a crashed writer no longer swallows the next entry
  (the torn line is closed before appending); a UTF-8 character cut in half
  no longer crashes `hod ledger` / `hod placebo`; best-effort lock on Windows
  (`<ledger>.lock`, untested in CI).
- Prereg: `hod prereg --seeds 42,43,44` commits to the generation salts in
  advance; bound sets must use exactly those salts, all of them, which blocks
  "seed shopping" (generate many bound sets, keep the best). `prereg` files
  are validated on load; `--min-ratio` outside [0, 1] and a negative
  `--flat-tol` are clean errors.
- Pipeline: refuses a mix of bound and unbound sets (it used to fall back to
  the weak mtime check for all of them) and a bound `seed-N` directory whose
  salt is not N; coverage per seed and pooled is measured against the
  manifests; `--wave1` must be >= 1, with a warning when it is too small for
  the CI decision to ever stop; malformed manifests, calibration, categories
  and prereg files are one-line errors.

### Packaging and CI

- `pyproject.toml`: SPDX `license = "Apache-2.0"` plus `license-files`
  (setuptools announced that the table form stops building after
  2027-02-18); classifiers for Python 3.13 and 3.14.
- `hod.__version__` is a literal checked against `pyproject.toml` by the
  tests; reading it from installed metadata reported a stale installed copy
  when running from a checkout.
- CI: test matrix 3.10 to 3.14; new `package` job builds the wheel, installs
  it in a clean venv outside the checkout and runs `hod --version` and
  `hod demo`. Release: a manual run is a build-only dry run (it used to fail
  the tag check), and only a `v*` tag publishes.

## v0.1.1 — 2026-10-23

- README: installation now leads with `pip install hod` (the package is on PyPI); the Layout section no longer lists `runs/` as shipped content (`runs/` is local output, gitignored); the Canary section states the canary runner is external to this package and documents the harness-side variables it uses.
- CI: new `test.yml` workflow (pytest on Python 3.10/3.11/3.12 plus pyflakes lint); the release workflow now runs the test job first and publishes only if it passes.
- Catalog default generalized: `--catalog` now defaults to `~/hod/catalog` instead of a path inside the author's workspace; the previous default only existed on the author's machine.
- Lint fixes: removed an unused `re` import (`audit.py`) and an unused variable assignment (`demo.py`).

## v0.1.0 — 2026-10-22

Audit corrections and improvements incorporated before launch, in the
subsections below.

- Held-out task generator (`hod generate`): seeded generation of new tasks in
  catalog format (setup, instruction, expected behavior, auto-fail, 0–2
  rubric) from 8 families, with a similarity gate against the fixed suite
  (5-gram Jaccard, fail-closed) and a manifest with ids, families and
  hashes. Same seed reproduces the same set.
- Gap measurement (`hod gap`): takes suite and held-out scores before and
  after a change and reports means, deltas, the gap (suite minus held-out)
  and a verdict: GENUINE, PARTIAL, OVERFIT, or NO IMPROVEMENT. Small held-out
  samples are flagged as indicative.
- Contamination audit (`hod audit`): verifies manifest and hashes, checks
  cross-family near-duplicates, duplicate tasks (setup + instruction) and
  held-out similarity against the suite.
- Packaging: unified `hod` CLI
  (generate, gap, audit sections). From a checkout it runs without
  installing via `python3 -m hod <section>`; the canary scripts use the
  checkout directly.
- Robustness layer: hypothesis preregistration with predicted-vs-observed
  check and peeking refusal; difficulty calibration and per-task Q
  (validity x difficulty x novelty) with saturated-task filtering and a
  NO SIGNAL report; per-category gaps; multi-seed measurement with
  stability verdicts; placebo controls with empirical noise calibration
  and significance flags; a JSONL change ledger with adaptive family
  weights; blind scoring with key-based unblinding; second-rater and
  retest agreement (Cohen's kappa); wave-1 gating; and a `hod pipeline`
  command running all layers end to end. Canary hooks: multi-seed
  generation, blind templates, frozen prereg copies and automatic ledger
  appends, all event-triggered, no cron, no daemon.
- Canary integration: the harness canary generates a held-out set on every
  run, reports the held-out table, gap and verdict next to the suite table,
  records seed and candidate hash, stays backward compatible with old runs
  and degrades softly if generation or the verdict computation fails.
  The verdict is advisory by default; `HELDOUT_GATE=blocking` makes OVERFIT
  block.
- Synthetic positive controls (4/4): planted overfitting is flagged, genuine
  improvement passes, partial and no-improvement cases are correct.
- First real baseline: 8 held-out tasks, all scored 2 (saturated at the
  ceiling, same as the fixed suite). At the ceiling, the instrument's value
  is regression and overfit detection, not improvement measurement.

Validation status (honest): synthetic controls pass and one real baseline
exists. Real-world validation is still pending: second rater on baseline
scores, test-retest noise floor, controls on real harness changes, and
promotion of the blocking gate after 3–5 changes with a stable signal.

### Corrections incorporated before launch

Note: relative to the initial design, the same seed generates a different
set (sampling without replacement and gate with containment).

- generate: variants sampled without replacement (with --n 8, v0.1.0 repeated
  a task in ~55% of seeds); fail-closed when --n exceeds the number of
  distinct tasks, leaving no partial files; similarity gate =
  max(Jaccard, containment); NFC normalization + casefold with any Unicode
  letter; short text does not become an empty set; --family-weights rejects
  keys that are not families; --prereg derives the seed from the prereg hash.
- gap: paired deltas (only tasks with base and candidate), error on an empty
  set, float tolerance at boundaries, REGRESSION verdict (suite flat,
  held-out fell), gap significance against the placebo (with empirical
  p-value) separate from effect significance; with no placebos nothing is
  declared significant; score files accept the `hod unblind` format.
- audit: explicit checks (no assert, holds under `python -O`), clean error
  for missing manifest, empty catalog and empty set.
- blind: cryptographically sourced key by default, answer key in t01..tNN
  order (the order does not reveal the key), unblind requires a complete
  answer key.
- ledger: append with lock and fsync, corrupted line ignored with a warning,
  per-family weights (by_family field) with shrinkage, not counting wave 1
  nor a gap within noise.
- placebo: ignores wave 1, records suite_delta and held_delta noise.
- prereg: "held rises" with a flat suite can match again; --flat-tol;
  strong verification by hash + derived seed.
- pipeline: strong binding to the prereg when the set was generated with
  --prereg; a score file with an id outside the manifest is an error;
  refuses held-base equal to the calibration filter scores; wave 1
  stratified by family; seeds in numeric order; ledger with by_family and
  significance.
- calibrate: keeps the filter scores in the manifest, cleans leftovers in
  --out, refuses --out equal to the source, exit code 2 when everything
  saturates.
- agree: validates scores, quadratic weighted kappa, marks kappa undefined.
- tests: the 3 script-format files are now collected by pytest;
  tests/test_auditoria.py with one test per bug.

### Improvements incorporated before launch

Note: relative to the initial design, the same seed generates a different set
(new and parametric families); the default verdict of `gap` and `pipeline`
is now by confidence interval and may yield INCONCLUSIVE (`--decision point`
returns to the initial design's rule).

- families: 8 parametric families (2 new: verify-first and
  ambiguity), 13,272 combinations, import-time check that every
  axis appears in the text; `VARIANTS[f]` is an on-demand computed
  indexable space.
- generate: validity gate, novelty gate (setup + instruction against the
  set and the `--exclude` buffer), `--seeds` batch seeds disjoint from
  each other, manifest with `novelty`, `rejected`, `space_size`; frontier
  sets (`--frontier`, Ornith-1.5 weights) and stratified-quota weights
  instead of family-by-family sampling.
- stats (new): paired bootstrap with 95% CI, placebo noise floor,
  power and required n, CI verdict, per-family regression guard.
- gap: `verdict_ci` (default), `verdict_point`, `stats`, minimum 90%
  coverage, `--decision`, `--boot`, `--strict`; report with CI table.
- pipeline: `--decision`, pooled verdict with all seeds together
  (`pooled` in `summary.json`), new ledger fields.
- calibrate: Ornith frontier (`frontier`, `q_frontier`, `--target`,
  `--frontier-out`).
- sanity (new): does the evaluator separate a good harness from a bad one?
- improve (new): self-improvement loop with pluggable proposer and evaluator,
  rotating held-out, per-round placebo, sanity, verify-then-commit gates,
  lineage ledger with lessons, budget and stopping, sealed final proof.
- toy + demo (new): deterministic toy harness and `hod demo`.
- tests: 69 (5 in the initial design, 37 after the corrections).
