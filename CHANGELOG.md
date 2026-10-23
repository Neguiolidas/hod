# Changelog

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
