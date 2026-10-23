# Harness Overfit Detector (HOD)

**Detect when an agent-harness improvement is real and when it is just fitted to the eval suite.**

Every harness improvement is usually measured on the same fixed set of tasks. A change tuned to those tasks can raise the score without improving anything in general, and nothing detects it. HOD adds a second lens: tasks the change was never tuned for, and the gap between the two measurements.

## Installation

From PyPI (Python 3.10 or newer):

```bash
pip install hod
```

From the repo checkout (no install needed, run from the repo root):

```bash
python3 -m hod --help
```

Editable install from a checkout, for the `hod` command:

```bash
pip install -e .
```

## Usage

The `hod` command has three measurement sections, a robustness layer and an automation layer (including the `improve` self-improvement loop). From a clone without installing, replace `hod` with `python3 -m hod` in every command below. Paths under `runs/` in the examples are outputs the commands create on your machine; `runs/` is not part of the checkout.

### generate: held-out tasks

Generates new tasks in catalog format (setup, instruction, expected behavior, auto-fail, 0–2 rubric) from 8 parametric families, reproducibly by seed. Each family is a set of setup and instruction templates times parameter axes (files, values, requests, material differences), so the space is 13,272 combinations instead of the 18 fixed variants of the initial design. Every candidate task passes three gates before it is emitted:

- validity: no empty field, no unfilled placeholder, complete 0/1/2 rubric;
- similarity against the fixed suite (max of Jaccard and containment on 5-grams, ceiling 0.30);
- novelty: setup + instruction compared with every task already in the set and with an exclusion buffer (`--exclude DIR`, sets already used), Jaccard on 3-grams, ceiling 0.60.

If no candidate passes, generation fails closed instead of emitting a contaminated or duplicated task. Under the novelty gate the space is estimated at roughly 1,500 mutually distinct tasks (about 70 to 800 per family). With `--seeds`, each seed's set excludes the previous seeds' tasks, so multi-seed sets are disjoint.

```bash
# Generate 8 held-out tasks (seed 42 reproduces the same set)
hod generate --seed 42 --n 8 --out runs/heldout-42

# Against a different fixed suite
hod generate --seed 42 --n 8 --out runs/heldout-42 --catalog path/to/catalog

# A fresh set that shares nothing with sets already used
hod generate --seed 43 --n 40 --out runs/heldout-43 --exclude runs/heldout-42

# Frontier sets (Ornith-1.5 style): families where the baseline succeeds
# near the target rate get more tasks; saturated families keep a floor
hod calibrate --heldout runs/heldout-42 --baseline held-base.json --frontier-out frontier.json
hod generate --seed 44 --n 40 --out runs/heldout-44 --frontier frontier.json --frontier-target 0.35
```

With weights (`--family-weights` or `--frontier`), each family gets a quota proportional to its weight (largest remainder), so the set stays stratified.

The output directory holds one `.md` file per task plus `manifest.json` (id, family, file, sha256, max similarity).

### gap: suite vs held-out

From suite and held-out scores before and after a change, reports means, deltas, the gap (suite minus held-out), 95% confidence intervals from a paired bootstrap, the placebo noise floor, coverage, power (minimum detectable effect and the held-out size you would need) and a verdict. Score files are JSON maps of `{task_id: 0|1|2}`. `--decision point` reproduces the 0.1 point rule; `--strict` asks for more evidence before GENUINE.

```bash
hod gap --suite-base scores/suite-base.json --suite-cand scores/suite-cand.json \
  --held-base scores/held-base.json --held-cand scores/held-cand.json --out gap-report.md
```

### audit: contamination check

Verifies manifest and hashes, cross-family near-duplicates, duplicate tasks (setup + instruction) and held-out similarity against the suite.

```bash
hod audit --heldout runs/heldout-42 [--catalog path/to/catalog]
```

## Robustness

Beyond the three measurement sections, HOD has a robustness layer. Every layer is optional except through `hod pipeline`, which runs all of them.

### prereg: hypothesis first

The change author declares what they expect before seeing the held-out set. The pipeline refuses to measure if the prereg file is newer than the generated sets (peeking). After the gap, the prediction is checked against the observation.

```bash
hod prereg --change "new routing rule" --expect-held up --min-ratio 0.75 --out prereg.json
hod gap ... --prereg prereg.json
```

### calibrate: difficulty and Q

Baseline scores on the candidate tasks reveal which tasks discriminate and which are saturated. Difficulty is (2 - baseline) / 2, novelty is 1 - max similarity, and Q is validity x difficulty x novelty. Saturated tasks (baseline 2) are filtered out; if all saturate, the report says NO SIGNAL instead of measuring a gap on noise.

```bash
hod calibrate --heldout runs/heldout-42 --baseline held-base.json \
  --out runs/heldout-42-cal --report runs/cal-42.md
```

### placebo: empirical noise

A placebo run is a pipeline run with a no-effect change. Its gap is pure noise. `hod placebo` aggregates placebo gaps from the ledger into a calibration file, and `hod gap --calibration` flags whether an effect is bigger than any observed no-op effect.

```bash
hod placebo --ledger runs/ledger.jsonl --out calibration.json
hod gap ... --calibration calibration.json
```

### ledger: the feedback loop

Every pipeline run and every canary report appends one JSONL line to the ledger (change, verdicts, gap, hypothesis, per-category verdicts). `hod ledger` reports totals, overfit rate and family weights: weight is 1 + the family's overfit rate, so the next generation looks harder where overfitting appeared most. `hod generate --family-weights weights.json` uses them; same seed plus same weights reproduces the same set.

```bash
hod ledger --ledger runs/ledger.jsonl --weights-out weights.json
```

### sanity: does the evaluator separate good from bad?

Before trusting any gap, score a known-good and a known-bad harness on the same tasks. `hod sanity` passes only if good − bad has a 95% CI above zero (and, with `--manifest`, if no family has good ≤ bad). An evaluator that accepts both versions measures nothing.

```bash
hod sanity --good scores-good.json --bad scores-bad.json --manifest runs/heldout-42/manifest.json
```

### agree: second rater and retest

`hod agree` compares two score files (second rater, or the same set scored later) and reports percent equal, mean absolute difference and Cohen's kappa. The mechanism is ready; the second rater's actual scores are future data.

```bash
hod agree --a scores-rater1.json --b scores-rater2.json
```

### blind: anonymous scoring

`hod blind` replaces task ids with shuffled t01..tNN ids and writes the key elsewhere; `hod unblind` restores real ids after scoring. Honest limit: with a single agent, the key exists on disk and the blinding is instruction-enforced (do not open the key); with a second rater who never sees the key, it is real blinding.

```bash
hod blind --heldout runs/heldout-42 --template run/heldout-blind.json --key keys/heldout-key.json
hod unblind --scores run/heldout-blind.json --key keys/heldout-key.json --out run/heldout.json
```

## Automation

One command runs all layers end to end: prereg timing check, wave 1 gate, multi-seed gaps, hypothesis check, placebo significance, ledger append, report. Exit codes: 0 completed, 3 stopped at wave 1 (OVERFIT, no expansion), 1 validation failure.

```bash
hod pipeline --change "new routing rule" --prereg prereg.json \
  --heldout-dir runs/heldout --scores-dir runs/scores \
  --seeds 42,43,44 --ledger runs/ledger.jsonl --out runs/pipe-01
```

Because multi-seed sets are now disjoint, the pipeline also pools all held-out tasks into one bootstrap and reports the pooled CI verdict (`pooled` in `summary.json`).

### improve: self-improvement loop

`hod improve` runs a proposer (your command) that suggests harness changes and accepts only the ones HOD calls GENUINE. Each round: it checks that the evaluator has not changed (hash), asks the proposer for k candidates (the proposer sees the champion, its public suite scores and aggregate history, never held-out tasks), generates a fresh held-out set that excludes every task used before, measures the champion and a placebo rerun of it (noise floor), optionally checks evaluator sanity, then runs each candidate through the gates in order: your `--validate` command, coverage, CI verdict = GENUINE, no family regression on the held-out or the suite. The champion is replaced (atomic write) only after all gates pass. A sealed final-exam set, generated before round 1 and excluded from every round, compares the final champion with the initial harness at the end. Everything goes to `lineage.jsonl`: each candidate, the gate that stopped it, its numbers and a lesson with its invalidation trigger.

With `--frontier-target 0.35`, each round's held-out set leans toward the families where the champion's public suite success is near the target (the sealed final exam stays uniform).

Stop rules: `--rounds`, `--budget` (candidate evaluations), `--patience` (rounds without an acceptance), an empty proposal, held-out exhausted, failed sanity, changed evaluator.

```bash
hod improve --initial harness.json \
  --eval "my-eval --config {config} --tasks {tasks} --seed {seed} --out {out}" \
  --propose "my-proposer --champion {champion} --history {history} --k {k} --round {round} --out {out}" \
  --catalog path/to/catalog --out runs/rsi-01 \
  --sanity-good good.json --sanity-bad bad.json --validate "my-check {config}"
```

`{tasks}` is a JSON list of `{"id", "family", "text"}`; the evaluator writes `{task_id: 0|1|2}`; the proposer writes a JSON list of candidate configs (each may carry `"change"`, a description). Exit codes: 0 something accepted, 2 nothing accepted, 1 error.

### demo: offline proof

`hod demo --out runs/demo [--sweep 30] [--via-commands]` runs `improve` against a deterministic toy harness (`hod/toy.py`: per-family skill knobs that transfer, a "memorize the suite" knob that does not, a cosmetic knob) with a scripted proposer that offers a genuine, an overfit, a placebo, a trade-off (one family zeroed) and a mixed change every round. No network, no model.

Hooks, not cron and not a daemon: automation is event-triggered. The canary run generates multi-seed sets, blind templates and the frozen prereg copy; the canary report computes per-seed and per-category gaps, calibrates against ledger placebos and appends to the ledger by itself. Nothing polls, nothing runs on a schedule.

## How the verdict works

Default (`--decision ci`). "Up" means the 95% CI is above zero and the point estimate is above the placebo noise floor; "down" is the mirror image. t50 = held − ½·suite and t75 = held − ¾·suite test the transfer as linear combinations, which stay stable when the suite gain is small.

| Condition | Verdict |
| --- | --- |
| suite not up, and suite or held-out down | REGRESSION |
| suite not up, point estimate above the floor but CI crosses zero | INCONCLUSIVE |
| suite not up | NO IMPROVEMENT |
| suite up, held-out down, or t50 CI entirely below zero | OVERFIT |
| suite up, held-out up, t75 ≥ 0 (with `--strict`: also t50 CI above zero) | GENUINE |
| suite up, held-out up, otherwise | PARTIAL |
| suite up, held-out CI does not separate transfer from overfit | INCONCLUSIVE |

Coverage below 90% (tasks without a score) makes the verdict INCONCLUSIVE: a missing score is "not run", never "passed". The 0.1 point rule is still in every report as `verdict_point`:

| Suite delta | Held-out delta | Point verdict |
| --- | --- | --- |
| ≤ 0 | held-out falls | REGRESSION |
| ≤ 0 | otherwise | NO IMPROVEMENT |
| > 0 | ≤ 0, or less than half the suite gain | OVERFIT |
| > 0 | between half and three quarters of the suite gain | PARTIAL |
| > 0 | at least three quarters of the suite gain | GENUINE |

With a small held-out sample (under 8 tasks) the point verdict is flagged as indicative, and the CI verdict warns below 30 tasks and estimates the size you would need.

Held-out tasks must never enter the fixed suite: once they do, they stop being held-out. The candidate change must be frozen before the held-out set is generated; if it changes after, the run is invalid.

## Canary integration

HOD does not ship a canary runner: the canary is the harness that embeds HOD, not this package. On every harness run, the harness wrapper calls `hod generate` to build a fresh held-out set and `hod gap` / `hod audit` to report the held-out table, gap and verdict next to the suite table. The verdict is advisory by default; `HELDOUT_GATE=blocking` makes `OVERFIT` (aggregate) block. Old runs without held-out data still report normally, and a failed generation or verdict computation degrades softly instead of breaking the suite gate. A harness canary typically wires HOD like this: it uses the repo checkout directly (`PYTHONPATH=<repo> python3 -m hod generate`), so it works without a pip install. With `HOD_SEEDS="42,43,44"` it generates one set per seed and reports the aggregate verdict and stability; with `HOD_BLIND=1` it writes blind templates and keeps keys outside the run dirs; with a prereg file argument it freezes the hypothesis copy in the run dirs. The report appends to `runs/ledger.jsonl` by itself unless `HOD_NO_LEDGER=1`. The `HELDOUT_GATE`, `HOD_SEEDS`, `HOD_BLIND` and `HOD_NO_LEDGER` variables are read by the harness-side wrapper, not by HOD itself (the package contains no reference to them); HOD only exposes the CLI commands the wrapper calls.

## Layout

```text
hod/         the package: cli, generate, gap, stats, audit, families,
             prereg, calibrate, placebo, ledger, agree, blind, sanity,
             pipeline, improve, toy, demo
tests/       synthetic positive controls + CLI smoke test + robustness +
             pipeline e2e + audit regressions + stats, generator, improve
runs/        NOT shipped: created locally by the commands above
             (held-out sets, ledger, reports); gitignored
```

## License

Apache 2.0. See `LICENSE`.
