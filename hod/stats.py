"""Gap statistics: paired bootstrap, confidence intervals, placebo noise
floor, power and verdict by CI.

Why: with 8 to 30 tasks, the v0.1 point gap swings a lot between sets.
A verdict that flips from GENUINE to OVERFIT when the seed changes is not
a verdict. Here every number comes with a 95% CI and the verdict only
asserts what the intervals support; when they do not support it, it says
INCONCLUSIVE and how many tasks would be missing.

Method:
- Suite and held-out are independent samples of tasks. Each bootstrap
  replica resamples tasks with replacement inside each set, keeping the
  (base, candidate) pair of the same task together (paired).
- For each replica: suite delta, held-out delta, gap = suite - held,
  and the linear combinations t50 = held - 0.5*suite and t75 = held -
  0.75*suite (transfer of at least half / three quarters of the gain).
  Testing transfer by linear combination avoids the held/suite ratio,
  which explodes when the suite rises little.
- 95% percentile CI, fixed seed: the same scoreboard always gives the same CI.
- Noise floor: the largest |delta| observed in placebos (no-effect changes)
  in the suite and in the held-out. A delta inside the floor does not
  count as movement, even with a CI outside zero.
- Power: standard error of the held-out delta (deviation of the replicas)
  and the minimum detectable effect with 80% power and two-sided 5% alpha
  (2.8 x SE). If the held-out cannot detect three quarters of the suite
  gain, it warns and estimates the n needed.

Verdicts by CI (constants in gap.py):
- REGRESSION: the suite or the held-out dropped with the whole CI below zero and
  beyond the floor, without a significant improvement in the suite.
- NO IMPROVEMENT: the suite did not rise beyond the floor.
- INCONCLUSIVE: the suite seems to rise but the CI crosses zero; or the suite
  rose and the held-out CI does not separate "transferred" from "not transferred".
- OVERFIT: the suite rose and (the held-out dropped, or with 95% confidence the
  held-out captured less than half of the gain: the t50 CI entirely below 0).
- GENUINE: the suite rose, the held-out rose (CI above 0 and beyond the floor)
  and the point transfer is at least three quarters (t75 >= 0). With
  `strict=True` (`hod gap --strict`) it also requires, with 95%
  confidence, that the transfer be at least half (t50 CI above 0); that
  asks for many more tasks.
- PARTIAL: the held-out truly rose, but without reaching the GENUINE criterion.
"""

import random

GENUINE_RATIO = 0.75
PARTIAL_RATIO = 0.5
DEFAULT_BOOT = 2000
DEFAULT_SEED = 20261022
MIN_N_WARN = 30
Z_POWER = 2.8  # 1.96 + 0.84: alpha 5% two-sided, power 80%


def _paired(base, cand):
    ids = sorted(set(base) & set(cand))
    return [base[t] for t in ids], [cand[t] for t in ids], ids


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _percentile(sorted_xs, q):
    if not sorted_xs:
        return 0.0
    k = (len(sorted_xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


def _ci(xs, alpha=0.05):
    s = sorted(xs)
    return [round(_percentile(s, alpha / 2), 4),
            round(_percentile(s, 1 - alpha / 2), 4)]


def _sd(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def bootstrap(suite_base, suite_cand, held_base, held_cand,
              n_boot=DEFAULT_BOOT, seed=DEFAULT_SEED, alpha=0.05):
    """CI paired of suite_delta, held_delta, gap, t50 and t75."""
    sb, sc, s_ids = _paired(suite_base, suite_cand)
    hb, hc, h_ids = _paired(held_base, held_cand)
    if not s_ids or not h_ids:
        raise ValueError("bootstrap needs paired tasks in the suite and "
                         "in the held-out")
    s_diff = [c - b for b, c in zip(sb, sc)]
    h_diff = [c - b for b, c in zip(hb, hc)]
    rng = random.Random(seed)
    ns, nh = len(s_diff), len(h_diff)
    reps = {"suite_delta": [], "held_delta": [], "gap": [], "t50": [],
            "t75": []}
    for _ in range(n_boot):
        ds = sum(s_diff[rng.randrange(ns)] for _ in range(ns)) / ns
        dh = sum(h_diff[rng.randrange(nh)] for _ in range(nh)) / nh
        reps["suite_delta"].append(ds)
        reps["held_delta"].append(dh)
        reps["gap"].append(ds - dh)
        reps["t50"].append(dh - PARTIAL_RATIO * ds)
        reps["t75"].append(dh - GENUINE_RATIO * ds)
    ds, dh = _mean(s_diff), _mean(h_diff)
    point = {"suite_delta": ds, "held_delta": dh, "gap": ds - dh,
             "t50": dh - PARTIAL_RATIO * ds, "t75": dh - GENUINE_RATIO * ds}
    out = {
        "n_suite": ns,
        "n_held": nh,
        "n_boot": n_boot,
        "seed": seed,
        "alpha": alpha,
        "point": {k: round(v, 4) for k, v in point.items()},
        "ci": {k: _ci(v, alpha) for k, v in reps.items()},
        "se": {k: round(_sd(v), 4) for k, v in reps.items()},
        "p_held_le_0": round(sum(1 for x in reps["held_delta"] if x <= 0)
                             / n_boot, 4),
    }
    return out


def noise_floor(calibration):
    """Noise floor (suite, held) the partir of placebo.stats()."""
    if not calibration or not calibration.get("n_placebo"):
        return {"suite": 0.0, "held": 0.0, "n_placebo": 0, "source": "none"}
    return {
        "suite": float(calibration.get("max_abs_suite_delta") or 0.0),
        "held": float(calibration.get("max_abs_held_delta") or 0.0),
        "n_placebo": calibration["n_placebo"],
        "source": "largest |delta| across %d placebo(s)" % calibration["n_placebo"],
    }


def power(boot):
    """Effect minimum detectable and suggested n for the held-out."""
    se_h = boot["se"]["held_delta"]
    mde = Z_POWER * se_h
    target = GENUINE_RATIO * boot["point"]["suite_delta"]
    warnings = []
    n_needed = None
    if boot["n_held"] < MIN_N_WARN:
        warnings.append("held-out with n=%d (< %d): wide CI, use more tasks"
                        % (boot["n_held"], MIN_N_WARN))
    if se_h == 0:
        warnings.append("no variation across tasks in the held-out: the CI "
                        "does not measure evaluation noise; run placebos")
    if boot["n_suite"] < MIN_N_WARN:
        warnings.append("suite with n=%d (< %d)" % (boot["n_suite"], MIN_N_WARN))
    if target > 0 and mde > target:
        n_needed = int(boot["n_held"] * (mde / target) ** 2 + 0.999)
        warnings.append(
            "low power: the held-out only detects effects ≥ %.3f and three quarters "
            "of the suite gain are %.3f; would need ~%d tasks in the held-out"
            % (mde, target, n_needed))
    return {"mde_held_80": round(mde, 4), "target_held": round(target, 4),
            "n_held_needed": n_needed, "warnings": warnings}


def verdict_ci(boot, floor=None, strict=False):
    """Verdict sustentado by the ICs and by the noise floor. Returns
    (verdict, reasons)."""
    floor = floor or {"suite": 0.0, "held": 0.0}
    p, ci = boot["point"], boot["ci"]
    s_lo, s_hi = ci["suite_delta"]
    h_lo, h_hi = ci["held_delta"]
    eps = 1e-9
    suite_up = s_lo > eps and p["suite_delta"] > floor["suite"] + eps
    suite_down = s_hi < -eps and -p["suite_delta"] > floor["suite"] + eps
    held_up = h_lo > eps and p["held_delta"] > floor["held"] + eps
    held_down = h_hi < -eps and -p["held_delta"] > floor["held"] + eps
    why = []
    if not suite_up:
        if suite_down or held_down:
            why.append("significant drop (%s)" % (
                "suite" if suite_down else "held-out"))
            return "REGRESSION", why
        if p["suite_delta"] > floor["suite"] + eps:
            why.append("suite rose %.3f but the CI [%.3f, %.3f] crosses zero"
                       % (p["suite_delta"], s_lo, s_hi))
            return "INCONCLUSIVE", why
        why.append("suite did not rise above the noise floor (%.3f)"
                   % floor["suite"])
        if held_up:
            why.append("the held-out rose without the suite (suite at ceiling?)")
        return "NO IMPROVEMENT", why
    if held_down:
        why.append("suite rose and the held-out dropped (CI [%.3f, %.3f])"
                   % (h_lo, h_hi))
        return "OVERFIT", why
    t50_lo, t50_hi = ci["t50"]
    if t50_hi < -eps:
        why.append("with 95%% confidence the held-out captured less than half "
                   "of the gain (CI of held − 0.5·suite: [%.3f, %.3f])"
                   % (t50_lo, t50_hi))
        return "OVERFIT", why
    if held_up:
        if p["t75"] >= -eps and (t50_lo > eps or not strict):
            why.append("held-out rose (CI [%.3f, %.3f]) and captured %s of the "
                       "suite gain" % (h_lo, h_hi,
                                           "≥ ¾ (≥ ½ with 95%% confidence)"
                                           if t50_lo > eps else "≥ ¾ at the point"))
            return "GENUINE", why
        if p["t75"] >= -eps:
            why.append("held-out rose and captured ≥ ¾ at the point, but strict mode "
                       "requires ≥ ½ with 95%% confidence (CI of t50 "
                       "[%.3f, %.3f])" % (t50_lo, t50_hi))
        else:
            why.append("held-out truly rose, but captured less than ¾ of the "
                       "suite gain (held %.3f, suite %.3f)"
                       % (p["held_delta"], p["suite_delta"]))
        return "PARTIAL", why
    why.append("the held-out CI [%.3f, %.3f] does not separate transfer from "
               "overfit; enlarge the held-out" % (h_lo, h_hi))
    return "INCONCLUSIVE", why


FAMILY_MARGIN = 0.25
MIN_FAMILY_N = 8  # below that, the point margin is pure noise; only the CI counts


def family_regressions(held_base, held_cand, family_of, floor_held=0.0,
                       margin=FAMILY_MARGIN, n_boot=1000, seed=DEFAULT_SEED,
                       alpha=0.05):
    """Held-out families that got worse. From the Google blog: the
    main question is not "did it rise 2%?", it is "did anything get worse?".

    A family regresses if (a) the delta CI stays entirely below zero and
    beyond the noise floor, or (b) the point delta drops more than `margin`
    (default 0.25 points), with at least MIN_FAMILY_N tasks. Per family n
    is small and the CI alone lets real drops pass; (b) is a deliberately
    conservative non-inferiority margin: in an automatic loop, rejecting
    a good improvement costs the round, accepting a regression costs
    the harness."""
    fams = sorted({family_of[t] for t in held_base if t in family_of})
    out = {}
    for i, fam in enumerate(fams):
        ids = [t for t in held_base if family_of.get(t) == fam and t in held_cand]
        if not ids:
            continue
        diffs = [held_cand[t] - held_base[t] for t in ids]
        rng = random.Random(seed + i)
        n = len(diffs)
        reps = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                for _ in range(n_boot)]
        lo, hi = _ci(reps, alpha)
        d = _mean(diffs)
        strong = hi < 0 and -d > floor_held
        margin_hit = (margin is not None and n >= MIN_FAMILY_N
                      and d < -max(margin, floor_held))
        out[fam] = {"n": n, "delta": round(d, 4), "ci": [lo, hi],
                    "regressed": strong or margin_hit,
                    "rule": "ic" if strong else ("margem" if margin_hit else None)}
    return out


def summarize(suite_base, suite_cand, held_base, held_cand, calibration=None,
              n_boot=DEFAULT_BOOT, seed=DEFAULT_SEED, strict=False):
    boot = bootstrap(suite_base, suite_cand, held_base, held_cand,
                     n_boot=n_boot, seed=seed)
    floor = noise_floor(calibration)
    verdict, why = verdict_ci(boot, floor, strict=strict)
    pw = power(boot)
    return {"verdict": verdict, "why": why, "bootstrap": boot,
            "noise_floor": floor, "power": pw, "strict": strict}
