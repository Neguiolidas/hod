#!/usr/bin/env python3
"""Control positive of the instrumento: the detector must acertar 3 cases
synthetic known before merecer confidence in cases real.

- Overfit planted: suite rises, held-out stopped -> OVERFIT.
- Improvement genuine: suite and held-out rise together -> Genuine.
- Case partial: held-out captures little more of the half of the gain -> Partial.
- Without improvement: suite not rises -> Without Improvement.

Usage: python3 test_positive_control.py (code 0 = all controls pass).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hod.gap import analyze  # noqa: E402


def scores(n, value):
    return {"t%02d" % i: value for i in range(n)}


def check(name, result, expected):
    got = result["verdict"]
    assert got == expected, "%s: esperado %s, veio %s" % (name, expected, got)
    print("ok: %s -> %s" % (name, got))


def main():
    suite_base = scores(21, 1)
    held_base = scores(8, 1)

    check(
        "overfit plantado",
        analyze(suite_base, scores(21, 2), held_base, scores(8, 1)),
        "OVERFIT",
    )
    check(
        "genuine improvement",
        analyze(suite_base, scores(21, 2), held_base, scores(8, 2)),
        "GENUINE",
    )
    check(
        "caso parcial",
        analyze(
            scores(4, 0),
            scores(4, 2),
            {"h0": 0, "h1": 0, "h2": 1, "h3": 1},
            {"h0": 1, "h1": 1, "h2": 2, "h3": 2},
        ),
        "PARTIAL",
    )
    check(
        "sem melhora",
        analyze(suite_base, scores(21, 1), held_base, scores(8, 2)),
        "NO IMPROVEMENT",
    )
    print("positive controls: 4/4")
    return 0


def test_main():
    """Wrapper for the pytest coletar the test in format of script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
