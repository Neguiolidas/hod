#!/usr/bin/env python3
"""Positive controls of the instrument: the detector must get known
synthetic cases right before it deserves confidence on real cases.

- Planted overfit: suite rises, held-out flat -> OVERFIT.
- Genuine improvement: suite and held-out rise together -> GENUINE.
- Partial case: held-out captures a bit more than half of the gain -> PARTIAL.
- The same partial pattern on 4 tasks per side -> INCONCLUSIVE (too few
  tasks for a CI: nothing may be asserted).
- No improvement: suite does not rise -> NO IMPROVEMENT.

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
    assert got == expected, "%s: expected %s, got %s" % (name, expected, got)
    print("ok: %s -> %s" % (name, got))


def main():
    suite_base = scores(21, 1)
    held_base = scores(8, 1)

    check(
        "planted overfit",
        analyze(suite_base, scores(21, 2), held_base, scores(8, 1)),
        "OVERFIT",
    )
    check(
        "genuine improvement",
        analyze(suite_base, scores(21, 2), held_base, scores(8, 2)),
        "GENUINE",
    )
    check(
        "partial case",
        analyze(
            scores(8, 0),
            scores(8, 2),
            {"h%d" % i: i % 2 for i in range(8)},
            {"h%d" % i: i % 2 + 1 for i in range(8)},
        ),
        "PARTIAL",
    )
    # the same pattern with 4 tasks per side: too few for a CI, nothing asserted
    check(
        "partial case, n=4",
        analyze(
            scores(4, 0),
            scores(4, 2),
            {"h0": 0, "h1": 0, "h2": 1, "h3": 1},
            {"h0": 1, "h1": 1, "h2": 2, "h3": 2},
        ),
        "INCONCLUSIVE",
    )
    check(
        "no improvement",
        analyze(suite_base, scores(21, 1), held_base, scores(8, 2)),
        "NO IMPROVEMENT",
    )
    print("positive controls: 5/5")
    return 0


def test_main():
    """Wrapper so pytest collects the script-format test."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
