"""Reading and validating scoreboards.

Accepts the flat format {task_id: score} and the canary / `hod unblind`
format {"tasks": {task_id: {"nota": n}}, "behavior": {...}}. Scores are
integers 0, 1 or 2: bool, None and non-integer floats are rejected with a
clear error. (The "nota" key is kept as-is: it is part of the file format.)
"""

import json
from pathlib import Path

VALID = (0, 1, 2)


def validate_score(task_id, score, origin="scoreboard"):
    if isinstance(score, bool) or score is None:
        raise ValueError("invalid score in %s: %s=%r" % (origin, task_id, score))
    if isinstance(score, float):
        if not score.is_integer():
            raise ValueError(
                "invalid score in %s: %s=%r" % (origin, task_id, score)
            )
        score = int(score)
    if not isinstance(score, int) or score not in VALID:
        raise ValueError("invalid score in %s: %s=%r" % (origin, task_id, score))
    return score


def flatten(data, origin="scoreboard"):
    if not isinstance(data, dict):
        raise ValueError("%s: expected a JSON object" % origin)
    if "tasks" in data and isinstance(data["tasks"], dict):
        raw = {}
        for task_id, value in data["tasks"].items():
            raw[task_id] = value.get("nota") if isinstance(value, dict) else value
    else:
        raw = data
    return {t: validate_score(t, s, origin) for t, s in raw.items()}


def _no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key %r" % key)
        out[key] = value
    return out


def read_json(path, what="file", expect=None):
    """Reads a JSON file with clean ValueErrors (missing file, directory,
    bad encoding, invalid JSON, duplicate keys, wrong top-level type).

    Duplicate keys are an error, not "last one wins": in a scoreboard,
    {"t0": 0, "t0": 2} would silently keep one of two contradictory scores."""
    path = Path(path)
    if not path.exists():
        raise ValueError("%s missing: %s" % (what, path))
    if not path.is_file():
        raise ValueError("%s is not a file: %s" % (what, path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=_no_duplicate_keys)
    except OSError as e:
        raise ValueError("cannot read %s %s: %s" % (what, path, e))
    except ValueError as e:  # JSONDecodeError, UnicodeDecodeError, duplicates
        raise ValueError("invalid JSON in %s %s: %s" % (what, path, e))
    if expect is not None and not isinstance(data, expect):
        raise ValueError("%s %s: expected a JSON %s" % (
            what, path, "object" if expect is dict else "list"))
    return data


def load_scores(path):
    return flatten(read_json(path, "scoreboard file"), str(path))
