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


def load_scores(path):
    path = Path(path)
    if not path.exists():
        raise ValueError("scoreboard file missing: %s" % path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError("invalid JSON in %s: %s" % (path, e))
    return flatten(data, str(path))
