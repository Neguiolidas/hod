"""Reading and validating held-out manifests.

Every module that follows a manifest to task files goes through here, so a
malformed or hostile manifest fails with a clean ValueError instead of a
traceback, and a `file` entry can never point outside the set directory
(`"../x.md"`, absolute paths, separators). Task ids and files must be unique:
a repeated id would let one task hide another from every per-id check.
"""

import sys
from pathlib import Path

try:
    from .scores import read_json
except ImportError:  # direct execution of a module file, outside the package
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scores import read_json  # noqa: E402


def safe_task_file(name, origin="manifest"):
    """A task file name must be a plain name inside the set directory."""
    if (not isinstance(name, str) or not name or name in (".", "..")
            or name.startswith(".") or "/" in name or "\\" in name
            or "\0" in name or Path(name).name != name
            or Path(name).is_absolute()):
        raise ValueError("%s: unsafe task file name %r (must be a plain file "
                         "name inside the set directory)" % (origin, name))
    return name


def validate(data, origin="manifest"):
    """Checks the structure of a parsed manifest; returns it unchanged."""
    if not isinstance(data, dict):
        raise ValueError("%s: expected a JSON object" % origin)
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("%s: missing 'tasks' list" % origin)
    ids, files = set(), set()
    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError("%s: task #%d is not an object" % (origin, i))
        for key in ("id", "family", "file"):
            if not isinstance(task.get(key), str) or not task[key]:
                raise ValueError("%s: task #%d without a valid %r"
                                 % (origin, i, key))
        safe_task_file(task["file"], origin)
        if task["id"] in ids:
            raise ValueError("%s: duplicate task id %r" % (origin, task["id"]))
        if task["file"] in files:
            raise ValueError("%s: duplicate task file %r" % (origin, task["file"]))
        ids.add(task["id"])
        files.add(task["file"])
    return data


def load(path):
    """Loads and validates `path` (a manifest.json or a set directory)."""
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    return validate(read_json(path, "manifest"), str(path))
