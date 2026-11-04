"""Harness Overfit Detector (HOD): detects when a harness improvement
is real and when it is just fitting to the evaluation suite.
"""

# Single literal, kept equal to pyproject.toml by tests/test_cli.py. Reading
# it from importlib.metadata reported a stale installed copy when running
# from a checkout (`python3 -m hod --version`).
__version__ = "0.1.1"
