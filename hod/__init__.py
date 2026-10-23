"""Harness Overfit Detector (HOD): detects when a harness improvement
is real and when it is just fitting to the evaluation suite.
"""

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("hod")
except Exception:
    __version__ = "0.1.1"
