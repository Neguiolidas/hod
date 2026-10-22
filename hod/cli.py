#!/usr/bin/env python3
"""Unified HOD CLI: one command, organized sections.

Usage:
    hod generate --seed 42 --n 8 --out runs/heldout-42
    hod gap --suite-base A --suite-cand B --held-base C --held-cand D --out report.md
    hod audit --heldout runs/heldout-42 [--catalog ...]
    hod pipeline --change "..." --heldout-dir runs/heldout --scores-dir runs/scores --out runs/pipe-01
    hod improve --initial h.json --eval "CMD" --propose "CMD" --catalog suite/ --out runs/rsi
    hod demo --out runs/demo
    (from the clone, without installing: python3 -m hod <section> ...)

Each section delegates to its own module, which remains the sole owner
of its parser: no duplicated arguments.
"""


import sys

from hod import __version__
from hod import agree as agree_mod
from hod import audit as audit_mod
from hod import blind as blind_mod
from hod import calibrate as calibrate_mod
from hod import demo as demo_mod
from hod import gap as gap_mod
from hod import generate as generate_mod
from hod import improve as improve_mod
from hod import ledger as ledger_mod
from hod import pipeline as pipeline_mod
from hod import placebo as placebo_mod
from hod import prereg as prereg_mod
from hod import sanity as sanity_mod

GROUPS = [
    (
        "measurement",
        [
            ("generate", "Generates held-out tasks (8 parametric families) with disjointness and novelty checks.", generate_mod),
            ("gap", "Measures the suite x held-out gap with bootstrap CI and emits the verdict.", gap_mod),
            ("audit", "Audits contamination of a held-out set.", audit_mod),
        ],
    ),
    (
        "robustness",
        [
            ("prereg", "Registers the hypothesis before seeing the held-out.", prereg_mod),
            ("calibrate", "Calibrates difficulty and Q, filters saturated tasks.", calibrate_mod),
            ("placebo", "Noise statistics from the ledger.", placebo_mod),
            ("ledger", "Ledger report and family weights.", ledger_mod),
            ("agree", "Agreement between annotators or retest.", agree_mod),
            ("sanity", "Checks whether the evaluator separates a good harness from a bad one.", sanity_mod),
            ("blind", "Blind answer key for unbiased annotation.", blind_mod),
            ("unblind", "Restores real ids after blind annotation.", blind_mod),
        ],
    ),
    (
        "automation",
        [
            ("pipeline", "Runs all layers at once.", pipeline_mod),
            ("improve", "Self-improvement loop: only accepts GENUINE changes.", improve_mod),
            ("demo", "Offline demo of improve with the toy harness.", demo_mod),
        ],
    ),
]

SECTIONS = {}
for _, items in GROUPS:
    for name, _, module in items:
        SECTIONS[name] = module


def _help():
    lines = ["usage: hod [-h] [--version] {section} ...", ""]
    for group, items in GROUPS:
        lines.append("%s:" % group)
        for name, help_text, _ in items:
            lines.append("  %-10s %s" % (name, help_text))
        lines.append("")
    lines.append("help for a section: hod <section> --help")
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(_help())
        return 0
    if args[0] == "--version":
        print("hod " + __version__)
        return 0
    section = args[0]
    if section not in SECTIONS:
        print("unknown section: %s" % section, file=sys.stderr)
        print(_help(), file=sys.stderr)
        return 2
    if section == "blind":
        return blind_mod.main_blind(args[1:], prog="hod blind")
    if section == "unblind":
        return blind_mod.main_unblind(args[1:], prog="hod unblind")
    module = SECTIONS[section]
    return module.main(args[1:], prog="hod " + section)
