#!/usr/bin/env python3
"""Test of smoke of the CLI unificado: generate + gap + audit via hod.cli,
and version of the package batendo with the pyproject.

Usage: python3 tests/test_cli.py (code 0 = everything passes).
"""

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hod import __version__  # noqa: E402
from hod import cli  # noqa: E402


def main():
    repo = Path(__file__).resolve().parent.parent
    version = re.search(
        r'^version = "(.*)"', (repo / "pyproject.toml").read_text(), re.M
    ).group(1)
    assert __version__ == version, (
        "package version %s != pyproject %s" % (__version__, version)
    )
    print("ok: version %s matches pyproject" % version)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        catalog = tmp / "catalog"
        catalog.mkdir()
        (catalog / "a.md").write_text(
            "# Example task\n\n### Setup\nCreate a file.\n\n"
            "### Instruction\nEscreva oi.\n"
        )
        (catalog / "b.md").write_text(
            "# Another task\n\n### Setup\nOpen the editor.\n\n"
            "### Instruction\nEscreva tchau.\n"
        )
        held = tmp / "held"
        rc = cli.main(
            [
                "generate", "--seed", "7", "--n", "4",
                "--out", str(held), "--catalog", str(catalog),
            ]
        )
        assert rc == 0, "generate failed"
        files = sorted(held.glob("heldout-*.md"))
        assert len(files) == 4, "esperava 4 tasks, vieram %d" % len(files)
        assert (held / "manifest.json").exists()
        print("ok: generate emitted 4 tasks with manifest")
        rc = cli.main(
            ["audit", "--heldout", str(held), "--catalog", str(catalog)]
        )
        assert rc == 0, "audit failed"
        print("ok: audit passed on the generated set")
        sb, sc, hb, hc = (
            tmp / "sb.json", tmp / "sc.json", tmp / "hb.json", tmp / "hc.json"
        )
        sb.write_text(json.dumps({"s0": 1, "s1": 1}))
        sc.write_text(json.dumps({"s0": 2, "s1": 2}))
        hb.write_text(json.dumps({"h0": 1, "h1": 1}))
        hc.write_text(json.dumps({"h0": 1, "h1": 1}))
        report = tmp / "report.md"
        rc = cli.main(
            [
                "gap", "--suite-base", str(sb), "--suite-cand", str(sc),
                "--held-base", str(hb), "--held-cand", str(hc),
                "--out", str(report),
            ]
        )
        assert rc == 0, "gap failed"
        assert "OVERFIT" in report.read_text(), "expected verdict OVERFIT"
        print("ok: gap emitted OVERFIT in the planted case")
        assert cli.main(["--version"]) == 0
        assert cli.main([]) == 0
        assert cli.main(["invalid-section"]) == 2
        print("ok: --version, help and invalid section")
    print("CLI smoke: all pass")
    return 0


def test_main():
    """Wrapper for the pytest coletar the test in format of script."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
