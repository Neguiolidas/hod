"""Entry point for `python3 -m hod` (clone usage, without installing)."""

import sys

from hod.cli import main

if __name__ == "__main__":
    sys.exit(main())
