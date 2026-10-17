"""
Backwards-compatible entry point: equivalent to ``python -m src.cli run``.
"""

import sys

from src.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
