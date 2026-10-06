"""Regenerate every deterministic test fixture (scope s.20)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent / "fixtures"


def main() -> int:
    for script in sorted(HERE.glob("gen_*.py")):
        print(f"== {script.name}")
        runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
