"""CI gate: every scope test ID T001..T055 must name at least one test."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [f"T{i:03d}" for i in range(1, 56)]


def main() -> int:
    found: dict[str, list[str]] = {t: [] for t in REQUIRED}
    pattern = re.compile(r"def (test_([Tt]\d{3})\w*)")
    for path in (ROOT / "tests").rglob("test_*.py"):
        for m in pattern.finditer(path.read_text(encoding="utf-8")):
            tid = m.group(2).upper()
            if tid in found:
                found[tid].append(f"{path.relative_to(ROOT)}::{m.group(1)}")
    missing = [t for t, hits in found.items() if not hits]
    for t in REQUIRED:
        print(f"{t}: {len(found[t])} test(s)")
    if missing:
        print(f"MISSING test IDs: {missing}", file=sys.stderr)
        return 1
    print("all 55 scope test IDs are covered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
