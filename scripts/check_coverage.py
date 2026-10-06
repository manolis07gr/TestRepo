"""CI gate: per-package coverage thresholds (scope s.21) from coverage.py JSON output.

>= 90% for domain, accounting (portfolio), execution simulator and risk;
>= 80% overall core src, excluding thin vendor network clients and CLI glue.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

STRICT = {
    "src/cma/domain/": 90.0,
    "src/cma/portfolio/": 90.0,
    "src/cma/execution/simulator/": 90.0,
    "src/cma/risk/": 90.0,
}
OVERALL_MIN = 80.0
EXCLUDED = (
    "src/cma/adapters/kalshi/client.py",
    "src/cma/adapters/polymarket/client.py",
    "src/cma/cli/",
    "src/cma/research/paper_run.py",
    "src/cma/ingestion/collector.py",
)


def _pct(files: dict[str, dict[str, dict[str, int]]], prefix: str = "") -> float:
    covered = total = 0
    for name, data in files.items():
        if prefix and not name.startswith(prefix):
            continue
        if not prefix and name.startswith(EXCLUDED):
            continue
        s = data["summary"]
        covered += s["covered_lines"] + s.get("covered_branches", 0)
        total += s["num_statements"] + s.get("num_branches", 0)
    return 100.0 * covered / total if total else 100.0


def main(path: str = "coverage.json") -> int:
    files = json.loads(Path(path).read_text())["files"]
    files = {k[k.index("src/") :] if "src/" in k else k: v for k, v in files.items()}
    ok = True
    for prefix, minimum in STRICT.items():
        pct = _pct(files, prefix)
        flag = "OK " if pct >= minimum else "LOW"
        ok &= pct >= minimum
        print(f"{flag} {prefix:<32} {pct:6.2f}% (min {minimum}%)")
    overall = _pct(files)
    ok &= overall >= OVERALL_MIN
    print(
        f"{'OK ' if overall >= OVERALL_MIN else 'LOW'} overall core src {overall:6.2f}% "
        f"(min {OVERALL_MIN}%)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
