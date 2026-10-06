"""CI gate: fail on credential-shaped strings in tracked files (no network, no deps)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "private key block": re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----\s*\n\s*[A-Za-z0-9+/=]{40,}"
    ),
    "aws access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "assigned secret": re.compile(
        r"(?i)\b(api[_-]?secret|secret[_-]?key|private[_-]?key|passphrase)\b\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"
    ),
}
ALLOW = ("tests/security/", "scripts/secret_scan.py", "src/cma/security.py")


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], check=True, capture_output=True, text=True
    ).stdout.split()
    return [ROOT / p for p in out]


def main() -> int:
    findings = []
    for path in tracked_files():
        rel = str(path.relative_to(ROOT))
        if rel.startswith(ALLOW) or path.suffix in {".parquet", ".gz", ".png", ".jpg"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for name, pat in PATTERNS.items():
            for m in pat.finditer(text):
                findings.append(f"{rel}: {name}: {m.group(0)[:30]}...")
    for f in findings:
        print(f, file=sys.stderr)
    print(f"secret scan: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
