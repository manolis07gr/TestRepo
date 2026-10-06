"""Live staleness study: collect Kalshi BTC + Coinbase, then measure stale quotes.

  python scripts/live_study.py collect --duration 7000      # raw capture (needs network)
  python scripts/live_study.py analyze --out reports/live_study

``collect`` runs the production collector with config/base.yaml + config/live_study.yaml and
prints a one-line health summary every ``--report-every`` seconds. ``analyze`` streams the
raw store once (memory scales with top-of-book changes, not raw messages) and writes
summary.json / summary.md (see cma.research.live_study for the method).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from cma.config import load_config
from cma.research.live_study import (
    REFERENCE,
    StudyConfig,
    above_contracts,
    analyze_quotes,
    stream_quotes,
    write_outputs,
)
from cma.storage.contracts import ContractStore
from cma.storage.db import open_database

CONFIGS = ("config/base.yaml", "config/live_study.yaml")


def _brief(report: dict[str, Any]) -> str:
    feeds = ", ".join(
        f"{f['name']}={f['state']}/{f['messages']}msg"
        + (f"/err:{f['last_error']}" if f.get("last_error") else "")
        for f in report.get("feeds", [])
    )
    books = report.get("books", [])
    valid = sum(1 for b in books if b.get("valid"))
    quarantined = sum(int(f.get("quarantined", 0)) for f in report.get("feeds", []))
    return (
        f"status={report.get('status')} feeds[{feeds}] books={valid}/{len(books)} valid "
        f"quarantined={quarantined}"
    )


def cmd_collect(args: argparse.Namespace) -> int:
    from cma.ingestion.collector import build_collector

    cfg = load_config(*CONFIGS)
    Path(cfg.storage.raw_dir).mkdir(parents=True, exist_ok=True)
    db = open_database(cfg.storage.db_url)
    # discover new hourly / 15-minute markets within a minute of listing
    collector = build_collector(cfg, db=db, raw_root=cfg.storage.raw_dir, discovery_refresh_s=60.0)

    async def run() -> None:
        await collector.start()
        started = time.monotonic()
        try:
            while time.monotonic() - started < args.duration:
                await asyncio.sleep(min(args.report_every, args.duration))
                print(
                    f"[{time.monotonic() - started:7.0f}s] {_brief(collector.health_report())}",
                    flush=True,
                )
        finally:
            await collector.stop()
            await collector.aclose()

    asyncio.run(run())
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    cfg = load_config(*CONFIGS)
    db = open_database(cfg.storage.db_url)
    contracts = ContractStore(db).load()
    above = above_contracts(contracts)
    wanted = {REFERENCE} | {c.instrument_id for c in above}
    started = time.monotonic()
    quotes, n_raw = stream_quotes(Path(cfg.storage.raw_dir), wanted)
    changes = sum(q.ts.size for q in quotes.values())
    print(
        f"contracts {len(contracts)} (above-strike {len(above)}), raw messages {n_raw}, "
        f"books {len(quotes)}, top-of-book changes {changes} "
        f"in {time.monotonic() - started:.0f}s",
        flush=True,
    )
    result = analyze_quotes(quotes, contracts, cfg=StudyConfig())
    result["raw_messages"] = n_raw
    paths = write_outputs(result, Path(args.out))
    print(json.dumps({"outputs": [str(p) for p in paths]}))
    print(paths[1].read_text())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--duration", type=float, default=7000.0)
    c.add_argument("--report-every", type=float, default=60.0)
    a = sub.add_parser("analyze")
    a.add_argument("--out", default="reports/live_study")
    args = ap.parse_args()
    return cmd_collect(args) if args.cmd == "collect" else cmd_analyze(args)


if __name__ == "__main__":
    sys.exit(main())
