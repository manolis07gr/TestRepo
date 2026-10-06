"""Hold-to-settlement study on settled Kalshi BTC markets (history, no live data needed).

  python scripts/settlement_study.py fetch --since 2026-01-01          # resumable download
  python scripts/settlement_study.py analyze --out reports/settlement_study

``fetch`` stores settled markets, their 1-minute YES bid/ask candles, Coinbase BTC-USD
1-minute candles and Deribit DVOL under ``--data`` (git-ignored). Kalshi requests are signed
when KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY are set (higher read limit); the key is never
printed. ``analyze`` runs cma.research.settlement_study (see its docstring for the method).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from cma.research.settlement_data import fetch_all


def _ts(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())


def cmd_fetch(args: argparse.Namespace) -> int:
    since = _ts(args.since)
    until = _ts(args.until) if args.until else int(time.time())
    t0 = time.time()
    summary = asyncio.run(
        fetch_all(
            series=args.series,
            since_ts=since,
            until_ts=until,
            data_dir=Path(args.data),
            kalshi_rate_per_s=args.kalshi_rate,
            signed=not args.unsigned,
        )
    )
    summary["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(summary))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--series", nargs="+", default=["KXBTC15M"])
    f.add_argument("--since", required=True, help="UTC date, e.g. 2026-01-01")
    f.add_argument("--until", default=None, help="UTC date (default: now)")
    f.add_argument("--data", default="data/settlement")
    f.add_argument("--kalshi-rate", type=float, default=10.0, help="requests per second")
    f.add_argument("--unsigned", action="store_true", help="do not sign Kalshi requests")
    f.set_defaults(func=cmd_fetch)
    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
