"""Round-trip times from this machine to the venues (read-only; prints timings only).

  python scripts/latency_probe.py --n 30

Times ``n`` HTTPS GETs on one kept-alive connection to Kalshi's public exchange-status
endpoint and to Coinbase's BTC-USD ticker, plus a fresh TCP+TLS connect to each. Run it on
the machine that would collect or trade: the live study's latency grid (0-1000 ms) should
be read against these numbers. No credentials are used and nothing is written.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import httpx

TARGETS = {
    "kalshi": "https://external-api.kalshi.com/trade-api/v2/exchange/status",
    "coinbase": "https://api.exchange.coinbase.com/products/BTC-USD/ticker",
}


def _ms(samples: list[float]) -> dict[str, float | int]:
    s = sorted(samples)
    return {
        "n": len(s),
        "min_ms": round(s[0], 1),
        "median_ms": round(statistics.median(s), 1),
        "p90_ms": round(s[min(len(s) - 1, int(0.9 * len(s)))], 1),
    }


def probe(url: str, n: int) -> dict[str, object]:
    out: dict[str, object] = {}
    t0 = time.perf_counter()
    with httpx.Client(timeout=10.0, headers={"User-Agent": "cma-latency-probe"}) as cl:
        first = cl.get(url)
        out["first_request_with_connect_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        out["status"] = first.status_code
        rtts = []
        for _ in range(n):
            t = time.perf_counter()
            cl.get(url)
            rtts.append(1000 * (time.perf_counter() - t))
    out["kept_alive"] = _ms(rtts)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    result = {name: probe(url, args.n) for name, url in TARGETS.items()}
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
