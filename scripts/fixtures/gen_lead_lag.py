"""Generate the deterministic lead-lag fixtures (scope s.20; tests T037/T038).

Writes, in long format with columns ``series`` ("x"/"y"), ``ts_ns`` (int64 UTC ns) and
``value`` (float64):

* ``tests/fixtures/lead_lag_2s.parquet`` - X is a Brownian log-price (log of a BTC-like
  price, sigma = 1e-4 per sqrt-second) sampled at Poisson times (~5/s) for 2 h; Y(t) =
  X(t - 2 s) + iid N(0, 1e-4^2) observation noise, observed at its own Poisson times
  (~2/s).
* ``tests/fixtures/lead_lag_null.parquet`` - two independent random walks with the same
  sampling and noise.

Generation parameters are stored in the parquet schema metadata. Re-running the script
reproduces byte-identical data.

Usage: ``python scripts/fixtures/gen_lead_lag.py [--out-dir tests/fixtures]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cma.models.lead_lag.synthetic import SyntheticPair, independent_pair, lead_lag_pair

REPO_ROOT = Path(__file__).resolve().parents[2]
DURATION_S = 7_200.0
X_RATE_HZ = 5.0
Y_RATE_HZ = 2.0
SIGMA = 1e-4  # per sqrt(second), log-price units
NOISE_STD = 1e-4
LAG_MS = 2_000
LEAD_LAG_SEED = 20261005
NULL_SEED = 20261006


def _write(pair: SyntheticPair, path: Path, params: dict[str, object]) -> None:
    table = pa.Table.from_pandas(pair.to_frame(), preserve_index=False)
    meta = {b"cma_fixture": json.dumps(params, sort_keys=True).encode()}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **meta})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {table.num_rows} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the lead-lag parquet fixtures.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "tests" / "fixtures")
    args = parser.parse_args()
    common: dict[str, object] = {
        "duration_s": DURATION_S,
        "x_rate_hz": X_RATE_HZ,
        "y_rate_hz": Y_RATE_HZ,
        "sigma": SIGMA,
        "noise_std": NOISE_STD,
    }
    lead_lag = lead_lag_pair(
        seed=LEAD_LAG_SEED,
        duration_s=DURATION_S,
        lag_ms=LAG_MS,
        x_rate_hz=X_RATE_HZ,
        y_rate_hz=Y_RATE_HZ,
        sigma=SIGMA,
        noise_std=NOISE_STD,
    )
    _write(
        lead_lag,
        args.out_dir / "lead_lag_2s.parquet",
        {"kind": "lead_lag", "seed": LEAD_LAG_SEED, "lag_ms": LAG_MS, **common},
    )
    null = independent_pair(
        seed=NULL_SEED,
        duration_s=DURATION_S,
        x_rate_hz=X_RATE_HZ,
        y_rate_hz=Y_RATE_HZ,
        sigma=SIGMA,
        noise_std=NOISE_STD,
    )
    _write(
        null, args.out_dir / "lead_lag_null.parquet", {"kind": "null", "seed": NULL_SEED, **common}
    )


if __name__ == "__main__":
    main()
