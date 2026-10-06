"""No-lookahead enforcement (scope s.2, s.8, T005).

A feature or estimate computed for decision time ``t`` must declare the maximum source
timestamp it used; anything stamped after ``t`` is rejected.
"""

from __future__ import annotations

from collections.abc import Mapping

from cma.domain.errors import LookaheadError
from cma.domain.models import FeatureVector


def assert_watermark(watermarks: Mapping[str, int], decision_ts_ns: int, *, what: str = "") -> int:
    """Return the max watermark; raise LookaheadError if any source is after the decision."""
    if not watermarks:
        raise LookaheadError(f"{what or 'feature'} declares no source watermarks")
    offenders = {k: v for k, v in watermarks.items() if v > decision_ts_ns}
    if offenders:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(offenders.items()))
        raise LookaheadError(
            f"{what or 'feature'} uses data after decision time {decision_ts_ns}: {detail}"
        )
    return max(watermarks.values())


def check_feature_vector(fv: FeatureVector, decision_ts_ns: int) -> FeatureVector:
    if fv.asof_ts_ns > decision_ts_ns:
        raise LookaheadError(
            f"feature vector as-of {fv.asof_ts_ns} is after decision time {decision_ts_ns}"
        )
    assert_watermark(fv.source_watermarks, decision_ts_ns, what=f"features[{fv.contract_id}]")
    return fv
