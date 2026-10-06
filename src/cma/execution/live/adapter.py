"""Live order routing: interface only, hard-disabled in v1 (scope s.2, s.17, T053).

The adapter refuses to initialise unless an explicit multi-factor gate is satisfied, and
in v1 it refuses *even then*: ``V1_LIVE_TRADING_PERMITTED`` is a code constant, not
configuration, so no config file, environment variable or CLI flag can enable it. Lifting
it requires a code change reviewed under a separately scoped live-money phase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from cma.config import AppConfig
from cma.domain.enums import ExecutionMode
from cma.domain.errors import LiveTradingDisabledError
from cma.domain.models import SimOrder

V1_LIVE_TRADING_PERMITTED = False
REQUIRED_ACK_PHRASE = "I UNDERSTAND THIS ROUTES REAL MONEY"


@dataclass(frozen=True)
class LiveGateStatus:
    config_enabled: bool
    mode_live: bool
    env_ack: bool
    approval_file_present: bool
    code_permits: bool

    @property
    def satisfied(self) -> bool:
        return all(
            (
                self.config_enabled,
                self.mode_live,
                self.env_ack,
                self.approval_file_present,
                self.code_permits,
            )
        )

    def missing(self) -> list[str]:
        names = {
            "config_enabled": "live_execution_enabled=true",
            "mode_live": "mode=LIVE",
            "env_ack": "CMA_LIVE_ACK environment acknowledgement",
            "approval_file_present": "signed approval file",
            "code_permits": "v1 code constant (live trading is not permitted in v1)",
        }
        return [label for attr, label in names.items() if not getattr(self, attr)]


def evaluate_live_gate(config: AppConfig, approval_file: Path | None = None) -> LiveGateStatus:
    return LiveGateStatus(
        config_enabled=config.live_execution_enabled,
        mode_live=config.mode is ExecutionMode.LIVE,
        env_ack=os.environ.get("CMA_LIVE_ACK") == REQUIRED_ACK_PHRASE,
        approval_file_present=bool(approval_file and approval_file.is_file()),
        code_permits=V1_LIVE_TRADING_PERMITTED,
    )


class LiveExecutionAdapter:
    """Would route real orders. Cannot be constructed in v1."""

    def __init__(self, config: AppConfig, approval_file: Path | None = None) -> None:
        status = evaluate_live_gate(config, approval_file)
        if not status.satisfied:
            raise LiveTradingDisabledError(
                "live execution refused; missing: " + ", ".join(status.missing())
            )
        raise LiveTradingDisabledError(  # pragma: no cover - unreachable while v1 constant holds
            "live execution is not implemented in v1"
        )

    def place_order(self, order: SimOrder) -> str:  # pragma: no cover - never constructible
        raise LiveTradingDisabledError("live order placement is disabled in v1")

    def cancel_order(self, order_id: str) -> None:  # pragma: no cover - never constructible
        raise LiveTradingDisabledError("live order cancellation is disabled in v1")
