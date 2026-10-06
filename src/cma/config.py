"""Typed, layered configuration (YAML -> pydantic), plus a stable config hash.

Secrets never live in configuration files: any key that looks like a credential and
carries a value is rejected (secrets come from the environment, see ``cma.security``).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cma.domain.enums import ExecutionMode
from cma.domain.errors import ConfigError

MANDATORY_LATENCY_GRID_MS: tuple[int, ...] = (0, 100, 250, 500, 1000, 2000, 5000)
_SECRET_KEY_HINTS = ("secret", "password", "private_key", "api_key", "token", "passphrase")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataQualityConfig(_Model):
    max_clock_drift_ms: int = 50
    fail_on_sequence_gap: bool = True
    max_reference_age_ms: int = 1_000
    max_prediction_feed_age_ms: int = 5_000
    max_forward_fill_ms: int = 2_000


class CostScenario(_Model):
    name: str
    fee_multiplier: Decimal = Decimal(1)
    extra_slippage_ticks: int = 0
    extra_latency_ms: int = 0


class SimulationConfig(_Model):
    latency_ms: list[int] = Field(default_factory=lambda: [100, 250, 500, 1000, 2000, 5000])
    include_zero_latency_diagnostic: bool = True
    compute_latency_ms: int = 5
    ack_latency_ms: int = 50
    cancel_latency_ms: int | None = None  # defaults to the outbound latency
    extra_feed_latency_ms: int = 0
    queue_model: Literal["conservative", "fifo_proxy", "trade_through"] = "conservative"
    allow_partial_fills: bool = True
    fill_on_cross: bool = True
    latency_jitter: Literal["none", "lognormal"] = "none"
    latency_jitter_sigma: float = 0.25
    seed: int = 7

    @field_validator("latency_ms")
    @classmethod
    def _non_negative(cls, v: list[int]) -> list[int]:
        if any(x < 0 for x in v):
            raise ValueError("latencies must be non-negative")
        return sorted(set(v))

    def stress_grid(self) -> list[int]:
        grid = set(self.latency_ms)
        if self.include_zero_latency_diagnostic:
            grid.add(0)
        return sorted(grid)


class SignalConfig(_Model):
    min_net_edge_bps: Decimal = Decimal(100)
    require_reviewed_mapping: bool = True
    ttl_ms: int = 1_000
    adverse_selection_buffer_bps: Decimal = Decimal(0)
    uncertainty_buffer_bps: Decimal = Decimal(0)
    default_order_qty: Decimal = Decimal(10)
    max_levels_to_walk: int = 5


class RiskConfig(_Model):
    initial_nav: Decimal = Decimal(10_000)
    max_contract_nav_pct: Decimal = Decimal(2)
    max_event_nav_pct: Decimal = Decimal(5)
    max_total_open_nav_pct: Decimal = Decimal(10)
    daily_loss_stop_pct: Decimal = Decimal(2)

    @model_validator(mode="after")
    def _ordered(self) -> RiskConfig:
        for name in ("max_contract_nav_pct", "max_event_nav_pct", "max_total_open_nav_pct"):
            if not (Decimal(0) < getattr(self, name) <= Decimal(100)):
                raise ValueError(f"{name} must be in (0, 100]")
        if self.initial_nav <= 0:
            raise ValueError("initial_nav must be positive")
        return self


class ResearchConfig(_Model):
    final_test_locked: bool = True
    walk_forward: bool = True
    multiple_testing_log: bool = True
    final_test_fraction: float = 0.2
    validation_fraction: float = 0.2
    walk_forward_folds: int = 4
    embargo_ms: int = 60_000
    fdr_alpha: float = 0.05
    bootstrap_samples: int = 2_000
    min_forward_paper_days: int = 14
    cost_scenarios: list[CostScenario] = Field(
        default_factory=lambda: [
            CostScenario(name="base"),
            CostScenario(name="fees_x1.5", fee_multiplier=Decimal("1.5")),
            CostScenario(name="slip_+1tick", extra_slippage_ticks=1),
            CostScenario(
                name="fees_x2_slip_+1tick", fee_multiplier=Decimal(2), extra_slippage_ticks=1
            ),
        ]
    )


class StorageConfig(_Model):
    raw_dir: str = "data/raw"
    db_url: str = "sqlite:///data/cma.sqlite"
    reports_dir: str = "reports"
    quarantine_dir: str = "data/quarantine"
    mappings_dir: str = "config/mappings"


class VenueConfig(_Model):
    enabled: bool = True
    rest_url: str | None = None
    ws_url: str | None = None
    fee_schedule: str = "zero"
    series: list[str] = Field(default_factory=list)
    instruments: list[str] = Field(default_factory=list)
    rate_limit_per_s: float = 5.0
    credential_env: dict[str, str] = Field(default_factory=dict)  # name -> ENV VAR NAME


class AppConfig(_Model):
    mode: ExecutionMode = ExecutionMode.PAPER
    live_execution_enabled: bool = False
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    venues: dict[str, VenueConfig] = Field(default_factory=dict)
    strategies: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def hash(self) -> str:
        return config_hash(self)


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _reject_inline_secrets(data: Any, path: str = "") -> None:
    if isinstance(data, Mapping):
        for key, value in data.items():
            key_l = str(key).lower()
            here = f"{path}.{key}" if path else str(key)
            if key_l == "credential_env":
                # values are environment-variable NAMES, never the secrets themselves
                for name, env_var in (value or {}).items():
                    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", str(env_var)):
                        raise ConfigError(
                            f"{here}.{name} must be an environment variable name, not a value"
                        )
                continue
            if (
                any(h in key_l for h in _SECRET_KEY_HINTS)
                and isinstance(value, str | int)
                and str(value)
            ):
                raise ConfigError(
                    f"config key {here!r} looks like a secret; secrets must come from the "
                    "environment (use venues.<name>.credential_env)"
                )
            _reject_inline_secrets(value, here)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            _reject_inline_secrets(item, f"{path}[{i}]")


def load_config(*paths: str | Path, overrides: Mapping[str, Any] | None = None) -> AppConfig:
    """Load and deep-merge YAML files left-to-right, then apply ``overrides``."""
    merged: dict[str, Any] = {}
    for p in paths:
        text = Path(p).read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{p}: top level must be a mapping")
        _reject_inline_secrets(data)
        merged = _deep_merge(merged, data)
    if overrides:
        _reject_inline_secrets(overrides)
        merged = _deep_merge(merged, overrides)
    try:
        return AppConfig.model_validate(merged)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _canonical(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj.normalize()) if obj == obj.to_integral_value() else str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, list | tuple):
        return [_canonical(v) for v in obj]
    return obj


def config_hash(cfg: AppConfig) -> str:
    payload = json.dumps(_canonical(cfg.model_dump(mode="python")), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()
