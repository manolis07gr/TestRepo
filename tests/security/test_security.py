"""T053 live trading impossible in v1; T054 no secret leakage in logs/config snapshots."""

from __future__ import annotations

import ast
import io
import json
import logging
from pathlib import Path

import pytest

from cma.backtest.core import StaticMappings, TradingCore
from cma.config import AppConfig, load_config
from cma.domain.enums import ExecutionMode
from cma.domain.errors import ConfigError, LiveTradingDisabledError
from cma.domain.time import ManualClock
from cma.execution.latency import LatencyModel, LatencyProfile
from cma.execution.live import adapter as live
from cma.execution.paper.executor import PaperTradingSession
from cma.security import SECRETS, RedactingFilter, Secret, load_secret, redact, redact_mapping

pytestmark = pytest.mark.security
SRC = Path(__file__).resolve().parents[2] / "src" / "cma"
REPO = Path(__file__).resolve().parents[2]


def test_T053_default_config_cannot_construct_live_adapter() -> None:
    cfg = load_config(REPO / "config" / "base.yaml")
    assert cfg.live_execution_enabled is False
    with pytest.raises(LiveTradingDisabledError):
        live.LiveExecutionAdapter(cfg)


def test_T053_even_full_gate_cannot_enable_live_in_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig.model_validate({"mode": "LIVE", "live_execution_enabled": True})
    approval = tmp_path / "approval.sig"
    approval.write_text("signed")
    monkeypatch.setenv("CMA_LIVE_ACK", live.REQUIRED_ACK_PHRASE)
    status = live.evaluate_live_gate(cfg, approval)
    assert status.missing() == ["v1 code constant (live trading is not permitted in v1)"]
    assert live.V1_LIVE_TRADING_PERMITTED is False
    with pytest.raises(LiveTradingDisabledError):
        live.LiveExecutionAdapter(cfg, approval)


def test_T053_core_and_paper_refuse_live_mode() -> None:
    cfg = AppConfig.model_validate({"mode": "LIVE", "live_execution_enabled": True})
    with pytest.raises(LiveTradingDisabledError):
        TradingCore(
            config=cfg,
            strategies=[],
            contracts={},
            mappings=StaticMappings({}),
            latency=LatencyModel(LatencyProfile(outbound_ms=1)),
            scheduler=None,  # type: ignore[arg-type]
            mode=ExecutionMode.LIVE,
        )
    with pytest.raises(LiveTradingDisabledError):
        PaperTradingSession(
            config=cfg,
            strategies=[],
            contracts={},
            mappings=StaticMappings({}),
            latency=LatencyModel(LatencyProfile(outbound_ms=1)),
            clock=ManualClock(),
        )
    with pytest.raises(ConfigError):
        PaperTradingSession(
            config=AppConfig.model_validate({"mode": "BACKTEST"}),
            strategies=[],
            contracts={},
            mappings=StaticMappings({}),
            latency=LatencyModel(LatencyProfile(outbound_ms=1)),
            clock=ManualClock(),
        )


def test_T053_no_order_placement_code_outside_disabled_adapter() -> None:
    """Static check: nothing in src/ references a venue order-entry endpoint or defines an
    order-placement function, except the hard-disabled live adapter."""
    import re

    order_entry = re.compile(
        r"/(portfolio/(batched_)?orders|orders?|cancel(-all|-market-orders|_orders)?)"
        r"(?=$|[/?'\"{])"  # path segment ends: Kalshi /portfolio/orders, CLOB /order(s), /cancel*
    )
    offenders = []
    for path in SRC.rglob("*.py"):
        if "live" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if order_entry.search(node.value):
                    offenders.append(f"{path}:{node.lineno}:{node.value[:60]}")
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in (
                "place_order",
                "post_order",
                "create_order",
                "submit_live_order",
            ):
                offenders.append(f"{path}:{node.lineno}:def {node.name}")
    assert not offenders, offenders
    assert order_entry.search("/portfolio/orders") and not order_entry.search("/orderbook")
    assert order_entry.search("https://clob.polymarket.com/order")
    assert not order_entry.search("void/cancel risk not priced")


def test_T054_secrets_never_reach_logs_or_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    SECRETS.clear()
    monkeypatch.setenv("KALSHI_API_KEY_ID", "key-id-1234567890")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM", "-----BEGIN PRIVATE KEY-----\nMIIEsecretbody\n-----END PRIVATE KEY-----")
    key = load_secret("KALSHI_API_KEY_ID")
    pem = load_secret("KALSHI_PRIVATE_KEY_PEM")
    assert key is not None and pem is not None
    assert "key-id-1234567890" not in repr(key) and "key-id" not in str(key)

    stream = io.StringIO()
    logger = logging.getLogger("cma.test.secrets")
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("connecting with key %s", key.reveal())
    logger.info("headers {'KALSHI-ACCESS-SIGNATURE': 'c2lnbmF0dXJlLXZhbHVl'}")
    logger.info("pem=%s", pem.reveal())
    logger.info("Authorization: Bearer abc.def.ghi")
    out = stream.getvalue()
    logger.removeHandler(handler)
    for leaked in ("key-id-1234567890", "MIIEsecretbody", "c2lnbmF0dXJlLXZhbHVl", "abc.def.ghi"):
        assert leaked not in out, out

    snapshot = redact_mapping({"venue": {"api_key_value": key.reveal(), "nested": [pem]}})
    dumped = json.dumps(snapshot)
    assert "key-id-1234567890" not in dumped and "MIIE" not in dumped
    assert redact(f"token={key.reveal()}") == "token=***REDACTED***"


def test_T054_config_files_cannot_carry_inline_secrets(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("venues:\n  kalshi:\n    api_secret: hunter2hunter2\n")
    with pytest.raises(ConfigError, match="secret"):
        load_config(bad)
    bad2 = tmp_path / "bad2.yaml"
    bad2.write_text("venues:\n  kalshi:\n    credential_env:\n      key_id: actual-secret-value\n")
    with pytest.raises(ConfigError, match="environment variable name"):
        load_config(bad2)
    ok = load_config(REPO / "config" / "base.yaml")
    dumped = ok.model_dump_json()
    assert "KALSHI_API_KEY_ID" in dumped  # only the env var *name* is stored


def test_T054_repository_contains_no_committed_secrets() -> None:
    import re

    patterns = [
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        re.compile(r"(?i)(api[_-]?secret|private[_-]?key)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
    ]
    roots = [REPO / "src", REPO / "config", REPO / "scripts"]
    hits = []
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json", ".toml", ".env"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                for pat in patterns:
                    for m in pat.finditer(text):
                        if "security.py" in path.name or "PRIVATE KEY-----.*?" in m.group(0):
                            continue
                        hits.append(f"{path}: {m.group(0)[:40]}")
    assert not hits, hits
    assert isinstance(Secret("X", "abcd"), Secret)
