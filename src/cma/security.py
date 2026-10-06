"""Secret handling and log redaction (scope s.17).

Secrets are read only from the environment, wrapped so that ``repr``/``str`` never reveal
them, and registered with the global redactor so any log line or config snapshot that
contains their value is scrubbed before it is written.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Mapping
from typing import Any

REDACTED = "***REDACTED***"

# Shapes of credentials that must never appear in logs even if not registered.
_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(
        r"(?i)\b(KALSHI-ACCESS-SIGNATURE|POLY_SIGNATURE|POLY_API_KEY|POLY_PASSPHRASE|"
        r"authorization|api[_-]?key|api[_-]?secret|secret|token|passphrase|signature)"
        r"(\"?\s*[:=]\s*\"?)([^\s\",}]+)"
    ),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
]


class _SecretRegistry:
    def __init__(self) -> None:
        self._values: set[str] = set()
        self._lock = threading.Lock()

    def add(self, value: str) -> None:
        if value and len(value) >= 4:
            with self._lock:
                self._values.add(value)

    def values(self) -> list[str]:
        with self._lock:
            return sorted(self._values, key=len, reverse=True)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()


SECRETS = _SecretRegistry()


class Secret:
    """Opaque wrapper; use :meth:`reveal` only at the point of signing a request."""

    __slots__ = ("_name", "_value")

    def __init__(self, name: str, value: str) -> None:
        self._name = name
        self._value = value
        SECRETS.add(value)

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"Secret({self._name}={REDACTED})"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Secret) and other._value == self._value

    def __hash__(self) -> int:
        return hash(("Secret", self._name))


def load_secret(env_var: str, *, required: bool = True) -> Secret | None:
    value = os.environ.get(env_var)
    if not value:
        if required:
            raise KeyError(f"secret environment variable {env_var} is not set")
        return None
    return Secret(env_var, value)


def redact(text: str) -> str:
    for value in SECRETS.values():
        text = text.replace(value, REDACTED)
    text = _PATTERNS[0].sub(REDACTED, text)
    text = _PATTERNS[1].sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    return _PATTERNS[2].sub(f"Bearer {REDACTED}", text)


def redact_mapping(data: Any) -> Any:
    """Recursively redact strings inside config snapshots / structured log payloads."""
    if isinstance(data, Mapping):
        return {k: redact_mapping(v) for k, v in data.items()}
    if isinstance(data, list | tuple):
        return [redact_mapping(v) for v in data]
    if isinstance(data, Secret):
        return REDACTED
    if isinstance(data, str):
        return redact(data)
    return data


class RedactingFilter(logging.Filter):
    """Logging filter that scrubs secrets from the fully formatted message."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        record.msg = redact(msg)
        record.args = None
        return True
