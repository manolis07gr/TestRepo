"""Structured JSON logging with mandatory secret redaction (scope s.16, s.17)."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, TextIO

from cma.security import RedactingFilter, redact_mapping

_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = {k: v for k, v in vars(record).items() if k not in _RESERVED}
        if extra:
            payload["extra"] = redact_mapping(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging(
    level: str = "INFO", *, json_format: bool = True, stream: TextIO | None = None
) -> logging.Logger:
    """Install a single redacting handler on the ``cma`` logger tree."""
    logger = logging.getLogger("cma")
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.addFilter(RedactingFilter())
    handler.setFormatter(
        JsonFormatter() if json_format else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
