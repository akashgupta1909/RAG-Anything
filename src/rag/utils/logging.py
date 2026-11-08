"""Logging setup: human-readable console output plus JSON-lines files in ``logs/``."""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
from datetime import UTC, datetime
from pathlib import Path

# Correlates every log line of one API request (set by the API middleware).
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)

_NOISY_LOGGERS = ("httpx", "httpcore", "huggingface_hub", "urllib3", "openai")
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"asctime", "message"}


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, including any ``extra={...}`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure the ``rag`` logger once per process (idempotent)."""
    logger = logging.getLogger("rag")
    if getattr(logger, "_configured", False):
        return
    logger.setLevel(level.upper())
    logger.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"
        )
    )
    console.addFilter(_RequestIdFilter())
    logger.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "rag.jsonl", maxBytes=10_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(_RequestIdFilter())
        logger.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    logger._configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """A child of the ``rag`` logger, e.g. ``get_logger(__name__)``."""
    return logging.getLogger(name if name.startswith("rag") else f"rag.{name}")
