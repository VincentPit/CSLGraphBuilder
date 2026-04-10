"""
Structured JSON Logging for GraphBuilder.

Provides a drop-in ``logging.Handler`` and a factory function that emit
every log record as a single-line JSON object, suitable for ingestion by
log aggregation systems (Datadog, Splunk, Cloud Logging, etc.).

Typical setup::

    from graphbuilder.infrastructure.logging.json_logger import configure_json_logging
    configure_json_logging(level="INFO", log_file="logs/app.log")

    import logging
    log = logging.getLogger(__name__)
    log.info("document processed", extra={"document_id": "abc", "chunks": 12})

Every emitted line looks like::

    {"timestamp": "2026-04-09T12:00:00.123Z", "level": "INFO",
     "logger": "graphbuilder.core", "message": "document processed",
     "document_id": "abc", "chunks": 12}
"""

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# Fields that are already captured at the top level; do not re-emit from `extra`.
_RESERVED = frozenset(
    {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """
    Formats ``LogRecord`` objects as single-line JSON strings.

    Parameters
    ----------
    service:
        Optional service name added to every record as ``"service"``.
    extra_fields:
        Static key-value pairs merged into every record.
    """

    def __init__(
        self,
        service: Optional[str] = "graphbuilder",
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._service = service
        self._extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }

        if self._service:
            payload["service"] = self._service

        # Merge static extras
        payload.update(self._extra_fields)

        # Merge dynamic extras from the log call
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        # Exception info
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exception"] = record.exc_text

        # Stack info
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str)


class JsonHandler(logging.StreamHandler):
    """
    ``StreamHandler`` that uses ``JsonFormatter`` by default.

    Parameters
    ----------
    stream:
        Output stream; defaults to ``sys.stdout``.
    service:
        Passed through to ``JsonFormatter``.
    extra_fields:
        Static extra fields passed through to ``JsonFormatter``.
    """

    def __init__(
        self,
        stream=None,
        service: Optional[str] = "graphbuilder",
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(stream or sys.stdout)
        self.setFormatter(JsonFormatter(service=service, extra_fields=extra_fields))


def configure_json_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    service: Optional[str] = "graphbuilder",
    extra_fields: Optional[Dict[str, Any]] = None,
    replace_existing_handlers: bool = True,
) -> logging.Logger:
    """
    Configure the root logger to emit structured JSON.

    Parameters
    ----------
    level:
        Logging level string, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
    log_file:
        Optional file path.  When provided, a ``FileHandler`` that also
        emits JSON is attached alongside the console handler.
    service:
        Service name embedded in every log record.
    extra_fields:
        Static key-value pairs merged into every record.
    replace_existing_handlers:
        When ``True`` (default) all existing root handlers are removed first
        so the application does not double-log.

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    root = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)

    if replace_existing_handlers:
        root.handlers.clear()

    # Console handler → stdout
    console_handler = JsonHandler(
        stream=sys.stdout, service=service, extra_fields=extra_fields
    )
    console_handler.setLevel(numeric_level)
    root.addHandler(console_handler)

    # Optional file handler
    if log_file:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            JsonFormatter(service=service, extra_fields=extra_fields)
        )
        file_handler.setLevel(numeric_level)
        root.addHandler(file_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Thin wrapper around ``logging.getLogger`` for convenience."""
    return logging.getLogger(name)
