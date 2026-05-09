"""
Observability primitives for the RAG Q&A path (§8 of docs/RAG_QA_PLAN.md).

Two roles:

1. **request_id propagation** — every chat turn enters the API and is
   stamped with a uuid4. The id is held in a ``contextvars.ContextVar``
   so it survives across ``await`` boundaries and reaches every logger
   in the same task without needing to be threaded through call sites.
   FastAPI middleware in ``api/middleware.py`` sets it; loggers read it
   via the filter installed by :func:`install_request_id_filter`.

2. **`graphbuilder.qa.*` logger namespace** — distinct from the existing
   ``graphbuilder.*`` namespaces so the chat path can be enabled or
   filtered independently. :func:`get_qa_logger` returns a logger with
   the request-id filter attached.

The metrics extension lives in ``metrics.py`` (the singleton already
in use) so callers don't need to know about a separate module.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import Optional

# ----------------------------------------------------------------------
# request_id contextvar
# ----------------------------------------------------------------------

_request_id_var: ContextVar[Optional[str]] = ContextVar(
    "graphbuilder_qa_request_id", default=None
)


def new_request_id() -> str:
    """Generate a fresh request id. Short uuid4 hex prefix for log-friendliness."""
    return f"req_{uuid.uuid4().hex[:12]}"


def set_request_id(request_id: Optional[str]) -> object:
    """Attach a request id to the current context. Returns a token that can
    be passed to :func:`reset_request_id` to undo the change (mirrors the
    ``ContextVar.set`` / ``reset`` pair). Pass ``None`` to clear."""
    return _request_id_var.set(request_id)


def reset_request_id(token) -> None:
    _request_id_var.reset(token)


def get_request_id() -> Optional[str]:
    """Return the request id bound to the current context, or ``None``."""
    return _request_id_var.get()


# ----------------------------------------------------------------------
# Logger filter
# ----------------------------------------------------------------------

class RequestIdFilter(logging.Filter):
    """Inject the current request_id into every LogRecord.

    The attribute name ``request_id`` is added unconditionally — if no id
    is set, the value is the literal string ``"-"``. That keeps log
    formatters that reference ``%(request_id)s`` from raising KeyError.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


_QA_NAMESPACES = (
    "graphbuilder.qa",
    "graphbuilder.qa.api",
    "graphbuilder.qa.planner",
    "graphbuilder.qa.retrieval",
    "graphbuilder.qa.memory",
    "graphbuilder.qa.llm",
    "graphbuilder.qa.tools",
    "graphbuilder.qa.mutations",
    "graphbuilder.qa.faithfulness",
)


def install_request_id_filter(extra_namespaces: Optional[list[str]] = None) -> None:
    """Idempotently attach a :class:`RequestIdFilter` to all qa loggers.

    Safe to call multiple times — adds at most one filter per logger by
    checking for an existing instance first.
    """
    namespaces = list(_QA_NAMESPACES) + list(extra_namespaces or [])
    for name in namespaces:
        logger = logging.getLogger(name)
        if not any(isinstance(f, RequestIdFilter) for f in logger.filters):
            logger.addFilter(RequestIdFilter())


def get_qa_logger(suffix: str) -> logging.Logger:
    """Return ``graphbuilder.qa.<suffix>`` with the request_id filter attached.

    The filter is installed on the namespace tree on first use, so log
    records emitted by this logger will carry the current request_id even
    if the caller never directly invokes ``install_request_id_filter``.
    """
    install_request_id_filter()
    return logging.getLogger(f"graphbuilder.qa.{suffix}")


__all__ = [
    "RequestIdFilter",
    "get_qa_logger",
    "get_request_id",
    "install_request_id_filter",
    "new_request_id",
    "reset_request_id",
    "set_request_id",
]
