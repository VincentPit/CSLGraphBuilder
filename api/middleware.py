"""FastAPI middleware — request-id propagation for the QA observability spine.

Every incoming request gets a ``request_id`` (uuid4 hex prefix). The id is:

- read from the ``X-Request-Id`` request header if the caller supplied one
  (so a frontend that wants its own correlation id can pass it through),
- otherwise generated server-side via :func:`new_request_id`,
- stored in a ``contextvars.ContextVar`` for the duration of the request
  so every logger and metric in the same task can read it,
- echoed back in the ``X-Request-Id`` response header for client-side
  log stitching.
"""

from __future__ import annotations

import sys
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Make sure the src/ package is importable when running from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphbuilder.infrastructure.services.qa_observability import (  # noqa: E402
    new_request_id,
    reset_request_id,
    set_request_id,
)


_REQUEST_ID_HEADER = "X-Request-Id"


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get(_REQUEST_ID_HEADER)
        request_id = incoming or new_request_id()

        token = set_request_id(request_id)
        # Attach to request.state so route handlers can read it without
        # touching the contextvar API.
        request.state.request_id = request_id
        try:
            response: Response = await call_next(request)
        finally:
            reset_request_id(token)
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response


__all__ = ["RequestIdMiddleware"]
