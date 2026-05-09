"""Tests for the QA observability primitives (§8 of docs/RAG_QA_PLAN.md).

Covers:
- request_id contextvar set/get/reset
- RequestIdFilter injects request_id into log records
- Concurrent asyncio tasks see independent request_ids (no cross-talk)
- get_qa_logger returns a properly-namespaced logger
- PipelineMetrics qa_* recorders update snapshot correctly
"""

from __future__ import annotations

import asyncio
import logging
import os

import pytest

os.environ.setdefault("LLM_API_KEY", "not-configured")

from graphbuilder.infrastructure.services.metrics import (  # noqa: E402
    PipelineMetrics,
)
from graphbuilder.infrastructure.services.qa_observability import (  # noqa: E402
    RequestIdFilter,
    get_qa_logger,
    get_request_id,
    install_request_id_filter,
    new_request_id,
    reset_request_id,
    set_request_id,
)


# ---------------------------------------------------------------- request_id


def test_new_request_id_has_expected_prefix():
    rid = new_request_id()
    assert rid.startswith("req_")
    assert len(rid) > len("req_")


def test_set_and_reset_request_id_round_trip():
    assert get_request_id() is None
    token = set_request_id("req_abc")
    try:
        assert get_request_id() == "req_abc"
    finally:
        reset_request_id(token)
    assert get_request_id() is None


def test_set_request_id_to_none_clears():
    token = set_request_id("req_x")
    try:
        token2 = set_request_id(None)
        try:
            assert get_request_id() is None
        finally:
            reset_request_id(token2)
        assert get_request_id() == "req_x"
    finally:
        reset_request_id(token)


async def test_concurrent_tasks_have_independent_request_ids():
    """ContextVar inheritance: each task captures its own snapshot."""
    seen: dict[str, str | None] = {}

    async def task(label: str, rid: str) -> None:
        token = set_request_id(rid)
        try:
            # Yield to the event loop to make sure interleaving doesn't bleed.
            await asyncio.sleep(0)
            seen[label] = get_request_id()
        finally:
            reset_request_id(token)

    await asyncio.gather(task("a", "req_a"), task("b", "req_b"), task("c", "req_c"))
    assert seen == {"a": "req_a", "b": "req_b", "c": "req_c"}
    # The outer task has no id set.
    assert get_request_id() is None


# ---------------------------------------------------------------- log filter


def test_request_id_filter_injects_attribute():
    filt = RequestIdFilter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__,
        lineno=1, msg="hello", args=(), exc_info=None,
    )
    token = set_request_id("req_filter")
    try:
        filt.filter(record)
        assert record.request_id == "req_filter"
    finally:
        reset_request_id(token)


def test_request_id_filter_defaults_to_dash_when_unset():
    filt = RequestIdFilter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__,
        lineno=1, msg="hello", args=(), exc_info=None,
    )
    # No id set → "-" sentinel keeps format strings safe.
    filt.filter(record)
    assert record.request_id == "-"


def test_install_request_id_filter_is_idempotent():
    install_request_id_filter()
    install_request_id_filter()
    install_request_id_filter()
    logger = logging.getLogger("graphbuilder.qa.api")
    rid_filters = [f for f in logger.filters if isinstance(f, RequestIdFilter)]
    assert len(rid_filters) == 1


def test_get_qa_logger_returns_namespaced_logger():
    logger = get_qa_logger("retrieval")
    assert logger.name == "graphbuilder.qa.retrieval"
    # Filter is wired up so format strings referencing %(request_id)s work.
    rid_filters = [f for f in logger.filters if isinstance(f, RequestIdFilter)]
    assert rid_filters, "RequestIdFilter not installed on qa logger namespace"


# ---------------------------------------------------------------- metrics


@pytest.fixture
def metrics() -> PipelineMetrics:
    return PipelineMetrics()


async def test_qa_request_counter(metrics):
    await metrics.record_qa_request(intent="lookup", status="ok")
    await metrics.record_qa_request(intent="lookup", status="ok")
    await metrics.record_qa_request(intent="relational", status="ok")
    snap = metrics.snapshot()["qa"]
    assert snap["requests"] == {"lookup|ok": 2, "relational|ok": 1}


async def test_qa_tool_call_and_mutation_counters(metrics):
    await metrics.record_qa_tool_call(tool="search_graph", outcome="ok")
    await metrics.record_qa_tool_call(tool="propose_entity", outcome="denied")
    await metrics.record_qa_mutation(tool="propose_entity", operation="created")
    snap = metrics.snapshot()["qa"]
    assert snap["tool_calls"]["search_graph|ok"] == 1
    assert snap["tool_calls"]["propose_entity|denied"] == 1
    assert snap["mutations"]["propose_entity|created"] == 1


async def test_qa_faithfulness_failure_increments(metrics):
    await metrics.record_qa_faithfulness_failure()
    await metrics.record_qa_faithfulness_failure(n=3)
    assert metrics.snapshot()["qa"]["faithfulness_failures"] == 4


async def test_qa_latency_histogram_summarises(metrics):
    for s in (0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
        await metrics.record_qa_latency(phase="total", seconds=s)
    h = metrics.snapshot()["qa"]["latency_seconds"]["total"]
    assert h["count"] == 10
    assert h["avg"] == pytest.approx(1.65, rel=0.01)
    assert h["max"] == 5.0
    # p50 between 0.5 and 1.0 (samples sorted: 0.1..0.5, 1.0..5.0)
    assert 0.5 <= h["p50"] <= 1.0
    # p95 close to the top of the range
    assert h["p95"] >= 4.0


async def test_qa_retrieval_hits_per_channel(metrics):
    await metrics.record_qa_retrieval_hits(channel="vector", count=12)
    await metrics.record_qa_retrieval_hits(channel="vector", count=8)
    await metrics.record_qa_retrieval_hits(channel="cypher", count=3)
    snap = metrics.snapshot()["qa"]["retrieval_hits"]
    assert snap["vector"]["count"] == 2
    assert snap["vector"]["avg"] == 10.0
    assert snap["cypher"]["count"] == 1


async def test_qa_active_sessions_and_pending_confirmations(metrics):
    await metrics.set_qa_active_sessions(7)
    await metrics.set_qa_pending_confirmations(2)
    snap = metrics.snapshot()["qa"]
    assert snap["active_sessions"] == 7
    assert snap["pending_confirmations"] == 2


async def test_qa_memory_overflow_drops(metrics):
    await metrics.record_qa_memory_overflow_drop(layer="sources")
    await metrics.record_qa_memory_overflow_drop(layer="sources", n=2)
    await metrics.record_qa_memory_overflow_drop(layer="summary")
    snap = metrics.snapshot()["qa"]["memory_overflow_drops"]
    assert snap == {"sources": 3, "summary": 1}


async def test_histogram_caps_memory_at_max_samples(metrics):
    """Unbounded growth would be a memory leak — verify the cap holds."""
    from graphbuilder.infrastructure.services.metrics import _HISTOGRAM_MAX_SAMPLES

    over_cap = _HISTOGRAM_MAX_SAMPLES + 50
    for i in range(over_cap):
        await metrics.record_qa_latency(phase="total", seconds=float(i))
    h = metrics.snapshot()["qa"]["latency_seconds"]["total"]
    assert h["count"] == _HISTOGRAM_MAX_SAMPLES
    # Most-recent retained: oldest sample dropped, so max should be the last one.
    assert h["max"] == float(over_cap - 1)
