"""Tests for api.job_store — covers state transitions, locking, persistence,
and cancellation. The store is process-global, so each test creates jobs
with unique IDs and never asserts on the global count.
"""

from __future__ import annotations

import threading
import time

import pytest

from api import job_store
from api.job_store import (
    JobStatus,
    add_event,
    begin_stage,
    complete_stage,
    create_job,
    fail_stage,
    get_job,
    is_cancelled,
    list_jobs,
    request_cancel,
    skip_stage,
    update_job,
)


def test_create_job_returns_pending_with_default_stages():
    job = create_job(kind="document")
    assert job.status == JobStatus.PENDING
    assert job.stages, "default stages should not be empty"
    assert get_job(job.job_id) is job


def test_update_job_clamps_progress():
    job = create_job(kind="document")
    update_job(job.job_id, progress=2.5)
    assert get_job(job.job_id).progress == 1.0
    update_job(job.job_id, progress=-1.0)
    assert get_job(job.job_id).progress == 0.0


def test_update_job_unknown_id_is_noop():
    update_job("does-not-exist", status=JobStatus.RUNNING)
    assert get_job("does-not-exist") is None


def test_request_cancel_sets_flag_and_emits_event():
    job = create_job(kind="document")
    assert is_cancelled(job.job_id) is False
    assert request_cancel(job.job_id) is True
    assert is_cancelled(job.job_id) is True
    # `request_cancel` adds a warn-level event explaining the action.
    msgs = [e.get("message", "") for e in get_job(job.job_id).events]
    assert any("Cancellation requested" in m for m in msgs)


def test_request_cancel_terminal_job_returns_false():
    job = create_job(kind="document")
    update_job(job.job_id, status=JobStatus.COMPLETED)
    assert request_cancel(job.job_id) is False


def test_stage_helpers_set_status_and_event():
    job = create_job(kind="document", stages=["fetch", "parse"])
    begin_stage(job.job_id, "fetch")
    assert get_job(job.job_id).stage_progress["fetch"] == "running"
    complete_stage(job.job_id, "fetch")
    assert get_job(job.job_id).stage_progress["fetch"] == "completed"
    skip_stage(job.job_id, "parse", message="nothing to parse")
    assert get_job(job.job_id).stage_progress["parse"] == "skipped"

    job2 = create_job(kind="document", stages=["fetch"])
    fail_stage(job2.job_id, "fetch", message="boom")
    assert get_job(job2.job_id).stage_progress["fetch"] == "failed"


def test_list_jobs_returns_newest_first():
    a = create_job(kind="document")
    time.sleep(0.001)
    b = create_job(kind="document")
    jobs = list_jobs(limit=10)
    ids = [j.job_id for j in jobs]
    # b was created after a, so b must come before a in the list.
    assert ids.index(b.job_id) < ids.index(a.job_id)


def test_concurrent_updates_do_not_lose_events():
    """Smoke-test that the RLock around add_event prevents lost appends.

    Without locking, `job.events.append(...)` from many threads racing on
    list mutation will sometimes lose events. With locking we expect
    every append to land.
    """
    job = create_job(kind="document")
    n_threads = 16
    n_per_thread = 50

    def worker(idx: int) -> None:
        for k in range(n_per_thread):
            add_event(job.job_id, message=f"thread-{idx}-msg-{k}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = get_job(job.job_id)
    assert len(final.events) >= n_threads * n_per_thread


def test_concurrent_cancel_and_query_dont_crash():
    """A cancel can land mid-read without raising.

    The lock makes is_cancelled / get_job consistent with request_cancel.
    """
    job = create_job(kind="document")
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            _ = is_cancelled(job.job_id)
            _ = get_job(job.job_id)

    rs = [threading.Thread(target=reader) for _ in range(8)]
    for t in rs:
        t.start()
    request_cancel(job.job_id)
    stop.set()
    for t in rs:
        t.join()
    assert is_cancelled(job.job_id) is True
