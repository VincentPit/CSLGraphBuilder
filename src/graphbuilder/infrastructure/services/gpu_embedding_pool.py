"""GPU-aware worker pool for parallel embedding inference.

Activated automatically by ``embedding_factory`` when ``torch.cuda.is_available()``
returns true. On CPU the pool is never instantiated and the existing
single-worker lock+executor path stays in effect (per the project
decision: multi-worker is GPU-only).

Architecture
------------
Each worker is a daemon thread that owns:
* its own ``SentenceTransformer`` copy (separate GPU tensors, no shared
  state with other workers)
* a dedicated ``torch.cuda.Stream`` so the GPU scheduler can interleave
  kernels across workers instead of serializing on the default stream

Workers pull from a shared ``queue.Queue`` of ``_Job`` items. "Whoever's
free first grabs the next job" is the load-balancing policy — the
simplest correct algorithm and exactly what the GPU wants (a busy
worker doesn't get a new job, an idle worker does). Per-worker job
counts are tracked for observability via ``jobs_in_flight``.

Sizing
------
At init we load + warm up one model on the target device, then read
``torch.cuda.max_memory_allocated`` to learn the per-worker footprint.
Worker count is::

    n = (free_mem * memory_fraction) // per_worker_bytes
    n = clamp(n, min_workers, max_workers)

Override knobs (env vars, all read by ``embedding_factory``):
    EMBEDDING_GPU_WORKERS           — exact count, bypasses sizing math
    EMBEDDING_GPU_MIN_WORKERS       — floor (default 1)
    EMBEDDING_GPU_MAX_WORKERS       — ceiling (default 8)
    EMBEDDING_GPU_MEMORY_FRACTION   — usable share of free mem (default 0.7)

Async bridge
------------
Public ``embed_async`` / ``embed_batch_async`` create an ``asyncio.Future``
on the calling loop, submit the job to the worker queue, and ``await``
the future. Worker threads resolve the future via
``loop.call_soon_threadsafe`` so the result lands back on the loop
thread cleanly — no asyncio primitives are touched off-loop.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Union


logger = logging.getLogger("graphbuilder.embeddings.gpu_pool")


@dataclass
class _Job:
    payload: Union[str, List[str]]
    is_batch: bool
    batch_size: int
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop


class GPUEmbeddingPool:
    """Multi-worker embedding pool that runs encodes truly in parallel on GPU.

    Workers are spawned in a background thread on construction so the
    constructor returns immediately. The first ``embed_async`` /
    ``embed_batch_async`` call awaits readiness via ``_init_event``
    before submitting; all later calls bypass that wait.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        min_workers: int = 1,
        max_workers: int = 8,
        memory_fraction: float = 0.7,
        explicit_workers: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._min_workers = max(1, int(min_workers))
        self._max_workers = max(self._min_workers, int(max_workers))
        self._mem_fraction = float(memory_fraction)
        self._explicit_workers = explicit_workers

        self._queue: "queue.Queue[Optional[_Job]]" = queue.Queue()
        self._workers: List[threading.Thread] = []
        self._models: List[Any] = []     # one SentenceTransformer per worker
        self._streams: List[Any] = []    # one CUDA stream per worker
        self._jobs_in_flight: List[int] = []
        self._counts_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._init_event = threading.Event()
        self._init_error: Optional[Exception] = None
        self._dim: int = 0

        # Boot in the background so the asyncio loop / FastAPI startup
        # doesn't block on model loading + warmup (~5–15 s for SapBERT).
        threading.Thread(
            target=self._initialize, daemon=True, name="GPUEmbeddingPool-init"
        ).start()

    # ------------------------------------------------------------------
    # Initialization (runs once in a background thread)
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            # Load + warm up one model so we can measure per-worker GPU
            # memory cost (model weights + peak activations for a typical
            # batch). The first model also becomes worker 0 so we don't
            # waste this load.
            torch.cuda.reset_peak_memory_stats(self.device)
            base_model = SentenceTransformer(self.model_name).to(self.device)
            base_stream = torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(base_stream):
                _ = base_model.encode(
                    ["warmup"] * 32,
                    batch_size=32,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            torch.cuda.synchronize(self.device)
            per_worker_bytes = int(torch.cuda.max_memory_allocated(self.device))
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            free_bytes = int(free_bytes)

            try:
                self._dim = int(base_model.get_sentence_embedding_dimension())
            except Exception:
                self._dim = 0

            # Decide worker count
            if self._explicit_workers is not None and self._explicit_workers > 0:
                n = int(self._explicit_workers)
            elif per_worker_bytes <= 0:
                n = self._min_workers
            else:
                budget = int(free_bytes * self._mem_fraction)
                n = budget // per_worker_bytes
            n = max(self._min_workers, min(self._max_workers, n))

            logger.info(
                "GPUEmbeddingPool: device=%s model=%s "
                "per_worker_bytes=%d free=%d -> %d workers",
                self.device, self.model_name, per_worker_bytes, free_bytes, n,
            )

            # Worker 0 reuses the warmed-up model; load the rest.
            self._models.append(base_model)
            self._streams.append(base_stream)
            self._jobs_in_flight.append(0)
            for _ in range(n - 1):
                m = SentenceTransformer(self.model_name).to(self.device)
                s = torch.cuda.Stream(device=self.device)
                self._models.append(m)
                self._streams.append(s)
                self._jobs_in_flight.append(0)

            for i in range(n):
                t = threading.Thread(
                    target=self._worker_loop,
                    args=(i,),
                    daemon=True,
                    name=f"GPUEmbeddingWorker-{i}",
                )
                t.start()
                self._workers.append(t)

            self._init_event.set()

        except Exception as exc:
            logger.error("GPU embedding pool init failed: %s", exc, exc_info=True)
            self._init_error = exc
            self._init_event.set()

    # ------------------------------------------------------------------
    # Worker loop (runs forever in each worker thread)
    # ------------------------------------------------------------------

    def _worker_loop(self, worker_id: int) -> None:
        import torch

        model = self._models[worker_id]
        stream = self._streams[worker_id]

        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Poison pill = clean shutdown signal
            if job is None:
                self._queue.task_done()
                break

            with self._counts_lock:
                self._jobs_in_flight[worker_id] += 1
            try:
                with torch.cuda.stream(stream):
                    if job.is_batch:
                        result = self._encode_batch(model, job.payload, job.batch_size)
                    else:
                        result = self._encode_single(model, job.payload)
                # Bridge back to the calling loop on its own thread.
                if not job.future.cancelled():
                    job.loop.call_soon_threadsafe(job.future.set_result, result)
            except Exception as exc:
                logger.exception("GPUEmbeddingWorker %d encode failed", worker_id)
                if not job.future.cancelled():
                    job.loop.call_soon_threadsafe(job.future.set_exception, exc)
            finally:
                with self._counts_lock:
                    self._jobs_in_flight[worker_id] -= 1
                self._queue.task_done()

    # ------------------------------------------------------------------
    # Encode helpers (mirror the sync embed / embed_batch semantics)
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_single(model: Any, text: str) -> Optional[List[float]]:
        text = (text or "").strip()
        if not text:
            return None
        vec = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vec.tolist()

    @staticmethod
    def _encode_batch(
        model: Any, texts: List[str], batch_size: int
    ) -> List[Optional[List[float]]]:
        if not texts:
            return []
        cleaned: List[str] = []
        keep_idx: List[int] = []
        for i, t in enumerate(texts):
            s = (t or "").strip()
            if s:
                cleaned.append(s)
                keep_idx.append(i)
        out: List[Optional[List[float]]] = [None] * len(texts)
        if not cleaned:
            return out
        vectors = model.encode(
            cleaned,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        for slot, vec in zip(keep_idx, vectors):
            out[slot] = vec.tolist()
        return out

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def _await_ready(self) -> None:
        if self._init_event.is_set():
            if self._init_error:
                raise self._init_error
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._init_event.wait)
        if self._init_error:
            raise self._init_error

    async def embed_async(self, text: str) -> Optional[List[float]]:
        await self._await_ready()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._queue.put(_Job(
            payload=text, is_batch=False, batch_size=0, future=fut, loop=loop,
        ))
        return await fut

    async def embed_batch_async(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Optional[List[float]]]:
        await self._await_ready()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._queue.put(_Job(
            payload=list(texts),
            is_batch=True,
            batch_size=batch_size,
            future=fut,
            loop=loop,
        ))
        return await fut

    # ------------------------------------------------------------------
    # Observability + lifecycle
    # ------------------------------------------------------------------

    @property
    def num_workers(self) -> int:
        return len(self._workers)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def jobs_in_flight(self) -> List[int]:
        with self._counts_lock:
            return list(self._jobs_in_flight)

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def shutdown(self, wait: bool = True, timeout: float = 5.0) -> None:
        """Signal workers to stop and (optionally) join them."""
        self._stop_event.set()
        for _ in self._workers:
            self._queue.put(None)  # poison pill
        if wait:
            for t in self._workers:
                t.join(timeout=timeout)
