"""Background AI jobs and the inference cache.

Model inference runs on one dedicated worker thread, never on the UI thread
and not on the preview thread either.  A preview render that needs an
analysis *waits* for the job; if the preview is restarted (for example
because a styling slider moved) the job keeps running and the new preview
waits for the same result, so tweaking styling never restarts inference.
A job is cancelled when the user cancels AI work, or when a newer request
for the same effect needs different inference and nobody waits any more.

Results are cached in memory (bounded) and, for expensive generations, on
disk, so composition changes and reopening a project are immediate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..engine.cancel import CancelToken
from ..engine.errors import Cancelled
from .paths import cache_dir

log = logging.getLogger(__name__)

CACHE_VERSION = 1


def make_key(*parts: Any) -> str:
    blob = json.dumps([CACHE_VERSION, *parts], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _nbytes(obj: Any) -> int:
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(_nbytes(v) for v in obj.values()) + 64
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(v) for v in obj) + 64
    return 64


class MemoryCache:
    def __init__(self, max_bytes: int = 1536 * 2**20):
        self.max_bytes = max_bytes
        self._d: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            v = self._d.get(key)
            if v is None:
                return None
            self._d.move_to_end(key)
            return v[0]

    def put(self, key: str, obj: Any) -> None:
        n = _nbytes(obj)
        if n > self.max_bytes // 2:
            return
        with self._lock:
            old = self._d.pop(key, None)
            if old:
                self._bytes -= old[1]
            self._d[key] = (obj, n)
            self._bytes += n
            while self._bytes > self.max_bytes and self._d:
                _, (_, m) = self._d.popitem(last=False)
                self._bytes -= m

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self._bytes = 0


class DiskCache:
    """Dict-of-arrays results stored as .npz files, trimmed least-recently-used first."""

    def __init__(self, root: Path | None = None, max_bytes: int = 4 * 2**30):
        self._root = root
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        r = self._root or cache_dir()
        r.mkdir(parents=True, exist_ok=True)
        return r

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.npz"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            with np.load(p, allow_pickle=False) as z:
                out = {k: z[k] for k in z.files}
            os.utime(p)
            return out
        except Exception:  # damaged file: forget it
            log.warning("Discarding damaged AI cache entry %s", p)
            p.unlink(missing_ok=True)
            return None

    def put(self, key: str, arrays: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        with self._lock:
            np.savez(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
            os.replace(tmp, p)
            self._trim()

    def _trim(self) -> None:
        files = [(f.stat().st_mtime, f.stat().st_size, f) for f in self.root.glob("*/*.npz")]
        total = sum(s for _, s, _ in files)
        for _, s, f in sorted(files):
            if total <= self.max_bytes:
                break
            f.unlink(missing_ok=True)
            total -= s

    def size(self) -> int:
        return sum(f.stat().st_size for f in self.root.glob("*/*.npz"))

    def clear(self) -> None:
        for f in self.root.glob("*/*.npz"):
            f.unlink(missing_ok=True)


@dataclass
class JobContext:
    """Passed to a job function: cancellation and progress reporting."""

    cancel: CancelToken
    _job: "Job"

    def check(self) -> None:
        self.cancel.check()

    def progress(self, fraction: float, message: str = "") -> None:
        self._job.set_progress(fraction, message)


@dataclass
class Job:
    key: str
    label: str
    fn: Callable[[JobContext], Any]
    owner: str | None
    persist: bool
    manager: "JobManager"
    cancel: CancelToken = field(default_factory=CancelToken)
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None
    waiters: int = 0
    fraction: float = 0.0
    message: str = ""
    started: float = 0.0
    id: int = 0
    _listeners: list = field(default_factory=list)

    def set_progress(self, fraction: float, message: str = "") -> None:
        self.fraction = float(min(max(fraction, 0.0), 1.0))
        if message:
            self.message = message
        self.manager._emit("progress", self)
        for cb in list(self._listeners):
            try:
                cb(self.fraction)
            except Exception:  # pragma: no cover
                pass


class JobManager:
    def __init__(self):
        self.memory = MemoryCache()
        self.disk = DiskCache()
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        self._listeners: list[Callable[[str, Job], None]] = []
        self._thread: threading.Thread | None = None
        self._next_id = 0
        self.current: Job | None = None

    # ------------------------------------------------------------ listeners
    def add_listener(self, fn: Callable[[str, Job], None]) -> None:
        self._listeners.append(fn)

    def remove_listener(self, fn) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _emit(self, event: str, job: Job) -> None:
        for fn in list(self._listeners):
            try:
                fn(event, job)
            except Exception:  # pragma: no cover - UI problems must not kill jobs
                log.exception("AI job listener failed")

    # --------------------------------------------------------------- cache
    def cached(self, key: str, persist: bool = False) -> Any:
        hit = self.memory.get(key)
        if hit is None and persist:
            hit = self.disk.get(key)
            if hit is not None:
                self.memory.put(key, hit)
        return hit

    def store(self, key: str, value: Any, persist: bool = False) -> None:
        self.memory.put(key, value)
        if persist and isinstance(value, dict):
            try:
                self.disk.put(key, value)
            except OSError as exc:  # disk full etc.: memory cache still works
                log.warning("Could not write AI cache: %s", exc)

    # ---------------------------------------------------------------- jobs
    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, name="glitchlab-ai", daemon=True)
            self._thread.start()

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            if job.cancel.cancelled:
                job.error = Cancelled()
                self._finish(job, "cancelled")
                continue
            self.current = job
            job.started = time.monotonic()
            self._emit("start", job)
            try:
                job.result = job.fn(JobContext(job.cancel, job))
                self.store(job.key, job.result, job.persist)
                self._finish(job, "done")
            except Cancelled as exc:
                job.error = exc
                self._finish(job, "cancelled")
            except BaseException as exc:  # reported to every waiter
                log.exception("AI job %s failed", job.label)
                job.error = exc
                self._finish(job, "error")
            finally:
                self.current = None

    def _finish(self, job: Job, event: str) -> None:
        with self._lock:
            if self._jobs.get(job.key) is job:
                del self._jobs[job.key]
        job.done.set()
        self._emit(event, job)

    def submit(self, key: str, label: str, fn: Callable[[JobContext], Any], *, owner: str | None = None,
               persist: bool = False, waiter: bool = False) -> Job:
        with self._lock:
            job = self._jobs.get(key)
            if job is not None and not job.cancel.cancelled:
                if waiter:
                    job.waiters += 1
                return job
            # A newer request from the same effect supersedes its abandoned older jobs.
            if owner is not None:
                for other in list(self._jobs.values()):
                    if other.owner == owner and other.key != key and other.waiters <= 0:
                        other.cancel.cancel()
            self._next_id += 1
            job = Job(key, label, fn, owner, persist, self, id=self._next_id)
            if waiter:
                job.waiters = 1
            self._jobs[key] = job
        self._ensure_thread()
        self._queue.put(job)
        self._emit("queued", job)
        return job

    def wait(self, job: Job, cancel: CancelToken | None = None,
             progress: Callable[[float], None] | None = None, *, counted: bool = False) -> Any:
        """Wait for ``job``. Cancelling ``cancel`` stops waiting but not the job."""
        with self._lock:
            if not counted:
                job.waiters += 1
            if progress is not None:
                job._listeners.append(progress)
        try:
            while not job.done.wait(0.05):
                if cancel is not None and cancel.cancelled:
                    raise Cancelled()
        finally:
            with self._lock:
                job.waiters -= 1
                if progress is not None and progress in job._listeners:
                    job._listeners.remove(progress)
        if job.error is not None:
            raise job.error
        return job.result

    def run(self, key: str, label: str, fn: Callable[[JobContext], Any], *, owner: str | None = None,
            persist: bool = False, cancel: CancelToken | None = None,
            progress: Callable[[float], None] | None = None) -> Any:
        hit = self.cached(key, persist)
        if hit is not None:
            return hit
        job = self.submit(key, label, fn, owner=owner, persist=persist, waiter=True)
        return self.wait(job, cancel, progress, counted=True)

    def cancel_all(self) -> int:
        with self._lock:
            jobs = list(self._jobs.values())
        for j in jobs:
            j.cancel.cancel()
        return len(jobs)

    def in_flight(self, key: str) -> bool:
        with self._lock:
            job = self._jobs.get(key)
            return job is not None and not job.cancel.cancelled

    def busy(self) -> bool:
        with self._lock:
            return bool(self._jobs)


# ------------------------------------------------------------ approvals
_APPROVED: set[str] = set()
_ALOCK = threading.Lock()


def approve(owner: str) -> None:
    """Allow the next expensive generation of effect ``owner`` in interactive previews."""
    with _ALOCK:
        _APPROVED.add(owner)


def take_approval(owner: str) -> bool:
    with _ALOCK:
        if owner in _APPROVED:
            _APPROVED.discard(owner)
            return True
        return False


_MANAGER: JobManager | None = None
_MLOCK = threading.Lock()


def manager() -> JobManager:
    global _MANAGER
    with _MLOCK:
        if _MANAGER is None:
            _MANAGER = JobManager()
        return _MANAGER
