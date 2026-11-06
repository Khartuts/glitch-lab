"""Background task runner.

Each *lane* (``preview``, ``full``, ``variations``, ``export``) has its own
single worker thread.  Submitting a new job to a lane cancels the previous
one (cooperatively, via its CancelToken - FFmpeg children are killed) and
results from superseded jobs are dropped, so the canvas never shows a stale
preview.  NumPy/OpenCV release the GIL for heavy work, keeping the UI fluid.
"""

from __future__ import annotations

import itertools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from PySide6.QtCore import QObject, Signal

from ..engine.cancel import CancelToken
from ..engine.errors import Cancelled

log = logging.getLogger(__name__)


class _Bridge(QObject):
    done = Signal(int, object, object)
    progress = Signal(int, float, str)


@dataclass
class Job:
    id: int
    lane: str
    token: CancelToken = field(default_factory=CancelToken)
    on_done: Callable[[Any], None] | None = None
    on_error: Callable[[BaseException], None] | None = None
    on_progress: Callable[[float, str], None] | None = None
    on_cancel: Callable[[], None] | None = None
    started: float = field(default_factory=time.monotonic)


class TaskRunner(QObject):
    LANES = ("preview", "full", "variations", "export")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pools = {lane: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"glitch-{lane}")
                       for lane in self.LANES}
        self._ids = itertools.count(1)
        self._jobs: dict[int, Job] = {}
        self._current: dict[str, int] = {}
        self._bridge = _Bridge()
        self._bridge.done.connect(self._finished)
        self._bridge.progress.connect(self._progress)

    def submit(self, lane: str, fn: Callable[[CancelToken, Callable[[float, str], None]], Any], *,
               on_done=None, on_error=None, on_progress=None, on_cancel=None) -> Job:
        """Run ``fn(token, progress)`` on the lane's worker, cancelling its previous job."""
        self.cancel(lane)
        job = Job(next(self._ids), lane, on_done=on_done, on_error=on_error,
                  on_progress=on_progress, on_cancel=on_cancel)
        self._jobs[job.id] = job
        self._current[lane] = job.id
        self._pools[lane].submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn) -> None:
        if job.token.cancelled:
            self._bridge.done.emit(job.id, None, Cancelled())
            return
        last = [0.0]

        def progress(frac: float, msg: str = "") -> None:
            now = time.monotonic()
            if now - last[0] > 0.05 or frac >= 1.0:
                last[0] = now
                self._bridge.progress.emit(job.id, float(frac), msg)

        try:
            result = fn(job.token, progress)
            self._bridge.done.emit(job.id, result, None)
        except BaseException as exc:  # delivered to the UI thread
            self._bridge.done.emit(job.id, None, exc)

    def _finished(self, job_id: int, result, error) -> None:
        job = self._jobs.pop(job_id, None)
        if job is None:
            return
        is_current = self._current.get(job.lane) == job_id
        if is_current:
            self._current.pop(job.lane, None)
        cancelled = isinstance(error, Cancelled) or job.token.cancelled
        if not is_current:
            return  # superseded by a newer job in the same lane: drop silently
        if cancelled:
            if job.on_cancel:
                try:
                    job.on_cancel()
                except Exception:
                    log.exception("on_cancel handler failed")
            return
        if error is not None:
            if job.on_error:
                job.on_error(error)
            else:
                log.error("Background job failed: %r", error)
            return
        if job.on_done:
            job.on_done(result)

    def _progress(self, job_id: int, frac: float, msg: str) -> None:
        job = self._jobs.get(job_id)
        if job and job.on_progress and self._current.get(job.lane) == job_id and not job.token.cancelled:
            job.on_progress(frac, msg)

    def cancel(self, lane: str) -> bool:
        jid = self._current.get(lane)
        if jid is None:
            return False
        job = self._jobs.get(jid)
        if job:
            job.token.cancel()
        return True

    def cancel_all(self) -> None:
        for lane in self.LANES:
            self.cancel(lane)

    def busy(self, lane: str) -> bool:
        return lane in self._current

    def shutdown(self) -> None:
        self.cancel_all()
        for pool in self._pools.values():
            pool.shutdown(wait=False, cancel_futures=True)
