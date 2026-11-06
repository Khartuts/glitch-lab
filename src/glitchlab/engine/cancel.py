"""Cooperative cancellation shared between UI and worker threads."""

from __future__ import annotations

import threading
import time

from .errors import Cancelled


class CancelToken:
    """Thread-safe cancellation flag.

    Effects call :meth:`check` between expensive steps; external processes
    (FFmpeg) poll :attr:`cancelled` and are killed when it becomes true.
    An optional deadline turns the token into a timeout.
    """

    def __init__(self, deadline: float | None = None):
        self._event = threading.Event()
        self._deadline = deadline

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        if self._event.is_set():
            return True
        if self._deadline is not None and time.monotonic() > self._deadline:
            return True
        return False

    def check(self) -> None:
        if self.cancelled:
            raise Cancelled()


NEVER_CANCELLED = CancelToken()
