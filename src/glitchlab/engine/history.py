"""Undo/redo history of document recipes.

Only small JSON recipes are stored - never image buffers - and the history is
bounded both in entry count and total size.  Rapid successive changes with
the same *coalesce key* (e.g. dragging one slider) collapse into one step.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass


@dataclass
class _Entry:
    state: str
    label: str
    key: str | None
    stamp: float


class History:
    def __init__(self, limit: int = 200, max_bytes: int = 64 * 1024 * 1024):
        self.limit = limit
        self.max_bytes = max_bytes
        self._entries: list[_Entry] = []
        self._pos = -1

    def reset(self, state: dict, label: str = "Initial state") -> None:
        self._entries = [_Entry(json.dumps(state), label, None, time.monotonic())]
        self._pos = 0

    def push(self, state: dict, label: str, coalesce_key: str | None = None, window: float = 1.2) -> None:
        s = json.dumps(state)
        now = time.monotonic()
        if self._pos >= 0 and self._entries[self._pos].state == s:
            return
        del self._entries[self._pos + 1:]
        top = self._entries[-1] if self._entries else None
        if (top is not None and coalesce_key is not None and top.key == coalesce_key
                and now - top.stamp < window and self._pos > 0):
            top.state = s
            top.stamp = now
            return
        self._entries.append(_Entry(s, label, coalesce_key, now))
        self._pos = len(self._entries) - 1
        self._trim()

    def _trim(self) -> None:
        while len(self._entries) > self.limit:
            self._entries.pop(0)
            self._pos -= 1
        total = sum(len(e.state) for e in self._entries)
        while total > self.max_bytes and len(self._entries) > 2:
            total -= len(self._entries[0].state)
            self._entries.pop(0)
            self._pos -= 1

    def break_coalescing(self) -> None:
        if self._entries:
            self._entries[-1].key = None

    @property
    def can_undo(self) -> bool:
        return self._pos > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self._pos < len(self._entries) - 1

    def undo_label(self) -> str:
        return self._entries[self._pos].label if self.can_undo else ""

    def redo_label(self) -> str:
        return self._entries[self._pos + 1].label if self.can_redo else ""

    def undo(self) -> dict | None:
        if not self.can_undo:
            return None
        self._pos -= 1
        self._entries[self._pos].key = None
        return json.loads(self._entries[self._pos].state)

    def redo(self) -> dict | None:
        if not self.can_redo:
            return None
        self._pos += 1
        return json.loads(self._entries[self._pos].state)

    def current(self) -> dict | None:
        if self._pos < 0:
            return None
        return json.loads(self._entries[self._pos].state)

    def __len__(self) -> int:
        return len(self._entries)
