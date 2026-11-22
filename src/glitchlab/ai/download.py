"""Small, dependency-free HTTPS downloader with progress, cancellation,
resume and checksum verification (used for model weights and the AI engine)."""

from __future__ import annotations

import hashlib
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .. import __version__
from ..engine.cancel import CancelToken
from ..engine.errors import Cancelled
from .errors import AIError

#: progress(done_bytes, total_bytes, label)
ByteProgress = Callable[[int, int, str], None]
CHUNK = 1 << 20
USER_AGENT = f"GlitchLab/{__version__} (local model download)"


class DownloadError(AIError):
    pass


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(path: Path, *, size: int | None, sha256: str | None = None, git_sha1: str | None = None) -> bool:
    try:
        st = path.stat()
    except OSError:
        return False
    if size is not None and st.st_size != size:
        return False
    if sha256:
        return file_sha256(path) == sha256
    if git_sha1:
        return git_blob_sha1(path.read_bytes()) == git_sha1
    return True


def download(url: str, dest: Path, *, size: int | None, sha256: str | None = None, git_sha1: str | None = None,
             cancel: CancelToken | None = None, progress: ByteProgress | None = None, label: str = "",
             base_done: int = 0, grand_total: int | None = None, retries: int = 4) -> None:
    """Download ``url`` to ``dest`` (atomically) and verify size and checksum.

    A partial ``dest.part`` is resumed with an HTTP range request.  Progress
    is reported as ``base_done + bytes of this file`` out of ``grand_total``.
    """
    cancel = cancel or CancelToken()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    total = grand_total if grand_total is not None else (size or 0)
    attempt = 0
    while True:
        cancel.check()
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have > size:
            part.unlink()
            have = 0
        if size is not None and have == size:
            break
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:  # server ignored the range: start again
                    have = 0
                mode = "ab" if have else "wb"
                with open(part, mode) as f:
                    done = have
                    last = 0.0
                    while True:
                        if cancel.cancelled:
                            raise Cancelled()
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        f.write(block)
                        done += len(block)
                        now = time.monotonic()
                        if progress and (now - last > 0.1):
                            last = now
                            progress(base_done + done, total, label)
            if size is None or part.stat().st_size >= size:
                break
            raise DownloadError(f"The download of {label or url} stopped early.")
        except Cancelled:
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, DownloadError) as exc:
            attempt += 1
            if attempt > retries:
                if isinstance(exc, urllib.error.HTTPError):
                    msg = f"The server refused the download of {label or url} (HTTP {exc.code})."
                else:
                    msg = (f"Could not download {label or url}. Check the internet connection "
                           f"(or a firewall/proxy) and try again.")
                raise DownloadError(msg, detail=repr(exc)) from exc
            for _ in range(int(2 ** attempt * 10)):
                cancel.check()
                time.sleep(0.1)
    if progress:
        progress(base_done + (size or part.stat().st_size), total, label)
    if not verify_file(part, size=size, sha256=sha256, git_sha1=git_sha1):
        part.unlink(missing_ok=True)
        raise DownloadError(f"The downloaded file {label or dest.name} is damaged (checksum mismatch). "
                            "Please try again.")
    os.replace(part, dest)
