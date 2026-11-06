"""External services used by effects: FFmpeg discovery/probing, managed
temporary directories and a cached loader for auxiliary images (palettes,
displacement maps).

FFmpeg is always invoked with an argument list (never a shell string), with
a timeout, and is killed if the render is cancelled.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cancel import CancelToken
from .errors import Cancelled, DependencyError, EffectError

log = logging.getLogger(__name__)

#: Audio filters used by the Sonify effect.
SONIFY_FILTERS = ("acrusher", "volume", "asoftclip", "bass", "treble", "highpass", "lowpass",
                  "bandpass", "aecho", "aphaser", "flanger", "tremolo")

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def app_base_dir() -> Path:
    """Directory of the running application (PyInstaller aware)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def bundled_ffmpeg_candidates() -> list[Path]:
    exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    base = app_base_dir()
    cands = [base / "ffmpeg" / exe, base / exe]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cands += [Path(meipass) / "ffmpeg" / exe, Path(meipass) / exe]
    return cands


@dataclass
class FFmpegInfo:
    path: str
    version: str
    filters: set[str] = field(default_factory=set)
    source: str = ""
    raw_alaw_ok: bool = False

    def missing(self, needed=SONIFY_FILTERS) -> list[str]:
        return [f for f in needed if f not in self.filters]


def _run_capture(args: list[str], timeout: float = 15.0) -> subprocess.CompletedProcess:
    return subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout, creationflags=_CREATE_NO_WINDOW, check=False)


def probe_ffmpeg(path: str) -> FFmpegInfo:
    """Run ``ffmpeg -version`` and ``ffmpeg -filters``; raise DependencyError on failure."""
    try:
        v = _run_capture([path, "-hide_banner", "-version"])
    except FileNotFoundError as exc:
        raise DependencyError(f"FFmpeg was not found at '{path}'.") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DependencyError(f"FFmpeg at '{path}' could not be started: {exc}") from exc
    if v.returncode != 0:
        raise DependencyError(f"'{path}' did not run as FFmpeg (exit code {v.returncode}).")
    first = v.stdout.decode("utf-8", "replace").splitlines()[:1]
    version = first[0] if first else "unknown version"
    if "ffmpeg" not in version.lower():
        raise DependencyError(f"'{path}' does not look like FFmpeg: {version!r}")
    f = _run_capture([path, "-hide_banner", "-filters"])
    filters: set[str] = set()
    for line in f.stdout.decode("utf-8", "replace").splitlines():
        m = re.match(r"^\s*[TSC.]{3}\s+(\S+)\s+\S+->\S+", line)
        if m:
            filters.add(m.group(1))
    info = FFmpegInfo(path=path, version=version, filters=filters)
    info.raw_alaw_ok = _alaw_roundtrip_ok(path)
    return info


def _alaw_roundtrip_ok(path: str) -> bool:
    """Functional probe: raw A-law in -> anull -> raw A-law out must be byte-exact.

    This verifies that this FFmpeg build accepts exactly the raw-audio
    arguments Sonify uses (``-f alaw -ar R -ac 1`` in and out)."""
    data = bytes(range(256)) * 16
    with tempfile.TemporaryDirectory(prefix="glitchlab-ffprobe-", ignore_cleanup_errors=True) as tmp:
        fin, fout = os.path.join(tmp, "in.alaw"), os.path.join(tmp, "out.alaw")
        with open(fin, "wb") as f:
            f.write(data)
        try:
            r = _run_capture([path, "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "alaw", "-ar", "8000",
                              "-ac", "1", "-i", fin, "-af", "anull", "-c:a", "pcm_alaw", "-f", "alaw", "-ar", "8000",
                              "-ac", "1", "-y", fout], timeout=20)
            if r.returncode != 0:
                log.warning("FFmpeg raw A-law probe failed: %s", r.stderr.decode("utf-8", "replace")[-500:])
                return False
            with open(fout, "rb") as f:
                return f.read() == data
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("FFmpeg raw A-law probe could not run: %s", exc)
            return False


class EngineServices:
    """Shared, thread-safe service container passed to effects."""

    def __init__(self, ffmpeg_path: str | None = None, temp_root: str | None = None):
        self._ffmpeg_override = ffmpeg_path or os.environ.get("GLITCHLAB_FFMPEG") or None
        self._ffmpeg_info: FFmpegInfo | None = None
        self._ffmpeg_error: DependencyError | None = None
        self._lock = threading.Lock()
        self.temp_root = temp_root
        self._aux_cache: dict[tuple, np.ndarray] = {}

    # FFmpeg -----------------------------------------------------------
    def set_ffmpeg_path(self, path: str | None) -> None:
        with self._lock:
            self._ffmpeg_override = path or None
            self._ffmpeg_info = None
            self._ffmpeg_error = None

    def locate_ffmpeg(self) -> tuple[str | None, str]:
        if self._ffmpeg_override:
            return self._ffmpeg_override, "configured path"
        for c in bundled_ffmpeg_candidates():
            if c.is_file():
                return str(c), "bundled with Glitch Lab"
        found = shutil.which("ffmpeg")
        if found:
            return found, "system PATH"
        return None, ""

    def ffmpeg(self, needed=SONIFY_FILTERS) -> FFmpegInfo:
        """Return probed FFmpeg info or raise an actionable DependencyError."""
        with self._lock:
            if self._ffmpeg_info is None and self._ffmpeg_error is None:
                path, source = self.locate_ffmpeg()
                if path is None:
                    self._ffmpeg_error = DependencyError(
                        "FFmpeg is required for Sonify but was not found. Place ffmpeg.exe in the "
                        "'ffmpeg' folder next to GlitchLab.exe, add it to PATH, or choose it in "
                        "Edit > Preferences > FFmpeg.")
                else:
                    try:
                        info = probe_ffmpeg(path)
                        info.source = source
                        self._ffmpeg_info = info
                        log.info("FFmpeg: %s (%s) %s", path, source, info.version)
                    except DependencyError as exc:
                        self._ffmpeg_error = exc
            if self._ffmpeg_error is not None:
                raise self._ffmpeg_error
            info = self._ffmpeg_info
        missing = info.missing(needed)
        if missing:
            raise DependencyError(
                f"The FFmpeg build at '{info.path}' lacks audio filter(s): {', '.join(missing)}. "
                "Install a full FFmpeg build (e.g. a current 'essentials' or LGPL build).")
        if not info.raw_alaw_ok:
            raise DependencyError(
                f"The FFmpeg build at '{info.path}' ({info.version}) failed the raw A-law round-trip check that "
                "Sonify relies on. Use a standard FFmpeg 4.4-8.x build (see docs/FFMPEG.md).")
        return info

    def ffmpeg_status(self) -> tuple[bool, str]:
        try:
            info = self.ffmpeg()
            return True, f"{info.version} ({info.source}: {info.path})"
        except DependencyError as exc:
            return False, exc.message

    def run_ffmpeg(self, args: list[str], *, cancel: CancelToken, timeout: float = 120.0) -> None:
        """Run FFmpeg with an argument list; kill it on cancel or timeout."""
        info = self.ffmpeg()
        cmd = [info.path, "-hide_banner", "-nostdin", "-loglevel", "error", *args]
        log.debug("FFmpeg: %s", cmd)
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, creationflags=_CREATE_NO_WINDOW)
        stderr_chunks: list[bytes] = []

        def _drain():
            try:
                for chunk in iter(lambda: proc.stderr.read(4096), b""):
                    if len(stderr_chunks) < 64:
                        stderr_chunks.append(chunk)
            except Exception:
                pass

        t = threading.Thread(target=_drain, daemon=True)
        t.start()
        start = time.monotonic()
        try:
            while True:
                try:
                    rc = proc.wait(timeout=0.05)
                    break
                except subprocess.TimeoutExpired:
                    if cancel.cancelled:
                        raise Cancelled()
                    if time.monotonic() - start > timeout:
                        raise EffectError(f"FFmpeg did not finish within {timeout:.0f} s and was stopped.")
        finally:
            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    log.error("FFmpeg process %s did not exit after kill", proc.pid)
            t.join(timeout=2)
            if proc.stderr:
                proc.stderr.close()
        if rc != 0:
            msg = b"".join(stderr_chunks).decode("utf-8", "replace").strip()
            log.error("FFmpeg failed (%s): %s", rc, msg)
            last = msg.splitlines()[-1] if msg else f"exit code {rc}"
            raise EffectError(f"FFmpeg rejected the audio filter chain: {last}", detail=msg)

    # Temp files ---------------------------------------------------------
    def temp_dir(self, prefix: str = "glitchlab-") -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory(prefix=prefix, dir=self.temp_root, ignore_cleanup_errors=True)

    # Auxiliary images -------------------------------------------------------
    def load_aux_image(self, path: str) -> np.ndarray:
        """Load (and cache) a secondary image as uint8 RGBA in the working space."""
        from .image_io import load_image

        try:
            st = os.stat(path)
        except OSError as exc:
            raise EffectError(f"Auxiliary image not found: {path}") from exc
        key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
        with self._lock:
            hit = self._aux_cache.get(key)
        if hit is not None:
            return hit
        img = load_image(path, max_pixels=16_000_000, downscale_to_fit=True).rgba
        img.flags.writeable = False
        with self._lock:
            if len(self._aux_cache) > 8:
                self._aux_cache.clear()
            self._aux_cache[key] = img
        return img


_DEFAULT: EngineServices | None = None


def default_services() -> EngineServices:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = EngineServices()
    return _DEFAULT
