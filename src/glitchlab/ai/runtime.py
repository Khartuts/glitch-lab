"""The optional AI engine: PyTorch + transformers + diffusers.

Development installs get it with ``pip install glitchlab[ai]``.  The frozen
Windows program cannot pip-install, so it downloads the exact wheels listed
in ``runtime_win_amd64.json`` (pinned versions and SHA-256 checksums, CPU
PyTorch for 64-bit Windows / Python 3.13) from PyPI, unpacks them into the
user's data folder and imports them from there.  numpy and Pillow are not
downloaded: the program already contains them.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import platform
import shutil
import sys
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..engine.cancel import CancelToken
from .download import ByteProgress, download
from .errors import AIHardwareError, AIUnavailable
from .paths import runtime_root

log = logging.getLogger(__name__)

REQUIRED = ("torch", "transformers", "diffusers", "safetensors")
MANIFEST = Path(__file__).with_name("runtime_win_amd64.json")
#: Parts of wheels that are never needed at run time (C++ headers, CMake files).
SKIP_PREFIXES = ("torch/include/", "torch/share/", "torch/bin/")

_lock = threading.Lock()
_torch = None
_activated: str | None = None


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def load_manifest() -> dict | None:
    """The wheel manifest if this platform/Python can use it, else ``None``."""
    try:
        m = json.loads(MANIFEST.read_text())
    except (OSError, ValueError):
        return None
    if sys.platform != "win32" or platform.machine().lower() not in ("amd64", "x86_64"):
        return None
    if f"{sys.version_info[0]}.{sys.version_info[1]}" != m.get("python"):
        return None
    return m


def install_dir(manifest: dict | None = None) -> Path | None:
    m = manifest or load_manifest()
    return None if m is None else runtime_root() / m["id"]


def installed() -> bool:
    d = install_dir()
    return bool(d and (d / "COMPLETE").exists())


def activate() -> bool:
    """Put a downloaded AI engine on ``sys.path`` (idempotent)."""
    global _activated
    d = install_dir()
    if d is None or not (d / "COMPLETE").exists():
        return False
    site = str(d / "site-packages")
    with _lock:
        if _activated == site:
            return True
        if site not in sys.path:
            sys.path.append(site)
        importlib.invalidate_caches()
        _activated = site
    log.info("AI engine activated from %s", site)
    return True


def engine_available() -> bool:
    """True when torch/transformers/diffusers can be imported (without importing them)."""
    activate()
    try:
        return all(importlib.util.find_spec(m) is not None for m in REQUIRED)
    except (ImportError, ValueError):
        return False


def can_download() -> bool:
    return load_manifest() is not None


def download_size() -> int:
    m = load_manifest()
    return sum(w["size"] for w in m["wheels"]) if m else 0


def _extract_wheel(path: Path, site: Path) -> None:
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or name.startswith(SKIP_PREFIXES):
                continue
            # "<pkg>.data/purelib|platlib/..." belongs in site-packages; scripts/headers/data are not needed.
            parts = name.split("/")
            if parts[0].endswith(".data"):
                if len(parts) > 2 and parts[1] in ("purelib", "platlib"):
                    name = "/".join(parts[2:])
                else:
                    continue
            target = site / name
            if ".." in Path(name).parts or Path(name).is_absolute():
                raise AIUnavailable("A downloaded AI-engine file has an unsafe path; installation stopped.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)


def install(progress: ByteProgress | None = None, cancel: CancelToken | None = None) -> Path:
    """Download, verify and unpack the AI engine. Returns its folder."""
    cancel = cancel or CancelToken()
    m = load_manifest()
    if m is None:
        raise AIUnavailable(
            "The AI engine download is only available for the 64-bit Windows program. "
            "In a Python installation use:  pip install glitchlab[ai]")
    final = runtime_root() / m["id"]
    if (final / "COMPLETE").exists():
        return final
    dl = runtime_root() / "downloads"
    staging = runtime_root() / (m["id"] + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    total = sum(w["size"] for w in m["wheels"])
    done = 0
    for w in m["wheels"]:
        cancel.check()
        dest = dl / w["file"]
        download(w["url"], dest, size=w["size"], sha256=w["sha256"], cancel=cancel, progress=progress,
                 label=f"{w['name']} {w['version']}", base_done=done, grand_total=total)
        done += w["size"]
    site = staging / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    for i, w in enumerate(m["wheels"]):
        cancel.check()
        if progress:
            progress(total, total, f"Unpacking {w['name']} ({i + 1}/{len(m['wheels'])})")
        _extract_wheel(dl / w["file"], site)
    (staging / "COMPLETE").write_text(json.dumps({"id": m["id"], "wheels": [w["file"] for w in m["wheels"]]}))
    shutil.rmtree(final, ignore_errors=True)
    os.replace(staging, final)
    shutil.rmtree(dl, ignore_errors=True)
    remove_old(keep=m["id"])
    activate()
    return final


def remove_old(keep: str) -> None:
    for p in runtime_root().iterdir():
        if p.is_dir() and p.name not in (keep, "downloads") and not p.name.startswith(keep):
            shutil.rmtree(p, ignore_errors=True)


def uninstall() -> None:
    d = install_dir()
    if d is not None:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- torch access
def import_torch():
    """Import PyTorch or raise a clear, user-facing error."""
    global _torch
    if _torch is not None:
        return _torch
    activate()
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise AIUnavailable(detail=repr(exc)) from exc
    except OSError as exc:  # DLL load failures on Windows
        raise AIHardwareError(
            "The AI engine could not start on this computer (a system library failed to load). Installing the "
            "latest Microsoft Visual C++ Redistributable (x64) usually fixes this.", detail=repr(exc)) from exc
    _torch = torch
    return torch


def select_device(preference: str = "auto") -> str:
    """'cpu', 'cuda' or 'mps' according to the preference and the hardware."""
    torch = import_torch()
    pref = (preference or "auto").lower()
    if pref == "cpu":
        return "cpu"
    cuda = bool(torch.cuda.is_available())
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if pref == "cuda":
        if not cuda:
            raise AIHardwareError("The NVIDIA graphics card (CUDA) was selected in Preferences, but this AI engine "
                                  "or computer cannot use it. Choose 'Automatic' or 'Processor (CPU)'.")
        return "cuda"
    if pref == "mps" and not mps:
        raise AIHardwareError("Apple GPU acceleration (MPS) is not available on this computer.")
    if pref == "mps" or (pref == "auto" and mps and not cuda):
        return "mps"
    return "cuda" if cuda else "cpu"


@dataclass
class EngineStatus:
    available: bool
    can_download: bool
    message: str
    versions: dict


def status() -> EngineStatus:
    if not engine_available():
        msg = ("Not installed. " + ("Use AI › Install AI engine… to download it "
                                    f"({download_size() / 2**20:.0f} MB)." if can_download()
                                    else "Install it with: pip install glitchlab[ai]"))
        return EngineStatus(False, can_download(), msg, {})
    versions = {}
    for mod in REQUIRED:
        try:
            from importlib.metadata import version  # noqa: PLC0415

            versions[mod] = version(mod)
        except Exception:  # pragma: no cover - metadata missing in odd installs
            versions[mod] = "?"
    return EngineStatus(True, can_download(), "Installed: " + ", ".join(f"{k} {v}" for k, v in versions.items()),
                        versions)
