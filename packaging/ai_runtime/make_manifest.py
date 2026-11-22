"""Build the AI-engine wheel manifest that Glitch Lab downloads on first use.

    uv pip compile packaging/ai_runtime/requirements.in --python-platform x86_64-pc-windows-msvc \
        --python-version 3.13 --only-binary :all: -o packaging/ai_runtime/requirements.txt
    python packaging/ai_runtime/make_manifest.py

For every pinned requirement the script asks PyPI for the wheel that a
64-bit Windows CPython 3.13 would install and records its URL, size and
SHA-256.  Packages that the frozen program already contains (numpy, Pillow)
are left out.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "src" / "glitchlab" / "ai" / "runtime_win_amd64.json"
BUNDLED = {"numpy", "pillow"}
PY = (3, 13)
PLATFORM = "win_amd64"


def pinned(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def main() -> int:
    tags = list(cpython_tags(PY, abis=[f"cp{PY[0]}{PY[1]}"], platforms=[PLATFORM]))
    tags += list(compatible_tags(PY, f"cp{PY[0]}{PY[1]}", platforms=[PLATFORM]))
    rank = {t: i for i, t in enumerate(tags)}
    wheels = []
    for name, version in pinned(HERE / "requirements.txt"):
        if canonicalize_name(name) in BUNDLED:
            continue
        data = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json"))
        best = None
        for f in data["urls"]:
            if f["packagetype"] != "bdist_wheel":
                continue
            _, _, _, ftags = parse_wheel_filename(f["filename"])
            r = min((rank[t] for t in ftags if t in rank), default=None)
            if r is not None and (best is None or r < best[0]):
                best = (r, f)
        if best is None:
            print(f"No Windows wheel for {name}=={version}", file=sys.stderr)
            return 1
        f = best[1]
        wheels.append({"name": canonicalize_name(name), "version": version, "file": f["filename"],
                       "url": f["url"], "size": f["size"], "sha256": f["digests"]["sha256"]})
        print(f"{name:22s} {version:12s} {f['size'] / 2**20:8.1f} MB  {f['filename']}")
    ident = hashlib.sha256(json.dumps(wheels, sort_keys=True).encode()).hexdigest()[:12]
    manifest = {"id": f"cpu-py{PY[0]}{PY[1]}-{ident}", "python": f"{PY[0]}.{PY[1]}", "platform": PLATFORM,
                "device": "cpu", "wheels": wheels}
    OUT.write_text(json.dumps(manifest, indent=1) + "\n")
    total = sum(w["size"] for w in wheels)
    print(f"{len(wheels)} wheels, {total / 2**20:.0f} MB -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
