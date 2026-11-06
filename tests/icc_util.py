"""Build a minimal, valid ICC v2 RGB matrix/TRC profile for tests.

The default primaries are Adobe RGB (1998) chromatically adapted to D50
(as stored in real Adobe RGB profiles) with a 2.2 gamma.
"""

import struct


def _s15(v: float) -> bytes:
    return struct.pack(">i", int(round(v * 65536)))


def _xyz(x, y, z) -> bytes:
    return b"XYZ " + b"\0" * 4 + _s15(x) + _s15(y) + _s15(z)


def _curve(gamma: float) -> bytes:
    return b"curv" + b"\0" * 4 + struct.pack(">I", 1) + struct.pack(">H", int(round(gamma * 256))) + b"\0\0"


def _desc(text: str) -> bytes:
    t = text.encode("ascii") + b"\0"
    body = b"desc" + b"\0" * 4 + struct.pack(">I", len(t)) + t
    body += struct.pack(">II", 0, 0) + struct.pack(">HB", 0, 0) + b"\0" * 67  # unicode + scriptcode (empty)
    return body


def _text(text: str) -> bytes:
    return b"text" + b"\0" * 4 + text.encode("ascii") + b"\0"


def make_rgb_profile(description: str = "Adobe RGB (1998) test", gamma: float = 2.2,
                     red=(0.60974, 0.31111, 0.01947), green=(0.20528, 0.62567, 0.06087),
                     blue=(0.14919, 0.06322, 0.74457)) -> bytes:
    tags = [
        (b"desc", _desc(description)),
        (b"cprt", _text("No copyright, test data")),
        (b"wtpt", _xyz(0.96420, 1.0, 0.82491)),
        (b"rXYZ", _xyz(*red)),
        (b"gXYZ", _xyz(*green)),
        (b"bXYZ", _xyz(*blue)),
        (b"rTRC", _curve(gamma)),
        (b"gTRC", _curve(gamma)),
        (b"bTRC", _curve(gamma)),
    ]
    offset = 128 + 4 + 12 * len(tags)
    table = b""
    data = b""
    for sig, body in tags:
        while (offset + len(data)) % 4:
            data += b"\0"
        table += sig + struct.pack(">II", offset + len(data), len(body))
        data += body
    while len(data) % 4:
        data += b"\0"
    size = offset + len(data)
    header = struct.pack(">I", size) + b"lcms" + struct.pack(">I", 0x02100000) + b"mntr" + b"RGB " + b"XYZ "
    header += struct.pack(">HHHHHH", 2026, 1, 1, 0, 0, 0) + b"acsp" + b"MSFT" + struct.pack(">I", 0)
    header += b"\0" * 8 + struct.pack(">II", 0, 0) + struct.pack(">I", 0)
    header += _s15(0.9642) + _s15(1.0) + _s15(0.82491) + b"lcms"
    header += b"\0" * (128 - len(header))
    return header + struct.pack(">I", len(tags)) + table + data
