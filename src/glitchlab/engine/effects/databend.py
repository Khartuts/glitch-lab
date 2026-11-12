"""Data-bending effects that operate on encoded or raw bytes.

HEX / byte corruption
---------------------
The image is genuinely *encoded* (BMP, binary PPM or JPEG), bytes of the
encoded file are searched/replaced or mutated, and the damaged file is
decoded again with Pillow.

Structure protection and length policy:

* **BMP** (24-bit BI_RGB, bottom-up): the 14-byte file header and 40-byte
  BITMAPINFOHEADER are never touched; only the pixel array is edited.
* **PPM** (P6, maxval 255): the ASCII header is never touched.
* For BMP/PPM the pixel payload has a fixed size.  Insertions shift all
  following bytes (the classic diagonal tear); afterwards the payload is
  **truncated** to its original length, or if shorter **padded** according
  to *Length policy* (source tail, zeros, or repeating the payload from its
  start).  The header therefore always stays consistent.
* **JPEG (experimental)**: only entropy-coded scan data between the SOS
  segment and EOI is edited.  0xFF bytes and the byte after them (marker /
  stuffing pairs) are protected and 0xFF is never written, so marker
  structure survives.  Up to *Attempts* seeded attempts are decoded (each with
  a time limit); a result is accepted only if it decodes to the right size
  and differs from the undamaged decode.  If no attempt succeeds an error
  is reported - the original is never returned as a "successful" glitch.
"""

from __future__ import annotations

import io
import logging
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import cv2
import numpy as np
from PIL import Image

from ..errors import EffectError
from ..params import FILE, HEX, Param, bparam, cparam, fparam, iparam, parse_hex_bytes
from .base import Effect, register

log = logging.getLogger(__name__)

_DECODE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="glitch-decode")


# --------------------------------------------------------------------- codecs
def encode_bmp(rgb: np.ndarray) -> tuple[bytes, int]:
    """24-bit bottom-up BMP. Returns (file bytes, header length)."""
    h, w = rgb.shape[:2]
    stride = (w * 3 + 3) & ~3
    pixels = np.zeros((h, stride), np.uint8)
    pixels[:, : w * 3] = rgb[::-1, :, ::-1].reshape(h, w * 3)  # bottom-up, BGR
    payload = pixels.tobytes()
    header = struct.pack("<2sIHHI", b"BM", 54 + len(payload), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(payload), 2835, 2835, 0, 0)
    return header + info + payload, 54


def encode_ppm(rgb: np.ndarray) -> tuple[bytes, int]:
    h, w = rgb.shape[:2]
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    return header + np.ascontiguousarray(rgb).tobytes(), len(header)


def encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb), "RGB").save(buf, "JPEG", quality=int(quality), subsampling=2,
                                                           optimize=False, progressive=False)
    return buf.getvalue()


def jpeg_scan_range(data: bytes) -> tuple[int, int]:
    """Return (start, end) of the entropy-coded data of the first scan."""
    i = 2
    n = len(data)
    if data[:2] != b"\xff\xd8":
        raise EffectError("Internal error: not a JPEG stream.")
    while i < n - 4:
        if data[i] != 0xFF:
            raise EffectError("Internal error: malformed JPEG marker sequence.")
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        seglen = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker == 0xDA:  # SOS
            start = i + 2 + seglen
            end = data.rfind(b"\xff\xd9")
            if end <= start:
                raise EffectError("Internal error: JPEG has no EOI marker.")
            return start, end
        i += 2 + seglen
    raise EffectError("Internal error: JPEG has no scan data.")


def _decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return np.array(im.convert("RGB"), dtype=np.uint8)


def decode_with_timeout(data: bytes, timeout: float = 10.0) -> np.ndarray:
    """Decode encoded image bytes in a guarded worker thread with a time limit."""
    fut = _DECODE_POOL.submit(_decode, data)
    try:
        return fut.result(timeout=timeout)
    except FutureTimeout:
        fut.cancel()
        raise EffectError(f"The damaged file took longer than {timeout:.0f} s to decode.") from None


# ------------------------------------------------------------ byte operations
def find_pattern(buf: np.ndarray, pattern: list[int | None], lo: int, hi: int,
                 protected: np.ndarray | None = None) -> np.ndarray:
    """Start offsets of non-overlapping matches of ``pattern`` within buf[lo:hi]."""
    L = len(pattern)
    if L == 0 or hi - lo < L:
        return np.zeros(0, np.int64)
    seg = buf[lo:hi]
    m = np.ones(len(seg) - L + 1, bool)
    for k, b in enumerate(pattern):
        if b is not None:
            m &= seg[k:len(seg) - L + 1 + k] == b
    if protected is not None:
        pseg = protected[lo:hi]
        for k in range(L):
            m &= ~pseg[k:len(seg) - L + 1 + k]
    cand = np.flatnonzero(m) + lo
    if L == 1 or cand.size == 0:
        return cand.astype(np.int64)
    keep = []
    last_end = -1
    for c in cand.tolist():
        if c >= last_end:
            keep.append(c)
            last_end = c + L
    return np.asarray(keep, np.int64)


def replace_at(buf: np.ndarray, starts: np.ndarray, search_len: int, repl: bytes) -> np.ndarray:
    if starts.size == 0:
        return buf
    rep = np.frombuffer(repl, np.uint8)
    pieces = []
    prev = 0
    for s in starts.tolist():
        pieces.append(buf[prev:s])
        pieces.append(rep)
        prev = s + search_len
    pieces.append(buf[prev:])
    return np.concatenate(pieces)


MUTATIONS = (("none", "None"), ("random", "Random value"), ("bitflip", "Flip one bit"),
             ("increment", "Increment"), ("swap", "Swap nearby"), ("insert", "Insert byte"),
             ("delete", "Delete byte"))


def mutate(buf: np.ndarray, mode: str, count: int, lo: int, hi: int, rng: np.random.Generator,
           protected: np.ndarray | None = None, forbid_ff: bool = False) -> np.ndarray:
    """Apply ``count`` seeded mutations at positions inside [lo, hi)."""
    if mode == "none" or count <= 0 or hi <= lo:
        return buf
    pos = rng.integers(lo, hi, size=count)
    if protected is not None:
        pos = pos[~protected[pos]]
    if pos.size == 0:
        return buf
    if mode in ("insert", "delete"):
        pos = np.unique(pos)
        if mode == "insert":
            vals = rng.integers(0, 256, size=pos.size).astype(np.uint8)
            if forbid_ff:
                vals[vals == 0xFF] = 0xFE
            return np.insert(buf, pos, vals)
        return np.delete(buf, pos)
    out = buf.copy()
    if mode == "random":
        vals = rng.integers(0, 256, size=pos.size).astype(np.uint8)
        out[pos] = vals
    elif mode == "bitflip":
        out[pos] ^= (1 << rng.integers(0, 8, size=pos.size)).astype(np.uint8)
    elif mode == "increment":
        out[pos] = (out[pos].astype(np.int32) + rng.integers(1, 32, size=pos.size)) % 256
    elif mode == "swap":
        other = np.clip(pos + rng.integers(1, 9, size=pos.size), lo, hi - 1)
        if protected is not None:
            ok = ~protected[other]
            pos, other = pos[ok], other[ok]
        tmp = out[pos].copy()
        out[pos] = out[other]
        out[other] = tmp
    else:
        raise EffectError(f"Unknown mutation mode {mode!r}")
    if forbid_ff:
        bad = out[pos] == 0xFF
        out[pos[bad]] = 0xFE
    return out


def normalise_length(payload: np.ndarray, original: np.ndarray, policy: str) -> np.ndarray:
    """Force a fixed-size payload back to ``len(original)`` bytes."""
    n = len(original)
    if len(payload) >= n:
        return payload[:n]
    missing = n - len(payload)
    if policy == "truncate_pad_zero":
        tail = np.zeros(missing, np.uint8)
    elif policy == "truncate_pad_repeat":
        reps = int(np.ceil(missing / max(1, len(payload)))) if len(payload) else 0
        tail = np.tile(payload, reps)[:missing] if len(payload) else np.zeros(missing, np.uint8)
    else:  # truncate_pad_source
        tail = original[len(payload):]
    return np.concatenate([payload, tail])


@register
class HexCorrupt(Effect):
    id = "hex_corrupt"
    name = "HEX / Byte Corruption"
    category = "Data Bending"
    description = ("Encodes the image as a real BMP, PPM or JPEG file, edits its bytes (hex search & replace "
                   "and seeded mutations) with headers protected, then decodes the damaged file.")
    supports_channels = True
    preview_note = "Byte damage depends on file size: the reduced preview only approximates the full render."
    params = (
        cparam("container", "File format", "bmp",
               (("bmp", "BMP (24-bit)"), ("ppm", "PPM (binary P6)"), ("jpeg", "JPEG scan data (experimental)")),
               tooltip="The encoded format whose bytes are corrupted."),
        Param("search", "Search hex", HEX, "DA", tooltip="Bytes to find, e.g. 'DA' or 'FF ?? 00' (?? = any byte). "
              "Empty disables search & replace.", group="Search & replace", flags=("wildcard",), randomizable=False),
        Param("replace", "Replace with", HEX, "0E DA", tooltip="Replacement bytes, e.g. '0E DA'. Different lengths "
              "insert/delete bytes and shift everything after.", group="Search & replace", randomizable=False),
        fparam("replace_prob", "Replace probability", 0.25, 0.0, 1.0, group="Search & replace",
               tooltip="Chance that each match is replaced (seeded)."),
        iparam("max_replacements", "Max replacements", 400, 0, 200000, group="Search & replace",
               tooltip="0 = unlimited"),
        cparam("mutation", "Mutation", "random", MUTATIONS, group="Random mutation"),
        fparam("density", "Density", 0.0004, 0.0, 0.02, decimals=5, step=0.0001, group="Random mutation",
               tooltip="Fraction of payload bytes mutated per iteration."),
        iparam("iterations", "Iterations", 1, 1, 20, group="Random mutation",
               tooltip="Mutation passes; each pass works on the result of the previous one."),
        fparam("region_start", "Region start", 0.0, 0.0, 1.0, group="Region",
               tooltip="Start of the edited region as a fraction of the payload."),
        fparam("region_end", "Region end", 1.0, 0.0, 1.0, group="Region"),
        cparam("length_policy", "Length policy", "truncate_pad_source",
               (("truncate_pad_source", "Truncate / pad with source tail"),
                ("truncate_pad_zero", "Truncate / pad with zeros"),
                ("truncate_pad_repeat", "Truncate / pad by repeating")), group="Region",
               tooltip="BMP/PPM payloads are fixed-size: longer results are truncated, shorter ones padded.",
               show_if=("container", ("bmp", "ppm"))),
        iparam("quality", "JPEG quality", 75, 10, 100, group="JPEG", show_if=("container", ("jpeg",))),
        iparam("attempts", "Attempts", 6, 1, 16, group="JPEG", show_if=("container", ("jpeg",)),
               tooltip="Seeded attempts before reporting failure."),
    )

    def validate(self, p):
        problems = []
        try:
            if p["search"].strip():
                parse_hex_bytes(p["search"], allow_wildcard=True)
        except ValueError as exc:
            problems.append(f"Search hex: {exc}")
        try:
            if p["replace"].strip():
                rep = parse_hex_bytes(p["replace"])
                if p["container"] == "jpeg" and 0xFF in rep:
                    problems.append("In JPEG mode the replacement may not contain FF (it would create markers).")
        except ValueError as exc:
            problems.append(f"Replace hex: {exc}")
        return problems  # an empty replacement deletes the matched bytes

    # ------------------------------------------------------------------
    def _edit(self, payload: np.ndarray, p, rng, protected=None, forbid_ff=False) -> tuple[np.ndarray, int, int]:
        n = len(payload)
        lo = int(round(min(p["region_start"], p["region_end"]) * n))
        hi = int(round(max(p["region_start"], p["region_end"]) * n))
        replaced = 0
        mutated = 0
        buf = payload
        search = p["search"].strip()
        if search:
            pat = parse_hex_bytes(search, allow_wildcard=True)
            rep = bytes(parse_hex_bytes(p["replace"])) if p["replace"].strip() else b""
            starts = find_pattern(buf, pat, lo, hi, protected)
            if starts.size:
                keep = rng.random(starts.size) < p["replace_prob"]
                starts = starts[keep]
                if p["max_replacements"] > 0:
                    starts = starts[: p["max_replacements"]]
                buf = replace_at(buf, starts, len(pat), rep)
                replaced = int(starts.size)
                if protected is not None and len(rep) != len(pat):
                    protected = jpeg_protect_mask(buf)
                hi = max(lo, min(len(buf), hi + replaced * (len(rep) - len(pat))))
        if p["mutation"] != "none" and p["density"] > 0:
            for _ in range(int(p["iterations"])):
                count = max(1, int(round(p["density"] * max(0, hi - lo))))
                before = len(buf)
                buf = mutate(buf, p["mutation"], count, lo, min(hi, len(buf)), rng, protected, forbid_ff)
                mutated += count
                if protected is not None and len(buf) != before:
                    protected = jpeg_protect_mask(buf)
                hi = max(lo, min(len(buf), hi + (len(buf) - before)))
        return buf, replaced, mutated

    def process(self, img, p, ctx):
        rgb = np.ascontiguousarray(img[..., :3])
        h, w = rgb.shape[:2]
        container = p["container"]
        if container in ("bmp", "ppm"):
            data, hlen = encode_bmp(rgb) if container == "bmp" else encode_ppm(rgb)
            arr = np.frombuffer(data, np.uint8)
            header, payload = arr[:hlen], arr[hlen:]
            ctx.check()
            new, replaced, mutated = self._edit(payload, p, ctx.rng)
            length_changed = len(new) - len(payload)
            new = normalise_length(new, payload, p["length_policy"])
            if p["search"].strip() and replaced == 0:
                ctx.warn(f"pattern '{p['search']}' was not found (or not selected) in the pixel payload")
            if np.array_equal(new, payload):
                ctx.warn("no bytes were changed; the image is unaltered")
            elif length_changed:
                log.debug("hex payload length changed by %+d bytes; normalised", length_changed)
            blob = np.concatenate([header, new]).tobytes()
            try:
                dec = decode_with_timeout(blob)
            except EffectError:
                raise
            except Exception as exc:
                raise EffectError(f"The corrupted {container.upper()} could not be decoded: {exc}") from exc
            if dec.shape != (h, w, 3):
                raise EffectError(f"The corrupted {container.upper()} decoded to the wrong size {dec.shape[1]}x{dec.shape[0]}.")
            out = img.copy()
            out[..., :3] = dec
            return out
        return self._process_jpeg(img, rgb, p, ctx)

    def _process_jpeg(self, img, rgb, p, ctx):
        h, w = rgb.shape[:2]
        data = encode_jpeg(rgb, p["quality"])
        clean = decode_with_timeout(data)
        start, end = jpeg_scan_range(data)
        arr = np.frombuffer(data, np.uint8)
        head, scan, tail = arr[:start], arr[start:end], arr[end:]
        protected = jpeg_protect_mask(scan)
        reasons = []
        for attempt in range(int(p["attempts"])):
            ctx.check()
            rng = ctx.sub_rng(attempt + 1)
            new_scan, replaced, mutated = self._edit(scan, p, rng, protected, forbid_ff=True)
            if replaced == 0 and mutated == 0:
                reasons.append("nothing to change (pattern not found and no mutation)")
                break
            blob = np.concatenate([head, new_scan, tail]).tobytes()
            try:
                dec = decode_with_timeout(blob)
            except EffectError as exc:
                reasons.append(exc.message)
                continue
            except Exception as exc:
                reasons.append(f"decoder error: {exc}")
                continue
            if dec.shape != (h, w, 3):
                reasons.append(f"decoded to wrong size {dec.shape[1]}x{dec.shape[0]}")
                continue
            if np.array_equal(dec, clean):
                reasons.append("damage had no visible effect")
                continue
            if attempt:
                ctx.warn(f"JPEG corruption succeeded on attempt {attempt + 1}")
            out = img.copy()
            out[..., :3] = dec
            return out
        uniq = sorted(set(reasons))
        raise EffectError(
            f"JPEG corruption failed after {len(reasons)} attempt(s): {'; '.join(uniq)}. "
            "Try a lower density, a different region or seed, or the BMP format.")


def jpeg_protect_mask(scan: np.ndarray) -> np.ndarray:
    """Protect 0xFF bytes and the byte that follows each (stuffing/RST pairs)."""
    ff = scan == 0xFF
    prot = ff.copy()
    prot[1:] |= ff[:-1]
    return prot


# ---------------------------------------------------------------------------
# Raw data reinterpretation
# ---------------------------------------------------------------------------

_ORDERS = ("rgb", "rbg", "grb", "gbr", "brg", "bgr")


@register
class DataReinterpret(Effect):
    id = "reinterpret"
    name = "Data Reinterpretation"
    category = "Data Bending"
    description = ("Reads the raw pixel bytes with the wrong channel order, row width, byte offset or "
                   "planar/interleaved layout, then reconstructs a valid image of the original size.")
    uses_seed = False
    preview_note = "Byte offsets are scaled with the preview (keeping the channel rotation)."
    params = (
        cparam("layout", "Read layout", "interleaved",
               (("interleaved", "Interleaved RGBRGB… (correct)"), ("planar_as_interleaved", "Planar read as interleaved"),
                ("interleaved_as_planar", "Interleaved read as planar"), ("as_gray", "RGB bytes read as greyscale"),
                ("as_rgba", "RGB bytes read as RGBA")),
               tooltip="How the byte stream is (mis)interpreted."),
        Param("channel_order", "Channel order", "choice", "rgb", choices=tuple((o, o.upper()) for o in _ORDERS)),
        iparam("width_delta", "Row width error", 2, -64, 64, unit="px",
               tooltip="Read rows as if the image were this many pixels wider (classic stride skew)."),
        iparam("offset", "Byte offset", 0, -4096, 4096, unit="bytes",
               tooltip="Start reading this many bytes later (negative = earlier)."),
        cparam("fill", "Missing bytes", "wrap", (("wrap", "Wrap around"), ("zero", "Black"), ("source", "Source bytes"))),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        rgb = np.ascontiguousarray(img[..., :3])
        stream = rgb.reshape(-1)
        layout = p["layout"]
        if layout == "planar_as_interleaved":
            stream = np.ascontiguousarray(np.moveaxis(rgb, -1, 0)).reshape(-1)
        # byte offset: whole pixels scale with the preview, the remainder
        # (channel rotation) is kept so preview and export look alike
        k = int(p["offset"])
        sign = -1 if k < 0 else 1
        whole, rem = divmod(abs(k), 3)
        off = sign * (int(round(whole * ctx.scale)) * 3 + rem)
        if off:
            if p["fill"] == "wrap":
                stream = np.roll(stream, -off)
            else:
                pad = np.zeros(abs(off), np.uint8) if p["fill"] == "zero" else stream[:abs(off)]
                stream = np.concatenate([stream[off:], pad]) if off > 0 else np.concatenate([pad, stream[:off]])
        bpp = 4 if layout == "as_rgba" else (1 if layout == "as_gray" else 3)
        # the skew slope (delta per row) is resolution independent, so it is not scaled
        rw = max(1, w + int(p["width_delta"]))
        need = h * rw * bpp if layout != "interleaved_as_planar" else h * w * 3
        if stream.size < need:
            extra = need - stream.size
            if p["fill"] == "zero":
                tail = np.zeros(extra, np.uint8)
            elif p["fill"] == "source":
                tail = np.resize(rgb.reshape(-1), extra)
            else:
                tail = np.resize(stream, extra)
            stream = np.concatenate([stream, tail])
        if layout == "interleaved_as_planar":
            planes = stream[: h * w * 3].reshape(3, h, w)
            out_rgb = np.moveaxis(planes, 0, -1)
        else:
            grid = stream[: h * rw * bpp].reshape(h, rw, bpp)[:, : min(w, rw)]
            if rw < w:
                grid = np.concatenate([grid, np.zeros((h, w - rw, bpp), np.uint8)], axis=1)
            if bpp == 1:
                out_rgb = np.repeat(grid, 3, axis=2)
            else:
                out_rgb = grid[..., :3]
        order = p["channel_order"]
        idx = ["rgb".index(c) for c in order]
        out_rgb = out_rgb[..., idx]
        out = img.copy()
        out[..., :3] = out_rgb
        return out


# ---------------------------------------------------------------------------
# Repeated JPEG compression
# ---------------------------------------------------------------------------

@register
class CompressionDecay(Effect):
    id = "jpeg_decay"
    name = "Compression Decay"
    category = "Data Bending"
    description = ("Real repeated JPEG encode/decode generations with seeded quality per generation, optional "
                   "downscale/upscale and 1-pixel shifts that compound block artefacts.")
    preview_note = "JPEG blocks are 8x8 pixels at every resolution, so block size looks larger in the export."
    params = (
        iparam("generations", "Generations", 8, 1, 60, tooltip="Number of encode/decode cycles."),
        iparam("quality_min", "Quality min", 10, 1, 100),
        iparam("quality_max", "Quality max", 35, 1, 100),
        cparam("subsampling", "Chroma subsampling", "4:2:0", (("4:4:4", "4:4:4"), ("4:2:2", "4:2:2"), ("4:2:0", "4:2:0"))),
        fparam("downscale", "Downscale between", 1.0, 0.1, 1.0,
               tooltip="Resize to this fraction and back each generation (1 = off)."),
        cparam("interp", "Resize filter", "linear", (("nearest", "Nearest"), ("linear", "Bilinear"), ("cubic", "Bicubic"))),
        iparam("shift_px", "Shift per generation", 0, 0, 16, unit="px",
               tooltip="Offset the image between generations so blocks misalign and pile up."),
    )

    def process(self, img, p, ctx):
        h, w = img.shape[:2]
        cur = np.ascontiguousarray(img[..., :3])
        qlo, qhi = sorted((p["quality_min"], p["quality_max"]))
        sub = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}[p["subsampling"]]
        interp = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}[p["interp"]]
        qualities = ctx.rng.integers(qlo, qhi + 1, size=p["generations"])
        shifts = ctx.rng.integers(-1, 2, size=(p["generations"], 2)) * p["shift_px"]
        for g in range(p["generations"]):
            ctx.check()
            ctx.progress(g / p["generations"])
            work = cur
            if p["downscale"] < 0.999:
                sw, sh = max(8, int(w * p["downscale"])), max(8, int(h * p["downscale"]))
                work = cv2.resize(work, (sw, sh), interpolation=cv2.INTER_AREA)
            dy, dx = int(shifts[g, 0]), int(shifts[g, 1])
            if dx or dy:
                work = np.roll(work, (dy, dx), axis=(0, 1))
            buf = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(work), "RGB").save(buf, "JPEG", quality=int(qualities[g]),
                                                                    subsampling=sub)
            dec = _decode(buf.getvalue())
            if dx or dy:
                dec = np.roll(dec, (-dy, -dx), axis=(0, 1))
            if dec.shape[:2] != (h, w):
                dec = cv2.resize(dec, (w, h), interpolation=interp)
            cur = dec
        out = img.copy()
        out[..., :3] = cur
        return out


# ---------------------------------------------------------------------------
# Text / file injection
# ---------------------------------------------------------------------------

MAX_INJECT_BYTES = 64 * 1024 * 1024
_INJECT_TARGETS = (
    ("bmp_pixels", "BMP pixel data"),
    ("bmp_palette", "8-bit BMP colour table (palette)"),
    ("ppm", "PPM pixel data"),
    ("jpeg_scan", "JPEG scan data (experimental)"),
    ("jpeg_tables", "JPEG quantisation tables"),
    ("raw", "Raw RGB bytes"),
)
_INJECT_MODES = (("overwrite", "Overwrite"), ("insert", "Insert (shifts data)"), ("xor", "XOR"), ("add", "Add"))


def read_injection_data(source: str, text: str, path: str) -> bytes:
    if source == "file":
        if not path or not os.path.isfile(path):
            raise EffectError(f"Injection file not found: {path or '(none)'}")
        with open(path, "rb") as fh:
            data = fh.read(MAX_INJECT_BYTES)
    else:
        data = (text or "").encode("utf-8")
    if not data:
        raise EffectError("Nothing to inject: the text or file is empty.")
    return data


def injection_chunks(data: bytes, count: int, chunk: int) -> list[bytes]:
    """``count`` consecutive pieces of ``data`` (wrapping around)."""
    n = len(data)
    chunk = max(1, min(chunk, n))
    out = []
    pos = 0
    for _ in range(count):
        piece = data[pos:pos + chunk]
        if len(piece) < chunk:
            piece += data[: chunk - len(piece)]
        out.append(piece)
        pos = (pos + chunk) % n
    return out


def injection_positions(lo: int, hi: int, count: int, placement: str, rng: np.random.Generator) -> np.ndarray:
    if hi <= lo:
        return np.zeros(0, np.int64)
    if placement == "even":
        return np.unique(np.linspace(lo, hi - 1, count).astype(np.int64))
    return np.sort(rng.integers(lo, hi, size=count))


def inject(buf: np.ndarray, positions: np.ndarray, chunks: list[bytes], mode: str,
           protected: np.ndarray | None = None, forbid_ff: bool = False) -> np.ndarray:
    """Write ``chunks`` into ``buf`` at ``positions`` (sorted)."""
    if mode == "insert":
        pieces, prev = [], 0
        for pos, ch in zip(positions.tolist(), chunks):
            if protected is not None:
                while pos < len(buf) and protected[pos]:
                    pos += 1
            arr = np.frombuffer(ch, np.uint8)
            if forbid_ff:
                arr = np.where(arr == 0xFF, 0xFE, arr).astype(np.uint8)
            pieces.append(buf[prev:pos])
            pieces.append(arr)
            prev = pos
        pieces.append(buf[prev:])
        return np.concatenate(pieces)
    out = buf.copy()
    for pos, ch in zip(positions.tolist(), chunks):
        arr = np.frombuffer(ch, np.uint8)[: max(0, len(out) - pos)]
        if arr.size == 0:
            continue
        sl = slice(pos, pos + arr.size)
        cur = out[sl]
        if mode == "xor":
            new = cur ^ arr
        elif mode == "add":
            new = ((cur.astype(np.int32) + arr) & 0xFF).astype(np.uint8)
        else:
            new = arr
        if protected is not None:
            new = np.where(protected[sl], cur, new)
        if forbid_ff:
            new = np.where((new == 0xFF) & (cur != 0xFF), 0xFE, new)
        out[sl] = new
    return out


def jpeg_dqt_value_positions(data: bytes, which: str) -> list[int]:
    """Byte offsets of every 8-bit quantisation-table value in a baseline JPEG."""
    out: list[int] = []
    i = 2
    n = len(data)
    while i < n - 4 and data[i] == 0xFF:
        marker = data[i + 1]
        seglen = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker == 0xDA:
            break
        if marker == 0xDB:
            j = i + 4
            end = i + 2 + seglen
            while j < end:
                pq, tq = data[j] >> 4, data[j] & 0x0F
                size = 128 if pq else 64
                want = which == "both" or (which == "luma" and tq == 0) or (which == "chroma" and tq != 0)
                if want and not pq:
                    out.extend(range(j + 1, j + 1 + 64))
                j += 1 + size
        i += 2 + seglen
    return out


@register
class DataInjection(Effect):
    id = "data_injection"
    name = "Text & File Injection"
    category = "Data Bending"
    description = ("Pastes text, lyrics, GPS coordinates - or the raw bytes of ANY file you load (txt, wav, mp3, "
                   "pdf …) - into the image's encoded data: pixel data, the BMP colour table, PPM, JPEG scan data "
                   "or JPEG quantisation tables. The decoder misreads them as neon blocks, colour remaps and "
                   "shifted scanlines.")
    supports_channels = True
    preview_note = "Bytes land at different image positions in the reduced preview than in the export."
    params = (
        cparam("source", "Inject", "text", (("text", "Typed text"), ("file", "Any file"))),
        Param("text", "Text", "text", "the medium is the message // 51.5007 N, 0.1246 W // la la la",
              show_if=("source", ("text",)), randomizable=False,
              tooltip="Lyrics, coordinates, anything - injected as UTF-8 bytes."),
        Param("data_file", "File", FILE, "", file_filter="All files (*)", show_if=("source", ("file",)),
              randomizable=False, tooltip="Any file: its raw bytes are injected (first 64 MB)."),
        Param("target", "Inject into", "choice", "bmp_pixels", choices=_INJECT_TARGETS),
        Param("mode", "Mode", "choice", "insert", choices=_INJECT_MODES,
              tooltip="Insert shifts all following data (diagonal tears); overwrite/XOR/add keep positions."),
        iparam("count", "Injections", 8, 1, 2000, tooltip="How many times a piece of the data is injected."),
        iparam("chunk", "Bytes per injection", 48, 1, 1_000_000,
               tooltip="Consecutive bytes taken from the text/file for each injection."),
        cparam("placement", "Placement", "random", (("random", "Random (seeded)"), ("even", "Evenly spaced"))),
        fparam("region_start", "Region start", 0.0, 0.0, 1.0, group="Region"),
        fparam("region_end", "Region end", 1.0, 0.0, 1.0, group="Region"),
        Param("length_policy", "Length policy", "choice", "truncate_pad_source",
              choices=(("truncate_pad_source", "Truncate / pad with source tail"),
                       ("truncate_pad_zero", "Truncate / pad with zeros"),
                       ("truncate_pad_repeat", "Truncate / pad by repeating")), group="Region",
              show_if=("target", ("bmp_pixels", "ppm", "raw"))),
        cparam("tables", "Tables", "both", (("both", "Luma + chroma"), ("luma", "Luma"), ("chroma", "Chroma")),
               group="JPEG", show_if=("target", ("jpeg_tables",))),
        iparam("quality", "JPEG quality", 80, 10, 100, group="JPEG", show_if=("target", ("jpeg_scan", "jpeg_tables"))),
        iparam("attempts", "Attempts", 6, 1, 16, group="JPEG", show_if=("target", ("jpeg_scan",))),
        iparam("palette_colors", "Palette colours", 256, 2, 256, group="BMP colour table",
               show_if=("target", ("bmp_palette",))),
    )

    def validate(self, p):
        if p["source"] == "file" and not p["data_file"]:
            return ["choose a file to inject (or switch to typed text)."]
        if p["source"] == "text" and not p["text"]:
            return ["type some text to inject."]
        return []

    def _region(self, n: int, p) -> tuple[int, int]:
        a, b = sorted((p["region_start"], p["region_end"]))
        return int(a * n), max(int(a * n) + 1, int(b * n))

    def process(self, img, p, ctx):
        data = read_injection_data(p["source"], p["text"], p["data_file"])
        chunks = injection_chunks(data, int(p["count"]), int(p["chunk"]))
        rgb = np.ascontiguousarray(img[..., :3])
        h, w = rgb.shape[:2]
        target, mode = p["target"], p["mode"]
        rng = ctx.rng
        if target in ("bmp_pixels", "ppm", "raw"):
            if target == "raw":
                header, payload = np.zeros(0, np.uint8), rgb.reshape(-1)
            else:
                blob, hlen = encode_bmp(rgb) if target == "bmp_pixels" else encode_ppm(rgb)
                arr = np.frombuffer(blob, np.uint8)
                header, payload = arr[:hlen], arr[hlen:]
            lo, hi = self._region(len(payload), p)
            pos = injection_positions(lo, hi, len(chunks), p["placement"], rng)
            new = normalise_length(inject(payload, pos, chunks, mode), payload, p["length_policy"])
            if np.array_equal(new, payload):
                ctx.warn("the injected bytes equal the existing bytes; nothing changed")
            if target == "raw":
                dec = new.reshape(h, w, 3)
            else:
                dec = decode_with_timeout(np.concatenate([header, new]).tobytes())
                if dec.shape != (h, w, 3):
                    raise EffectError("The injected file decoded to the wrong size.")
        elif target == "bmp_palette":
            im = Image.fromarray(rgb, "RGB").quantize(colors=int(p["palette_colors"]), method=Image.Quantize.MEDIANCUT,
                                                       dither=Image.Dither.NONE)
            buf = io.BytesIO()
            im.save(buf, "BMP")
            arr = np.frombuffer(buf.getvalue(), np.uint8)
            off = struct.unpack("<I", arr[10:14].tobytes())[0]
            info = struct.unpack("<I", arr[14:18].tobytes())[0]
            pal_lo, pal_hi = 14 + info, off  # BGRA colour table between the headers and the pixels
            table = arr[pal_lo:pal_hi]
            lo, hi = self._region(len(table), p)
            pos = injection_positions(lo, hi, len(chunks), p["placement"], rng)
            new_table = inject(table, pos, chunks, "overwrite" if mode == "insert" else mode)
            if mode == "insert":
                ctx.warn("insert mode keeps the colour table size: bytes were overwritten instead")
            dec = decode_with_timeout(np.concatenate([arr[:pal_lo], new_table, arr[pal_hi:]]).tobytes())
        elif target == "jpeg_tables":
            blob = encode_jpeg(rgb, p["quality"])
            vals = jpeg_dqt_value_positions(blob, p["tables"])
            if not vals:
                raise EffectError("No 8-bit quantisation tables found in the encoded JPEG.")
            arr = np.frombuffer(blob, np.uint8).copy()
            stream = np.frombuffer(b"".join(chunks), np.uint8)
            idx = np.array(vals, np.int64)
            if p["placement"] == "even" or len(stream) >= len(idx):
                chosen = idx[np.linspace(0, len(idx) - 1, min(len(idx), len(stream))).astype(int)]
            else:
                chosen = np.sort(rng.choice(idx, size=len(stream), replace=False))
            src = stream[: len(chosen)]
            cur = arr[chosen]
            if mode == "xor":
                new = cur ^ src
            elif mode == "add":
                new = ((cur.astype(np.int32) + src) & 0xFF).astype(np.uint8)
            else:
                new = src
            arr[chosen] = np.maximum(new, 1)  # a quantiser of 0 is invalid
            dec = decode_with_timeout(arr.tobytes())
        else:  # jpeg_scan
            dec = self._jpeg_scan(rgb, chunks, p, ctx)
        out = img.copy()
        out[..., :3] = dec
        return out

    def _jpeg_scan(self, rgb, chunks, p, ctx):
        h, w = rgb.shape[:2]
        blob = encode_jpeg(rgb, p["quality"])
        clean = decode_with_timeout(blob)
        start, end = jpeg_scan_range(blob)
        arr = np.frombuffer(blob, np.uint8)
        head, scan, tail = arr[:start], arr[start:end], arr[end:]
        prot = jpeg_protect_mask(scan)
        reasons = []
        for attempt in range(int(p["attempts"])):
            ctx.check()
            rng = ctx.sub_rng(attempt + 11)
            lo, hi = self._region(len(scan), p)
            pos = injection_positions(lo, hi, len(chunks), p["placement"], rng)
            new = inject(scan, pos, chunks, p["mode"], protected=prot, forbid_ff=True)
            try:
                dec = decode_with_timeout(np.concatenate([head, new, tail]).tobytes())
            except EffectError as exc:
                reasons.append(exc.message)
                continue
            except Exception as exc:
                reasons.append(f"decoder error: {exc}")
                continue
            if dec.shape != (h, w, 3):
                reasons.append("decoded to the wrong size")
                continue
            if np.array_equal(dec, clean):
                reasons.append("injection had no visible effect")
                continue
            return dec
        raise EffectError(f"JPEG injection failed after {len(reasons)} attempt(s): {'; '.join(sorted(set(reasons)))}. "
                          "Try fewer or smaller injections, another region, or a BMP target.")
