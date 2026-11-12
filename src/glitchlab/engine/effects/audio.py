"""Audio-style databending of pixels.

These effects use the same byte contract as Sonify - pixels -> YUV 4:2:0
planar bytes -> read as 8-bit **A-law** audio samples - but do the audio
processing in NumPy, so they behave identically on every platform:

* :class:`AudioReverbSmear` - convolution reverb (synthetic cathedral / hall /
  room impulse responses, reverse reverb, or your own impulse-response file).
* :class:`AudioFileBend` - bends the pixel-audio with an audio file you load
  (WAV natively; MP3, FLAC, OGG, M4A … are decoded by FFmpeg) or a built-in
  test signal: mix, add, ring-modulate, envelope-follow or gate-replace.

The A-law codec below is G.711 exactly as FFmpeg implements it (decode
formula and the 14-bit encode table), so a round trip through it equals a
round trip through FFmpeg's ``pcm_alaw`` coder.  Nothing is ever played.
"""

from __future__ import annotations

import logging
import os
import threading
import wave
from collections import OrderedDict

import numpy as np

from ..errors import DependencyError, EffectError
from ..params import FILE, Param, bparam, cparam, fparam, iparam
from .base import Effect, register
from .sonify import RATES, rgb_to_yuv420p, yuv420p_to_rgb

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# G.711 A-law (bit-exact with FFmpeg's pcm_alaw)
# ---------------------------------------------------------------------------


def _alaw2linear(a: int) -> int:
    a ^= 0x55
    t = a & 0x0F
    seg = (a & 0x70) >> 4
    t = ((t + t + 1 + 32) << (seg + 2)) if seg else ((t + t + 1) << 3)
    return t if (a & 0x80) else -t


ALAW_DECODE = np.array([_alaw2linear(i) for i in range(256)], np.int32)


def _build_encode_table() -> np.ndarray:
    mask = 0xD5
    table = np.zeros(16384, np.uint8)
    table[8192] = mask
    j = 1
    for i in range(127):
        v1 = _alaw2linear(i ^ mask)
        v2 = _alaw2linear((i + 1) ^ mask)
        v = (v1 + v2 + 4) >> 3
        while j < v:
            table[8192 - j] = i ^ (mask ^ 0x80)
            table[8192 + j] = i ^ mask
            j += 1
    while j < 8192:
        table[8192 - j] = 127 ^ (mask ^ 0x80)
        table[8192 + j] = 127 ^ mask
        j += 1
    table[0] = table[1]
    return table


ALAW_ENCODE_TABLE = _build_encode_table()


def alaw_decode(codes: np.ndarray) -> np.ndarray:
    """uint8 A-law codes -> float32 samples in [-1, 1)."""
    return ALAW_DECODE[np.asarray(codes, np.uint8)].astype(np.float32) / np.float32(32768.0)


def alaw_encode(samples: np.ndarray) -> np.ndarray:
    """float samples in [-1, 1] -> uint8 A-law codes (FFmpeg's 14-bit table)."""
    s = np.nan_to_num(np.asarray(samples, np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    v = np.clip(np.rint(s * 32768.0), -32768, 32767).astype(np.int32)
    return ALAW_ENCODE_TABLE[(v + 32768) >> 2]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

PLANES = (("all", "Y + U + V (whole stream)"), ("luma", "Luma plane only"), ("chroma", "Chroma planes only"))


class PixelAudio:
    """Pixels <-> A-law sample stream (same layout as Sonify)."""

    def __init__(self, img: np.ndarray, planes: str):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.yuv, self.padded = rgb_to_yuv420p(np.ascontiguousarray(img[..., :3]))
        self.src = self.yuv.reshape(-1)
        ysize = self.padded[0] * self.padded[1]
        self.sl = {"luma": slice(0, ysize), "chroma": slice(ysize, None)}.get(planes, slice(None))

    def samples(self) -> np.ndarray:
        return alaw_decode(self.src[self.sl])

    def rebuild(self, samples: np.ndarray) -> np.ndarray:
        merged = self.src.copy()
        merged[self.sl] = alaw_encode(samples)
        rgb = yuv420p_to_rgb(merged.reshape(self.yuv.shape), self.padded)
        out = self.img.copy()
        out[..., :3] = rgb[: self.h, : self.w]
        return out


def effective_rate(p, scale: float) -> int:
    rate = int(p["rate"])
    if p.get("scale_rate", True) and scale < 1.0:
        rate = max(1000, int(round(rate * scale)))
    return rate


def fft_convolve(x: np.ndarray, ir: np.ndarray, check=None, block: int = 1 << 19) -> np.ndarray:
    """Linear convolution truncated to len(x), by FFT overlap-add (bounded memory)."""
    n, m = len(x), len(ir)
    if n == 0 or m == 0:
        return np.zeros_like(x)
    size = 1 << int(np.ceil(np.log2(block + m - 1)))
    H = np.fft.rfft(ir.astype(np.float32), size)
    out = np.zeros(n + m - 1, np.float32)
    for start in range(0, n, block):
        if check:
            check()
        seg = x[start:start + block].astype(np.float32)
        y = np.fft.irfft(np.fft.rfft(seg, size) * H, size)[: len(seg) + m - 1]
        out[start:start + len(y)] += y.astype(np.float32)
    return out[:n]


def synthetic_ir(kind: str, rt60: float, predelay_ms: float, damping: float, rate: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Exponentially decaying noise impulse response with early reflections."""
    rt60 = float(max(0.02, rt60))
    length = int(min(rt60 * 1.3, 10.0) * rate) + 1
    t = np.arange(length, dtype=np.float32) / rate
    env = np.exp(-6.91 * t / rt60).astype(np.float32)
    noise = rng.normal(0.0, 1.0, length).astype(np.float32)
    if damping > 0:
        # duller tail: blend towards a smoothed copy as time goes on
        k = max(2, int(rate * 0.0006 * (1 + 10 * damping)))
        smooth = np.convolve(noise, np.ones(k, np.float32) / k, mode="same")
        d = np.clip(damping * t / max(t[-1], 1e-6), 0, 1)
        noise = noise * (1 - d) + smooth * d * np.sqrt(k)
    ir = noise * env
    early = {"cathedral": 9, "hall": 7, "room": 4, "plate": 0}.get(kind, 5)
    for i in range(early):
        pos = int(rate * rng.uniform(0.005, 0.08 if kind != "room" else 0.03))
        if pos < length:
            ir[pos] += rng.uniform(0.4, 1.0) * (1 if rng.random() < 0.5 else -1) * 3.0
    pre = int(rate * predelay_ms / 1000.0)
    ir = np.concatenate([np.zeros(pre, np.float32), ir]) if pre > 0 else ir
    return ir / max(float(np.sqrt(np.sum(ir * ir))), 1e-9)


_AUDIO_CACHE: OrderedDict = OrderedDict()
_AUDIO_LOCK = threading.Lock()
MAX_AUDIO_SECONDS = 600


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        ch, sw, sr, n = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        n = min(n, sr * MAX_AUDIO_SECONDS)
        raw = wf.readframes(n)
    if sw == 1:
        a = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        a = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        a = v.astype(np.float32) / float(1 << 23)
    elif sw == 4:
        a = np.frombuffer(raw, "<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        a = a[: len(a) // ch * ch].reshape(-1, ch).mean(axis=1)
    return a.astype(np.float32), sr


def _resample(a: np.ndarray, sr: int, rate: int) -> np.ndarray:
    if sr == rate or len(a) < 2:
        return a
    n = max(1, int(round(len(a) * rate / sr)))
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def load_audio(path: str, rate: int, services, cancel) -> np.ndarray:
    """Decode an audio file to mono float32 at ``rate`` Hz (cached).

    PCM WAV files are read directly; other formats (or unusual WAVs) are
    decoded with FFmpeg into raw 16-bit mono.
    """
    if not path or not os.path.isfile(path):
        raise EffectError(f"Audio file not found: {path or '(none)'}")
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size, rate)
    with _AUDIO_LOCK:
        if key in _AUDIO_CACHE:
            _AUDIO_CACHE.move_to_end(key)
            return _AUDIO_CACHE[key]
    audio = None
    if path.lower().endswith((".wav", ".wave")):
        try:
            a, sr = _read_wav(path)
            audio = _resample(a, sr, rate)
        except (wave.Error, ValueError, EOFError) as exc:
            log.info("WAV reader could not read %s (%s); trying FFmpeg", path, exc)
    if audio is None:
        if services is None:
            raise EffectError("Decoding this audio format needs FFmpeg.")
        try:
            services.ffmpeg(needed=())
        except DependencyError as exc:
            raise DependencyError("Only PCM .wav files can be read without FFmpeg. " + exc.message) from exc
        with services.temp_dir("glitchlab-audio-") as tmp:
            out = os.path.join(tmp, "decoded.s16")
            services.run_ffmpeg(["-i", path, "-vn", "-t", str(MAX_AUDIO_SECONDS), "-ac", "1", "-ar", str(rate),
                                 "-f", "s16le", "-c:a", "pcm_s16le", "-y", out], cancel=cancel, timeout=180)
            with open(out, "rb") as fh:
                audio = np.frombuffer(fh.read(), "<i2").astype(np.float32) / 32768.0
    if audio.size == 0:
        raise EffectError(f"'{os.path.basename(path)}' contains no audio samples.")
    with _AUDIO_LOCK:
        _AUDIO_CACHE[key] = audio
        while len(_AUDIO_CACHE) > 4:
            _AUDIO_CACHE.popitem(last=False)
    return audio


AUDIO_FILTER = ("Audio files (*.wav *.mp3 *.flac *.ogg *.oga *.m4a *.aac *.aif *.aiff *.opus *.wma);;"
                "All files (*)")


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------

@register
class AudioReverbSmear(Effect):
    id = "audio_reverb"
    name = "Audio Reverb Smearing"
    category = "Audio Bending"
    description = ("Treats the pixels as audio and runs a convolution reverb over them - a synthetic cathedral, "
                   "hall or room, or your own impulse-response file. Bright highlights cast ghostly, rhythmic "
                   "echoes across the frame; reverse reverb makes the ghosts appear before them.")
    supports_channels = True
    preview_note = "With 'Scale rate with preview' on, echo spacing follows the image size; details still differ."
    params = (
        cparam("space", "Space", "cathedral", (("cathedral", "Cathedral"), ("hall", "Concert hall"),
                                               ("room", "Small room"), ("plate", "Plate"), ("file", "Impulse-response file"))),
        Param("ir_file", "Impulse response", FILE, "", file_filter=AUDIO_FILTER, show_if=("space", ("file",)),
              randomizable=False, tooltip="Any audio file; its sound is used as the reverb's impulse response."),
        fparam("decay", "Decay time", 2.5, 0.05, 8.0, decimals=2, unit="s", show_if=("space", ("cathedral", "hall", "room", "plate"))),
        fparam("predelay", "Pre-delay", 30.0, 0.0, 500.0, decimals=0, unit="ms"),
        fparam("damping", "Damping", 0.5, 0.0, 1.0, show_if=("space", ("cathedral", "hall", "room", "plate"))),
        bparam("reverse", "Reverse reverb", False, tooltip="Reverse, reverberate, reverse back: ghosts swell before highlights."),
        fparam("wet", "Wet", 0.55, 0.0, 1.0),
        fparam("dry", "Dry", 0.75, 0.0, 1.0),
        cparam("rate", "Sample rate", "22050", RATES),
        bparam("scale_rate", "Scale rate with preview", True),
        cparam("planes", "Process", "all", PLANES),
    )

    def validate(self, p):
        if p["space"] == "file" and not p["ir_file"]:
            return ["choose an impulse-response audio file (or a synthetic space)."]
        return []

    def process(self, img, p, ctx):
        rate = effective_rate(p, ctx.scale)
        pa = PixelAudio(img, p["planes"])
        x = pa.samples()
        if p["space"] == "file":
            ir = load_audio(p["ir_file"], rate, ctx.services, ctx.cancel)[: rate * 10]
            pre = np.zeros(int(rate * p["predelay"] / 1000.0), np.float32)
            ir = np.concatenate([pre, ir])
            ir = ir / max(float(np.sqrt(np.sum(ir * ir))), 1e-9)
        else:
            ir = synthetic_ir(p["space"], p["decay"], p["predelay"], p["damping"], rate, ctx.sub_rng(7))
        ctx.check()
        src = x[::-1] if p["reverse"] else x
        y = fft_convolve(src - np.float32(src.mean()), ir, check=ctx.check)
        if p["reverse"]:
            y = y[::-1]
        # keep the reverb at a level comparable to the input
        rx, ry = float(np.std(x)) or 1e-6, float(np.std(y)) or 1e-6
        y = y * np.float32(rx / ry)
        out = np.float32(p["dry"]) * x + np.float32(p["wet"]) * y
        return pa.rebuild(np.clip(out, -1.0, 1.0))


BUILTIN_SIGNALS = (("beat", "Built-in drum beat"), ("sweep", "Built-in sine sweep"), ("noise", "Built-in noise bursts"))


def builtin_signal(kind: str, rate: int, bpm: float, rng: np.random.Generator) -> np.ndarray:
    """An 8-second synthetic sound (looped or stretched by the caller)."""
    seconds = 8.0
    m = int(seconds * rate)
    t = np.arange(m, dtype=np.float32) / rate
    if kind == "sweep":
        f0, f1 = 40.0, rate * 0.25
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * seconds))
        sig = 0.8 * np.sin(phase)
    elif kind == "noise":
        sig = rng.normal(0, 0.4, m).astype(np.float32)
        gate = (np.sin(2 * np.pi * t * bpm / 60.0) > 0.3).astype(np.float32)
        sig = sig * gate
    else:  # beat: kick on 1 and 3, snare on 2 and 4, hats on 8ths
        beat = 60.0 / bpm
        sig = np.zeros(m, np.float32)
        for k in range(int(seconds / beat * 2)):
            start = int(k * beat / 2 * rate)
            if start >= m:
                break
            tt = t[: m - start][: int(0.4 * rate)]
            if k % 4 == 0:
                hit = np.sin(2 * np.pi * 55 * tt * (1 + 2 * np.exp(-tt * 30))) * np.exp(-tt * 9)
            elif k % 4 == 2:
                hit = rng.normal(0, 0.6, len(tt)).astype(np.float32) * np.exp(-tt * 18)
            else:
                hit = rng.normal(0, 0.25, len(tt)).astype(np.float32) * np.exp(-tt * 60)
            sig[start:start + len(hit)] += hit.astype(np.float32)
    return np.clip(sig, -1, 1).astype(np.float32)


@register
class AudioFileBend(Effect):
    id = "audio_file_bend"
    name = "Audio File Bending"
    category = "Audio Bending"
    description = ("Bends the pixels with a sound: load any audio file (WAV, MP3, FLAC, OGG …) or use a built-in "
                   "beat, then mix, add, ring-modulate, envelope-follow or gate it into the pixel data read as audio.")
    supports_channels = True
    preview_note = "The audio is stretched or looped over the pixel stream; the preview uses fewer samples."
    params = (
        cparam("source", "Sound", "beat", BUILTIN_SIGNALS + (("file", "Audio file"),)),
        Param("audio_file", "Audio file", FILE, "", file_filter=AUDIO_FILTER, show_if=("source", ("file",)),
              randomizable=False, tooltip="WAV is read directly; other formats need FFmpeg (bundled)."),
        fparam("bpm", "Beat tempo", 120.0, 30.0, 400.0, decimals=0, unit="bpm", show_if=("source", ("beat", "noise"))),
        cparam("mode", "Combine", "mix", (("mix", "Mix"), ("add", "Add"), ("ring", "Ring modulate"),
                                          ("envelope", "Envelope follower"), ("gate", "Gate / replace"))),
        fparam("amount", "Amount", 0.3, 0.0, 1.0),
        fparam("gain_db", "Sound gain", 0.0, -30.0, 30.0, unit="dB"),
        cparam("fit", "Fit", "stretch", (("stretch", "Stretch over the image"), ("loop", "Loop at sample rate"))),
        fparam("offset", "Start offset", 0.0, 0.0, 600.0, decimals=1, unit="s"),
        cparam("rate", "Sample rate", "22050", RATES),
        bparam("scale_rate", "Scale rate with preview", True),
        cparam("planes", "Process", "all", PLANES),
    )

    def validate(self, p):
        if p["source"] == "file" and not p["audio_file"]:
            return ["choose an audio file (or a built-in sound)."]
        return []

    def process(self, img, p, ctx):
        rate = effective_rate(p, ctx.scale)
        pa = PixelAudio(img, p["planes"])
        x = pa.samples()
        n = len(x)
        if p["source"] == "file":
            aud = load_audio(p["audio_file"], rate, ctx.services, ctx.cancel)
        else:
            aud = builtin_signal(p["source"], rate, p["bpm"], ctx.sub_rng(3))
        start = int(p["offset"] * rate) % max(1, len(aud))
        aud = np.roll(aud, -start)
        if p["fit"] == "stretch" and len(aud) > 1:
            aud = np.interp(np.linspace(0, len(aud) - 1, n), np.arange(len(aud)), aud).astype(np.float32)
        else:
            aud = np.resize(aud, n).astype(np.float32)
        aud = aud * np.float32(10 ** (p["gain_db"] / 20.0))
        a = np.float32(p["amount"])
        mode = p["mode"]
        if mode == "mix":
            out = x * (1 - a) + aud * a
        elif mode == "add":
            out = x + aud * a
        elif mode == "ring":
            peak = max(float(np.max(np.abs(aud))), 1e-6)
            out = x * ((1 - a) + a * aud / peak)
        elif mode == "envelope":
            k = max(1, int(rate * 0.01))
            env = np.convolve(np.abs(aud), np.ones(k, np.float32) / k, mode="same")
            env = env / max(float(env.max()), 1e-6)
            out = x * ((1 - a) + a * env * 2.0)
        else:
            thr = (1.0 - a) * max(float(np.max(np.abs(aud))), 1e-6)
            out = np.where(np.abs(aud) > thr, aud, x)
        return pa.rebuild(np.clip(out, -1.0, 1.0))
