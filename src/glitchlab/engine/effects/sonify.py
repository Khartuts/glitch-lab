"""Sonify: pixels interpreted as audio and processed with FFmpeg filters.

Pipeline (all formats explicit)::

    RGB (alpha kept aside)
      -> pad to even width/height (edge replicate)
      -> YUV 4:2:0 planar bytes: Y plane (W*H), then U, then V (W/2*H/2 each),
         via OpenCV COLOR_RGB2YUV_I420 (BT.601)
      -> written to a temp file and read by FFmpeg as raw **A-law** audio:
         ``-f alaw -ar <rate> -ac 1`` (8-bit A-law, mono)
      -> audio filter chain (-af)
      -> encoded back to raw A-law (``-c:a pcm_alaw -f alaw``), same rate, mono
      -> bytes reinterpreted as YUV 4:2:0 -> RGB -> cropped to original size
      -> source alpha restored.

Length policy: some filters (echo) add a tail and others may shorten the
stream.  Output longer than the input is **truncated**; output shorter than
the input is completed with the **unprocessed source bytes** for the missing
positions.  The A-law round trip with no filter is byte-exact.

The pixel-derived audio is only ever written to temporary files that are
removed afterwards; it is never played.
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from ..errors import EffectError
from ..params import bparam, cparam, fparam, iparam
from .base import Effect, register

log = logging.getLogger(__name__)

RATES = (("8000", "8 kHz"), ("11025", "11.025 kHz"), ("22050", "22.05 kHz"), ("44100", "44.1 kHz"), ("48000", "48 kHz"))
CLIP_TYPES = (("none", "No soft clipping"), ("hard", "Hard"), ("tanh", "Tanh"), ("atan", "Arctan"),
              ("cubic", "Cubic"), ("exp", "Exponential"), ("sin", "Sine"))


def rgb_to_yuv420p(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad to even dimensions and convert to planar I420 bytes (H*3/2 x W)."""
    h, w = rgb.shape[:2]
    ph, pw = h + (h & 1), w + (w & 1)
    if (ph, pw) != (h, w):
        rgb = cv2.copyMakeBorder(rgb, 0, ph - h, 0, pw - w, cv2.BORDER_REPLICATE)
    yuv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2YUV_I420)
    return yuv, (ph, pw)


def yuv420p_to_rgb(yuv: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
    return rgb[:h, :w]


def fit_length(out: bytes, src: np.ndarray) -> np.ndarray:
    """Apply the documented length policy (truncate / complete with source bytes)."""
    o = np.frombuffer(out, np.uint8)
    n = src.size
    if o.size >= n:
        return o[:n].copy()
    return np.concatenate([o, src.reshape(-1)[o.size:]])


@register
class Sonify(Effect):
    id = "sonify"
    name = "Sonify (Pixels as Audio)"
    category = "Data Bending"
    description = ("Real visual databending: YUV420p pixel bytes are read as raw A-law audio, run through "
                   "FFmpeg audio filters (echo, phaser, bit crusher, EQ…) and read back as pixels.")
    requires_ffmpeg = True
    uses_seed = False
    supports_channels = True
    preview_note = ("With 'Scale rate with preview' on, the sample rate follows the preview size so filters "
                    "hit similar image structures; results still differ in detail from the export.")
    params = (
        cparam("rate", "Sample rate", "44100", RATES, tooltip="How many bytes count as one second of audio."),
        bparam("scale_rate", "Scale rate with preview", True),
        cparam("planes", "Process", "all", (("all", "Y + U + V (whole stream)"), ("luma", "Luma plane only"),
                                             ("chroma", "Chroma planes only"))),
        fparam("gain_db", "Gain", 0.0, -24.0, 24.0, unit="dB", group="Gain & clipping"),
        cparam("clip", "Soft clip", "none", CLIP_TYPES, group="Gain & clipping"),
        fparam("clip_threshold", "Clip threshold", 0.8, 0.05, 1.0, group="Gain & clipping",
               show_if=("clip", tuple(c[0] for c in CLIP_TYPES[1:]))),
        fparam("bass_gain", "Bass", 0.0, -30.0, 30.0, unit="dB", group="Bass & treble"),
        fparam("bass_freq", "Bass frequency", 100.0, 20.0, 2000.0, decimals=0, unit="Hz", group="Bass & treble"),
        fparam("treble_gain", "Treble", 0.0, -30.0, 30.0, unit="dB", group="Bass & treble"),
        fparam("treble_freq", "Treble frequency", 3000.0, 500.0, 16000.0, decimals=0, unit="Hz", group="Bass & treble"),
        bparam("hp", "High-pass", False, group="Filters"),
        fparam("hp_freq", "High-pass cutoff", 300.0, 10.0, 20000.0, decimals=0, unit="Hz", group="Filters",
               show_if=("hp", (True,))),
        bparam("lp", "Low-pass", False, group="Filters"),
        fparam("lp_freq", "Low-pass cutoff", 3000.0, 10.0, 20000.0, decimals=0, unit="Hz", group="Filters",
               show_if=("lp", (True,))),
        bparam("bp", "Band-pass", False, group="Filters"),
        fparam("bp_freq", "Band-pass centre", 1000.0, 10.0, 20000.0, decimals=0, unit="Hz", group="Filters",
               show_if=("bp", (True,))),
        fparam("bp_q", "Band-pass Q", 0.7, 0.1, 10.0, group="Filters", show_if=("bp", (True,))),
        bparam("crush", "Bit crusher", False, group="Bit crusher"),
        iparam("crush_bits", "Bits", 4, 1, 16, group="Bit crusher", show_if=("crush", (True,))),
        iparam("crush_samples", "Sample hold", 1, 1, 64, group="Bit crusher", show_if=("crush", (True,)),
               tooltip="Downsampling factor of the crusher."),
        fparam("crush_mix", "Crusher mix", 1.0, 0.0, 1.0, group="Bit crusher", show_if=("crush", (True,))),
        bparam("echo", "Echo", True, group="Echo"),
        fparam("echo_delay", "Delay", 120.0, 1.0, 2000.0, decimals=0, unit="ms", group="Echo", show_if=("echo", (True,))),
        fparam("echo_decay", "Decay", 0.45, 0.01, 0.99, group="Echo", show_if=("echo", (True,))),
        fparam("echo_in", "Input gain", 0.8, 0.0, 1.0, group="Echo", show_if=("echo", (True,))),
        fparam("echo_out", "Output gain", 0.9, 0.0, 1.0, group="Echo", show_if=("echo", (True,))),
        bparam("phaser", "Phaser", False, group="Phaser"),
        fparam("phaser_delay", "Delay", 3.0, 0.0, 5.0, unit="ms", group="Phaser", show_if=("phaser", (True,))),
        fparam("phaser_decay", "Decay", 0.4, 0.0, 0.99, group="Phaser", show_if=("phaser", (True,))),
        fparam("phaser_speed", "Speed", 0.5, 0.1, 2.0, unit="Hz", group="Phaser", show_if=("phaser", (True,))),
        cparam("phaser_type", "Shape", "t", (("t", "Triangular"), ("s", "Sinusoidal")), group="Phaser",
               show_if=("phaser", (True,))),
        bparam("flanger", "Flanger", False, group="Flanger"),
        fparam("flanger_delay", "Delay", 0.0, 0.0, 30.0, unit="ms", group="Flanger", show_if=("flanger", (True,))),
        fparam("flanger_depth", "Depth", 2.0, 0.0, 10.0, unit="ms", group="Flanger", show_if=("flanger", (True,))),
        fparam("flanger_regen", "Regeneration", 0.0, -95.0, 95.0, decimals=0, unit="%", group="Flanger",
               show_if=("flanger", (True,))),
        fparam("flanger_speed", "Speed", 0.5, 0.1, 10.0, unit="Hz", group="Flanger", show_if=("flanger", (True,))),
        bparam("tremolo", "Tremolo", False, group="Tremolo"),
        fparam("tremolo_freq", "Frequency", 5.0, 0.1, 2000.0, unit="Hz", group="Tremolo", show_if=("tremolo", (True,))),
        fparam("tremolo_depth", "Depth", 0.5, 0.0, 1.0, group="Tremolo", show_if=("tremolo", (True,))),
    )

    # ------------------------------------------------------------------
    @staticmethod
    def build_chain(p, rate: int) -> tuple[str, list[str]]:
        """Return (-af filter string, list of filter names used)."""
        nyq = rate * 0.45

        def f(v):
            return f"{max(1.0, min(float(v), nyq)):.3f}"

        parts: list[tuple[str, str]] = []
        if abs(p["gain_db"]) > 1e-6:
            parts.append(("volume", f"volume=volume={p['gain_db']:.3f}dB"))
        if abs(p["bass_gain"]) > 1e-6:
            parts.append(("bass", f"bass=g={p['bass_gain']:.3f}:f={f(p['bass_freq'])}"))
        if abs(p["treble_gain"]) > 1e-6:
            parts.append(("treble", f"treble=g={p['treble_gain']:.3f}:f={f(p['treble_freq'])}"))
        if p["hp"]:
            parts.append(("highpass", f"highpass=f={f(p['hp_freq'])}:p=2"))
        if p["lp"]:
            parts.append(("lowpass", f"lowpass=f={f(p['lp_freq'])}:p=2"))
        if p["bp"]:
            parts.append(("bandpass", f"bandpass=f={f(p['bp_freq'])}:width_type=q:w={p['bp_q']:.3f}"))
        if p["crush"]:
            parts.append(("acrusher", f"acrusher=bits={p['crush_bits']}:samples={p['crush_samples']}"
                                      f":mix={p['crush_mix']:.3f}:mode=lin:aa=0"))
        if p["echo"]:
            parts.append(("aecho", f"aecho=in_gain={p['echo_in']:.3f}:out_gain={p['echo_out']:.3f}"
                                   f":delays={p['echo_delay']:.1f}:decays={p['echo_decay']:.3f}"))
        if p["phaser"]:
            parts.append(("aphaser", f"aphaser=in_gain=0.4:out_gain=0.74:delay={p['phaser_delay']:.3f}"
                                     f":decay={p['phaser_decay']:.3f}:speed={p['phaser_speed']:.3f}:type={p['phaser_type']}"))
        if p["flanger"]:
            parts.append(("flanger", f"flanger=delay={p['flanger_delay']:.3f}:depth={p['flanger_depth']:.3f}"
                                     f":regen={p['flanger_regen']:.1f}:width=71:speed={p['flanger_speed']:.3f}"
                                     ":shape=sinusoidal"))
        if p["tremolo"]:
            parts.append(("tremolo", f"tremolo=f={max(0.1, min(p['tremolo_freq'], 20000.0)):.3f}"
                                     f":d={p['tremolo_depth']:.3f}"))
        if p["clip"] != "none":
            parts.append(("asoftclip", f"asoftclip=type={p['clip']}:threshold={p['clip_threshold']:.4f}"))
        return ",".join(s for _, s in parts), [n for n, _ in parts]

    def process(self, img, p, ctx):
        rate = int(p["rate"])
        if p["scale_rate"] and ctx.scale < 1.0:
            rate = max(1000, int(round(rate * ctx.scale)))
        chain, used = self.build_chain(p, rate)
        if not chain:
            ctx.warn("no audio filter is enabled; the image is unaltered")
            return img.copy()
        if ctx.services is None:
            raise EffectError("Sonify needs the FFmpeg service.")
        ctx.services.ffmpeg(needed=tuple(used))  # raises an actionable DependencyError
        h, w = img.shape[:2]
        yuv, padded = rgb_to_yuv420p(np.ascontiguousarray(img[..., :3]))
        ph, pw = padded
        ysize = ph * pw
        src = yuv.reshape(-1)
        if p["planes"] == "luma":
            stream = src[:ysize]
        elif p["planes"] == "chroma":
            stream = src[ysize:]
        else:
            stream = src
        timeout = 60.0 + stream.size / 1e6 * 4.0
        with ctx.services.temp_dir("glitchlab-sonify-") as tmp:
            fin = os.path.join(tmp, "pixels_in.alaw")
            fout = os.path.join(tmp, "pixels_out.alaw")
            stream.tofile(fin)
            ctx.check()
            ctx.services.run_ffmpeg(
                ["-f", "alaw", "-ar", str(rate), "-ac", "1", "-i", fin,
                 "-af", chain,
                 "-c:a", "pcm_alaw", "-f", "alaw", "-ar", str(rate), "-ac", "1", "-y", fout],
                cancel=ctx.cancel, timeout=timeout)
            try:
                with open(fout, "rb") as fh:
                    data = fh.read()
            except OSError as exc:
                raise EffectError("FFmpeg produced no output.") from exc
        processed = fit_length(data, stream)
        if len(data) != stream.size:
            log.debug("sonify: output %d bytes vs input %d (policy applied)", len(data), stream.size)
        merged = src.copy()
        if p["planes"] == "luma":
            merged[:ysize] = processed
        elif p["planes"] == "chroma":
            merged[ysize:] = processed
        else:
            merged = processed
        rgb = yuv420p_to_rgb(merged.reshape(yuv.shape), padded)
        out = img.copy()
        out[..., :3] = rgb[:h, :w]
        return out
