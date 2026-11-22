"""Built-in and user presets.

A preset is a real effect stack: effect ids, every parameter value, mix,
channel targeting and the stored seeds.  Applying a preset replaces (or is
appended to) the active layer's stack; seeds are copied so a preset always
produces the same result on the same image and resolution.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from .effects.base import REGISTRY
from .errors import ProjectError
from .model import Document, EffectInstance, new_uid

log = logging.getLogger(__name__)

PRESET_FORMAT = "glitchlab-preset"
PRESET_VERSION = 1


def _fx(effect: str, seed: int, mix: float = 1.0, channels: str = "rgb", **params) -> dict:
    return {"effect": effect, "seed": seed, "mix": mix, "channels": channels, "params": params}


BUILTIN_PRESETS: list[dict] = [
    {
        "name": "Broken Broadcast",
        "description": "Torn analogue-style scanlines, drifting rows, chroma fringing and blocky decay.",
        "stack": [
            _fx("row_drift", 4101, axis="rows", waveform="sine", amplitude=10, frequency=3.0, phase=0.0,
                chroma_phase=0.25, edge="wrap"),
            _fx("scanline_tear", 4102, bands=22, min_thickness=2, max_thickness=36, displacement=140,
                jitter=4, direction="both", rgb_split=6, edge="wrap"),
            _fx("channel_shift", 4103, space="ycbcr", r_dx=0, r_dy=0, g_dx=10, g_dy=0, b_dx=-10, b_dy=2,
                edge="clamp"),
            _fx("jpeg_decay", 4104, generations=6, quality_min=12, quality_max=35, subsampling="4:2:0",
                downscale=0.75, shift_px=1, mix=0.85),
        ],
    },
    {
        "name": "Acid Chromatic",
        "description": "Large channel offsets, false-colour remapping and hue-driven cross-modulation.",
        "stack": [
            _fx("channel_shift", 4201, space="rgb", r_dx=24, r_dy=-6, g_dx=0, g_dy=10, b_dx=-26, b_dy=0,
                edge="mirror"),
            _fx("cross_mod", 4202, source="hue", destination="g", mode="remap", amount=0.7, frequency=3.0),
            _fx("palette", 4203, mode="false_colour", gradient="acid", levels=6, dither="none", mix=0.65),
        ],
    },
    {
        "name": "Industrial Decay",
        "description": "Generations of harsh JPEG compression, stuck and bleeding blocks and dropped bit planes.",
        "stack": [
            _fx("jpeg_decay", 4301, generations=14, quality_min=5, quality_max=25, subsampling="4:2:0",
                downscale=0.6, shift_px=2),
            _fx("mosaic_bleed", 4302, block=24, amount=0.18, source="repeat", leakage=0.4, feather=0.25),
            _fx("block_scramble", 4303, tile=48, amount=0.08, mode="duplicate"),
            _fx("bitplane", 4304, operation="drop", planes=0b00000111, depth=8, mix=0.8),
        ],
    },
    {
        "name": "Corrupted Memory",
        "description": "Genuine BMP byte corruption, a misread row stride and scrambled tiles.",
        "stack": [
            _fx("hex_corrupt", 4401, container="bmp", search="DA", replace="0E DA", replace_prob=0.35,
                max_replacements=400, mutation="random", density=0.0005, iterations=2,
                region_start=0.1, region_end=1.0, length_policy="truncate_pad_source"),
            _fx("reinterpret", 4402, layout="interleaved", channel_order="gbr", width_delta=3, offset=0,
                mix=0.7),
            _fx("block_scramble", 4403, tile=32, amount=0.12, mode="mixed", offset_max=0.5),
        ],
    },
    {
        "name": "Spectral Ghost",
        "description": "Frequency-band boosting with phase noise, echoed by zooming feedback.",
        "stack": [
            _fx("spectral", 4501, band_low=0.02, band_high=0.35, gain=1.8, phase_noise=0.6,
                outside_gain=0.7, channels_mode="per_channel", normalize="clip"),
            _fx("feedback", 4502, iterations=10, zoom=1.03, rotation=0.6, dx=0, dy=0, decay=0.82,
                blend="screen", mix=0.8),
            _fx("channel_shift", 4503, space="rgb", r_dx=5, r_dy=0, g_dx=0, g_dy=0, b_dx=-5, b_dy=0,
                edge="clamp", mix=0.8),
        ],
    },
    {
        "name": "Scanner Failure",
        "description": "Slit-scan strip misfeeds, stepped column drift and vertical pixel sorting.",
        "stack": [
            _fx("slitscan", 4601, orientation="vertical", strip=6, order="drift", drift=0.25,
                wave_amount=40, wave_cycles=2.0, stretch=1.0),
            _fx("row_drift", 4602, axis="columns", waveform="steps", amplitude=30, frequency=6.0,
                phase=0.0, chroma_phase=0.0, edge="clamp"),
            _fx("pixel_sort", 4603, direction="vertical", interval="luminance", lower=0.55, upper=1.0,
                sort_by="luminance", max_span=180, span_random=0.6),
        ],
    },
    {
        "name": "Digital Rust",
        "description": "Gray-Scott reaction-diffusion corrosion, posterised palette and error trails.",
        "stack": [
            _fx("reaction_diffusion", 4701, iterations=1800, feed=0.0545, kill=0.062, sim_size=256,
                seed_mode="luminance", seed_threshold=0.45, blend="multiply", color_a="#2b1206",
                color_b="#f08a3c", mix=0.75),
            _fx("error_trails", 4702, levels=4, direction="down", w_down=0.5, w_down_left=0.25,
                w_down_right=0.25, gain=1.05, drift=1, clamp=96, mode="per_channel", mix=0.6),
        ],
    },
    {
        "name": "Photocopier Meltdown",
        "description": "High-contrast dithered toner, luminance-driven warping, cellular grime and feedback ghosting.",
        "stack": [
            _fx("displace_map", 4801, source="noise", amount_x=18, amount_y=8, noise_scale=6.0,
                smooth=2, edge="clamp"),
            _fx("palette", 4802, mode="posterize", levels=3, dither="ordered8", gradient="mono",
                channels="luma"),
            _fx("cellular", 4803, seed_mode="luminance", threshold=0.5, rule="B3678/S34678",
                steps=24, cells=220, output="darken", amount=0.55),
            _fx("feedback", 4804, iterations=4, zoom=1.0, rotation=0.0, dx=6, dy=2, decay=0.55,
                blend="multiply", mix=0.6),
        ],
    },
    # Version 1.1 presets
    {
        "name": "Pop Art Misprint",
        "description": "Ben-Day comic dots with a misregistered colour plate and a torn print line.",
        "stack": [
            _fx("benday", 5201, spacing=9, dot_size=0.72, angle=45.0, palette="pop", threshold=0.33,
                black_level=0.16, flatten=6, outline=0.85, outline_px=2, paper="#f6efdc"),
            _fx("channel_shift", 5202, space="rgb", r_dx=4, r_dy=2, g_dx=0, g_dy=0, b_dx=-3, b_dy=-1,
                edge="clamp"),
            _fx("scanline_tear", 5203, bands=3, min_thickness=6, max_thickness=30, displacement=40, jitter=0,
                direction="both", rgb_split=0, edge="clamp", mix=0.9),
        ],
    },
    {
        "name": "Comic Moiré",
        "description": "Halftone glitch hybrid: CMYK dot plates drifting in waves, misregistered and sliced.",
        "stack": [
            _fx("cmyk_drift", 5301, cell=8, shape="round", gcr=0.7, angle_offset=0.0, drift_plates="cm",
                drift_amp=7, drift_wavelength=160, drift_direction="both", drift_phase=0.15, c_dx=3, c_dy=0,
                m_dx=-2, m_dy=2, y_dx=0, y_dy=0, k_dx=0, k_dy=0, slice_bands=8, slice_shift=36,
                slice_plates="cm", paper="#f7f3e8"),
        ],
    },
    {
        "name": "VHS Night",
        "description": "Smeared tape colour, tracking errors and a curved CRT.",
        "stack": [
            _fx("row_drift", 5401, axis="rows", waveform="noise", amplitude=6, frequency=8.0, phase=0.0,
                chroma_phase=0.1, edge="clamp"),
            _fx("vhs_crt", 5402, chroma_blur=9, chroma_shift=4, saturation=1.25, sharpen=0.6, tracking=0.6,
                tracking_bands=4, noise=0.18, streaks=14, scanlines=0.4, scan_period=3, mask="aperture",
                mask_strength=0.3, bloom=0.35, vignette=0.45, curvature=0.1, border=True),
        ],
    },
    {
        "name": "Stained Glass Meltdown",
        "description": "Voronoi cells sorted dark to light, leaded, with lens fringing.",
        "stack": [
            _fx("voronoi_sort", 5501, cells=34, sort_by="luminance", direction="vertical", reverse=False,
                lead=2, lead_color="#141414"),
            _fx("chromatic", 5502, strength=9.0, falloff=2.0, cx=0.5, cy=0.5, green=0.0, blur=1.0),
        ],
    },
    {
        "name": "Cathedral Ghosts",
        "description": "Pixels played through a cathedral reverb: highlights echo across the frame.",
        "stack": [
            _fx("audio_reverb", 5601, space="cathedral", decay=3.5, predelay=40.0, damping=0.6, reverse=False,
                wet=0.5, dry=0.8, rate="22050", scale_rate=True, planes="luma"),
        ],
    },
    {
        "name": "Lyric Injection",
        "description": "Text pasted into the JPEG quantisation tables and inserted into the pixel data.",
        "stack": [
            _fx("data_injection", 5701, source="text", text="i can hear the colours glitch // 48.8584 N 2.2945 E",
                target="jpeg_tables", mode="xor", count=4, chunk=40, placement="even", tables="chroma",
                quality=85, mix=0.85),
            _fx("data_injection", 5702, source="text", text="i can hear the colours glitch // 48.8584 N 2.2945 E",
                target="bmp_pixels", mode="insert", count=5, chunk=33, placement="random",
                region_start=0.2, region_end=0.9, length_policy="truncate_pad_source"),
        ],
    },
    {
        "name": "Tectonic Portrait",
        "description": "Brightness contours sliced into sliding tectonic plates with drawn elevation lines.",
        "stack": [
            _fx("contour_tear", 5801, levels=16, smooth=8, every=3, phase=1, offset_x=26, offset_y=6,
                alternate=True, growth=0.6, lines=0.45, line_px=1, line_color="#101010", edge="clamp"),
        ],
    },
    # Version 1.2 presets: experimental / abstract aesthetics
    {
        "name": "Kaleidoscope Bloom",
        "description": "Ten mirrored wedges spun from the photo, with soft lens fringing.",
        "stack": [
            _fx("kaleidoscope", 6001, mode="wedge", segments=10, mirrored=True, cx=0.5, cy=0.5, rotation=0.0,
                zoom=1.3, source_angle=24.0, source_x=0.0, source_y=0.0, edge="mirror"),
            _fx("chromatic", 6002, strength=7.0, falloff=2.0, cx=0.5, cy=0.5, green=0.0, blur=1.0),
        ],
    },
    {
        "name": "Acid Kaleidoscope",
        "description": "Hexagonal mirror tiles flooded with cycling acid colour bands.",
        "stack": [
            _fx("kaleidoscope", 6101, mode="hex", tile=220, cx=0.5, cy=0.5, rotation=0.0, zoom=1.1,
                source_angle=0.0, source_x=0.05, source_y=-0.05, edge="mirror"),
            _fx("psychedelic_pop", 6102, palette="acid", bands=10, cycles=3.0, offset=0.1, smooth=4,
                saturation=1.2, detail=0.3, outline=0.6, outline_px=2, outline_color="#120424", swirl=0.0,
                swirl_radius=0.6, cx=0.5, cy=0.5, mix=0.85),
        ],
    },
    {
        "name": "Psychedelic Pop Ad",
        "description": "A 1970s poster: swirling seventies colour bands, echo outlines and a printed dot texture.",
        "stack": [
            _fx("psychedelic_pop", 6201, palette="seventies", bands=9, cycles=1.5, offset=0.0, smooth=8,
                saturation=1.2, detail=0.2, outline=0.85, outline_px=3, outline_color="#2a0f1f", swirl=220.0,
                swirl_radius=0.7, cx=0.5, cy=0.45),
            _fx("halftone", 6202, mode="cmyk", cell=5, angle=45.0, shape="round", dot_gain=1.0, softness=1.0,
                gcr=0.6, ink="#111111", paper="#fbf3e2", mix=0.35),
        ],
    },
    {
        "name": "Candy Pop Ad",
        "description": "Bubble-gum candy bands with thick echo outlines.",
        "stack": [
            _fx("psychedelic_pop", 6301, palette="candy", bands=7, cycles=1.0, offset=0.15, smooth=10,
                saturation=1.1, detail=0.35, outline=0.9, outline_px=4, outline_color="#3b0a45", swirl=0.0,
                swirl_radius=0.6, cx=0.5, cy=0.5),
        ],
    },
    {
        "name": "Dark Deco",
        "description": "Symmetrical noir-and-gold Art Deco with a sunburst and a double frame.",
        "stack": [
            _fx("kaleidoscope", 6401, mode="mirror", axis="vertical", flip_side=False, cx=0.5, cy=0.5,
                rotation=0.0, zoom=1.0, source_angle=0.0, source_x=0.0, source_y=0.0, edge="mirror"),
            _fx("dark_deco", 6402, palette="noir_gold", contrast=1.4, gold_level=0.66, pattern="sunburst",
                density=22, origin_y=1.0, pattern_opacity=0.55, line_px=2, panels=True, frame="double",
                frame_inset=0.04, vignette=0.55),
        ],
    },
    {
        "name": "Emerald Deco Fan",
        "description": "Emerald and gold fan motif with nested arcs.",
        "stack": [
            _fx("dark_deco", 6501, palette="emerald", contrast=1.3, gold_level=0.7, pattern="fan", density=22,
                origin_y=1.0, pattern_opacity=0.6, line_px=2, panels=True, frame="double", frame_inset=0.05,
                vignette=0.5),
        ],
    },
    {
        "name": "Luminous Poster",
        "description": "Soft violet light fields around a radiant source (inspired by Koichi Sato's gradient posters).",
        "stack": [
            _fx("luminous_gradient", 6601, palette="deep_blue", abstraction=0.6, glow_x=0.5, glow_y=0.4,
                glow_radius=0.32, glow=1.0, halo=0.6, steps=0, detail=0.1, grain=0.025, vignette=0.6),
        ],
    },
    {
        "name": "Dawn Gradient Poster",
        "description": "Gently stepped rose-to-cream gradient bands with a low glowing sun.",
        "stack": [
            _fx("luminous_gradient", 6701, palette="dawn", abstraction=0.5, glow_x=0.5, glow_y=0.7,
                glow_radius=0.4, glow=0.9, halo=0.45, steps=12, detail=0.1, grain=0.02, vignette=0.45),
        ],
    },
    {
        "name": "Y2K Clear Plastic",
        "description": "Translucent Bondi-blue jelly plastic with glossy highlights and bubbles.",
        "stack": [
            _fx("clear_plastic", 6801, tint="bondi", custom_tint="#9ad0ff", translucency=0.6, smooth=16,
                thickness=1.2, refraction=20, frost=2, gloss=1.1, shininess=56.0, light_angle=135.0, rim=0.6,
                caustics=0.4, bubbles=18),
            _fx("chromatic", 6802, strength=4.0, falloff=2.0, cx=0.5, cy=0.5, green=0.0, blur=0.5),
        ],
    },
    {
        "name": "Tangerine Jelly",
        "description": "Thick, frosted tangerine plastic.",
        "stack": [
            _fx("clear_plastic", 6901, tint="tangerine", custom_tint="#9ad0ff", translucency=0.45, smooth=26,
                thickness=1.8, refraction=28, frost=5, gloss=1.2, shininess=40.0, light_angle=120.0, rim=0.7,
                caustics=0.5, bubbles=8),
        ],
    },
    {
        "name": "Moody Corporate Futurism",
        "description": "Cold navy grade, cyan accents, perspective grid floor, HUD brackets and glass panels.",
        "stack": [
            _fx("corporate_futurism", 7001, grade=0.85, shadow="#04070d", mid="#18293d", highlight="#d9ecff",
                accent="#22e0ff", accent_amount=0.4, grid=True, horizon=0.62, grid_density=16, grid_opacity=0.4,
                haze=0.45, hud=10, panels=2, panel_blur=16, line_opacity=0.65, line_px=1, scan=0.25),
            _fx("vhs_crt", 7002, chroma_blur=2, chroma_shift=1, saturation=1.0, sharpen=0.2, tracking=0.0,
                tracking_bands=0, noise=0.05, streaks=0, scanlines=0.15, scan_period=3, mask="none",
                mask_strength=0.0, bloom=0.35, vignette=0.4, curvature=0.0, border=False),
        ],
    },
    {
        "name": "Hackers",
        "description": "Green code rain over the photo on a curved, scanlined CRT.",
        "stack": [
            _fx("hacker_terminal", 7101, mode="matrix", cell=12, charset="code", text="", colour="green",
                contrast=1.4, invert=False, background=0.15, density=0.8, trail=18, glow=0.6),
            _fx("vhs_crt", 7102, chroma_blur=2, chroma_shift=1, saturation=1.1, sharpen=0.3, tracking=0.15,
                tracking_bands=1, noise=0.06, streaks=3, scanlines=0.4, scan_period=3, mask="aperture",
                mask_strength=0.2, bloom=0.5, vignette=0.5, curvature=0.12, border=True),
        ],
    },
    {
        "name": "Mainframe ASCII",
        "description": "Amber ASCII-art portrait on an old terminal.",
        "stack": [
            _fx("hacker_terminal", 7201, mode="ascii", cell=10, charset="dense", text="", colour="amber",
                contrast=1.5, invert=False, background=0.08, density=0.75, trail=16, glow=0.5),
            _fx("vhs_crt", 7202, chroma_blur=0, chroma_shift=0, saturation=1.0, sharpen=0.0, tracking=0.0,
                tracking_bands=0, noise=0.04, streaks=0, scanlines=0.35, scan_period=3, mask="none",
                mask_strength=0.0, bloom=0.4, vignette=0.45, curvature=0.08, border=True),
        ],
    },
    {
        "name": "Hex Dump Portrait",
        "description": "The image drawn by a green hex dump of its own bytes.",
        "stack": [
            _fx("hacker_terminal", 7301, mode="hexdump", cell=9, charset="hex", text="", colour="green",
                contrast=1.6, invert=False, background=0.05, density=0.75, trail=16, glow=0.35),
        ],
    },
    # Milestone-1-only presets (they use only the first four effects).
    {
        "name": "Tape Snag",
        "description": "Scanline tearing plus a small RGB split.",
        "stack": [
            _fx("scanline_tear", 4901, bands=10, min_thickness=4, max_thickness=60, displacement=120,
                jitter=2, direction="right", rgb_split=3, edge="wrap"),
            _fx("channel_shift", 4902, space="rgb", r_dx=6, r_dy=0, g_dx=0, g_dy=0, b_dx=-6, b_dy=0,
                edge="wrap"),
        ],
    },
    {
        "name": "Sorted Sunset",
        "description": "Classic bright-span horizontal pixel sort with randomised span lengths.",
        "stack": [
            _fx("pixel_sort", 5001, direction="horizontal", interval="luminance", lower=0.45, upper=0.95,
                sort_by="luminance", max_span=260, span_random=0.7),
        ],
    },
    {
        "name": "Tile Riot",
        "description": "Mixed block scrambling at two scales.",
        "stack": [
            _fx("block_scramble", 5101, tile=96, amount=0.15, mode="shuffle"),
            _fx("block_scramble", 5102, tile=24, amount=0.2, mode="mixed", offset_max=0.6),
        ],
    },
]


# Version 1.3: presets built on the optional local-AI effects ("ai": True).  They
# need the AI engine and model weights; generative ones wait for "Generate".
AI_PRESETS: list[dict] = [
    {"name": "Faultline Portrait", "ai": True,
     "description": "The selected object comes loose from its outline in colour-separated bands (click the subject "
                    "with 'Pick on image').",
     "stack": [_fx("semantic_faultlines", 8001, width=56, bands=7, displacement=30, discontinuity=0.7,
                   separation=7, strata=0.55, interior="slip", slip=36, background="dim", background_amount=0.35),
               _fx("chromatic", 8002, strength=4.0, falloff=2.0, cx=0.5, cy=0.5, green=0.0, blur=0.5)]},
    {"name": "Shattered Silhouette", "ai": True,
     "description": "Shattered outline fragments over a desaturated world, the object filled with colour strata.",
     "stack": [_fx("semantic_faultlines", 8101, width=80, bands=9, displacement=55, rotation=9.0,
                   discontinuity=0.95, fragments=40, separation=10, strata=0.8, interior="strata",
                   background="desaturate", background_amount=0.8, gap_fill="colour", gap_colour="#0b0b10")]},
    {"name": "Depth Riot", "ai": True,
     "description": "Inside-out perspective: background slabs punch through the foreground with mirrored gaps.",
     "stack": [_fx("depth_mutiny", 8201, mode="inside_out", slabs=10, amplitude=48, direction=20.0, inversion=0.5,
                   jitter=0.4, collision="inverted", gap_fill="mirror")]},
    {"name": "Accordion Street", "ai": True,
     "description": "Surfaces fold into ribbons along the depth, viewed on an old monitor.",
     "stack": [_fx("depth_mutiny", 8301, mode="accordion", slabs=14, amplitude=44, direction=0.0, inversion=0.3,
                   jitter=0.25, collision="correct", gap_fill="stretch"),
               _fx("vhs_crt", 8302, chroma_blur=2, chroma_shift=1, saturation=1.05, sharpen=0.2, tracking=0.1,
                   tracking_bands=1, noise=0.04, streaks=1, scanlines=0.25, scan_period=3, mask="none",
                   mask_strength=0.0, bloom=0.3, vignette=0.35, curvature=0.0, border=False)]},
    {"name": "Contagion Bloom", "ai": True,
     "description": "Look-alike patches infect each other over three generations.",
     "stack": [_fx("feature_contagion", 8401, mode="similar", patch=56, distance=0.3, similarity=0.02,
                   density=0.18, generations=3, feather=0.4, colour_match=0.25)]},
    {"name": "Dissimilar Invasion", "ai": True,
     "description": "The least similar regions invade each other in hard-edged patches.",
     "stack": [_fx("feature_contagion", 8501, mode="dissimilar", patch=48, distance=0.2, similarity=0.05,
                   density=0.35, generations=1, feather=0.1, colour_match=0.0)]},
    {"name": "Perception Static", "ai": True,
     "description": "Where the model's views disagree, the image tears, splits and repeats.",
     "stack": [_fx("perception_breakdown", 8601, perturbation=0.6, threshold=0.45, contrast=0.6, target="high",
                   glitch="all", displacement=36, softness=0.3)]},
    {"name": "Latent Rot", "ai": True,
     "description": "Damaged VAE latents grow strange material transitions.",
     "stack": [_fx("latent_bending", 8701, mutation="mixed", tile=4, channels="all", density=0.35, amplitude=0.6,
                   composite="residual")]},
    {"name": "Copper Scars", "ai": True,
     "description": "Cracks reveal oxidised copper circuitry in porcelain (click Generate).",
     "stack": [_fx("hallucination_scars", 8801, shape="cracks", density=0.45, width=14, branching=0.6,
                   edge_style="burn", edge_width=2)]},
    {"name": "Chrome Scanline Wounds", "ai": True,
     "description": "Broken scanlines open onto black liquid chrome with luminous fibres (click Generate).",
     "stack": [_fx("hallucination_scars", 8901, prompt="black liquid chrome with tangled luminous fibres",
                   shape="scanlines", density=0.5, width=12, edge_style="glow", edge_width=2, bleed=0.25)]},
    {"name": "Ceramic vs Metal", "ai": True,
     "description": "Fragments of a frozen-ceramic and a molten-metal version haunt the photo (click Generate).",
     "stack": [_fx("counterfactual_residue", 9001, prompt_a="frozen ceramic", prompt_b="molten organic metal",
                   strength=0.55, threshold=0.35, mode="threshold", fragment=48)]},
    {"name": "Fading Memory", "ai": True,
     "description": "Four low-strength rememberings combined through contour bands (click Generate).",
     "stack": [_fx("recursive_misremembering", 9101, generations=4, strength=0.3, anchor=0.15,
                   combine="contour", memory="accumulated")]},
]
BUILTIN_PRESETS.extend(AI_PRESETS)


def is_ai_preset(preset: dict) -> bool:
    return bool(preset.get("ai")) or any(getattr(REGISTRY.get(fx.get("effect")), "ai_models", ())
                                         for fx in preset.get("stack", []))


def available_builtin_presets() -> list[dict]:
    """Built-in presets whose effects are all registered."""
    return [p for p in BUILTIN_PRESETS if all(fx["effect"] in REGISTRY for fx in p["stack"])]


def preset_stack(preset: dict) -> list[EffectInstance]:
    stack = []
    for fx in preset["stack"]:
        d = dict(fx)
        d["uid"] = new_uid()
        stack.append(EffectInstance.from_dict(d))
    return stack


def apply_preset(doc: Document, preset: dict, *, mode: str = "replace") -> Document:
    """Return a new document with the preset applied to the active layer."""
    new = doc.copy()
    stack = preset_stack(preset)
    layer = new.layer
    if mode == "append":
        layer.stack.extend(stack)
    else:
        layer.stack = stack
    if "layer" in preset:
        layer.blend = preset["layer"].get("blend", layer.blend)
        layer.opacity = float(preset["layer"].get("opacity", layer.opacity))
    new.preset_name = preset["name"]
    return new


def stack_to_preset(name: str, stack: list[EffectInstance], description: str = "") -> dict:
    return {
        "format": PRESET_FORMAT,
        "version": PRESET_VERSION,
        "name": name,
        "description": description,
        "stack": [{k: v for k, v in e.to_dict().items() if k != "uid"} for e in stack],
    }


def validate_preset(data: dict) -> dict:
    if not isinstance(data, dict) or "stack" not in data or "name" not in data:
        raise ProjectError("Not a Glitch Lab preset file.")
    if int(data.get("version", 1)) > PRESET_VERSION:
        raise ProjectError("This preset was saved by a newer version of Glitch Lab.")
    for fx in data["stack"]:
        if fx.get("effect") not in REGISTRY:
            raise ProjectError(f"Preset uses an unknown effect '{fx.get('effect')}'.")
    preset_stack(data)  # raises on bad content
    return data


def user_preset_dir() -> Path:
    from .settings import config_dir

    d = config_dir() / "presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9 _-]+", "", name).strip() or "preset"
    return s[:60]


def save_user_preset(preset: dict, folder: Path | None = None) -> Path:
    folder = folder or user_preset_dir()
    path = Path(folder) / f"{_safe_name(preset['name'])}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(preset, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_user_presets(folder: Path | None = None) -> list[dict]:
    folder = folder or user_preset_dir()
    out = []
    for p in sorted(Path(folder).glob("*.json")):
        try:
            out.append(validate_preset(json.loads(p.read_text(encoding="utf-8"))))
        except Exception as exc:
            log.warning("Skipping preset %s: %s", p, exc)
    return out


def delete_user_preset(name: str, folder: Path | None = None) -> None:
    folder = folder or user_preset_dir()
    p = Path(folder) / f"{_safe_name(name)}.json"
    if p.exists():
        p.unlink()
