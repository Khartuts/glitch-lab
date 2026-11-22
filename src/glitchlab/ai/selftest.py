"""Self-test of the local AI effects with the real models (used by the Windows
build and available as ``glitchlab-cli --ai-selftest``)."""

from __future__ import annotations

import os
import platform
import time
import traceback

import numpy as np

from .. import __version__

#: Fast settings so the test also finishes on a processor.
FAST = {
    "semantic_faultlines": {"objects": [[0.5, 0.55, 1, 1]]},
    "depth_mutiny": {},
    "feature_contagion": {"density": 0.4},
    "perception_breakdown": {"glitch": "all"},
    "latent_bending": {"resolution": 256},
    "hallucination_scars": {"steps": 4, "resolution": 256, "density": 0.6, "width": 6},
    "counterfactual_residue": {"steps": 4, "resolution": 256, "strength": 0.6},
    "recursive_misremembering": {"steps": 6, "generations": 2, "strength": 0.4, "resolution": 256},
}


def run_ai_selftest(image: np.ndarray | None = None, out_dir: str | None = None,
                    effects: list[str] | None = None) -> tuple[bool, str]:
    from ..engine import fixtures
    from ..engine.effects.base import REGISTRY
    from ..engine.model import Document, EffectInstance
    from ..engine.pipeline import image_hash, render_full
    from . import models, runtime

    lines = [f"Glitch Lab {__version__} AI self-test — {platform.platform()}"]
    failures = 0
    st = runtime.status()
    lines.append(f"AI engine: {st.message}")
    if not st.available:
        lines.append("RESULT: FAILED (AI engine not installed)")
        return False, "\n".join(lines)
    torch = runtime.import_torch()
    from .support import device

    lines.append(f"PyTorch {torch.__version__}, device {device()}, threads {torch.get_num_threads()}")
    for mid, spec in models.MODELS.items():
        lines.append(f"  model {mid}: {'installed' if models.is_installed(mid) else 'missing'} "
                     f"({spec.repo} @ {spec.revision[:10]})")
    img = image if image is not None else fixtures.photo_like(320, 240)
    ids = effects or [e for e in FAST if e in REGISTRY]
    for eid in ids:
        eff = REGISTRY[eid]
        if models.missing(eff.ai_models):
            failures += 1
            lines.append(f"FAIL  {eid}: model weights missing ({', '.join(models.missing(eff.ai_models))})")
            continue
        t0 = time.perf_counter()
        try:
            inst = EffectInstance.create(eid, seed=7, **FAST.get(eid, {}))
            res = render_full(img, Document.single([inst], 3))
            out = res.image
            t1 = time.perf_counter() - t0
            assert out.shape == img.shape and out.dtype == np.uint8
            assert np.isfinite(out).all()
            assert (out != img).any(), "no visible change"
            assert (out[..., 3] == img[..., 3]).all(), "alpha not preserved"
            t2 = time.perf_counter()
            again = render_full(img, Document.single([inst], 3)).image
            cached = time.perf_counter() - t2
            same = np.array_equal(out, again)
            assert same, "second render (from the cache) differs"
            lines.append(f"PASS  {eid}  ({t1:.1f}s, cached re-render {cached:.2f}s) {image_hash(out)[:12]}")
            if out_dir:
                from ..engine.image_io import save_png_bytes

                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, f"{eid}.png"), "wb") as f:
                    f.write(save_png_bytes(out))
        except Exception as exc:  # noqa: BLE001 - reported
            failures += 1
            lines.append(f"FAIL  {eid}: {exc!r}")
            lines.append("      " + traceback.format_exc().replace("\n", "\n      "))
    lines.append("RESULT: " + ("PASSED" if failures == 0 else f"FAILED ({failures} failure(s))"))
    return failures == 0, "\n".join(lines)
