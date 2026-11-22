"""AI effects with the real models.

Skipped unless PyTorch/transformers/diffusers are installed and the model
weights are present in ``GLITCHLAB_TEST_AI_DIR`` (an AI data folder as
created by ``glitchlab-cli --ai-download``).  Stable Diffusion tests also
need ``GLITCHLAB_TEST_AI_SLOW=1``.
"""

import os

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from glitchlab.ai import generate, jobs, models  # noqa: E402
from glitchlab.engine import fixtures  # noqa: E402
from glitchlab.engine.model import Document, EffectInstance  # noqa: E402
from glitchlab.engine.pipeline import render_document, render_full  # noqa: E402

AI_DIR = os.environ.get("GLITCHLAB_TEST_AI_DIR", "")
SLOW = os.environ.get("GLITCHLAB_TEST_AI_SLOW") == "1"


@pytest.fixture(autouse=True)
def ai_dir(monkeypatch):
    if not AI_DIR:
        pytest.skip("GLITCHLAB_TEST_AI_DIR not set")
    monkeypatch.setenv("GLITCHLAB_AI_DIR", AI_DIR)
    jobs.manager().memory.clear()


def need(*ids):
    miss = models.missing(ids)
    if miss:
        pytest.skip(f"models not installed: {miss}")


@pytest.fixture
def scene():
    return fixtures.photo_like(256, 192)


def run(img, eid, seed=3, **kw):
    return render_full(img, Document.single([EffectInstance.create(eid, seed=seed, **kw)], 1)).image


VISION = {"semantic_faultlines": ("sam2", {"objects": [[0.5, 0.5, 1, 1]]}), "depth_mutiny": ("depth", {}),
          "feature_contagion": ("dinov2", {"density": 0.5}), "perception_breakdown": ("dinov2", {})}


@pytest.mark.parametrize("eid", sorted(VISION))
def test_vision_effects_run_deterministically(scene, eid):
    mid, kw = VISION[eid]
    need(mid)
    a = run(scene, eid, **kw)
    b = run(scene, eid, **kw)
    assert np.array_equal(a, b)
    assert (a != scene).any()
    np.testing.assert_array_equal(a[..., 3], scene[..., 3])


@pytest.mark.parametrize("eid", sorted(VISION))
def test_preview_reuses_the_full_resolution_analysis(scene, eid, monkeypatch):
    mid, kw = VISION[eid]
    need(mid)
    doc = Document.single([EffectInstance.create(eid, seed=3, **kw)], 1)
    full = render_full(scene, doc).image
    calls = []
    real = jobs.JobManager.submit

    def counting(self, key, label, fn, **k):
        calls.append(label)
        return real(self, key, label, fn, **k)

    monkeypatch.setattr(jobs.JobManager, "submit", counting)
    small = scene[::2, ::2].copy()
    prev = render_document(small, doc, scale=0.5, full_size=(256, 192), full_source=scene).image
    assert calls == []  # no new inference for the preview
    import cv2

    ref = cv2.resize(full, (128, 96), interpolation=cv2.INTER_AREA).astype(int)
    assert np.abs(ref - prev.astype(int)).mean() < 40


def test_styling_changes_do_not_rerun_inference(scene, monkeypatch):
    need("depth")
    run(scene, "depth_mutiny")
    calls = []
    real = jobs.JobManager.submit
    monkeypatch.setattr(jobs.JobManager, "submit", lambda self, *a, **k: (calls.append(a[1]), real(self, *a, **k))[1])
    for kw in ({"amplitude": 80}, {"mode": "avalanche"}, {"slabs": 20, "collision": "inverted"}):
        run(scene, "depth_mutiny", **kw)
    assert calls == []


def test_object_points_change_the_mask(scene):
    need("sam2")
    a = run(scene, "semantic_faultlines", objects=[[0.5, 0.5, 1, 1]], view="mask")
    b = run(scene, "semantic_faultlines", objects=[[0.1, 0.1, 1, 1]], view="mask")
    assert not np.array_equal(a, b)


def test_feature_disagreement_map_is_finite(scene):
    need("dinov2")
    out = run(scene, "perception_breakdown", show_map="map")
    assert out[..., :3].std() > 5


def test_latent_baseline_reconstructs_and_zero_density_is_identity(scene):
    need("vae")
    base = run(scene, "latent_bending", view="baseline", resolution=256)
    mse = np.mean((base[..., :3].astype(float) - scene[..., :3].astype(float)) ** 2)
    assert 10 * np.log10(255 ** 2 / mse) > 20  # PSNR of the VAE round trip
    same = run(scene, "latent_bending", density=0.0, resolution=256)
    assert np.array_equal(same, scene)
    bent = run(scene, "latent_bending", density=0.6, amplitude=0.8, resolution=256)
    assert np.abs(bent.astype(int) - scene.astype(int)).mean() > 2


@pytest.mark.skipif(not SLOW, reason="set GLITCHLAB_TEST_AI_SLOW=1 for Stable Diffusion tests")
def test_scars_change_only_the_scars(scene):
    need("sd15_inpaint")
    kw = {"steps": 2, "resolution": 256, "width": 6, "density": 0.5}
    mask = run(scene, "hallucination_scars", view="mask", **kw)
    out = run(scene, "hallucination_scars", **kw)
    scar = (mask[..., :3] != scene[..., :3]).any(-1)
    assert (out[~scar] == scene[~scar]).all()  # outside the scars nothing changes
    assert (out[scar] != scene[scar]).any()


@pytest.mark.skipif(not SLOW, reason="set GLITCHLAB_TEST_AI_SLOW=1 for Stable Diffusion tests")
def test_identical_prompts_leave_no_residue(scene):
    need("sd15")
    out = run(scene, "counterfactual_residue", prompt_a="a photo", prompt_b="a photo", steps=2, resolution=256)
    assert np.array_equal(out, scene)  # same model, size, scheduler and noise -> identical interpretations


@pytest.mark.skipif(not SLOW, reason="set GLITCHLAB_TEST_AI_SLOW=1 for Stable Diffusion tests")
def test_misremembering_resumes_finished_generations(scene, monkeypatch):
    need("sd15")
    kw = {"steps": 3, "strength": 0.5, "resolution": 256}
    run(scene, "recursive_misremembering", generations=2, **kw)
    calls = []
    real = generate._img2img
    monkeypatch.setattr(generate, "_img2img", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    out = run(scene, "recursive_misremembering", generations=3, **kw)
    assert len(calls) == 1  # generations 1 and 2 came from the cache
    assert (out != scene).any()
