"""Local-AI infrastructure that does not need PyTorch: jobs and caches,
downloads and the AI-engine installer, model registry, pipeline hooks, and
the NumPy parts of the AI effects."""

import hashlib
import http.server
import io
import json
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

from glitchlab.ai import download, jobs, models, runtime
from glitchlab.ai.errors import AIUnavailable, ModelMissing
from glitchlab.engine import fixtures
from glitchlab.engine.cancel import CancelToken
from glitchlab.engine.effects.base import REGISTRY, Effect
from glitchlab.engine.errors import Cancelled
from glitchlab.engine.model import Document, EffectInstance
from glitchlab.engine.params import MASK, POINTS, Param
from glitchlab.engine.pipeline import render_document, render_full

AI_EFFECTS = sorted(e for e in REGISTRY if REGISTRY[e].ai_models)


# ------------------------------------------------------------------ registry
def test_eight_ai_effects_registered():
    assert AI_EFFECTS == sorted(["semantic_faultlines", "depth_mutiny", "feature_contagion", "perception_breakdown",
                                 "latent_bending", "hallucination_scars", "counterfactual_residue",
                                 "recursive_misremembering"])
    for eid in AI_EFFECTS:
        eff = REGISTRY[eid]
        assert eff.category == "AI Experimental"
        assert all(m in models.MODELS for m in eff.ai_models)
        EffectInstance.create(eid)  # defaults are valid


def test_models_are_pinned_with_checksums():
    for spec in models.MODELS.values():
        assert len(spec.revision) == 40
        assert spec.files and spec.size > 1_000_000
        for f in spec.files:
            assert f["size"] > 0 and (f["sha256"] or f["git_sha1"])
        assert spec.licence


def test_runtime_manifest_is_pinned():
    m = json.loads((Path(runtime.__file__).with_name("runtime_win_amd64.json")).read_text())
    names = {w["name"] for w in m["wheels"]}
    assert {"torch", "transformers", "diffusers", "safetensors"} <= names
    assert not names & {"numpy", "pillow"}  # bundled in the program
    for w in m["wheels"]:
        assert len(w["sha256"]) == 64 and w["url"].startswith("https://files.pythonhosted.org/")
        assert w["file"].endswith(".whl") and ("win_amd64" in w["file"] or "none-any" in w["file"])


def test_heavy_effects_are_the_generative_ones():
    assert {e for e in AI_EFFECTS if REGISTRY[e].heavy} == {"hallucination_scars", "counterfactual_residue",
                                                            "recursive_misremembering"}


# ------------------------------------------------------------------ params
def test_points_param_coercion():
    p = Param("o", "O", POINTS, [])
    assert p.coerce(None) == []
    v = p.coerce([[0.2, 1.7, 1, 2], [-1, 0.5, 0], ["x", 1], [0.5, 0.5, 1, 99]])
    assert v == [[0.2, 1.0, 1, 2], [0.0, 0.5, 0, 1], [0.5, 0.5, 1, 16]]
    assert len(p.coerce([[0.1, 0.1, 1, 1]] * 500)) == 64


def test_mask_param_coercion_roundtrip():
    p = Param("m", "M", MASK, {})
    v = p.coerce({"enabled": True, "shapes": [{"type": "rect", "x0": 0.1, "y0": 0.1, "x1": 0.5, "y1": 0.5}],
                  "bogus": 1})
    assert v["enabled"] and v["shapes"][0]["mode"] == "add" and "bogus" not in v
    assert p.coerce("nonsense")["shapes"] == []


# ------------------------------------------------------------------ jobs
def test_job_runs_once_and_is_cached():
    mgr = jobs.JobManager()
    calls = []

    def fn(job):
        calls.append(1)
        job.progress(0.5, "half")
        return {"x": np.arange(3)}

    a = mgr.run("k1", "test", fn)
    b = mgr.run("k1", "test", fn)
    assert np.array_equal(a["x"], b["x"]) and len(calls) == 1


def test_cancelled_waiter_does_not_cancel_the_job():
    mgr = jobs.JobManager()
    gate = threading.Event()

    def fn(job):
        gate.wait(5)
        return {"v": np.int32(7)}

    tok = CancelToken()
    threading.Timer(0.1, tok.cancel).start()
    with pytest.raises(Cancelled):
        mgr.run("k2", "slow", fn, owner="fx", cancel=tok)
    gate.set()
    assert int(mgr.run("k2", "slow", fn, owner="fx")["v"]) == 7  # the same job finished


def test_newer_request_supersedes_abandoned_job():
    mgr = jobs.JobManager()
    started = threading.Event()
    release = threading.Event()

    def slow(job):
        started.set()
        while True:
            job.check()
            if release.wait(0.02):
                return 1

    old = mgr.submit("old", "old", slow, owner="fx")
    started.wait(2)
    new = mgr.submit("new", "new", lambda job: 2, owner="fx")
    assert old.cancel.cancelled
    time.sleep(0.15)  # the old job notices the cancellation at its next check
    release.set()
    assert mgr.wait(new) == 2
    with pytest.raises(Cancelled):
        mgr.wait(old)


def test_cancel_all_and_errors_propagate():
    mgr = jobs.JobManager()

    def boom(job):
        raise ValueError("bad")

    with pytest.raises(ValueError):
        mgr.run("k3", "boom", boom)
    ev = threading.Event()
    j = mgr.submit("k4", "wait", lambda job: (ev.wait(2), job.check()))
    time.sleep(0.05)
    assert mgr.cancel_all() >= 1
    ev.set()
    with pytest.raises(Cancelled):
        mgr.wait(j)


def test_approvals_are_single_use():
    jobs.approve("fx1")
    assert jobs.take_approval("fx1")
    assert not jobs.take_approval("fx1")


def test_disk_cache_roundtrip_and_trim(tmp_path):
    dc = jobs.DiskCache(tmp_path, max_bytes=25_000)
    for i in range(5):
        dc.put(f"{i:02d}" + "a" * 30, {"img": np.full((100, 100), i, np.uint8)})
        time.sleep(0.01)
    assert dc.size() <= 25_000
    assert dc.get("04" + "a" * 30)["img"][0, 0] == 4
    assert dc.get("00" + "a" * 30) is None


# --------------------------------------------------------------- downloads
class _Server:
    def __init__(self, files: dict[str, bytes]):
        files_ = files

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                data = files_.get(self.path)
                if data is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def no_proxy(monkeypatch):
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


def test_download_verifies_checksum(tmp_path, no_proxy):
    data = b"glitch" * 1000
    srv = _Server({"/f.bin": data})
    try:
        seen = []
        download.download(srv.url + "/f.bin", tmp_path / "f.bin", size=len(data),
                          sha256=hashlib.sha256(data).hexdigest(), progress=lambda d, t, l: seen.append(d))
        assert (tmp_path / "f.bin").read_bytes() == data and seen[-1] == len(data)
        with pytest.raises(download.DownloadError, match="damaged"):
            download.download(srv.url + "/f.bin", tmp_path / "g.bin", size=len(data), sha256="0" * 64)
        assert not (tmp_path / "g.bin").exists()
        tok = CancelToken()
        tok.cancel()
        with pytest.raises(Cancelled):
            download.download(srv.url + "/f.bin", tmp_path / "h.bin", size=len(data), cancel=tok)
    finally:
        srv.close()


def test_git_blob_sha1_matches_git():
    assert download.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def _wheel(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, d in files.items():
            z.writestr(n, d)
    return buf.getvalue()


def test_engine_install_from_manifest(tmp_path, monkeypatch, no_proxy):
    w1 = _wheel({"fakepkg/__init__.py": b"VALUE = 42\n", "fakepkg-1.0.dist-info/METADATA": b"Name: fakepkg\n",
                 "torch/include/huge.h": b"x", "fake2-1.0.data/purelib/fake2.py": b"Y = 1\n",
                 "fake2-1.0.data/scripts/tool": b"#!"})
    srv = _Server({"/fakepkg.whl": w1})
    manifest = {"id": "cpu-test-1", "python": "3.13", "platform": "win_amd64",
                "wheels": [{"name": "fakepkg", "version": "1.0", "file": "fakepkg-1.0-py3-none-any.whl",
                            "url": srv.url + "/fakepkg.whl", "size": len(w1),
                            "sha256": hashlib.sha256(w1).hexdigest()}]}
    monkeypatch.setattr(runtime, "load_manifest", lambda: manifest)
    monkeypatch.setenv("GLITCHLAB_AI_DIR", str(tmp_path / "ai"))
    try:
        root = runtime.install()
    finally:
        srv.close()
    site = root / "site-packages"
    assert (site / "fakepkg" / "__init__.py").exists()
    assert (site / "fake2.py").exists()
    assert not (site / "torch" / "include").exists()
    assert not (site / "fake2-1.0.data").exists()
    assert runtime.installed() and runtime.activate()
    import sys

    assert str(site) in sys.path
    sys.path.remove(str(site))
    runtime._activated = None


def test_wheel_with_unsafe_path_is_refused(tmp_path):
    p = tmp_path / "bad.whl"
    p.write_bytes(_wheel({"../evil.py": b"x"}))
    with pytest.raises(AIUnavailable):
        runtime._extract_wheel(p, tmp_path / "site")


def test_model_install_and_sharing(tmp_path, monkeypatch, no_proxy):
    a, b = b"weights-A" * 50, b"config"
    srv = _Server({"/repoA/a.safetensors": a, "/repoA/config.json": b, "/repoB/a.safetensors": a,
                   "/repoB/config.json": b})
    monkeypatch.setenv("GLITCHLAB_AI_DIR", str(tmp_path / "ai"))

    def spec(mid, repo):
        files = ({"path": "a.safetensors", "size": len(a), "sha256": hashlib.sha256(a).hexdigest(), "git_sha1": None},
                 {"path": "config.json", "size": len(b), "sha256": None, "git_sha1": download.git_blob_sha1(b)})
        return models.ModelSpec(mid, mid, "test", repo, "f" * 40, files, "MIT", 10)

    monkeypatch.setattr(models, "MODELS", {"ma": spec("ma", "repoA"), "mb": spec("mb", "repoB")})
    monkeypatch.setattr(models.ModelSpec, "file_url", lambda self, path: f"{srv.url}/{self.repo}/{path}")
    try:
        with pytest.raises(ModelMissing):
            models.require("ma")
        models.install(["ma"])
        assert models.is_installed("ma") and not models.is_installed("mb")
    finally:
        srv.close()
    # The identical weight file is shared (hard link or copy) instead of downloaded again:
    # the second server only has the small config file.
    srv2 = _Server({"/repoB/config.json": b})
    monkeypatch.setattr(models.ModelSpec, "file_url", lambda self, path: f"{srv2.url}/{self.repo}/{path}")
    monkeypatch.setattr(download, "BACKOFF", 0.0)
    try:
        models.install(["mb"])
    finally:
        srv2.close()
    assert models.is_installed("mb")
    assert (models.model_dir("mb") / "a.safetensors").read_bytes() == a
    models.delete("ma")
    assert not models.is_installed("ma") and models.is_installed("mb")


def test_model_missing_message_names_size_and_licence():
    msg = ModelMissing(["depth"]).message
    assert "Depth Anything V2 Small" in msg and "MB" in msg and "Apache-2.0" in msg and "AI › Models" in msg


# --------------------------------------------------------- pipeline hooks
class _Probe(Effect):
    id = "_ai_probe"
    name = "Probe"
    category = "Simulation"
    seen: list = []

    def process(self, img, p, ctx):
        full = ctx.full_input()
        _Probe.seen.append((ctx.input_key, full.shape, img.shape, ctx.interactive))
        return img.copy()


@pytest.fixture
def probe():
    REGISTRY[_Probe.id] = _Probe()
    _Probe.seen.clear()
    yield _Probe
    REGISTRY.pop(_Probe.id, None)


def test_input_key_is_scale_independent_and_full_input_is_full_res(probe):
    full = fixtures.photo_like(320, 240)
    small = full[::2, ::2].copy()
    pre = EffectInstance.create("channel_shift", seed=1)
    doc = Document.single([pre, EffectInstance.create(probe.id, seed=2)], 3)
    render_full(full, doc)
    render_document(small, doc, scale=0.5, full_size=(320, 240), full_source=full, on_error="skip")
    (k1, f1, i1, inter1), (k2, f2, i2, inter2) = probe.seen
    assert k1 == k2
    assert f1 == f2 == (240, 320, 4) and i2 == (120, 160, 4)
    assert not inter1 and inter2


def test_input_key_depends_on_the_prefix_and_image(probe):
    full = fixtures.photo_like(160, 120)
    other = fixtures.transparent_circle(160, 120)
    for img, seed in ((full, 1), (full, 2), (other, 1)):
        render_full(img, Document.single([EffectInstance.create("channel_shift", seed=seed),
                                          EffectInstance.create(probe.id, seed=9)], 3))
    keys = [s[0] for s in probe.seen]
    assert len(set(keys)) == 3


@pytest.mark.parametrize("eid", AI_EFFECTS)
def test_ai_effects_report_missing_engine_clearly(eid, monkeypatch, tmp_path):
    from glitchlab.ai import support

    monkeypatch.setattr(runtime, "engine_available", lambda: False)
    monkeypatch.setattr(support.runtime, "engine_available", lambda: False)
    doc = Document.single([EffectInstance.create(eid, seed=1)], 1)
    img = fixtures.photo_like(64, 48)
    with pytest.raises(AIUnavailable):
        render_full(img, doc)
    res = render_document(img, doc, scale=1.0, on_error="skip")
    assert res.failures and res.failures[0].code == "AIUnavailable"
    assert np.array_equal(res.image, img)  # nothing is substituted


@pytest.mark.parametrize("eid", AI_EFFECTS)
def test_ai_effects_report_missing_models_clearly(eid, monkeypatch, tmp_path):
    from glitchlab.ai import support

    monkeypatch.setattr(support.runtime, "engine_available", lambda: True)
    monkeypatch.setenv("GLITCHLAB_AI_DIR", str(tmp_path / "empty"))
    doc = Document.single([EffectInstance.create(eid, seed=1)], 1)
    res = render_document(fixtures.photo_like(64, 48), doc, scale=1.0, on_error="skip")
    assert res.failures and res.failures[0].code == "ModelMissing", res.failures
    assert "AI › Models" in res.failures[0].message


# ------------------------------------------------------- NumPy algorithms
def test_splat_identity_and_priority():
    from glitchlab.engine.effects.ai_vision import _splat

    img = fixtures.photo_like(40, 30)
    z = np.zeros((30, 40), np.float32)
    out, holes = _splat(img, z, z, z)
    assert np.array_equal(out, img) and not holes.any()
    dx = np.zeros_like(z)
    dx[:, :20] = 5.0  # left half moves right onto the right half
    prio_left = np.where(np.arange(40)[None, :] < 20, 0.9, 0.1).astype(np.float32).repeat(30, 0)
    out, holes = _splat(img, dx, z, prio_left)
    assert np.array_equal(out[:, 24], img[:, 19])  # higher priority wins the collision
    assert holes[:, :5].all()
    out2, _ = _splat(img, dx, z, 1 - prio_left)
    assert np.array_equal(out2[:, 24], img[:, 24])


def test_directional_fill_stretches_along_direction():
    from glitchlab.engine.effects.ai_vision import _directional_fill

    img = np.zeros((10, 10, 4), np.uint8)
    img[:, :3] = 200
    holes = np.zeros((10, 10), bool)
    holes[:, 3:6] = True
    out = _directional_fill(img, holes, -1.0, 0.0, 8)  # look left for the filler
    assert (out[:, 3:6, 0] == 200).all()


def test_choose_donors_respects_distance_and_similarity():
    from glitchlab.engine.effects.ai_vision import _choose_donors

    rng = np.random.default_rng(0)
    gw, gh = 10, 8
    F = rng.normal(size=(gw * gh, 16)).astype(np.float32)
    F[5] = F[70] + 0.001  # patch 5 and 70 are near-identical and far apart
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    d = _choose_donors(F, F, np.array([5]), gw, gh, 3.0, 1 / 80, False, rng)
    assert d[0] == 70
    d = _choose_donors(F, F, np.arange(80), gw, gh, 2.5, 0.05, False, rng)
    di, dj = np.arange(80) // gw, np.arange(80) % gw
    si, sj = d // gw, d % gw
    assert (np.hypot(si - di, sj - dj)[d >= 0] >= 2.5).all()
    d2 = _choose_donors(F, F, np.array([5]), gw, gh, 1.5, 1 / 80, True, rng)
    assert d2[0] != 70


def test_align_grid_flip_and_scale():
    from glitchlab.ai.analysis import align_grid

    g = np.arange(6 * 8, dtype=np.float32).reshape(6, 8, 1).repeat(4, -1)
    back, valid = align_grid(g[:, ::-1].copy(), "flip")
    assert np.array_equal(back, g) and valid.all()
    same, valid = align_grid(g, "scale:1.0000")
    assert np.allclose(same, g) and valid[2:-2, 2:-2].all()
    _, valid = align_grid(g, "scale:1.2000")
    assert not valid[0, 0] and valid[3, 4]
    _, valid = align_grid(g, "scale:0.8000")
    assert not valid[0, 0]


def test_view_transforms_are_exact_or_bounded():
    from glitchlab.ai.analysis import transform_view, view_names

    rgb = fixtures.photo_like(64, 48)[..., :3]
    assert np.array_equal(transform_view(transform_view(rgb, "flip"), "flip"), rgb)
    names = view_names(0.5)
    assert names[0] == "flip" and len(names) == 5
    for v in names:
        assert transform_view(rgb, v).shape == rgb.shape


def test_group_points_by_object():
    from glitchlab.ai.analysis import group_points

    g = group_points([[0.1, 0.1, 1, 1], [0.2, 0.2, 0, 1], [0.5, 0.5, 1, 2], [0.9, 0.9, 0, 3]])
    assert [k for k, _, _ in g] == [1, 2]  # object 3 has no positive point
    assert g[0][2] == [1, 0]


def test_mutate_latent_bounded_and_regional():
    from glitchlab.engine.effects.ai_generative import mutate_latent

    rng = np.random.default_rng(1)
    z = rng.normal(size=(4, 32, 40)).astype(np.float32)
    for kind in ("tile_swap", "offset", "permute", "amplify", "quantise", "noise", "mixed"):
        m = mutate_latent(z, kind, 4, [0, 1, 2, 3], 0.6, 1.0, np.random.default_rng(2), None)
        assert m.shape == z.shape and np.isfinite(m).all() and np.abs(m).max() <= 8 * z.std() + 1e-5
        assert not np.array_equal(m, z), kind
    region = np.zeros((32, 40), np.float32)
    region[:, :20] = 1
    m = mutate_latent(z, "noise", 4, [0, 1], 1.0, 1.0, np.random.default_rng(3), region)
    assert np.array_equal(m[:, :, 20:], z[:, :, 20:])
    assert np.array_equal(m[2:], z[2:])  # untouched channels


def test_parse_channels_and_selection():
    from glitchlab.engine.effects.ai_generative import parse_channels, parse_selection

    assert parse_channels("all", 4) == [0, 1, 2, 3]
    assert parse_channels("0, 2-3", 4) == [0, 2, 3]
    with pytest.raises(ValueError, match="0-3"):
        parse_channels("5", 4)
    assert parse_selection("all", "", 4) == [1, 2, 3, 4]
    assert parse_selection("every_other", "", 5) == [2, 4]
    assert parse_selection("first_last", "", 3) == [1, 3]
    assert parse_selection("custom", "3, 9, x, 1", 4) == [1, 3]
    assert parse_selection("custom", "", 4) == [4]


def test_scar_geometry_is_resolution_independent():
    from glitchlab.engine.effects.ai_generative import rasterize_scars, scar_geometry

    for shape in ("cracks", "scanlines", "wounds", "mixed"):
        g1 = scar_geometry(np.random.default_rng(5), shape, 0.5, 0.5, 4 / 3)
        g2 = scar_geometry(np.random.default_rng(5), shape, 0.5, 0.5, 4 / 3)
        assert len(g1["lines"]) == len(g2["lines"])
        big = rasterize_scars(g1, 800, 600, 10)
        small = rasterize_scars(g1, 400, 300, 5)
        assert big.max() > 0.9
        down = big.reshape(300, 2, 400, 2).mean((1, 3))
        assert np.abs(down - small).mean() < 0.03, shape


def test_misremembering_limits_work():
    eff = REGISTRY["recursive_misremembering"]
    p = eff.coerce_params({"generations": 12, "steps": 50, "strength": 0.7})
    assert eff.validate(p) and "400" in eff.validate(p)[0]
    assert "denoising steps" in eff.info(eff.coerce_params({}))


def test_prompt_and_seed_schedules():
    from glitchlab.ai.generate import prompt_for, seed_for

    pr = ["a", "b", "c"]
    assert [prompt_for(pr, "constant", g, 6) for g in range(6)] == ["a"] * 6
    assert [prompt_for(pr, "alternate", g, 6) for g in range(6)] == ["a", "b", "c", "a", "b", "c"]
    assert [prompt_for(pr, "progressive", g, 6) for g in range(6)] == ["a", "a", "b", "b", "c", "c"]
    assert seed_for(10, "fixed", 3) == 10 and seed_for(10, "increment", 3) == 13
    assert seed_for(10, "random", 3) == seed_for(10, "random", 3) != seed_for(10, "random", 4)


def test_generation_waits_for_approval_in_previews(monkeypatch):
    from glitchlab.ai import generate

    class Ctx:
        interactive = True
        owner = "fx-gate"

    with pytest.raises(generate.GenerationPending):
        generate._gate(Ctx(), False, "nokey", False, "Test")
    jobs.approve("fx-gate")
    generate._gate(Ctx(), False, "nokey", False, "Test")  # approved once
    generate._gate(Ctx(), True, "nokey", False, "Test")   # automatic
    Ctx.interactive = False
    generate._gate(Ctx(), False, "nokey", False, "Test")  # exports always generate


def test_project_records_ai_model_revisions(tmp_path):
    from glitchlab.engine.image_io import LoadedImage
    from glitchlab.engine.project import load_project, project_from_image, save_project

    li = LoadedImage(rgba=fixtures.photo_like(64, 48), path=None, format="PNG", original_mode="RGBA")
    doc = Document.single([EffectInstance.create("depth_mutiny", seed=1)], 1)
    proj = project_from_image(li, doc)
    path = tmp_path / "p.glab"
    save_project(str(path), proj)
    loaded = load_project(str(path))
    assert loaded.ai_models["depth"]["revision"] == models.MODELS["depth"].revision
    assert loaded.model_revision_notes() == []
    loaded.ai_models["depth"]["revision"] = "0" * 40
    assert "may differ" in loaded.model_revision_notes()[0]


def test_ai_presets_are_valid():
    from glitchlab.engine.presets import AI_PRESETS, is_ai_preset

    assert len(AI_PRESETS) == 12
    for p in AI_PRESETS:
        assert is_ai_preset(p)
        for fx in p["stack"]:
            names = {q.name for q in REGISTRY[fx["effect"]].params}
            assert set(fx["params"]) <= names, (p["name"], set(fx["params"]) - names)
