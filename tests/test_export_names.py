"""Bug 4: saving to an existing name -> overwrite or save as new numbered file."""

import os

import numpy as np
import pytest

from glitchlab.engine import fixtures
from glitchlab.engine.errors import ExportError
from glitchlab.engine.export import (
    SourceInfo,
    batch_export,
    next_free_path,
    plan_batch_paths,
    resolve_batch_paths,
)
from glitchlab.engine.image_io import ExportOptions, load_image
from glitchlab.engine.model import Document, EffectInstance


def touch(p):
    with open(p, "wb") as f:
        f.write(b"x")


def test_next_free_path_numbering(tmp_path):
    p = str(tmp_path / "art.png")
    touch(p)
    assert next_free_path(p) == str(tmp_path / "art_01.png")
    touch(tmp_path / "art_01.png")
    touch(tmp_path / "art_02.png")
    assert next_free_path(p) == str(tmp_path / "art_03.png")
    # a name that already carries a number continues from it
    touch(tmp_path / "pic_07.png")
    assert next_free_path(str(tmp_path / "pic_07.png")) == str(tmp_path / "pic_08.png")


def test_next_free_path_respects_recipe_and_reserved(tmp_path):
    p = str(tmp_path / "a.png")
    touch(p)
    touch(tmp_path / "a_01.json")  # recipe of _01 exists -> _01 is not free when recipes are written
    assert next_free_path(p) == str(tmp_path / "a_01.png")
    assert next_free_path(p, with_recipe=True) == str(tmp_path / "a_02.png")
    reserved = {os.path.normcase(os.path.abspath(str(tmp_path / "a_01.png")))}
    assert next_free_path(p, reserved=reserved) == str(tmp_path / "a_02.png")


def test_resolve_batch_paths_policies(tmp_path):
    paths = [str(tmp_path / f"v{i}.png") for i in (1, 2, 3)]
    touch(paths[0])
    touch(paths[2])
    with pytest.raises(ExportError):
        resolve_batch_paths(paths, "error", with_recipe=False)
    assert resolve_batch_paths(paths, "overwrite", with_recipe=False) == paths
    numbered = resolve_batch_paths(paths, "number", with_recipe=False)
    assert numbered == [str(tmp_path / "v1_01.png"), paths[1], str(tmp_path / "v3_01.png")]


def test_batch_export_twice_numbers_new_files(tmp_path):
    img = fixtures.photo_like(48, 32)
    docs = [Document.single([EffectInstance.create("channel_shift", seed=1, r_dx=3)], 5),
            Document.single([EffectInstance.create("channel_shift", seed=1, r_dx=9)], 6)]
    info = SourceInfo(name="src", width=48, height=32)
    opts = ExportOptions(format="PNG")
    first = batch_export(img, docs, str(tmp_path), "out_{index}", opts, source=info, write_recipe=True)
    before = [load_image(r.path).rgba for r in first]
    with pytest.raises(ExportError):
        batch_export(img, docs, str(tmp_path), "out_{index}", opts, source=info, write_recipe=True)
    second = batch_export(img, docs, str(tmp_path), "out_{index}", opts, source=info, write_recipe=True,
                          on_conflict="number")
    # names that already end in a number continue the sequence (01, 02 exist -> 03, 04)
    assert [os.path.basename(r.path) for r in second] == ["out_03.png", "out_04.png"]
    assert all(os.path.exists(r.recipe_path) for r in second)
    for r, img0 in zip(first, before):  # originals untouched
        assert np.array_equal(load_image(r.path).rgba, img0)
    third = batch_export(img, docs, str(tmp_path), "out_{index}", opts, source=info, write_recipe=True,
                         on_conflict="overwrite")
    assert [r.path for r in third] == [r.path for r in first]


def test_plan_batch_paths_unique_within_batch(tmp_path):
    docs = [Document.single([], 1), Document.single([], 1)]
    paths = plan_batch_paths(docs, str(tmp_path), "same", ExportOptions(), source=SourceInfo(name="x"))
    assert len(set(paths)) == 2
