import shutil
from concurrent.futures import Future, ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from pet_immich.data import Features, Sample
from pet_immich.extract import extract

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def test_extraction_caches_normalized_views(extraction_state: Path) -> None:
    samples = [Sample("crop", "asset", (0, 0, 1, 1))]
    actual = extract(samples, extraction_state, "http://test", workers=1)
    assert (actual.ids, actual.valid.tolist()) == (["crop"], [True])
    np.testing.assert_allclose(np.linalg.norm(actual.raw, axis=1), [1], atol=1e-6)
    repeated = extract(samples, extraction_state, "http://test", workers=1)
    np.testing.assert_array_equal(actual.face, repeated.face)


def test_extraction_resumes_without_repeating_cached_crops(extraction_state: Path) -> None:
    Features(
        ["old"],
        np.ones((1, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.zeros(1, dtype=np.uint64),
        np.ones(1, dtype=bool),
        "previous-model",
    ).save(extraction_state / "features.npz")
    with pytest.raises(ValueError, match="models changed"):
        extract([Sample("crop", "asset", (0, 0, 1, 1))], extraction_state, "http://test")


def test_refresh_replaces_incompatible_features(extraction_state: Path) -> None:
    (extraction_state / "features.npz").write_bytes(b"obsolete-cache")
    actual = extract([Sample("crop", "asset", (0, 0, 1, 1))], extraction_state, "http://test", refresh=True)
    assert actual.ids == ["crop"]


def test_extraction_supports_miewid_weights(extraction_state: Path, miewid_weights: Path) -> None:
    (extraction_state / "models" / "encoder.onnx").unlink()
    shutil.copyfile(miewid_weights, extraction_state / "models" / "encoder.safetensors")
    actual = extract([Sample("crop", "asset", (0, 0, 1, 1))], extraction_state, "http://test", workers=1)
    assert actual.raw.shape == (1, 2152)
    np.testing.assert_allclose(np.linalg.norm(actual.raw, axis=1), [1], atol=1e-6)


@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "resume"])
def test_parallel_extraction_merges_unique_crops(
    extraction_state: Path,
    mocker: MockerFixture,
    *,
    resume: bool,
) -> None:
    samples = [Sample(f"crop-{index}", f"asset-{index}", (0, 0, 1, 1)) for index in range(30)]
    baseline = extract(samples[:1], extraction_state, "http://test", workers=1)
    if not resume:
        (extraction_state / "features.npz").unlink()
    futures = []
    for index in range(2):
        selected = samples[index::2]
        result = Features(
            [sample.id for sample in selected],
            np.repeat(baseline.raw, len(selected), axis=0),
            np.repeat(baseline.body, len(selected), axis=0),
            np.repeat(baseline.face, len(selected), axis=0),
            np.repeat(baseline.hashes, len(selected)),
            np.repeat(baseline.valid, len(selected)),
            baseline.signature,
        )
        future = Future()
        future.set_result(result)
        futures.append(future)
    executor = mocker.create_autospec(ProcessPoolExecutor, instance=True)
    executor.__enter__.return_value = executor
    executor.submit.side_effect = futures
    mocker.patch("pet_immich.extract.ProcessPoolExecutor", return_value=executor)
    actual = extract(samples, extraction_state, "http://test", workers=2)
    assert sorted(actual.ids) == sorted(sample.id for sample in samples)
    np.testing.assert_allclose(actual.raw, np.repeat(baseline.raw, 30, axis=0))


@pytest.mark.parametrize(
    ("count", "eligible"),
    [pytest.param(0, True, id="missed-segmentation"), pytest.param(2, False, id="multiple-animals")],
)
def test_extraction_rejects_ambiguity_without_requiring_segmentation(
    extraction_state: Path, mocker: MockerFixture, count: int, *, eligible: bool
) -> None:
    segmenter = mocker.create_autospec(YOLO, instance=True)
    segmenter.predict.return_value = [
        Results(
            np.zeros((16, 16, 3), dtype=np.uint8),
            "synthetic",
            {16: "dog"},
            masks=torch.ones(count, 16, 16) if count else None,
        )
    ]
    mocker.patch("ultralytics.YOLO", return_value=segmenter)
    actual = extract([Sample("crop", "asset", (0, 0, 1, 1))], extraction_state, "http://test", workers=1)
    assert actual.valid.tolist() == [eligible]
