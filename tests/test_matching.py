from typing import TYPE_CHECKING

import httpx
import numpy as np
import pytest
import torch

from pet_immich.matching import LocalMatcher, match_candidates

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from pet_immich.data import Features, Sample


@pytest.mark.parametrize(
    ("matching_state", "expected"), [(2, 0.7), (0, 0)], indirect=["matching_state"], ids=["matches", "no-keypoints"]
)
def test_matching_caches_pair_scores(
    matching_state: tuple[list[Sample], Features], expected: float, tmp_path: Path
) -> None:
    samples, features = matching_state
    scores = match_candidates(samples, features, tmp_path, "http://127.0.0.1", limit=2)
    np.testing.assert_allclose(scores[np.triu_indices(3, 1)], expected, atol=1e-6)
    repeated = match_candidates(samples, features, tmp_path, "http://127.0.0.1", limit=2)
    np.testing.assert_array_equal(repeated, scores)


@pytest.mark.parametrize("matching_state", [2], indirect=True)
def test_matching_rejects_changed_cache(matching_state: tuple[list[Sample], Features], tmp_path: Path) -> None:
    samples, features = matching_state
    match_candidates(samples, features, tmp_path, "http://127.0.0.1", limit=1)
    with pytest.raises(ValueError, match="cache changed"):
        match_candidates(samples, features, tmp_path, "http://127.0.0.1", limit=2)


@pytest.mark.parametrize("limit", [0, 1], ids=["invalid-limit", "misaligned"])
def test_matching_requires_aligned_inputs(
    population: tuple[list[Sample], Features], tmp_path: Path, limit: int
) -> None:
    samples, features = population
    with pytest.raises(ValueError, match="aligned samples"):
        match_candidates(samples[::-1] if limit else samples, features, tmp_path, "http://127.0.0.1", limit=limit)


@pytest.mark.parametrize("matching_state", [2], indirect=True)
@pytest.mark.parametrize("masked", [True, False], ids=["foreground", "mask-unavailable"])
def test_matching_filters_background_points(
    matching_state: tuple[list[Sample], Features], tmp_path: Path, mocker: MockerFixture, *, masked: bool
) -> None:
    from ultralytics import YOLO  # ruff: ignore[import-outside-top-level] - optional native boundary.
    from ultralytics.engine.results import Results  # ruff: ignore[import-outside-top-level] - native boundary.

    samples, _ = matching_state
    weights = tmp_path / "mask.pt"
    weights.write_bytes(b"synthetic")
    segmenter = mocker.create_autospec(YOLO, instance=True)
    segmenter.predict.return_value = [
        Results(
            np.zeros((64, 64, 3), dtype=np.uint8),
            "synthetic",
            {16: "dog"},
            masks=torch.zeros((1, 64, 64)) if masked else None,
        )
    ]
    mocker.patch("ultralytics.YOLO", return_value=segmenter)
    matcher = LocalMatcher(weights)
    with httpx.Client() as client:
        values = matcher.describe(samples[0], tmp_path, client, "http://127.0.0.1")
    assert values["keypoints"].shape == (1, 0 if masked else 2, 2)


@pytest.mark.parametrize("matching_state", [2], indirect=True)
@pytest.mark.usefixtures("matching_state")
def test_matcher_learns_identity_correspondences() -> None:
    from kornia.feature.lightglue import LightGlue  # ruff: ignore[import-outside-top-level] - native model.

    torch.manual_seed(42)
    matcher = LocalMatcher()
    matcher.matcher = LightGlue(
        features=None,  # ty: ignore[invalid-argument-type] - Kornia explicitly supports uninitialized weights.
        input_dim=128,
        descriptor_dim=32,
        n_layers=1,
        num_heads=4,
        depth_confidence=-1,
        width_confidence=-1,
    )
    descriptors = [
        {
            "keypoints": torch.rand(1, 8, 2) * 64,
            "descriptors": torch.randn(1, 8, 128),
            "image_size": torch.tensor([[64, 64]]),
        }
        for _ in range(5)
    ]
    losses = matcher.fit(
        descriptors,
        np.asarray(["first", "first", "second", "second", "__unknown__"]),
        np.arange(5, dtype=np.int32),
        np.eye(5, dtype=np.float32),
        steps=20,
    )
    assert np.mean(losses[-5:]) < np.mean(losses[:5])


@pytest.mark.parametrize("matching_state", [0, 2], indirect=True, ids=["no-keypoints", "same-event"])
def test_matcher_rejects_missing_training_pairs(matching_state: tuple[list[Sample], Features], tmp_path: Path) -> None:
    samples, features = matching_state
    matcher = LocalMatcher()
    with httpx.Client() as client:
        descriptors = [matcher.describe(sample, tmp_path, client, "http://127.0.0.1") for sample in samples]
    with pytest.raises(ValueError, match="cross-event positives"):
        matcher.fit(
            descriptors,
            np.asarray(["first", "first", "second"]),
            np.arange(3, dtype=np.int32) if not descriptors[0]["keypoints"].shape[1] else np.zeros(3, dtype=np.int32),
            features.raw,
        )
