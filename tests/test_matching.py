import io
import warnings
from typing import TYPE_CHECKING

import httpx
import numpy as np
import pytest
import torch
from PIL import Image

from pet_immich.data import Features, Sample
from pet_immich.matching import match_candidates

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


@pytest.fixture
def matching_state(
    mocker: MockerFixture, client_type: type[httpx.Client], request: pytest.FixtureRequest
) -> tuple[list[Sample], Features]:
    # Kornia imports TorchScript operators even though these tests use eager inference boundaries.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="`torch.jit.script` is not supported in Python 3.14", category=FutureWarning
        )
        from kornia.feature import DISK, LightGlue  # ruff: ignore[import-outside-top-level] - optional native boundary.
        from kornia.feature.disk.structs import DISKFeatures  # ruff: ignore[import-outside-top-level] - native boundary.

    count = request.param
    detector = mocker.create_autospec(DISK, instance=True)
    detector.eval.return_value = detector
    detector.state_dict.return_value = {"weight": torch.ones(1)}
    detector.return_value = [DISKFeatures(torch.zeros(count, 2), torch.ones(count, 128), torch.ones(count))]
    mocker.patch("kornia.feature.DISK.from_pretrained", return_value=detector)
    matcher = mocker.create_autospec(LightGlue, instance=True)
    matcher.eval.return_value = matcher
    matcher.state_dict.return_value = {"weight": torch.ones(1)}
    matcher.return_value = {"scores": [torch.tensor([0.8, 0.6])]}
    mocker.patch("kornia.feature.LightGlue", return_value=matcher)
    pixels = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(pixels, format="PNG")
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=pixels.getvalue()))
    mocker.patch("pet_immich.matching.httpx.Client", side_effect=lambda **_: client_type(transport=transport))
    samples = [Sample(f"crop-{index}", f"photo-{index}", (0, 0, 1, 1)) for index in range(3)]
    vectors = np.eye(3, dtype=np.float32)
    features = Features(
        [sample.id for sample in samples],
        vectors,
        vectors,
        vectors,
        np.zeros(3, dtype=np.uint64),
        np.ones(3, dtype=bool),
        "synthetic",
    )
    return samples, features


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
