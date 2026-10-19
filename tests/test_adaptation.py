import io
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pet_immich.adaptation import AdaptedMatches

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from pet_immich.data import Features, Sample


@pytest.fixture
def adaptation_state(
    matching_state: tuple[list[Sample], Features], mocker: MockerFixture
) -> tuple[list[Sample], Features]:
    from kornia.feature.lightglue import LightGlue  # ruff: ignore[import-outside-top-level] - native model.

    model = LightGlue(
        features=None,  # ty: ignore[invalid-argument-type] - Kornia explicitly supports uninitialized weights.
        input_dim=128,
        descriptor_dim=32,
        n_layers=1,
        num_heads=4,
        depth_confidence=-1,
        width_confidence=-1,
    )
    mocker.patch("kornia.feature.LightGlue", return_value=model)
    samples, features = matching_state
    return [replace(sample, label="first" if index < 2 else "second") for index, sample in enumerate(samples)], features


@pytest.mark.parametrize("matching_state", [2], indirect=True)
@pytest.mark.parametrize("steps", [0, 1], ids=["frozen", "trained"])
def test_adaptation_resumes_pair_scores(
    adaptation_state: tuple[list[Sample], Features], tmp_path: Path, steps: int
) -> None:
    samples, features = adaptation_state
    channel = AdaptedMatches(samples, features, tmp_path, "http://127.0.0.1", steps=steps)
    scores = channel(np.arange(3, dtype=np.int64))
    assert np.isfinite(scores).sum() == 6
    np.testing.assert_array_equal(channel(np.arange(3, dtype=np.int64)), scores)
    path = next(tmp_path.glob("*.npz"))
    incomplete = scores.copy()
    incomplete[0, 1] = incomplete[1, 0] = np.nan
    stream = io.BytesIO()
    np.savez_compressed(stream, scores=incomplete)
    path.write_bytes(stream.getvalue())
    np.testing.assert_allclose(channel(np.arange(3, dtype=np.int64)), scores, equal_nan=True)
