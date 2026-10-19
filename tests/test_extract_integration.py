from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pet_immich.data import Features, load_samples
from pet_immich.extract import extract

pytestmark = pytest.mark.skipif("PET_TEST_STATE" not in os.environ, reason="Local model and crop service required")


@pytest.mark.parametrize("workers", [1, 2], ids=["serial", "parallel"])
def test_extraction_matches_saved_features(tmp_path: Path, workers: int) -> None:
    source = Path(os.environ["PET_TEST_STATE"])
    reference = Features.load(source / "features.npz")
    by_id = {sample.id: sample for sample in load_samples(source / "samples.json")}
    samples = [by_id[key] for key in reference.ids[:26]]
    actual = extract(
        samples, tmp_path, os.environ["PET_TAGGER_URL"], workers=workers, models_directory=source / "models"
    )
    indices = {key: index for index, key in enumerate(actual.ids)}
    np.testing.assert_allclose(actual.raw[[indices[sample.id] for sample in samples]], reference.raw[:26], atol=1e-5)
    repeated = extract(
        samples, tmp_path, os.environ["PET_TAGGER_URL"], workers=workers, models_directory=source / "models"
    )
    np.testing.assert_array_equal(actual.face, repeated.face)


def test_model_change_requires_cache_refresh(tmp_path: Path) -> None:
    source = Path(os.environ["PET_TEST_STATE"])
    features = Features.load(source / "features.npz")
    features.signature = "previous-model"
    features.save(tmp_path / "features.npz")
    with pytest.raises(ValueError, match="models changed"):
        extract(
            load_samples(source / "samples.json"),
            tmp_path,
            os.environ["PET_TAGGER_URL"],
            models_directory=source / "models",
        )
