from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from pet_immich.data import Features, Sample
from pet_immich.fusion import train_fusion

if TYPE_CHECKING:
    from pet_immich.fusion import FusionModel


@pytest.fixture
def fusion_model(population: tuple[list[Sample], Features]) -> FusionModel:
    samples, features = population
    return train_fusion(samples, features, features)[0]


def test_fusion_predicts_with_reordered_large_cache(
    population: tuple[list[Sample], Features], fusion_model: FusionModel
) -> None:
    samples, features = population
    reordered = Features(
        features.ids[::-1],
        features.raw[::-1],
        features.body[::-1],
        features.face[::-1],
        features.hashes[::-1],
        features.valid[::-1],
        features.signature,
    )
    assert fusion_model.predict(samples, features, reordered).tolist() == ["pet-a"] * 36 + ["pet-b"] * 36 + [""] * 36


def test_fusion_reports_held_out_errors(population: tuple[list[Sample], Features]) -> None:
    samples, features = population
    _, report = train_fusion(samples, features, features)
    assert (report["passed"], report["unique_photos"], report["errors"]) == (True, 72, [])


def test_fusion_rejects_missing_features(population: tuple[list[Sample], Features], fusion_model: FusionModel) -> None:
    samples, features = population
    with pytest.raises(ValueError, match="missing 108 crops"):
        fusion_model.predict(samples, features, replace(features, ids=[]))


def test_fusion_rejects_changed_encoder(population: tuple[list[Sample], Features], fusion_model: FusionModel) -> None:
    samples, features = population
    with pytest.raises(ValueError, match="extractors differ"):
        fusion_model.predict(samples, features, replace(features, signature="changed"))
