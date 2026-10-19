from dataclasses import replace

import numpy as np
import pytest

from pet_immich.benchmark import compare
from pet_immich.data import UNKNOWN, Features, Matrix, Sample


def test_compositions_recover_identities_on_shared_events(population: tuple[list[Sample], Features]) -> None:
    samples, features = population
    similarity = features.raw @ features.raw.T
    results = compare(samples, features, {"first": similarity, "second": similarity})
    assert {name: (row["unique_photos"], row["unknown_accepted"], row["passed"]) for name, row in results.items()} == {
        "first": (72, 0, True),
        "second": (72, 0, True),
        "first+second": (72, 0, True),
    }


def test_duplicate_comparison_does_not_read_test_labels(population: tuple[list[Sample], Features]) -> None:
    samples, features = population
    results = compare(samples, features, {"empty": np.zeros((108, 108), dtype=np.float32)}, group_duplicates=True)
    assert {name: row["unique_photos"] for name, row in results.items()} == {"empty": 0, "empty+duplicates": 0}


@pytest.mark.parametrize("missing", [False, True], ids=["uninformative", "unavailable"])
def test_comparison_abstains_without_identity_evidence(
    population: tuple[list[Sample], Features], *, missing: bool
) -> None:
    samples, features = population
    results = compare(samples, features, {"empty": np.full((108, 108), np.nan if missing else 0, dtype=np.float32)})
    assert (results["empty"]["unique_photos"], results["empty"]["passed"]) == (0, False)


def test_comparison_counts_ineligible_crops_in_recall(population: tuple[list[Sample], Features]) -> None:
    samples, features = population
    features.valid[:18] = False
    result = compare(samples, features, {"encoder": features.raw @ features.raw.T})["encoder"]
    assert (result["identities"]["pet-a"]["recall"], result["ineligible_crops"]) == (0.5, 18)


def test_comparison_counts_photos_once(population: tuple[list[Sample], Features]) -> None:
    samples, features = population
    samples = [replace(sample, asset_id=f"photo-{index // 2}") for index, sample in enumerate(samples)]
    result = compare(samples, features, {"encoder": features.raw @ features.raw.T})["encoder"]
    assert result["unique_photos"] == 36


@pytest.mark.parametrize("channels", [{}, {"wrong": np.zeros((1, 1), dtype=np.float32)}], ids=["empty", "misaligned"])
def test_comparison_rejects_misaligned_channels(
    population: tuple[list[Sample], Features], channels: dict[str, Matrix]
) -> None:
    samples, features = population
    with pytest.raises(ValueError, match="square and aligned"):
        compare(samples, features, channels)


@pytest.mark.parametrize("excluded", [UNKNOWN, "pet-a"], ids=["no-unknown", "one-identity"])
def test_comparison_requires_unknown_and_two_identities(
    population: tuple[list[Sample], Features], excluded: str
) -> None:
    samples, features = population
    with pytest.raises(ValueError, match="two identities and confirmed unknown"):
        compare(
            [sample for sample in samples if sample.label != excluded],
            features,
            {"encoder": features.raw @ features.raw.T},
        )
