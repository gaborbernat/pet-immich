from dataclasses import replace

import numpy as np
import pytest

from pet_immich.data import UNCERTAIN, UNKNOWN, Features, Sample
from pet_immich.grouping import propagate


@pytest.mark.parametrize(
    ("label", "prediction", "expected"),
    [
        pytest.param(None, "dog", ["dog", "dog"], id="accepted-anchor"),
        pytest.param("other", "dog", ["other", "other"], id="correction-wins"),
        pytest.param(UNKNOWN, "dog", ["", ""], id="unknown"),
        pytest.param(UNCERTAIN, "dog", ["", ""], id="uncertain"),
        pytest.param(None, "", ["", ""], id="no-anchor"),
    ],
)
def test_propagation_respects_confirmed_labels(label: str | None, prediction: str, expected: list[str]) -> None:
    samples = [Sample("first", "one", (0, 0, 1, 1), label=label), Sample("second", "two", (0, 0, 1, 1))]
    vectors = np.ones((2, 1), dtype=np.float32)
    features = Features(
        [sample.id for sample in samples],
        vectors,
        vectors,
        vectors,
        np.zeros(2, dtype=np.uint64),
        np.ones(2, dtype=bool),
        "synthetic",
    )
    assert propagate(samples, features, np.asarray([prediction, ""])).tolist() == expected


@pytest.mark.parametrize("failure", ["same-photo", "different-hash", "invalid", "different-appearance", "conflict"])
def test_propagation_rejects_unsupported_matches(failure: str) -> None:
    samples = [Sample(str(index), str(index), (0, 0, 1, 1)) for index in range(3)]
    vectors = np.ones((3, 1), dtype=np.float32)
    features = Features(
        [sample.id for sample in samples],
        vectors,
        vectors,
        vectors,
        np.zeros(3, dtype=np.uint64),
        np.ones(3, dtype=bool),
        "synthetic",
    )
    predicted = np.asarray(["first", "", "first"], dtype="U128")
    if failure == "same-photo":
        samples = [replace(sample, asset_id="one") for sample in samples]
    elif failure == "different-hash":
        features.hashes[1] = 2**64 - 1
    elif failure == "invalid":
        features.valid[1] = False
    elif failure == "different-appearance":
        features.raw[1] = -1
    else:
        predicted[2] = "second"
    assert not propagate(samples, features, predicted)[1]


def test_propagation_rejects_misaligned_inputs(population: tuple[list[Sample], Features]) -> None:
    samples, features = population
    with pytest.raises(ValueError, match="aligned"):
        propagate(samples[::-1], features, np.full(len(samples), ""))


def test_propagation_does_not_chain_through_new_labels() -> None:
    samples = [Sample(str(index), str(index), (0, 0, 1, 1)) for index in range(3)]
    vectors = np.ones((3, 1), dtype=np.float32)
    features = Features(
        [sample.id for sample in samples],
        vectors,
        vectors,
        vectors,
        np.asarray([0, 15, 255], dtype=np.uint64),
        np.ones(3, dtype=bool),
        "synthetic",
    )
    assert propagate(samples, features, np.asarray(["dog", "", ""])).tolist() == ["dog", "dog", ""]
