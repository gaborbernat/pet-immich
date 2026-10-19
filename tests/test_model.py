from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pet_immich.data import UNKNOWN, Features, Sample, normalize
from pet_immich.metric import MetricSVC
from pet_immich.model import capture_groups, metrics, train

if TYPE_CHECKING:
    from pathlib import Path


def test_training_rejects_unknowns(population: tuple[list[Sample], Features]) -> None:
    model, report = train(*population)
    assert (model.approved, report["unknown_accepted"], model.predict(np.eye(3, dtype=np.float32)).tolist()) == (
        True,
        0,
        ["pet-a", "pet-b", ""],
    )


def test_training_requires_negative_examples(population: tuple[list[Sample], Features]) -> None:
    samples, features = population
    with pytest.raises(ValueError, match="confirmed unknown"):
        train(samples[:72], features)


def test_training_requires_independent_events(population: tuple[list[Sample], Features]) -> None:
    samples, features = population
    with pytest.raises(ValueError, match="five independent"):
        train(samples[:74], features)


def test_grouping_keeps_bursts_and_duplicates_together() -> None:
    samples = [
        Sample(str(index), str(index), (0, 0, 1, 1), captured=date)
        for index, date in enumerate([
            "2020-01-01T00:00:59Z",
            "2020-01-01T00:01:01Z",
            "2021-01-01T00:00:00Z",
            "2022-01-01T00:00:00Z",
        ])
    ]
    groups = capture_groups(
        samples,
        np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32),
        np.asarray([0, 0, 1, 2**64 - 1], dtype=np.uint64),
    )
    assert groups.tolist() == [0, 0, 0, 1]


def test_metrics_counts_bursts_once() -> None:
    report = metrics(
        np.asarray(["pet-a", "pet-a", "pet-b"]),
        np.asarray(["pet-a"] * 3),
        np.asarray([0, 0, 1], dtype=np.int32),
        ["pet-a"],
    )
    assert report["identities"]["pet-a"]["event_precision"] == pytest.approx(0.5)


def test_features_round_trip(population: tuple[list[Sample], Features], tmp_path: Path) -> None:
    _, features = population
    features.save(tmp_path / "features.npz")
    restored = Features.load(tmp_path / "features.npz")
    assert restored.ids == features.ids
    np.testing.assert_array_equal(restored.matrix("raw"), features.raw)


def test_features_invalid_mode(population: tuple[list[Sample], Features]) -> None:
    with pytest.raises(ValueError, match="Unknown feature"):
        population[1].matrix("invalid")


def test_failed_validation_never_approves_model(population: tuple[list[Sample], Features]) -> None:
    samples, features = population
    random = np.random.default_rng(7)
    features.raw[:] = normalize(random.normal(size=features.raw.shape).astype(np.float32))
    features.body[:] = features.raw
    features.face[:] = features.raw
    model, report = train(samples, features)
    assert (model.approved, report["passed"]) == (False, False)


def test_metrics_handles_abstention() -> None:
    report = metrics(
        np.asarray(["pet-a", UNKNOWN]), np.asarray(["", ""], dtype=str), np.asarray([0, 1], dtype=np.int32), ["pet-a"]
    )
    assert report == {
        "identities": {
            "pet-a": {
                "correct": 0,
                "accepted": 0,
                "incorrect": 0,
                "known": 1,
                "missed": 1,
                "precision": 0,
                "recall": 0,
                "event_precision": 0,
                "events": 0,
                "precision_lower_95": 0,
            }
        },
        "unknown_accepted": 0,
        "coverage": 0,
    }


@pytest.mark.parametrize("method", ["decision_function", "predict"], ids=["margins", "labels"])
def test_metric_margins_separate_known_identities(population: tuple[list[Sample], Features], method: str) -> None:
    samples, features = population
    estimator = MetricSVC().fit(features.raw, np.asarray([sample.label for sample in samples], dtype=str))
    result = getattr(estimator, method)(np.eye(3, dtype=np.float32))
    predicted = estimator.classes_[result.argmax(axis=1)] if method == "decision_function" else result
    assert predicted.tolist() == ["pet-a", "pet-b", UNKNOWN]


def test_training_rejects_capture_overlap_that_breaks_inner_folds(population: tuple[list[Sample], Features]) -> None:
    samples, features = population
    groups = np.fromstring(
        "8 11 5 10 10 10 11 7 10 5 10 11 11 11 10 11 7 5 8 8 5 5 11 8 11 11 8 7 5 11 10 7 7 8 5 8 "
        "11 0 0 7 1 11 7 7 11 1 7 0 7 1 0 6 6 11 7 1 6 11 11 6 0 6 1 7 6 1 11 1 11 11 1 6 "
        "8 11 11 2 11 11 7 11 11 1 2 2 11 7 7 11 11 7 8 8 8 7 11 7 8 7 2 7 1 11 8 1 1 2 8 2 ",
        sep=" ",
        dtype=np.int32,
    )
    samples = [
        replace(sample, captured=f"2020-01-{group + 1:02}T00:00:00Z")
        for sample, group in zip(samples, groups, strict=True)
    ]
    with pytest.raises(ValueError, match="Insufficient independent"):
        train(samples, features)


@pytest.mark.parametrize(("cost", "expected"), [(1, "pet-a"), (5, "")], ids=["recover-majority", "reject-ambiguity"])
def test_calibration_balances_recovery_against_incorrect_assignments(
    population: tuple[list[Sample], Features],
    cost: float,
    expected: str,
) -> None:
    samples, features = population
    samples = [replace(sample, label="pet-a") if 36 <= index < 60 else sample for index, sample in enumerate(samples)]
    for values in (features.raw, features.body, features.face):
        values[:72] = [1, 0, 0]
    model, _ = train(samples, features, error_cost=cost)
    assert model.predict(np.asarray([[1, 0, 0]], dtype=np.float32)).tolist() == [expected]


@pytest.mark.parametrize("cost", [0, -1, float("nan"), float("inf")], ids=["zero", "negative", "nan", "infinite"])
def test_training_rejects_invalid_error_cost(population: tuple[list[Sample], Features], cost: float) -> None:
    with pytest.raises(ValueError, match="Error cost"):
        train(*population, error_cost=cost)


@pytest.mark.parametrize("fraction", [0, -1, 1.1, float("nan")], ids=["zero", "negative", "too-large", "nan"])
def test_training_rejects_invalid_fraction(population: tuple[list[Sample], Features], fraction: float) -> None:
    with pytest.raises(ValueError, match="Training fraction"):
        train(*population, training_fraction=fraction)


def test_learning_curve_preserves_test_events(population: tuple[list[Sample], Features]) -> None:
    _, full = train(*population)
    _, reduced = train(*population, training_fraction=0.5)
    assert [row["test_samples"] for row in reduced["folds"]] == [row["test_samples"] for row in full["folds"]]
    assert all(
        small["training_samples"] < large["training_samples"]
        for small, large in zip(reduced["folds"], full["folds"], strict=True)
    )
