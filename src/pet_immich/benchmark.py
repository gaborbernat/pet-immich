"""Compare compositions on shared events without selecting thresholds on their test predictions."""

import itertools
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, TypedDict

import numpy as np
from sklearn.linear_model import LogisticRegression

from pet_immich.data import UNCERTAIN, UNKNOWN, Features
from pet_immich.grouping import propagate
from pet_immich.model import capture_groups, event_weights, metrics, passes, splits, validate_error_cost

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

    from pet_immich.data import Matrix, Sample
    from pet_immich.model import IdentityMetrics


def compare(
    samples: list[Sample],
    features: Features,
    channels: dict[str, Matrix],
    *,
    group_duplicates: bool = False,
    error_cost: float = 5,
    learned: Mapping[str, Callable[[NDArray[np.int64]], Matrix]] | None = None,
) -> dict[str, Comparison]:
    """Channels contain frozen pair similarities in feature-ID order, including unavailable pairs as NaN."""
    validate_error_cost(error_cost)
    size = len(features.ids)
    if not channels or any(values.shape != (size, size) for values in channels.values()):
        msg = "Every similarity channel must be square and aligned with feature IDs"
        raise ValueError(msg)
    indices = {value: index for index, value in enumerate(features.ids)}
    labeled = [sample for sample in samples if sample.label not in {None, UNCERTAIN} and sample.id in indices]
    rows = np.asarray([indices[sample.id] for sample in labeled], dtype=np.int64)
    targets = np.asarray([sample.label for sample in labeled], dtype=str)
    groups = capture_groups(labeled, features.raw[rows], features.hashes[rows])
    names = sorted(set(targets) - {UNKNOWN})
    if len(names) <= 1 or UNKNOWN not in targets:
        msg = "Comparison requires two identities and confirmed unknown animals"
        raise ValueError(msg)
    folds = splits(targets, groups, 5)
    results: dict[str, Comparison] = {}
    for count in range(1, len(channels) + 1):
        for composition in itertools.combinations(channels, count):
            values = np.stack([channels[name][np.ix_(rows, rows)] for name in composition], axis=-1)
            predicted = np.full(len(rows), "", dtype="U128")
            for training, testing in folds:
                predicted[testing] = _predict_fold(
                    values,
                    targets,
                    groups,
                    training,
                    testing,
                    error_cost,
                    {
                        column: (lambda reference, channel=learned[name]: channel(rows[reference])[np.ix_(rows, rows)])
                        for column, name in enumerate(composition)
                        if learned is not None and name in learned
                    },
                )
            predicted[~features.valid[rows]] = ""
            variants = {"+".join(composition): predicted}
            if group_duplicates:
                subset = Features(
                    [sample.id for sample in labeled],
                    features.raw[rows],
                    features.body[rows],
                    features.face[rows],
                    features.hashes[rows],
                    features.valid[rows],
                    features.signature,
                )
                variants["+".join((*composition, "duplicates"))] = propagate(
                    [replace(sample, label=None) for sample in labeled], subset, predicted
                )
            for name, predictions in variants.items():
                results[name] = _summarize(
                    labeled, targets, groups, names, predictions, features.valid[rows], error_cost
                )
    return results


def _summarize(
    samples: list[Sample],
    targets: NDArray[np.str_],
    groups: NDArray[np.int32],
    names: list[str],
    predicted: NDArray[np.str_],
    valid: NDArray[np.bool_],
    error_cost: float,
) -> Comparison:
    report = metrics(targets, predicted, groups, names)
    return {
        "identities": report["identities"],
        "unknown_accepted": report["unknown_accepted"],
        "coverage": report["coverage"],
        "unique_photos": len({sample.asset_id for sample, label in zip(samples, predicted, strict=True) if label}),
        "ineligible_crops": int(np.sum(~valid)),
        "samples": len(samples),
        "events": len(set(groups)),
        "passed": passes(report, error_cost),
        "error_cost": error_cost,
        "errors": [
            {"crop": sample.id, "expected": str(target), "predicted": str(prediction), "event": int(group)}
            for sample, target, prediction, group in zip(samples, targets, predicted, groups, strict=True)
            if prediction != target and (prediction or target != UNKNOWN)
        ],
    }


def _predict_fold(
    values: Matrix,
    targets: NDArray[np.str_],
    groups: NDArray[np.int32],
    training: NDArray[np.int64],
    testing: NDArray[np.int64],
    error_cost: float,
    learned: dict[int, Callable[[NDArray[np.int64]], Matrix]],
) -> NDArray[np.str_]:
    votes = []
    for fitting, calibration in splits(targets[training], groups[training], 3):
        reference = training[fitting]
        held = training[calibration]
        for column, channel in learned.items():
            values[..., column] = channel(reference)
        model = PairFusion.fit(values, targets, groups, reference)
        thresholds = calibrate(model.scores(values[held]), targets[held], groups[held], model.names, error_cost)
        votes.append(decide(model.scores(values[testing]), model.names, thresholds))
    result = np.full(len(testing), "", dtype="U128")
    stacked = np.asarray(votes)
    minimum_votes: Final = 2
    for name in sorted(set(targets) - {UNKNOWN}):
        result[
            (np.sum(stacked == name, axis=0) >= minimum_votes) & np.all((stacked == name) | (stacked == ""), axis=0)
        ] = name
    return result


@dataclass
class PairFusion:
    """Learn channel weights from cross-event pairs; unknown animals supply negatives only."""

    classifier: LogisticRegression
    reference: NDArray[np.int64]
    targets: NDArray[np.str_]
    groups: NDArray[np.int32]
    names: list[str]

    @classmethod
    def fit(
        cls, values: Matrix, targets: NDArray[np.str_], groups: NDArray[np.int32], reference: NDArray[np.int64]
    ) -> PairFusion:
        """Balance capture events so repeated frames cannot dominate channel weights."""
        first, second = np.where(np.triu(groups[reference, None] != groups[None, reference], 1))
        same = (targets[reference[first]] == targets[reference[second]]) & (targets[reference[first]] != UNKNOWN)
        pairs = values[reference[first], reference[second]]
        weights = event_weights(groups[reference])
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000).fit(
            _pair_inputs(pairs), same, sample_weight=weights[first] * weights[second]
        )
        return cls(
            classifier,
            reference,
            targets[reference],
            groups[reference],
            sorted(set(targets[reference].tolist()) - {UNKNOWN}),
        )

    def scores(self, values: Matrix) -> Matrix:
        """Require support across reference events rather than repeated frames."""
        pairs = values[:, self.reference]
        scores = self.classifier.predict_proba(_pair_inputs(pairs.reshape(-1, pairs.shape[-1])))[:, 1].reshape(
            pairs.shape[:2]
        )
        scores[~np.isfinite(pairs).any(axis=-1)] = 0
        result = np.empty((len(values), len(self.names)), dtype=np.float32)
        for column, name in enumerate(self.names):
            events = np.stack(
                [
                    scores[:, (self.targets == name) & (self.groups == group)].max(axis=1)
                    for group in np.unique(self.groups[self.targets == name])
                ],
                axis=1,
            )
            # Repeated frames from one event cannot fill all reference positions.
            result[:, column] = np.sort(events, axis=1)[:, -3:].mean(axis=1)
        return result


def _pair_inputs(values: Matrix) -> Matrix:
    return np.concatenate((np.nan_to_num(values, nan=0, posinf=0, neginf=0), np.isfinite(values)), axis=1)


def calibrate(
    scores: Matrix, targets: NDArray[np.str_], groups: NDArray[np.int32], names: list[str], error_cost: float
) -> dict[str, tuple[float, float]]:
    """Choose acceptance and runner-up margin using calibration events only."""
    winners = np.asarray(names)[scores.argmax(axis=1)]
    confidence = scores.max(axis=1)
    margins = np.diff(np.sort(scores, axis=1)[:, -2:], axis=1).ravel()
    thresholds = {}
    weights = event_weights(groups)
    for name in names:
        best = 0.0
        threshold = (1.01, 1.01)
        for cutoff, margin in itertools.product(np.unique(confidence), (0, 0.025, 0.05, 0.1, 0.2)):
            accepted = (winners == name) & (confidence >= cutoff) & (margins >= margin)
            minimum_events: Final = 5
            if len(set(groups[accepted])) < minimum_events:
                continue
            correct = targets[accepted] == name
            coverage = float(np.sum(weights[accepted] * np.where(correct, 1, -error_cost)))
            if coverage > best:
                best = coverage
                threshold = (float(cutoff), margin)
        thresholds[name] = threshold
    return thresholds


def decide(scores: Matrix, names: list[str], thresholds: dict[str, tuple[float, float]]) -> NDArray[np.str_]:
    """Reject weak winners and close runner-up identities."""
    winners = np.asarray(names)[scores.argmax(axis=1)]
    confidence = scores.max(axis=1)
    margins = np.diff(np.sort(scores, axis=1)[:, -2:], axis=1).ravel()
    return np.asarray([
        name if score >= thresholds[name][0] and margin >= thresholds[name][1] else ""
        for name, score, margin in zip(winners, confidence, margins, strict=True)
    ])


class Comparison(TypedDict):
    """Retain extraction failures in the denominator of each comparison."""

    identities: dict[str, IdentityMetrics]
    unknown_accepted: int
    coverage: float
    unique_photos: int
    ineligible_crops: int
    samples: int
    events: int
    passed: bool
    error_cost: float
    errors: list[PredictionError]


class PredictionError(TypedDict):
    """Retain private crop IDs so failures can be grouped without asking for repeated burst labels."""

    crop: str
    expected: str
    predicted: str
    event: int


__all__ = ["Comparison", "PairFusion", "calibrate", "compare", "decide"]
