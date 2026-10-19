"""Reject uncertain identities using calibration separated by capture event."""

import operator
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Final, NotRequired, TypedDict

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.stats import beta
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC

from pet_immich.data import UNCERTAIN, UNKNOWN
from pet_immich.metric import MetricSVC

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pet_immich.data import Features, Matrix, Sample


_MINIMUM_SIMILARITY: Final = 0.4
_MINIMUM_GROUPS: Final = 5


def train(
    samples: list[Sample], features: Features, *, error_cost: float = 5, training_fraction: float = 1
) -> tuple[IdentityModel, _Report]:
    """Select thresholds inside each fold so outer observations remain evaluation data."""
    validate_error_cost(error_cost)
    if not 0 < training_fraction <= 1:
        msg = "Training fraction must be greater than zero and at most one"
        raise ValueError(msg)
    indices = {value: index for index, value in enumerate(features.ids)}
    labeled = [sample for sample in samples if sample.label not in {None, UNCERTAIN} and sample.id in indices]
    rows = np.asarray([indices[sample.id] for sample in labeled], dtype=np.int64)
    targets = np.asarray([sample.label for sample in labeled], dtype=str)
    groups = capture_groups(labeled, features.raw[rows], features.hashes[rows])
    names = sorted(set(targets) - {UNKNOWN})
    if len(names) <= 1 or UNKNOWN not in targets:
        msg = "Training requires two identities and confirmed unknown animals"
        raise ValueError(msg)
    for name in np.unique(targets):
        if len(set(groups[targets == name])) < _MINIMUM_GROUPS:
            msg = f"At least five independent capture groups are required for {name}"
            raise ValueError(msg)
    matrices = {mode: features.matrix(mode)[rows] for mode in ("raw", "body", "face", "fused")}
    matrices["metric"] = matrices["fused"]
    matrices["rbf10"] = matrices["fused"]
    matrices["linear"] = matrices["fused"]
    predicted = np.full(len(rows), "", dtype="U128")
    decisions: list[_Fold] = []
    for fold, (available, testing) in enumerate(splits(targets, groups, 5)):
        training = _subsample_training(available, targets, groups, training_fraction)
        candidate, selection = select_model(
            {name: values[training] for name, values in matrices.items()},
            targets[training],
            groups[training],
            error_cost,
        )
        predicted[testing] = np.where(
            features.valid[rows[testing]], candidate.predict(matrices[candidate.mode][testing]), ""
        )
        decisions.append({
            "fold": fold,
            "selection": selection,
            "test_samples": len(testing),
            "training_samples": len(training),
        })
    report = metrics(targets, predicted, groups, names)
    report["folds"] = decisions
    report["samples"] = len(rows)
    report["groups"] = len(set(groups))
    report["class_counts"] = dict(Counter(targets.tolist()))
    report["method"] = "nested grouped CV; inner model/threshold selection; outer evaluation only"
    report["training_fraction"] = training_fraction
    report["error_cost"] = error_cost
    report["passed"] = passes(report, error_cost)
    final, selection = select_model(matrices, targets, groups, error_cost)
    final.signature = features.signature
    final.approved = report["passed"] and training_fraction == 1
    report["final_selection"] = selection
    return final, report


def _subsample_training(
    training: NDArray[np.int64],
    targets: NDArray[np.str_],
    groups: NDArray[np.int32],
    fraction: float,
) -> NDArray[np.int64]:
    # Retain the rare unknown examples while measuring the value of more known-pet events.
    selected = set(groups[training][targets[training] == UNKNOWN])
    random = np.random.default_rng(42)
    for name in sorted(set(targets) - {UNKNOWN}):
        events = random.permutation(np.unique(groups[training][targets[training] == name]))
        selected.update(events[: int(np.ceil(len(events) * fraction))])
    return training[np.isin(groups[training], list(selected))]


def validate_error_cost(error_cost: float) -> None:
    """Reject costs that would reward mistakes or make comparisons undefined."""
    if not np.isfinite(error_cost) or error_cost <= 0:
        msg = "Error cost must be finite and greater than zero"
        raise ValueError(msg)


def passes(report: _Report, error_cost: float = 5) -> bool:
    """Require evidence of net benefit for each identity under the selected error cost."""
    minimum_events: Final = 10
    return all(
        row["event_precision"] - error_cost * (1 - row["event_precision"]) > 0 and row["events"] >= minimum_events
        for row in report["identities"].values()
    )


def capture_groups(samples: list[Sample], raw: Matrix, hashes: NDArray[np.uint64]) -> NDArray[np.int32]:
    """Keep bursts and near-duplicates together to avoid optimistic validation."""
    seconds = np.asarray([
        datetime.fromisoformat(sample.captured).timestamp() if sample.captured else np.nan for sample in samples
    ])
    burst_seconds: Final = 180
    adjacency = np.abs(seconds[:, None] - seconds[None, :]) <= burst_seconds
    assets = np.asarray([sample.asset_id for sample in samples])
    adjacency |= assets[:, None] == assets[None, :]
    similarity = raw @ raw.T
    duplicate_similarity: Final = 0.98
    duplicate_bits: Final = 4
    for first, second in zip(*np.where(np.triu(similarity >= duplicate_similarity, 1)), strict=True):
        if (int(hashes[first]) ^ int(hashes[second])).bit_count() <= duplicate_bits:
            adjacency[first, second] = adjacency[second, first] = True
    return connected_components(adjacency, directed=False)[1]


def splits(
    targets: NDArray[np.str_], groups: NDArray[np.int32], count: int
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Keep related captures together while retaining each class across folds."""
    count = min(count, *(len(set(groups[targets == name])) for name in np.unique(targets)))
    minimum_folds: Final = 2
    if count < minimum_folds:
        msg = "Insufficient independent groups for calibration"
        raise ValueError(msg)
    return list(StratifiedGroupKFold(count, shuffle=True, random_state=42).split(targets, targets, groups))


def select_model(
    matrices: dict[str, Matrix], targets: NDArray[np.str_], groups: NDArray[np.int32], error_cost: float
) -> tuple[IdentityModel, _Selection]:
    candidates = []
    for mode, values in matrices.items():
        members = []
        accepted = np.full(len(targets), "", dtype="U128")
        for training, calibration in splits(targets, groups, 3):
            classifier = fit(values[training], targets[training], groups[training], mode)
            probabilities = classifier.predict_proba(values[calibration])
            winners = classifier.classes_[probabilities.argmax(axis=1)]
            confidence = probabilities.max(axis=1)
            similarities = nearest(values[calibration], winners, values[training], targets[training])
            thresholds = _thresholds(
                winners, confidence, similarities, targets[calibration], groups[calibration], error_cost
            )
            member = IdentityModel(classifier, mode, thresholds, values[training], targets[training])
            accepted[calibration] = member.predict(values[calibration])
            members.append(member)
        weights = event_weights(groups)
        utility = float(
            np.mean([
                (
                    np.sum(weights[(accepted == name) & (targets == name)])
                    - error_cost * np.sum(weights[(accepted == name) & (targets != name)])
                )
                / np.sum(weights[targets == name])
                for name in np.unique(targets)
                if name != UNKNOWN
            ])
        )
        candidates.append((utility, members))
    utility, members = max(candidates, key=operator.itemgetter(0))
    first = members[0]
    return (
        IdentityModel(first.classifier, first.mode, {}, first.reference, first.targets, members=members),
        {"mode": first.mode, "thresholds": [member.thresholds for member in members], "inner_event_utility": utility},
    )


def _thresholds(
    winners: NDArray[np.str_],
    confidence: Matrix,
    similarities: Matrix,
    targets: NDArray[np.str_],
    groups: NDArray[np.int32],
    error_cost: float,
) -> dict[str, float]:
    weights = event_weights(groups)
    thresholds = {}
    for name in np.unique(targets):
        if name == UNKNOWN:
            continue
        threshold = 1.01
        best = 0.0
        for cutoff in np.unique(confidence):
            mask = (winners == name) & (confidence >= cutoff) & (similarities >= _MINIMUM_SIMILARITY)
            if len(set(groups[mask])) < _MINIMUM_GROUPS:
                continue
            if (benefit := float(np.sum(weights[mask] * np.where(targets[mask] == name, 1, -error_cost)))) > best:
                best = benefit
                threshold = float(cutoff)
        thresholds[str(name)] = threshold
    return thresholds


def fit(values: Matrix, targets: NDArray[np.str_], groups: NDArray[np.int32], mode: str) -> CalibratedClassifierCV:
    classifier = CalibratedClassifierCV(
        MetricSVC()
        if mode == "metric"
        else SVC(
            C=10 if mode == "rbf10" else 1, kernel="linear" if mode == "linear" else "rbf", class_weight="balanced"
        ),
        method="sigmoid",
        cv=splits(targets, groups, 3),
    )
    if mode == "metric":
        classifier.fit(values, targets, sample_weight=event_weights(groups), capture_groups=groups)
    else:
        classifier.fit(values, targets, sample_weight=event_weights(groups))
    return classifier


def event_weights(groups: NDArray[np.int32]) -> NDArray[np.float64]:
    """Give each event equal weight regardless of burst size."""
    counts = Counter(groups)
    weights = np.asarray([1 / counts[group] for group in groups])
    return weights * len(groups) / weights.sum()


def nearest(values: Matrix, winners: NDArray[np.str_], reference: Matrix, targets: NDArray[np.str_]) -> Matrix:
    scores = values @ reference.T
    scores[winners[:, None] != targets[None, :]] = -1
    return scores.max(axis=1)


def metrics(
    targets: NDArray[np.str_], predicted: NDArray[np.str_], groups: NDArray[np.int32], names: list[str]
) -> _Report:
    """Report event-weighted precision so bursts do not inflate apparent reliability."""
    identities: dict[str, IdentityMetrics] = {}
    for name in names:
        selected = predicted == name
        correct = int(np.sum(selected & (targets == name)))
        total = int(selected.sum())
        identities[name] = {
            "correct": correct,
            "accepted": total,
            "incorrect": total - correct,
            "known": int(np.sum(targets == name)),
            "missed": int(np.sum(targets == name)) - correct,
            "precision": correct / total if total else 0,
            "recall": correct / int(np.sum(targets == name)),
            "event_precision": float(np.average(targets[selected] == name, weights=event_weights(groups[selected])))
            if total
            else 0,
            "events": len(set(groups[selected])),
            "precision_lower_95": float(beta.ppf(0.05, correct, total - correct + 1)) if correct else 0,
        }
    return {
        "identities": identities,
        "unknown_accepted": int(np.sum((targets == UNKNOWN) & (np.char.str_len(predicted) > 0))),
        "coverage": float(np.mean(np.char.str_len(predicted) > 0)),
    }


@dataclass
class IdentityModel:
    """Keep each classifier frozen with the thresholds calibrated against it."""

    classifier: CalibratedClassifierCV
    mode: str
    thresholds: dict[str, float]
    reference: Matrix
    targets: NDArray[np.str_]
    signature: str = ""
    approved: bool = False
    members: list[IdentityModel] = field(default_factory=list)

    def predict(self, values: Matrix) -> NDArray[np.str_]:
        """Abstain on disagreements and distant examples instead of forcing an identity."""
        if self.members:
            minimum_votes: Final = 2
            votes = np.asarray([member.predict(values) for member in self.members])
            result = np.full(len(values), "", dtype="U128")
            for name in self.classifier.classes_:
                if name != UNKNOWN:
                    result[
                        (np.sum(votes == name, axis=0) >= minimum_votes)
                        & np.all((votes == name) | (votes == ""), axis=0)
                    ] = name
            return result
        probabilities = self.classifier.predict_proba(values)
        winners = self.classifier.classes_[probabilities.argmax(axis=1)]
        accepted = np.asarray(
            [
                probability >= self.thresholds.get(str(name), 1.01)
                for name, probability in zip(winners, probabilities.max(axis=1), strict=True)
            ],
            dtype=bool,
        )
        accepted &= nearest(values, winners, self.reference, self.targets) >= _MINIMUM_SIMILARITY
        return np.where(accepted, winners, "")


class IdentityMetrics(TypedDict):
    """Keep per-identity errors visible beside aggregate coverage."""

    correct: int
    accepted: int
    incorrect: int
    known: int
    missed: int
    precision: float
    recall: float
    event_precision: float
    events: int
    precision_lower_95: float


class _Selection(TypedDict):
    mode: str
    thresholds: list[dict[str, float]]
    inner_event_utility: float


class _Fold(TypedDict):
    fold: int
    selection: _Selection
    test_samples: int
    training_samples: int


class _Report(TypedDict):
    identities: dict[str, IdentityMetrics]
    unknown_accepted: int
    coverage: float
    folds: NotRequired[list[_Fold]]
    samples: NotRequired[int]
    groups: NotRequired[int]
    class_counts: NotRequired[dict[str, int]]
    method: NotRequired[str]
    passed: NotRequired[bool]
    error_cost: NotRequired[float]
    training_fraction: NotRequired[float]
    final_selection: NotRequired[_Selection]


__all__ = [
    "IdentityMetrics",
    "IdentityModel",
    "capture_groups",
    "event_weights",
    "metrics",
    "passes",
    "splits",
    "train",
    "validate_error_cost",
]
