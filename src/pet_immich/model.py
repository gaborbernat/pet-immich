from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

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


def train(samples: list[Sample], features: Features) -> tuple[IdentityModel, dict]:
    indices = {value: index for index, value in enumerate(features.ids)}
    labeled = [sample for sample in samples if sample.label not in {None, UNCERTAIN} and sample.id in indices]
    rows = np.asarray([indices[sample.id] for sample in labeled], dtype=np.int64)
    targets = np.asarray([sample.label for sample in labeled], dtype=str)
    groups = capture_groups(labeled, features.raw[rows], features.hashes[rows])
    names = sorted(set(targets) - {UNKNOWN})
    if len(names) < 2 or UNKNOWN not in targets:
        raise ValueError("Training requires two identities and confirmed unknown animals")
    for name in np.unique(targets):
        if len(set(groups[targets == name])) < 5:
            raise ValueError(f"At least five independent capture groups are required for {name}")
    matrices = {mode: features.matrix(mode)[rows] for mode in ("raw", "body", "face", "fused")}
    matrices["metric"] = matrices["fused"]
    matrices["rbf10"] = matrices["fused"]
    matrices["linear"] = matrices["fused"]
    predicted = np.full(len(rows), "", dtype="U128")
    decisions = []
    for fold, (training, testing) in enumerate(splits(targets, groups, 5)):
        candidate, selection = select_model(
            {name: values[training] for name, values in matrices.items()}, targets[training], groups[training]
        )
        predicted[testing] = np.where(
            features.valid[rows[testing]], candidate.predict(matrices[candidate.mode][testing]), ""
        )
        decisions.append({"fold": fold, "selection": selection, "test_samples": len(testing)})
    report = metrics(targets, predicted, groups, names)
    report["folds"] = decisions
    report["samples"] = len(rows)
    report["groups"] = len(set(groups))
    report["class_counts"] = dict(Counter(targets.tolist()))
    report["method"] = "nested grouped CV; inner model/threshold selection; outer evaluation only"
    report["passed"] = all(
        row["precision"] >= 0.95 and row["event_precision"] >= 0.95 and row["events"] >= 10
        for row in report["identities"].values()
    )
    final, selection = select_model(matrices, targets, groups)
    final.signature = features.signature
    final.approved = report["passed"]
    report["final_selection"] = selection
    return final, report


def capture_groups(samples: list[Sample], raw: Matrix, hashes: NDArray[np.uint64]) -> NDArray[np.int32]:
    seconds = np.asarray(
        [
            datetime.fromisoformat(sample.captured.replace("Z", "+00:00")).timestamp() if sample.captured else np.nan
            for sample in samples
        ]
    )
    adjacency = np.abs(seconds[:, None] - seconds[None, :]) <= 180
    assets = np.asarray([sample.asset_id for sample in samples])
    adjacency |= assets[:, None] == assets[None, :]
    similarity = raw @ raw.T
    for first, second in zip(*np.where(np.triu(similarity >= 0.98, 1)), strict=True):
        if (int(hashes[first]) ^ int(hashes[second])).bit_count() <= 4:
            adjacency[first, second] = adjacency[second, first] = True
    return connected_components(adjacency, directed=False)[1]


def splits(targets: NDArray[np.str_], groups: NDArray[np.int32], count: int) -> list[tuple]:
    count = min(count, min(len(set(groups[targets == name])) for name in np.unique(targets)))
    if count < 2:
        raise ValueError("Insufficient independent groups for calibration")
    return list(StratifiedGroupKFold(count, shuffle=True, random_state=42).split(targets, targets, groups))


def select_model(
    matrices: dict[str, Matrix], targets: NDArray[np.str_], groups: NDArray[np.int32]
) -> tuple[IdentityModel, dict]:
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
            thresholds = {}
            for name in classifier.classes_:
                if name == UNKNOWN:
                    continue
                threshold = 1.01
                for cutoff in np.unique(np.maximum(confidence, 0.7)):
                    mask = (winners == name) & (confidence >= cutoff) & (similarities >= 0.4)
                    if len(set(groups[calibration][mask])) < 5:
                        continue
                    if np.all(targets[calibration][mask] == name):
                        threshold = float(cutoff)
                        break
                thresholds[str(name)] = threshold
            member = IdentityModel(classifier, mode, thresholds, values[training], targets[training])
            accepted[calibration] = member.predict(values[calibration])
            members.append(member)
        weights = event_weights(groups)
        coverage = float(
            np.mean(
                [
                    np.sum(weights[(accepted == name) & (targets == name)]) / np.sum(weights[targets == name])
                    for name in np.unique(targets)
                    if name != UNKNOWN
                ]
            )
        )
        candidates.append((coverage, members))
    coverage, members = max(candidates, key=lambda row: row[0])
    first = members[0]
    return (
        IdentityModel(first.classifier, first.mode, {}, first.reference, first.targets, members=members),
        {"mode": first.mode, "thresholds": [member.thresholds for member in members], "inner_event_coverage": coverage},
    )


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
    counts = Counter(groups)
    weights = np.asarray([1 / counts[group] for group in groups])
    return weights * len(groups) / weights.sum()


def nearest(values: Matrix, winners: NDArray[np.str_], reference: Matrix, targets: NDArray[np.str_]) -> Matrix:
    scores = values @ reference.T
    scores[winners[:, None] != targets[None, :]] = -1
    return scores.max(axis=1)


def metrics(
    targets: NDArray[np.str_], predicted: NDArray[np.str_], groups: NDArray[np.int32], names: list[str]
) -> dict:
    identities = {}
    for name in names:
        selected = predicted == name
        correct = int(np.sum(selected & (targets == name)))
        total = int(selected.sum())
        identities[name] = {
            "correct": correct,
            "accepted": total,
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
        "unknown_accepted": int(np.sum((targets == UNKNOWN) & (predicted != ""))),
        "coverage": float(np.mean(predicted != "")),
    }


@dataclass
class IdentityModel:
    classifier: CalibratedClassifierCV
    mode: str
    thresholds: dict[str, float]
    reference: Matrix
    targets: NDArray[np.str_]
    signature: str = ""
    approved: bool = False
    members: list[IdentityModel] = field(default_factory=list)

    def predict(self, values: Matrix) -> NDArray[np.str_]:
        if self.members:
            votes = np.asarray([member.predict(values) for member in self.members])
            result = np.full(len(values), "", dtype="U128")
            for name in self.classifier.classes_:
                if name != UNKNOWN:
                    result[(np.sum(votes == name, axis=0) >= 2) & np.all((votes == name) | (votes == ""), axis=0)] = (
                        name
                    )
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
        accepted &= nearest(values, winners, self.reference, self.targets) >= 0.4
        return np.where(accepted, winners, "")


__all__ = ["IdentityModel", "capture_groups", "metrics", "train"]
