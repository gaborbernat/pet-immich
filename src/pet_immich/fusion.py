"""Deploy the same reference-gallery fusion evaluated by grouped comparison."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import numpy as np

from pet_immich.benchmark import PairFusion, calibrate, compare, decide
from pet_immich.data import UNCERTAIN, Features
from pet_immich.grouping import propagate
from pet_immich.model import capture_groups, splits

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pet_immich.benchmark import Comparison
    from pet_immich.data import Matrix, Sample


def train_fusion(
    samples: list[Sample],
    features: Features,
    large: Features,
    *,
    error_cost: float = 2,
) -> tuple[FusionModel, Comparison]:
    """Evaluate the chosen composition with thresholds fitted only on calibration events."""
    indexed = {sample.id: sample for sample in samples}
    rows = np.asarray(
        [index for index, key in enumerate(features.ids) if indexed[key].label not in {None, UNCERTAIN}], dtype=np.int64
    )
    samples = [indexed[features.ids[index]] for index in rows]
    subset = Features(
        [sample.id for sample in samples],
        features.raw[rows],
        features.body[rows],
        features.face[rows],
        features.hashes[rows],
        features.valid[rows],
        features.signature,
    )
    gallery = {"face": subset.face, "large": _align_large(subset, large)}
    channels = {name: vectors @ vectors.T for name, vectors in gallery.items()}
    report = compare(samples, subset, channels, group_duplicates=True, error_cost=error_cost)["face+large+duplicates"]
    targets = np.asarray([sample.label for sample in samples], dtype=str)
    groups = capture_groups(samples, subset.raw, subset.hashes)
    values = np.stack(list(channels.values()), axis=-1)
    members = []
    for reference, held in splits(targets, groups, 3):
        member = PairFusion.fit(values, targets, groups, reference)
        thresholds = calibrate(member.scores(values[held]), targets[held], groups[held], member.names, error_cost)
        members.append((member, thresholds))
    return FusionModel(gallery, members, features.signature, large.signature, report["passed"]), report


@dataclass
class FusionModel:
    """Store training galleries and calibration thresholds together for reproducible inference."""

    gallery: dict[str, Matrix]
    members: list[tuple[PairFusion, dict[str, tuple[float, float]]]]
    signature: str
    large_signature: str
    approved: bool

    def predict(self, samples: list[Sample], features: Features, large: Features) -> NDArray[np.str_]:
        """Apply the evaluated voting rule before direct duplicate propagation."""
        if (features.signature, large.signature) != (self.signature, self.large_signature):
            msg = "Feature extractors differ from the validated fusion model"
            raise ValueError(msg)
        vectors = {"face": features.face, "large": _align_large(features, large)}
        values = np.stack([vectors[name] @ gallery.T for name, gallery in self.gallery.items()], axis=-1)
        votes = np.asarray([
            decide(member.scores(values), member.names, thresholds) for member, thresholds in self.members
        ])
        predicted = np.full(len(features.ids), "", dtype="U128")
        minimum_votes: Final = 2
        for name in self.members[0][0].names:
            predicted[
                (np.sum(votes == name, axis=0) >= minimum_votes) & np.all((votes == name) | (votes == ""), axis=0)
            ] = name
        indexed = {sample.id: sample for sample in samples}
        return propagate([replace(indexed[key], label=None) for key in features.ids], features, predicted)


def _align_large(features: Features, large: Features) -> Matrix:
    indices = {key: index for index, key in enumerate(large.ids)}
    missing = set(features.ids) - indices.keys()
    if missing:
        msg = f"Large encoder cache is missing {len(missing)} crops; run extraction first"
        raise ValueError(msg)
    return large.matrix("fused")[[indices[key] for key in features.ids]]


__all__ = ["FusionModel", "train_fusion"]
