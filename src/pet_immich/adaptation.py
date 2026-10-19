"""Fit local correspondence models inside each calibration fold and resume expensive pair scoring."""

import hashlib
import io
import json
import sys
from typing import TYPE_CHECKING

import httpx
import numpy as np
import torch

from pet_immich.benchmark import compare
from pet_immich.data import UNCERTAIN, Features, load_samples, private_write
from pet_immich.matching import LocalMatcher
from pet_immich.model import capture_groups
from pet_immich.network import local_request

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pet_immich.data import Matrix, Sample


def compare_adaptation(directory: Path, tagger_url: str, *, foreground: bool = False, steps: int = 300) -> None:
    """Compare frozen and adapted local matching on identical reference and calibration events."""
    torch.set_num_threads(2)
    features = Features.load(directory / "features.npz")
    indexed = {sample.id: sample for sample in load_samples(directory / "samples.json")}
    rows = np.asarray(
        [index for index, key in enumerate(features.ids) if indexed[key].label not in {None, UNCERTAIN}], dtype=np.int64
    )
    samples = [indexed[features.ids[index]] for index in rows]
    features = Features(
        [sample.id for sample in samples],
        features.raw[rows],
        features.body[rows],
        features.face[rows],
        features.hashes[rows],
        features.valid[rows],
        features.signature,
    )
    experiment = directory / ("foreground-matcher" if foreground else "raw-matcher")
    weights = directory / "models" / "segment.pt" if foreground else None
    learned = {
        name: AdaptedMatches(samples, features, experiment, tagger_url, mask_weights=weights, steps=count)
        for name, count in (("frozen", 0), ("adapted", steps))
    }
    vectors = features.matrix("fused")
    channels = {
        "global": vectors @ vectors.T,
        **{name: np.full((len(samples), len(samples)), np.nan, dtype=np.float32) for name in learned},
    }
    report = compare(samples, features, channels, learned=learned, group_duplicates=True, error_cost=2)
    private_write(experiment / "comparison.json", json.dumps(report, indent=2).encode())
    sys.stdout.write(json.dumps(report, indent=2) + "\n")


class AdaptedMatches:
    """Keep validation identities out of matcher training, including threshold-calibration identities."""

    def __init__(
        self,
        samples: list[Sample],
        features: Features,
        directory: Path,
        tagger_url: str,
        *,
        mask_weights: Path | None = None,
        steps: int = 300,
    ) -> None:
        """Bind checkpoints to labels, feature values, preprocessing, and optimization settings."""
        self.samples = samples
        self.features = features
        self.directory = directory
        self.tagger_url = tagger_url
        self.steps = steps
        self.matcher = LocalMatcher(mask_weights)
        self.initial = {name: value.clone() for name, value in self.matcher.matcher.state_dict().items()}
        self.targets = np.asarray([sample.label or "" for sample in samples])
        self.groups = capture_groups(samples, features.raw, features.hashes)
        checksum = hashlib.sha256(features.raw.tobytes())
        checksum.update(json.dumps([features.ids, self.targets.tolist(), self.groups.tolist(), steps]).encode())
        checksum.update(self.matcher.signature.encode())
        self.signature = checksum.hexdigest()

    def __call__(self, reference: NDArray[np.int64]) -> Matrix:
        """Shortlist against training references only, matching the information available at deployment."""
        key = hashlib.sha256(self.signature.encode() + (reference.tobytes() if self.steps else b"")).hexdigest()
        path = self.directory / f"{key}.npz"
        scores = np.full((len(self.samples), len(self.samples)), np.nan, dtype=np.float32)
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                scores = archive["scores"]
        similarity = self.features.raw @ self.features.raw.T
        np.fill_diagonal(similarity, -np.inf)
        candidates = reference[np.argsort(-similarity[:, reference], axis=1)[:, :20]]
        pairs = sorted({
            tuple(sorted((index, int(other))))
            for index, others in enumerate(candidates.tolist())
            for other in others
            if index != other
        })
        eligible = np.zeros(scores.shape, dtype=bool)
        for first, second in pairs:
            eligible[first, second] = eligible[second, first] = True
        missing = [(first, second) for first, second in pairs if np.isnan(scores[first, second])]
        if not missing:
            return np.where(eligible, scores, np.nan)
        with httpx.Client(trust_env=False, event_hooks={"request": [local_request]}, timeout=60) as client:
            descriptors = [
                self.matcher.describe(sample, self.directory, client, self.tagger_url) for sample in self.samples
            ]
        self.matcher.matcher.load_state_dict(self.initial)
        weights = self.directory / f"{key}.pt"
        if weights.exists():
            self.matcher.matcher.load_state_dict(torch.load(weights, weights_only=True))
        elif self.steps:
            losses = self.matcher.fit(
                [descriptors[index] for index in reference],
                self.targets[reference],
                self.groups[reference],
                similarity[np.ix_(reference, reference)],
                steps=self.steps,
            )
            stream = io.BytesIO()
            torch.save(self.matcher.matcher.state_dict(), stream)
            private_write(weights, stream.getvalue())
            private_write(
                self.directory / f"{key}.json", json.dumps({"losses": losses, "reference": reference.tolist()}).encode()
            )
        for index, (first, second) in enumerate(missing, 1):
            scores[first, second] = scores[second, first] = self.matcher.score(descriptors[first], descriptors[second])
            if index % 100 == 0 or index == len(missing):
                stream = io.BytesIO()
                np.savez_compressed(stream, scores=scores)
                private_write(path, stream.getvalue())
                sys.stdout.write(f"Matcher fold {key[:8]}: {index}/{len(missing)} pairs\n")
                sys.stdout.flush()
        return np.where(eligible, scores, np.nan)


__all__ = ["AdaptedMatches", "compare_adaptation"]
