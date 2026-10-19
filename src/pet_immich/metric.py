"""Learn identity features using positives from independent capture events."""

import numpy as np
import torch
from numpy.typing import NDArray  # ruff: ignore[typing-only-third-party-import] - sklearn resolves method types.
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from torch import nn
from torch.nn import functional

from pet_immich.data import UNKNOWN, Matrix, normalize


class MetricSVC(ClassifierMixin, BaseEstimator):
    """Compare learned identity features with the frozen embedding baseline."""

    def fit(
        self,
        values: Matrix,
        targets: NDArray[np.str_],
        sample_weight: NDArray[np.float64] | None = None,
        capture_groups: NDArray[np.int32] | None = None,
    ) -> MetricSVC:
        """Train cross-event positives so burst similarity cannot dominate identity learning."""
        torch.set_num_threads(4)
        torch.manual_seed(42)
        self.classes_, encoded = np.unique(targets, return_inverse=True)
        self.projection_ = nn.Linear(values.shape[1], 64, bias=False)
        head = nn.Linear(64, len(self.classes_), bias=False)
        optimizer = torch.optim.AdamW([*self.projection_.parameters(), *head.parameters()], lr=0.002, weight_decay=0.05)
        inputs = torch.from_numpy(values.astype(np.float32))
        weights = (
            torch.from_numpy(sample_weight.astype(np.float32)) if sample_weight is not None else torch.ones(len(values))
        )
        known = torch.tensor(targets != UNKNOWN)
        positives = torch.from_numpy(encoded[:, None] == encoded[None, :]) & known[:, None] & known[None, :]
        positives.fill_diagonal_(fill_value=False)
        if capture_groups is not None:
            events = torch.from_numpy(capture_groups)
            positives &= events[:, None] != events[None, :]
        for _ in range(60):
            projected = functional.normalize(self.projection_(functional.dropout(inputs, p=0.15, training=True)), dim=1)
            logits = functional.linear(projected, functional.normalize(head.weight, dim=1)) * 16
            classification = (
                functional.cross_entropy(logits, torch.from_numpy(encoded.astype(np.int64)), reduction="none") * weights
            ).mean()
            similarity = projected @ projected.T / 0.15
            similarity = similarity.masked_fill(torch.eye(len(values), dtype=torch.bool), -10000)
            log_probabilities = similarity - similarity.logsumexp(dim=1, keepdim=True)
            contrastive = -(log_probabilities * positives).sum(dim=1) / positives.sum(dim=1).clamp_min(1)
            loss = classification + 0.2 * (contrastive[positives.any(dim=1)] * weights[positives.any(dim=1)]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.projection_.eval()
        self.svc_ = SVC(C=1, kernel="rbf", class_weight="balanced").fit(
            self.transform(values), targets, sample_weight=sample_weight
        )
        return self

    def transform(self, values: Matrix) -> Matrix:
        """Retain pretrained features alongside the learned projection to limit overfitting."""
        with torch.no_grad():
            projected = functional.normalize(
                self.projection_(torch.from_numpy(values.astype(np.float32))), dim=1
            ).numpy()
        return normalize(np.concatenate((values, projected), axis=1))

    def decision_function(self, values: Matrix) -> Matrix:
        """Expose margins for calibration on captures excluded from classifier training."""
        return self.svc_.decision_function(self.transform(values))

    def predict(self, values: Matrix) -> NDArray[np.str_]:
        """Keep the estimator compatible with scikit-learn validation and scoring."""
        return self.svc_.predict(self.transform(values))


__all__ = ["MetricSVC"]
