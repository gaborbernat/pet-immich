"""Extend accepted identities only across direct visual duplicates with no conflicting evidence."""

from typing import TYPE_CHECKING, Final

import numpy as np

from pet_immich.data import UNCERTAIN, UNKNOWN

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pet_immich.data import Features, Sample


def propagate(samples: list[Sample], features: Features, predicted: NDArray[np.str_]) -> NDArray[np.str_]:
    """Never use a propagated label as an anchor, or a timestamp as evidence of identity."""
    if [sample.id for sample in samples] != features.ids or len(predicted) != len(samples):
        msg = "Duplicate propagation requires aligned samples, features, and predictions"
        raise ValueError(msg)
    anchors = np.asarray(
        [sample.label if sample.label is not None else label for sample, label in zip(samples, predicted, strict=True)],
        dtype="U128",
    )
    result = anchors.copy()
    assets = np.asarray([sample.asset_id for sample in samples])
    minimum_similarity: Final = 0.98
    maximum_hash_bits: Final = 4
    for start in range(0, len(samples), 256):
        similarity = features.raw[start : start + 256] @ features.raw.T
        for offset, row in enumerate(similarity):
            index = start + offset
            if anchors[index] or not features.valid[index]:
                continue
            neighbors = np.flatnonzero(
                (row >= minimum_similarity)
                & (assets != assets[index])
                & (np.char.str_len(anchors) > 0)
                & features.valid
            )
            labels = {
                str(anchors[other])
                for other in neighbors
                if (int(features.hashes[index]) ^ int(features.hashes[other])).bit_count() <= maximum_hash_bits
            }
            if len(labels) == 1 and not labels & {UNKNOWN, UNCERTAIN}:
                result[index] = labels.pop()
    result[~features.valid | np.isin(anchors, [UNKNOWN, UNCERTAIN])] = ""
    return result


__all__ = ["propagate"]
