import numpy as np
import pytest

from pet_immich.data import UNKNOWN, Features, Sample, normalize


@pytest.fixture
def population() -> tuple[list[Sample], Features]:
    random = np.random.default_rng(120)
    samples = [
        Sample(f"sample-{index}", f"asset-{index}", (0, 0, 1, 1), label=name)
        for index, name in enumerate(["pet-a"] * 36 + ["pet-b"] * 36 + [UNKNOWN] * 36)
    ]
    vectors = normalize(
        np.repeat(np.eye(3), 36, axis=0).astype(np.float32) + random.normal(0, 0.03, (108, 3)).astype(np.float32)
    )
    features = Features(
        [sample.id for sample in samples],
        vectors,
        vectors,
        vectors,
        random.integers(0, 2**64 - 1, 108, dtype=np.uint64),
        np.ones(108, dtype=bool),
        "synthetic",
    )
    return samples, features
