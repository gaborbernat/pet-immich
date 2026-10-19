from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from pet_immich.extract import encode
from pet_immich.miewid import load_miewid

if TYPE_CHECKING:
    from pathlib import Path


def test_miewid_distinguishes_images_using_loaded_backbone(miewid_weights: Path) -> None:
    model = load_miewid(miewid_weights)
    values = encode(model, [Image.new("RGB", (64, 64), "red"), Image.new("RGB", (64, 64), "blue")])
    assert values.shape == (2, 2152)
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), [1, 1], atol=1e-6)
    assert not np.allclose(values[0], values[1])
