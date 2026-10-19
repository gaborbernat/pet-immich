import numpy as np
import pytest
from PIL import Image

from pet_immich.extract import difference_hash, face_crop


@pytest.mark.parametrize(
    "box",
    [
        pytest.param([0.25, 0.25, 0.75, 0.75], id="ordered"),
        pytest.param([0.75, 0.25, 0.25, 0.75], id="reversed-x"),
        pytest.param([0.25, 0.75, 0.75, 0.25], id="reversed-y"),
    ],
)
def test_face_crop_undoes_letterbox(box: list[float]) -> None:
    image = Image.new("RGB", (400, 200))
    head = face_crop(image, np.asarray(box, dtype=np.float32))
    assert head.size == (240, 200)


def test_invalid_face_box_falls_back_to_animal() -> None:
    image = Image.new("RGB", (100, 100))
    assert face_crop(image, np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)) is image


def test_hash_preserves_gradient_direction() -> None:
    gradient = Image.fromarray(np.tile(np.arange(9, dtype=np.uint8), (8, 1)))
    assert (
        int(difference_hash(gradient)),
        int(difference_hash(gradient.transpose(Image.Transpose.FLIP_LEFT_RIGHT))),
    ) == (2**64 - 1, 0)
