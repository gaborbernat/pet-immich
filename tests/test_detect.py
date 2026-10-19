import json
from typing import TYPE_CHECKING

import httpx
import numpy as np
import pytest
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from pet_immich.data import load_samples
from pet_immich.detect import scan_missing

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def test_detection_adds_missing_crops_without_duplicate_scales(detection_state: tuple[Path, httpx.Client]) -> None:
    directory, client = detection_state
    scan_missing(directory, client, "http://127.0.0.1", directory / "models/detector.pt")
    samples = load_samples(directory / "samples.json")
    assert (len(samples), [sample.box for sample in samples if sample.asset_id == "photo-1"]) == (
        51,
        [[0.125, 0.125, 0.375, 0.375], [0.625, 0.625, 0.875, 0.875]],
    )


def test_detection_resume_preserves_existing_geometry(detection_state: tuple[Path, httpx.Client]) -> None:
    directory, client = detection_state
    scan_missing(directory, client, "http://127.0.0.1", directory / "models/detector.pt")
    samples = load_samples(directory / "samples.json")
    scan_missing(directory, client, "http://127.0.0.1", directory / "models/detector.pt")
    assert load_samples(directory / "samples.json") == samples


def test_detection_rejects_changed_weights(detection_state: tuple[Path, httpx.Client]) -> None:
    directory, client = detection_state
    scan_missing(directory, client, "http://127.0.0.1", directory / "models/detector.pt")
    (directory / "models/detector.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Detector changed"):
        scan_missing(directory, client, "http://127.0.0.1", directory / "models/detector.pt")


@pytest.mark.parametrize("boxes", [None, torch.empty((0, 6))], ids=["no-boxes", "empty-boxes"])
def test_detection_records_photos_without_dogs(
    detection_state: tuple[Path, httpx.Client],
    mocker: MockerFixture,
    boxes: torch.Tensor | None,
) -> None:
    directory, client = detection_state
    model = mocker.create_autospec(YOLO, instance=True)
    model.predict.return_value = [Results(np.zeros((64, 64, 3), dtype=np.uint8), "synthetic", {16: "dog"}, boxes=boxes)]
    mocker.patch("ultralytics.YOLO", return_value=model)
    scan_missing(directory, client, "http://127.0.0.1", directory / "models/detector.pt")
    assert [sample.id for sample in load_samples(directory / "samples.json")] == ["existing"]


def test_detection_records_unreadable_previews(
    detection_state: tuple[Path, httpx.Client], mocker: MockerFixture, client_type: type[httpx.Client]
) -> None:
    directory, client = detection_state
    mocker.patch(
        "pet_immich.detect.httpx.Client",
        return_value=client_type(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"invalid"))),
    )
    scan_missing(directory, client, "http://127.0.0.1", directory / "models/detector.pt")
    assert set(json.loads((directory / "detection-failures.json").read_text()).values()) == {"unreadable preview"}
