"""Recover animal crops missed by the primary detector without replacing existing geometry."""

import hashlib
import io
import json
import sys
from typing import TYPE_CHECKING, Final, cast

import httpx
import numpy as np
from PIL import Image, UnidentifiedImageError

from pet_immich.data import Sample, digest, load_samples, private_write, save_samples
from pet_immich.network import local_request

if TYPE_CHECKING:
    from pathlib import Path

    from ultralytics.engine.results import Results


def scan_missing(directory: Path, client: httpx.Client, tagger_url: str, weights: Path) -> None:
    """Checkpoint separately from discovery so a second detector can revisit missed photos."""
    signature = hashlib.sha256(f"{digest(weights)}|640,1280|0.15|0.5".encode()).hexdigest()
    checkpoint = directory / "detection.json"
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {"signature": signature, "seen": []}
    if state["signature"] != signature:
        msg = "Detector changed; use a separate detection state"
        raise ValueError(msg)
    samples = load_samples(directory / "samples.json")
    seen = set(state["seen"]) | {sample.asset_id for sample in samples}
    failed = {}
    detector = _Detector(weights)
    page = 1
    with httpx.Client(trust_env=False, event_hooks={"request": [local_request]}, timeout=60) as provider:
        while True:
            response = client.post("/api/search/metadata", json={"type": "IMAGE", "size": 1000, "page": page})
            response.raise_for_status()
            assets = response.json()["assets"]
            for asset in assets["items"]:
                if asset["id"] in seen:
                    continue
                response = provider.get(f"{tagger_url}/api/crop/{asset['id']}")
                response.raise_for_status()
                try:
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                except UnidentifiedImageError:
                    failed[asset["id"]] = "unreadable preview"
                    private_write(directory / "detection-failures.json", json.dumps(failed).encode())
                    continue
                for box in detector.boxes(image):
                    key = hashlib.sha256(f"{signature}|{asset['id']}|{box}".encode()).hexdigest()[:16]
                    samples.append(Sample(f"{asset['id']}:detect-{key}", asset["id"], box, asset["fileCreatedAt"]))
                seen.add(asset["id"])
                if len(seen) % 25 == 0:
                    _checkpoint(directory, samples, signature, seen)
            _checkpoint(directory, samples, signature, seen)
            if not assets.get("nextPage"):
                return
            page = int(assets["nextPage"])


def _checkpoint(directory: Path, samples: list[Sample], signature: str, seen: set[str]) -> None:
    save_samples(directory / "samples.json", samples)
    private_write(directory / "detection.json", json.dumps({"signature": signature, "seen": sorted(seen)}).encode())
    sys.stdout.write(f"Second detector: {len(seen)} photos scanned or cached, {len(samples)} crops\n")
    sys.stdout.flush()


class _Detector:
    def __init__(self, weights: Path) -> None:
        from ultralytics import YOLO  # ruff: ignore[import-outside-top-level] - optional native detector.

        self.model = YOLO(str(weights))

    def boxes(self, image: Image.Image) -> list[tuple[float, float, float, float]]:
        candidates = []
        for size in (640, 1280):
            result = cast(
                "list[Results]",
                self.model.predict(image, imgsz=size, classes=[16], conf=0.15, verbose=False, device="cpu"),
            )[0]
            boxes = result.boxes
            if boxes is not None:
                candidates.extend(zip(boxes.conf.tolist(), boxes.xyxyn.tolist(), strict=True))
        accepted: list[tuple[float, float, float, float]] = []
        for _, box in sorted(candidates, reverse=True):
            if accepted:
                existing = np.asarray(accepted)
                overlap = np.maximum(0, np.minimum(existing[:, 2:], box[2:]) - np.maximum(existing[:, :2], box[:2]))
                intersection = overlap.prod(axis=1)
                area = (box[2] - box[0]) * (box[3] - box[1])
                union = (existing[:, 2:] - existing[:, :2]).prod(axis=1) + area - intersection
                minimum_overlap: Final = 0.5
                if np.any(intersection / union >= minimum_overlap):
                    continue
            accepted.append(tuple(box))
        return accepted


__all__ = ["scan_missing"]
