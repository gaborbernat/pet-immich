"""Cache learned local matches so composition experiments can reuse expensive inference."""

import hashlib
import io
import sys
import warnings
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import httpx
import numpy as np
import torch
from PIL import Image

from pet_immich.data import private_write

if TYPE_CHECKING:
    from pathlib import Path

    from pet_immich.data import Features, Matrix, Sample


def match_candidates(
    samples: list[Sample], features: Features, directory: Path, tagger_url: str, limit: int = 20
) -> Matrix:
    """Missing pairs remain NaN; a computed zero means the matcher found no correspondences."""
    if [sample.id for sample in samples] != features.ids or limit < 1:
        msg = "Matching requires aligned samples and a positive candidate limit"
        raise ValueError(msg)
    matcher = _Matcher()
    signature = hashlib.sha256((matcher.signature + features.signature + str(limit)).encode()).hexdigest()
    path = directory / "local-matches.npz"
    scores = np.full((len(samples), len(samples)), np.nan, dtype=np.float32)
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["signature"]) != signature or archive["ids"].tolist() != features.ids:
                msg = "Local match cache changed; use a separate experiment directory"
                raise ValueError(msg)
            scores = archive["scores"]
    similarity = features.raw @ features.raw.T
    np.fill_diagonal(similarity, -np.inf)
    candidates = np.argsort(-similarity, axis=1)[:, : min(limit, len(samples) - 1)]
    pairs = sorted({
        tuple(sorted((index, int(other)))) for index, others in enumerate(candidates.tolist()) for other in others
    })
    with httpx.Client(timeout=60) as client:
        for index, (first, second) in enumerate(pairs, 1):
            if np.isnan(scores[first, second]):
                left = matcher.describe(samples[first], directory, client, tagger_url)
                right = matcher.describe(samples[second], directory, client, tagger_url)
                scores[first, second] = scores[second, first] = matcher.score(left, right)
            if index % 100 == 0 or index == len(pairs):
                stream = io.BytesIO()
                np.savez_compressed(stream, ids=features.ids, scores=scores, signature=signature)
                private_write(path, stream.getvalue())
                sys.stdout.write(f"Matched {index}/{len(pairs)} pairs\n")
                sys.stdout.flush()
    return scores


class _Matcher:
    def __init__(self) -> None:
        # Kornia still scripts operators at import time; torch warns about Python 3.14 even for eager inference.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="`torch.jit.script` is not supported in Python 3.14", category=FutureWarning
            )
            from kornia.feature import DISK, LightGlue  # ruff: ignore[import-outside-top-level] - optional inference.

        self.detector = DISK.from_pretrained("depth").eval()
        self.matcher = LightGlue("disk", depth_confidence=-1, width_confidence=-1).eval()
        checksum = hashlib.sha256(b"disk-512px-128points-lightglue-v1")
        for model in (self.detector, self.matcher):
            for name, value in model.state_dict().items():
                checksum.update(name.encode())
                checksum.update(value.cpu().numpy().tobytes())
        self.signature = checksum.hexdigest()

    def describe(
        self, sample: Sample, directory: Path, client: httpx.Client, tagger_url: str
    ) -> dict[str, torch.Tensor]:
        key = hashlib.sha256(f"{self.signature}|{sample.asset_id}|{sample.box}".encode()).hexdigest()
        path = directory / "local-descriptors" / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                return {name: torch.from_numpy(archive[name]) for name in archive.files}
        query = urlencode({"bbox": ",".join(map(str, sample.box))})
        response = client.get(f"{tagger_url}/api/crop/{sample.asset_id}?{query}")
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.BILINEAR)
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255)[None]
        with torch.inference_mode():
            features = self.detector(pixels, n=128, pad_if_not_divisible=True)[0]
        values: dict[str, torch.Tensor] = {
            "keypoints": features.keypoints[None],
            "descriptors": features.descriptors[None],
            "image_size": torch.tensor([image.size]),
        }
        stream = io.BytesIO()
        np.savez_compressed(stream, **{name: value.numpy() for name, value in values.items()})
        private_write(path, stream.getvalue())
        return values

    def score(self, first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> float:
        count = min(first["keypoints"].shape[1], second["keypoints"].shape[1])
        if not count:
            return 0
        with torch.inference_mode():
            result = self.matcher({"image0": first, "image1": second})
        return float(result["scores"][0].sum()) / count


__all__ = ["match_candidates"]
