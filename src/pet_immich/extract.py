"""Reuse local embeddings only while their extraction models remain unchanged."""

import hashlib
import io
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Final, cast
from urllib.parse import urlencode

import httpx
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from pet_immich.data import Features, digest, normalize

if TYPE_CHECKING:
    from pathlib import Path

    from ultralytics.engine.results import Results

    from pet_immich.data import Matrix, Sample


_CHECKPOINT_SIZE: Final = 25


def extract(
    samples: list[Sample],
    directory: Path,
    tagger_url: str,
    *,
    refresh: bool = False,
    workers: int = 4,
    models_directory: Path | None = None,
) -> Features:
    """Checkpoint by crop ID so interrupted backfills can resume without repeating inference."""
    models = models_directory or directory / "models"
    signature = hashlib.sha256(
        "|".join(digest(models / name) for name in ("encoder.onnx", "segment.pt", "localizer.tflite")).encode()
    ).hexdigest()
    cache = directory / "features.npz"
    existing = Features.load(cache) if cache.exists() and not refresh else None
    if existing is not None and existing.signature != signature:
        msg = "Feature models changed; use --refresh to rebuild the feature cache"
        raise ValueError(msg)
    stored = {value: index for index, value in enumerate(existing.ids)} if existing is not None else {}
    pending = [sample for sample in samples if sample.id not in stored]
    if not pending and existing is not None:
        return existing
    if workers > 1 and len(pending) > _CHECKPOINT_SIZE:
        return parallel_extract(pending, directory, tagger_url, models, existing, workers, refresh=refresh)
    return _serial_extract(pending, cache, tagger_url, models, existing, signature)


def _serial_extract(
    pending: list[Sample], cache: Path, tagger_url: str, models: Path, existing: Features | None, signature: str
) -> Features:
    extractor = _Extractor(models)
    ids = list(existing.ids) if existing is not None else []
    raw = list(existing.raw) if existing is not None else []
    body = list(existing.body) if existing is not None else []
    faces = list(existing.face) if existing is not None else []
    hashes = list(existing.hashes) if existing is not None else []
    valid = list(existing.valid) if existing is not None else []
    with httpx.Client(timeout=60) as client:
        for index, sample in enumerate(pending, 1):
            query = urlencode({"bbox": ",".join(map(str, sample.box))})
            response = client.get(f"{tagger_url}/api/crop/{sample.asset_id}?{query}")
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            vectors, single = extractor.views(image)
            ids.append(sample.id)
            raw.append(vectors[0])
            body.append(vectors[1])
            faces.append(vectors[2])
            hashes.append(difference_hash(image))
            valid.append(single)
            if index % _CHECKPOINT_SIZE == 0 or index == len(pending):
                Features(
                    ids,
                    np.asarray(raw),
                    np.asarray(body),
                    np.asarray(faces),
                    np.asarray(hashes, dtype=np.uint64),
                    np.asarray(valid),
                    signature,
                ).save(cache)
                sys.stdout.write(f"Extracted {index}/{len(pending)} crops\n")
                sys.stdout.flush()
    return Features.load(cache)


def parallel_extract(
    samples: list[Sample],
    directory: Path,
    tagger_url: str,
    models: Path,
    existing: Features | None,
    workers: int,
    *,
    refresh: bool,
) -> Features:
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = [
            pool.submit(
                extract,
                samples[index::workers],
                directory / "workers" / str(index),
                tagger_url,
                workers=1,
                models_directory=models,
                refresh=refresh,
            )
            for index in range(workers)
        ]
        for job in as_completed(jobs):
            result = job.result()
            if existing is None:
                existing = result
            else:
                indices = np.asarray(
                    [index for index, key in enumerate(result.ids) if key not in set(existing.ids)], dtype=np.int64
                )
                existing = Features(
                    existing.ids + [result.ids[index] for index in indices],
                    np.concatenate((existing.raw, result.raw[indices])),
                    np.concatenate((existing.body, result.body[indices])),
                    np.concatenate((existing.face, result.face[indices])),
                    np.concatenate((existing.hashes, result.hashes[indices])),
                    np.concatenate((existing.valid, result.valid[indices])),
                    result.signature,
                )
            existing.save(directory / "features.npz")
    return Features.load(directory / "features.npz")


class _Extractor:
    def __init__(self, models: Path) -> None:
        from ai_edge_litert.interpreter import Interpreter  # ruff: ignore[import-outside-top-level] - optional runtime.
        from ultralytics import YOLO  # ruff: ignore[import-outside-top-level] - optional model.

        options = ort.SessionOptions()  # ty: ignore[possibly-missing-attribute] - native ONNX export.
        options.intra_op_num_threads = 2
        self.encoder = ort.InferenceSession(
            str(models / "encoder.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.segmenter = YOLO(str(models / "segment.pt"))
        self.localizer = Interpreter(model_path=str(models / "localizer.tflite"), num_threads=2)
        self.localizer.allocate_tensors()

    def views(self, image: Image.Image) -> tuple[Matrix, bool]:
        detection = cast(
            "list[Results]", self.segmenter.predict(image, classes=[16], conf=0.25, verbose=False, device="cpu")
        )[0]
        masks = detection.masks
        single = masks is not None and len(masks) == 1
        foreground = image
        if masks is not None and single:
            mask = Image.fromarray((masks.data[0].cpu().numpy() * 255).astype(np.uint8))
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
            foreground = Image.composite(image, Image.new("RGB", image.size, (127, 127, 127)), mask)
        square = ImageOps.pad(image, (224, 224), color=(0, 0, 0))
        self.localizer.set_tensor(
            self.localizer.get_input_details()[0]["index"], (np.asarray(square, dtype=np.float32) / 255)[None]
        )
        self.localizer.invoke()
        box = self.localizer.get_tensor(self.localizer.get_output_details()[0]["index"])[0]
        head = face_crop(image, box)
        vectors = encode(self.encoder, [image, foreground, head])
        return vectors, single


def face_crop(image: Image.Image, box: Matrix) -> Image.Image:
    """Undo letterboxing before cropping; keep the animal when localization is unusable."""
    scale = 224 / max(image.size)
    offset = np.asarray([(224 - image.width * scale) / 2, (224 - image.height * scale) / 2] * 2)
    coordinates = (np.asarray(box) * 224 - offset) / scale
    coordinates += np.asarray([-1, -1, 1, 1]) * max(coordinates[2:] - coordinates[:2]) * 0.1
    coordinates = np.clip(coordinates, 0, [image.width, image.height] * 2)
    minimum_head_pixels: Final = 16
    if not np.isfinite(coordinates).all() or np.min(coordinates[2:] - coordinates[:2]) < minimum_head_pixels:
        return image
    return image.crop(tuple(coordinates))


def encode(session: ort.InferenceSession, images: list[Image.Image]) -> Matrix:
    """Use the encoder training normalization for every animal view."""
    pixels = np.stack([
        np.asarray(image.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32) / 255 for image in images
    ])
    tensors = ((pixels - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(0, 3, 1, 2).astype(np.float32)
    return normalize(np.asarray(session.run(None, {session.get_inputs()[0].name: tensors})[0], dtype=np.float32))


def difference_hash(image: Image.Image) -> np.uint64:
    """Detect near-duplicate captures without storing image pixels in the feature cache."""
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return np.uint64(int.from_bytes(np.packbits(pixels[:, 1:] > pixels[:, :-1]).tobytes(), "big"))


__all__ = ["difference_hash", "encode", "extract", "face_crop"]
