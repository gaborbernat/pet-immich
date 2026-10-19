import hashlib
import io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import urlencode

import httpx
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from pet_immich.data import Features, digest, normalize

if TYPE_CHECKING:
    from pet_immich.data import Matrix, Sample


def extract(
    samples: list[Sample],
    directory: Path,
    tagger_url: str,
    *,
    refresh: bool = False,
    workers: int = 4,
    models_directory: Path | None = None,
) -> Features:
    from ai_edge_litert.interpreter import Interpreter
    from ultralytics import YOLO
    from ultralytics.engine.results import Results

    models = models_directory or directory / "models"
    signature = hashlib.sha256(
        "|".join(digest(models / name) for name in ("encoder.onnx", "segment.pt", "localizer.tflite")).encode()
    ).hexdigest()
    cache = directory / "features.npz"
    existing = Features.load(cache) if cache.exists() and not refresh else None
    if existing is not None and existing.signature != signature:
        raise ValueError("Feature models changed; use --refresh to rebuild the feature cache")
    stored = {value: index for index, value in enumerate(existing.ids)} if existing is not None else {}
    pending = [sample for sample in samples if sample.id not in stored]
    if not pending and existing is not None:
        return existing
    if workers > 1 and len(pending) > 25:
        return parallel_extract(pending, directory, tagger_url, models, existing, workers, refresh=refresh)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    encoder = ort.InferenceSession(
        str(models / "encoder.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
    )
    segmenter = YOLO(str(models / "segment.pt"))
    localizer = Interpreter(model_path=str(models / "localizer.tflite"), num_threads=2)
    localizer.allocate_tensors()
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
            detection = cast(
                list[Results], segmenter.predict(image, classes=[16], conf=0.25, verbose=False, device="cpu")
            )[0]
            masks = detection.masks
            single = masks is not None and len(masks) == 1
            foreground = image
            if masks is not None and single:
                mask = Image.fromarray((masks.data[0].cpu().numpy() * 255).astype(np.uint8))
                mask = mask.resize(image.size, Image.Resampling.NEAREST)
                foreground = Image.composite(image, Image.new("RGB", image.size, (127, 127, 127)), mask)
            square = ImageOps.pad(image, (224, 224), color=(0, 0, 0))
            localizer.set_tensor(
                localizer.get_input_details()[0]["index"], (np.asarray(square, dtype=np.float32) / 255)[None]
            )
            localizer.invoke()
            box = localizer.get_tensor(localizer.get_output_details()[0]["index"])[0]
            head = face_crop(image, box)
            vectors = encode(encoder, [image, foreground, head])
            ids.append(sample.id)
            raw.append(vectors[0])
            body.append(vectors[1])
            faces.append(vectors[2])
            hashes.append(difference_hash(image))
            valid.append(single)
            if index % 25 == 0 or index == len(pending):
                Features(
                    ids,
                    np.asarray(raw),
                    np.asarray(body),
                    np.asarray(faces),
                    np.asarray(hashes, dtype=np.uint64),
                    np.asarray(valid),
                    signature,
                ).save(cache)
                print(f"Extracted {index}/{len(pending)} crops", flush=True)
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


def face_crop(image: Image.Image, box: Matrix) -> Image.Image:
    scale = 224 / max(image.size)
    offset = np.asarray([(224 - image.width * scale) / 2, (224 - image.height * scale) / 2] * 2)
    coordinates = (np.asarray(box) * 224 - offset) / scale
    coordinates += np.asarray([-1, -1, 1, 1]) * max(coordinates[2:] - coordinates[:2]) * 0.1
    coordinates = np.clip(coordinates, 0, [image.width, image.height] * 2)
    if not np.isfinite(coordinates).all() or np.min(coordinates[2:] - coordinates[:2]) < 16:
        return image
    return image.crop(tuple(coordinates))


def encode(session: ort.InferenceSession, images: list[Image.Image]) -> Matrix:
    pixels = np.stack(
        [np.asarray(image.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32) / 255 for image in images]
    )
    tensors = ((pixels - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(0, 3, 1, 2).astype(np.float32)
    return normalize(np.asarray(session.run(None, {session.get_inputs()[0].name: tensors})[0], dtype=np.float32))


def difference_hash(image: Image.Image) -> np.uint64:
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return np.uint64(int.from_bytes(np.packbits(pixels[:, 1:] > pixels[:, :-1]).tobytes(), "big"))


__all__ = ["difference_hash", "encode", "extract", "face_crop"]
