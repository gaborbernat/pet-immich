from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

UNKNOWN: Final = "__unknown__"
UNCERTAIN: Final = "__uncertain__"
type Matrix = NDArray[np.float32]


def load_samples(path: Path) -> list[Sample]:
    return [Sample(**row) for row in json.loads(path.read_text())]


def save_samples(path: Path, samples: list[Sample]) -> None:
    private_write(path, json.dumps([asdict(sample) for sample in samples]).encode())


def private_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as output:
        output.write(content)
    temporary.replace(path)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class Sample:
    id: str
    asset_id: str
    box: tuple[float, float, float, float]
    captured: str = ""
    label: str | None = None


@dataclass
class Features:
    ids: list[str]
    raw: Matrix
    body: Matrix
    face: Matrix
    hashes: NDArray[np.uint64]
    valid: NDArray[np.bool_]
    signature: str

    def matrix(self, mode: str) -> Matrix:
        if mode == "raw":
            return self.raw
        if mode == "body":
            return self.body
        if mode == "face":
            return self.face
        if mode in {"fused", "metric", "rbf10", "linear"}:
            return normalize(np.concatenate((self.raw, self.body, self.face), axis=1))
        raise ValueError(f"Unknown feature mode: {mode}")

    def save(self, path: Path) -> None:
        import io

        stream = io.BytesIO()
        np.savez_compressed(stream, **asdict(self))
        private_write(path, stream.getvalue())

    @classmethod
    def load(cls, path: Path) -> Features:
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                archive["ids"].tolist(),
                archive["raw"],
                archive["body"],
                archive["face"],
                archive["hashes"],
                archive["valid"],
                str(archive["signature"]),
            )


def normalize(values: Matrix) -> Matrix:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


__all__ = [
    "UNKNOWN",
    "UNCERTAIN",
    "Features",
    "Matrix",
    "Sample",
    "digest",
    "load_samples",
    "normalize",
    "private_write",
    "save_samples",
]
