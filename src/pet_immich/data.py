"""Preserve crop identities and extraction provenance across restarts."""

import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pathlib import Path

UNKNOWN: Final = "__unknown__"
UNCERTAIN: Final = "__uncertain__"
type Matrix = NDArray[np.float32]


def load_samples(path: Path) -> list[Sample]:
    """Restore confirmed labels without deriving identities from filenames."""
    return [Sample(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


def save_samples(path: Path, samples: list[Sample]) -> None:
    """Keep labels outside model artifacts so corrections survive retraining."""
    private_write(path, json.dumps([asdict(sample) for sample in samples]).encode())


def private_write(path: Path, content: bytes) -> None:
    """Replace checkpoints atomically without exposing private data to other users."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as output:
        output.write(content)
    temporary.replace(path)


def digest(path: Path) -> str:
    """Bind cached embeddings to the exact model bytes that produced them."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class Sample:
    """Keep crop geometry with its identity to prevent labels drifting between detections."""

    id: str
    asset_id: str
    box: tuple[float, float, float, float]
    captured: str = ""
    label: str | None = None


@dataclass
class Features:
    """Keep feature rows aligned with crop IDs and extractor provenance."""

    ids: list[str]
    raw: Matrix
    body: Matrix
    face: Matrix
    hashes: NDArray[np.uint64]
    valid: NDArray[np.bool_]
    signature: str

    def matrix(self, mode: str) -> Matrix:
        """Normalize concatenated views so feature count does not change their scale."""
        if mode == "raw":
            return self.raw
        if mode == "body":
            return self.body
        if mode == "face":
            return self.face
        if mode in {"fused", "metric", "rbf10", "linear"}:
            return normalize(np.concatenate((self.raw, self.body, self.face), axis=1))
        msg = f"Unknown feature mode: {mode}"
        raise ValueError(msg)

    def save(self, path: Path) -> None:
        """Avoid pickle when storing arrays from external inference engines."""
        stream = io.BytesIO()
        np.savez_compressed(stream, **asdict(self))
        private_write(path, stream.getvalue())

    @classmethod
    def load(cls, path: Path) -> Features:
        """Reject object arrays rather than executing pickled cache contents."""
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
    """Keep zero vectors finite while making cosine matching independent of magnitude."""
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


__all__ = [
    "UNCERTAIN",
    "UNKNOWN",
    "Features",
    "Matrix",
    "Sample",
    "digest",
    "load_samples",
    "normalize",
    "private_write",
    "save_samples",
]
