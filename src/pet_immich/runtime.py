"""Preserve confirmed assignments while applying a validated local identity model."""

import hashlib
import io
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

import httpx
import joblib
from filelock import FileLock

from pet_immich.data import UNCERTAIN, UNKNOWN, Features, Sample, load_samples, private_write, save_samples
from pet_immich.extract import extract

if TYPE_CHECKING:
    from pathlib import Path

    from pet_immich.model import IdentityModel


def run(directory: Path, *, apply: bool, sync: bool) -> None:
    """Require approval and matching feature provenance before planning any writes."""
    with FileLock(directory / "run.lock", timeout=0):
        artifact = (directory / "approved.joblib").read_bytes()
        model = cast("IdentityModel", joblib.load(io.BytesIO(artifact)))
        if not model.approved:
            msg = "Model has not passed validation"
            raise ValueError(msg)
        identities = cast("dict[str, str]", json.loads((directory / "identities.json").read_text()))
        with httpx.Client(
            base_url=os.environ["IMMICH_URL"].rstrip("/"),
            headers={"x-api-key": os.environ["IMMICH_API_KEY"]},
            timeout=60,
        ) as client:
            if sync:
                with httpx.Client(
                    base_url=os.environ["IMMICH_URL"].rstrip("/"),
                    timeout=60,
                    headers={"x-api-key": os.environ.get("IMMICH_READ_API_KEY", os.environ["IMMICH_API_KEY"])},
                ) as reader:
                    discover(directory, reader, os.environ["PET_TAGGER_URL"])
                extract(load_samples(directory / "samples.json"), directory, os.environ["PET_TAGGER_URL"])
            samples = load_samples(directory / "samples.json")
            features = Features.load(directory / "features.npz")
            if model.signature != features.signature:
                msg = "Feature extractor differs from the validated model"
                raise ValueError(msg)
            rows = {key: index for index, key in enumerate(features.ids)}
            predictions = model.predict(features.matrix(model.mode))
            planned = []
            for sample in samples:
                if sample.id not in rows or sample.label in {UNCERTAIN, UNKNOWN}:
                    continue
                index = rows[sample.id]
                name = str(predictions[index])
                if not name or not features.valid[index] or sample.label not in {None, name}:
                    continue
                if name not in identities:
                    msg = "Missing identity mapping for a predicted class"
                    raise ValueError(msg)
                faces = fetch_faces(client, sample.asset_id)
                planned.append(Assignment(sample, identities[name], face_action(sample, identities[name], faces)))
            private_write(directory / "plan.json", json.dumps([asdict(item) for item in planned]).encode())
            sys.stdout.write(json.dumps(dict(Counter(item.action for item in planned))) + "\n")
            if apply:
                apply_assignments(client, planned, directory, hashlib.sha256(artifact).hexdigest())


def discover(directory: Path, client: httpx.Client, tagger_url: str) -> None:
    """Scan by asset ID so late uploads of old photographs are not missed."""
    seen_path = directory / "scanned.json"
    seen = set(json.loads(seen_path.read_text())) if seen_path.exists() else set()
    samples = load_samples(directory / "samples.json")
    seen.update(sample.asset_id for sample in samples)
    page = 1
    with httpx.Client(base_url=tagger_url, timeout=120) as tagger:
        while True:
            response = client.post("/api/search/metadata", json={"type": "IMAGE", "size": 1000, "page": page})
            response.raise_for_status()
            assets = response.json()["assets"]
            for asset in assets["items"]:
                if asset["id"] in seen:
                    continue
                detection = tagger.get(f"/api/asset/{asset['id']}/crops")
                detection.raise_for_status()
                samples.extend(
                    Sample(f"{asset['id']}:{index}", asset["id"], tuple(box), asset["fileCreatedAt"])
                    for index, box in enumerate(detection.json()["crops"])
                )
                seen.add(asset["id"])
            save_samples(directory / "samples.json", samples)
            private_write(seen_path, json.dumps(sorted(seen)).encode())
            sys.stdout.write(f"Discovery page {page}: {len(seen)} scanned assets, {len(samples)} cached crops\n")
            sys.stdout.flush()
            if not assets.get("nextPage"):
                return
            page = int(assets["nextPage"])


def fetch_faces(client: httpx.Client, asset_id: str) -> list[Face]:
    response = client.get("/api/faces", params={"id": asset_id})
    response.raise_for_status()
    return cast("list[Face]", response.json())


def face_action(sample: Sample, person: str, faces: list[Face]) -> str:
    """Protect existing faces before adding another identity to the same crop."""
    for face in faces:
        horizontal = (face["boundingBoxX1"] + face["boundingBoxX2"]) / (2 * face["imageWidth"])
        vertical = (face["boundingBoxY1"] + face["boundingBoxY2"]) / (2 * face["imageHeight"])
        if sample.box[0] <= horizontal <= sample.box[2] and sample.box[1] <= vertical <= sample.box[3]:
            if (face.get("person") or {}).get("id") == person:
                return "exists"
            return "conflict"
    return "create"


def apply_assignments(client: httpx.Client, planned: list[Assignment], directory: Path, version: str) -> None:
    """Verify server state after each write so retries do not duplicate faces."""
    audit_path = directory / "applied.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else []
    for assignment in planned:
        sample = assignment.sample
        if (
            assignment.action != "create"
            or face_action(sample, assignment.person, fetch_faces(client, sample.asset_id)) != "create"
        ):
            continue
        left, top, right, bottom = sample.box
        response = client.post(
            "/api/faces",
            json={
                "assetId": sample.asset_id,
                "personId": assignment.person,
                "imageWidth": 10000,
                "imageHeight": 10000,
                "x": int(left * 10000),
                "y": int(top * 10000),
                "width": max(1, int((right - left) * 10000)),
                "height": max(1, int((bottom - top) * 10000)),
            },
        )
        response.raise_for_status()
        faces = fetch_faces(client, sample.asset_id)
        if face_action(sample, assignment.person, faces) != "exists":
            msg = "Immich did not retain the requested face assignment"
            raise RuntimeError(msg)
        audit.append({
            "sample": asdict(sample),
            "person": assignment.person,
            "model": version,
            "face_ids": [face["id"] for face in faces if face_action(sample, assignment.person, [face]) == "exists"],
            "verified": True,
        })
        private_write(audit_path, json.dumps(audit).encode())


@dataclass(frozen=True)
class Assignment:
    """Keep the intended person with the crop for read-back verification."""

    sample: Sample
    person: str
    action: str


class Face(TypedDict):
    """Represent Immich's pixel-space boxes without assuming a person is assigned."""

    id: NotRequired[str]
    imageWidth: int
    imageHeight: int
    boundingBoxX1: int
    boundingBoxX2: int
    boundingBoxY1: int
    boundingBoxY2: int
    person: dict[str, str] | None


__all__ = ["Assignment", "Face", "apply_assignments", "discover", "face_action", "run"]
