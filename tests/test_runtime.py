import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import httpx
import joblib
import numpy as np
import pytest

from pet_immich.data import Features, Sample, digest, load_samples, save_samples
from pet_immich.runtime import Assignment, Face, apply_assignments, discover, face_action, run

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


@pytest.mark.parametrize(("person", "expected"), [("pet-a", "exists"), ("pet-b", "conflict")])
def test_existing_face_protects_assignment(person: str, expected: str) -> None:
    assert (
        face_action(
            Sample("crop", "asset", (0, 0, 1, 1)),
            "pet-a",
            [
                {
                    "imageWidth": 100,
                    "imageHeight": 100,
                    "boundingBoxX1": 10,
                    "boundingBoxX2": 90,
                    "boundingBoxY1": 10,
                    "boundingBoxY2": 90,
                    "person": {"id": person},
                }
            ],
        )
        == expected
    )


def test_distinct_animals_can_share_photo() -> None:
    assert (
        face_action(
            Sample("right", "asset", (0.6, 0, 1, 1)),
            "pet-b",
            [
                {
                    "imageWidth": 100,
                    "imageHeight": 100,
                    "boundingBoxX1": 10,
                    "boundingBoxX2": 40,
                    "boundingBoxY1": 10,
                    "boundingBoxY2": 90,
                    "person": {"id": "pet-a"},
                }
            ],
        )
        == "create"
    )


def test_application_is_idempotent(tmp_path: Path) -> None:
    faces: list[Face] = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content)
            faces.append({
                "id": "face",
                "imageWidth": 10000,
                "imageHeight": 10000,
                "boundingBoxX1": body["x"],
                "boundingBoxX2": body["x"] + body["width"],
                "boundingBoxY1": body["y"],
                "boundingBoxY2": body["y"] + body["height"],
                "person": {"id": body["personId"]},
            })
            return httpx.Response(204)
        return httpx.Response(200, json=faces)

    with httpx.Client(base_url="http://127.0.0.1", transport=httpx.MockTransport(respond)) as client:
        plan = [Assignment(Sample("crop", "asset", (0, 0, 1, 1)), "pet-a", "create")]
        apply_assignments(client, plan, tmp_path, "version")
        apply_assignments(client, plan, tmp_path, "version")
    assert len(faces) == len(json.loads((tmp_path / "applied.json").read_text())) == 1


def test_failed_readback_does_not_record_success(tmp_path: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={} if request.method == "POST" else [])

    with (
        httpx.Client(base_url="http://127.0.0.1", transport=httpx.MockTransport(respond)) as client,
        pytest.raises(RuntimeError, match="did not retain"),
    ):
        apply_assignments(
            client, [Assignment(Sample("crop", "asset", (0, 0, 1, 1)), "pet-a", "create")], tmp_path, "version"
        )
    assert not (tmp_path / "applied.json").exists()


def test_discovery_handles_old_photos_uploaded_later(tmp_path: Path, mocker: MockerFixture) -> None:
    save_samples(tmp_path / "samples.json", [])
    original = httpx.Client

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/search/metadata":
            page = json.loads(request.content)["page"]
            return httpx.Response(
                200,
                json={
                    "assets": {
                        "items": [{"id": f"asset-{page}", "fileCreatedAt": "2000-01-01T00:00:00Z"}],
                        "nextPage": "2" if page == 1 else None,
                    }
                },
            )
        return httpx.Response(200, json={"crops": [[0, 0, 1, 1]]})

    with original(base_url="http://127.0.0.1", transport=httpx.MockTransport(respond)) as client:
        mocker.patch(
            "pet_immich.runtime.httpx.Client",
            return_value=original(transport=httpx.MockTransport(respond), base_url="http://tagger"),
        )
        discover(tmp_path, client, "http://tagger")
    assert [sample.asset_id for sample in load_samples(tmp_path / "samples.json")] == ["asset-1", "asset-2"]


def test_plan_preserves_confirmed_labels(
    inference_state: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    monkeypatch.setenv("IMMICH_URL", "http://127.0.0.1")
    monkeypatch.setenv("IMMICH_API_KEY", "synthetic")
    mocker.patch(
        "pet_immich.runtime.httpx.Client",
        return_value=httpx.Client(base_url="http://127.0.0.1", transport=httpx.MockTransport(respond)),
    )
    samples = load_samples(inference_state / "samples.json")
    samples[0] = replace(samples[0], label="pet-b")
    save_samples(inference_state / "samples.json", samples)
    run(inference_state, apply=False, sync=False)
    plan = json.loads((inference_state / "plan.json").read_text())
    assert (
        samples[0].id in {item["sample"]["id"] for item in plan},
        {item["person"] for item in plan},
    ) == (False, {"identity-a", "identity-b"})


def test_unapproved_model_cannot_write(inference_state: Path) -> None:
    model = joblib.load(inference_state / "approved.joblib")
    model.approved = False
    joblib.dump(model, inference_state / "approved.joblib")
    with pytest.raises(ValueError, match="not passed"):
        run(inference_state, apply=True, sync=False)


@pytest.mark.parametrize("failure", ["signature", "identity"], ids=["changed-encoder", "unmapped-person"])
def test_runtime_refuses_incompatible_state(
    inference_state: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    monkeypatch.setenv("IMMICH_URL", "http://127.0.0.1")
    monkeypatch.setenv("IMMICH_API_KEY", "synthetic")
    if failure == "signature":
        features = Features.load(inference_state / "features.npz")
        features.signature = "different"
        features.save(inference_state / "features.npz")
    else:
        (inference_state / "identities.json").write_text("{}")
    with pytest.raises(ValueError, match=r"differs|mapping"):
        run(inference_state, apply=True, sync=False)
    assert not (inference_state / "applied.json").exists()


@pytest.mark.usefixtures("extraction_state")
def test_runtime_sync_uses_cache_and_applies_new_assignments(
    inference_state: Path,
    client_type: type[httpx.Client],
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    signature = hashlib.sha256(
        (
            "views-v2|"
            + "|".join(
                digest(inference_state / "models" / name) for name in ("encoder.onnx", "segment.pt", "localizer.tflite")
            )
        ).encode()
    ).hexdigest()
    features = Features.load(inference_state / "features.npz")
    features.signature = signature
    for field in (features.raw, features.body, features.face):
        field[:] = np.repeat(np.eye(3, dtype=np.float32), 36, axis=0)
    features.save(inference_state / "features.npz")
    model = joblib.load(inference_state / "approved.joblib")
    model.signature = signature
    joblib.dump(model, inference_state / "approved.joblib")
    faces: dict[str, list[Face]] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/search/metadata":
            return httpx.Response(200, json={"assets": {"items": [{"id": "asset-0"}], "nextPage": None}})
        if request.method == "POST":
            body = json.loads(request.content)
            faces[body["assetId"]] = [
                {
                    "id": body["assetId"],
                    "imageWidth": 10000,
                    "imageHeight": 10000,
                    "boundingBoxX1": body["x"],
                    "boundingBoxX2": body["x"] + body["width"],
                    "boundingBoxY1": body["y"],
                    "boundingBoxY2": body["y"] + body["height"],
                    "person": {"id": body["personId"]},
                }
            ]
            return httpx.Response(204)
        return httpx.Response(200, json=faces.get(request.url.params["id"], []))

    monkeypatch.setenv("IMMICH_URL", "http://127.0.0.1")
    monkeypatch.setenv("IMMICH_API_KEY", "synthetic")
    monkeypatch.setenv("PET_TAGGER_URL", "http://tagger")
    original = client_type
    mocker.patch(
        "httpx.Client",
        side_effect=lambda *args, **kwargs: original(*args, **kwargs, transport=httpx.MockTransport(respond)),
    )
    run(inference_state, apply=True, sync=True)
    assert len(json.loads((inference_state / "applied.json").read_text())) == 72
