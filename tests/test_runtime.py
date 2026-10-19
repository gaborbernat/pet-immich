from __future__ import annotations

import json
from pathlib import Path

import httpx
import joblib
import pytest
from pytest_mock import MockerFixture

from pet_immich.data import Features, Sample, load_samples, save_samples
from pet_immich.model import train
from pet_immich.runtime import Assignment, apply_assignments, discover, face_action, run


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
    faces: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content)
            faces.append(
                {
                    "id": "face",
                    "imageWidth": 10000,
                    "imageHeight": 10000,
                    "boundingBoxX1": body["x"],
                    "boundingBoxX2": body["x"] + body["width"],
                    "boundingBoxY1": body["y"],
                    "boundingBoxY2": body["y"] + body["height"],
                    "person": {"id": body["personId"]},
                }
            )
            return httpx.Response(204)
        return httpx.Response(200, json=faces)

    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        plan = [Assignment(Sample("crop", "asset", (0, 0, 1, 1)), "pet-a", "create")]
        apply_assignments(client, plan, tmp_path, "version")
        apply_assignments(client, plan, tmp_path, "version")
    assert len(faces) == len(json.loads((tmp_path / "applied.json").read_text())) == 1


def test_failed_readback_does_not_record_success(tmp_path: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={} if request.method == "POST" else [])

    with (
        httpx.Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client,
        pytest.raises(RuntimeError, match="did not retain"),
    ):
        apply_assignments(
            client, [Assignment(Sample("crop", "asset", (0, 0, 1, 1)), "pet-a", "create")], tmp_path, "version"
        )
    assert not (tmp_path / "applied.json").exists()


def test_discovery_handles_old_photos_uploaded_later(tmp_path: Path) -> None:
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

    with original(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        from unittest.mock import patch

        with patch(
            "pet_immich.runtime.httpx.Client",
            return_value=original(transport=httpx.MockTransport(respond), base_url="http://tagger"),
        ):
            discover(tmp_path, client, "http://tagger")
    assert [sample.asset_id for sample in load_samples(tmp_path / "samples.json")] == ["asset-1", "asset-2"]


@pytest.fixture
def inference_state(tmp_path: Path, population: tuple[list[Sample], Features]) -> Path:
    samples, features = population
    model, _ = train(samples, features)
    joblib.dump(model, tmp_path / "approved.joblib")
    save_samples(tmp_path / "samples.json", samples)
    features.save(tmp_path / "features.npz")
    (tmp_path / "identities.json").write_text(json.dumps({"pet-a": "identity-a", "pet-b": "identity-b"}))
    return tmp_path


def test_plan_preserves_confirmed_labels(
    inference_state: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    monkeypatch.setenv("IMMICH_URL", "http://test")
    monkeypatch.setenv("IMMICH_API_KEY", "synthetic")
    mocker.patch(
        "pet_immich.runtime.httpx.Client",
        return_value=httpx.Client(base_url="http://test", transport=httpx.MockTransport(respond)),
    )
    run(inference_state, apply=False, sync=False)
    plan = json.loads((inference_state / "plan.json").read_text())
    assert {item["person"] for item in plan} == {"identity-a", "identity-b"}
    assert all(item["action"] == "create" for item in plan)


def test_unapproved_model_cannot_write(inference_state: Path) -> None:
    model = joblib.load(inference_state / "approved.joblib")
    model.approved = False
    joblib.dump(model, inference_state / "approved.joblib")
    with pytest.raises(ValueError, match="not passed"):
        run(inference_state, apply=True, sync=False)
