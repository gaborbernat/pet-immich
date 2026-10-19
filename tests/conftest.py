import io
import json
from typing import TYPE_CHECKING

import httpx
import joblib
import numpy as np
import onnx
import pytest
import timm
import torch
from ai_edge_litert.interpreter import Interpreter
from onnx import TensorProto, helper
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from pet_immich.data import Features, Sample, save_samples
from pet_immich.model import train

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


from pet_immich.data import UNKNOWN, normalize


@pytest.fixture
def population() -> tuple[list[Sample], Features]:
    random = np.random.default_rng(120)
    samples = [
        Sample(f"sample-{index}", f"asset-{index}", (0, 0, 1, 1), label=name)
        for index, name in enumerate(["pet-a"] * 36 + ["pet-b"] * 36 + [UNKNOWN] * 36)
    ]
    vectors = normalize(
        np.repeat(np.eye(3), 36, axis=0).astype(np.float32) + random.normal(0, 0.03, (108, 3)).astype(np.float32)
    )
    features = Features(
        [sample.id for sample in samples],
        vectors,
        vectors,
        vectors,
        random.integers(0, 2**64 - 1, 108, dtype=np.uint64),
        np.ones(108, dtype=bool),
        "synthetic",
    )
    return samples, features


@pytest.fixture
def extraction_state(tmp_path: Path, mocker: MockerFixture, client_type: type[httpx.Client]) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["input"], ["embedding"], axes=[2, 3], keepdims=0)],
        "synthetic-encoder",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, 224, 224])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, 3])],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), models / "encoder.onnx"
    )
    (models / "segment.pt").write_bytes(b"synthetic-segmentation-boundary")
    (models / "localizer.tflite").write_bytes(b"synthetic-localizer-boundary")
    segmenter = mocker.create_autospec(YOLO, instance=True)
    segmenter.predict.return_value = [
        Results(np.zeros((16, 16, 3), dtype=np.uint8), "synthetic", {16: "dog"}, masks=torch.ones(1, 16, 16))
    ]
    mocker.patch("ultralytics.YOLO", return_value=segmenter)
    localizer = mocker.create_autospec(Interpreter, instance=True)
    localizer.get_input_details.return_value = [{"index": 0}]
    localizer.get_output_details.return_value = [{"index": 1}]
    localizer.get_tensor.return_value = np.asarray([[0.1, 0.1, 0.9, 0.9]], dtype=np.float32)
    mocker.patch("ai_edge_litert.interpreter.Interpreter", return_value=localizer)
    pixels = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(pixels, format="PNG")
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=pixels.getvalue()))
    mocker.patch("pet_immich.extract.httpx.Client", return_value=client_type(transport=transport))
    return tmp_path


@pytest.fixture
def inference_state(tmp_path: Path, population: tuple[list[Sample], Features]) -> Path:
    samples, features = population
    model, _ = train(samples, features)
    joblib.dump(model, tmp_path / "approved.joblib")
    save_samples(tmp_path / "samples.json", samples)
    features.save(tmp_path / "features.npz")
    (tmp_path / "identities.json").write_text(json.dumps({"pet-a": "identity-a", "pet-b": "identity-b"}))
    return tmp_path


@pytest.fixture
def client_type() -> type[httpx.Client]:
    return httpx.Client


@pytest.fixture
def miewid_weights(tmp_path: Path, mocker: MockerFixture) -> Path:
    backbone = timm.create_model("efficientnetv2_rw_m", pretrained=False, num_classes=0)
    weights = {f"backbone.{name}": value for name, value in backbone.state_dict().items()}
    weights["backbone.global_pool.p"] = torch.ones(1) * 3
    weights.update({f"bn.{name}": value for name, value in torch.nn.BatchNorm1d(2152).state_dict().items()})
    weights["final.weight"] = torch.zeros(10, 2152)
    destination = tmp_path / "encoder.safetensors"
    destination.write_bytes(b"synthetic-miewid-checkpoint")
    mocker.patch("safetensors.torch.load_file", return_value=weights)
    return destination
