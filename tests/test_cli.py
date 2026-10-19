import json
import pickle  # ruff: ignore[suspicious-pickle-import] - constructs trusted legacy importer fixtures.
import sqlite3
import subprocess  # ruff: ignore[suspicious-subprocess-import] - verifies the public CLI without invoking a shell.
import sys
from contextlib import closing
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
import pytest
from torch.jit import TracerWarning
from transformers import BertConfig, BertModel, BitImageProcessor, Dinov2Config, Dinov2Model

from pet_immich.cli import main
from pet_immich.data import Features, Sample, load_samples, save_samples
from pet_immich.encoder import export_encoder

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "arguments",
    [[], ["import"], ["export-encoder"], ["compare"]],
    ids=["missing-command", "missing-import-paths", "missing-source", "missing-similarities"],
)
def test_cli_requires_inputs(arguments: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["pet-immich", *arguments, "--state", str(tmp_path)])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2


def test_cli_trains_and_promotes(inference_state: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["pet-immich", "train", "--state", str(inference_state)])
    main()
    assert (inference_state / "approved.joblib").read_bytes() == (inference_state / "candidate.joblib").read_bytes()
    assert json.loads((inference_state / "validation.json").read_text())["passed"]


@pytest.mark.parametrize("aligned", [True, False], ids=["aligned", "reordered"])
def test_cli_compares_only_aligned_channels(
    inference_state: Path, monkeypatch: pytest.MonkeyPatch, *, aligned: bool
) -> None:
    features = Features.load(inference_state / "features.npz")
    similarities = inference_state / "similarities.npz"
    np.savez(similarities, ids=features.ids if aligned else features.ids[::-1], raw=features.raw @ features.raw.T)
    monkeypatch.setattr(
        "sys.argv", ["pet-immich", "compare", "--state", str(inference_state), "--similarities", str(similarities)]
    )
    if aligned:
        main()
        assert json.loads((inference_state / "comparison.json").read_text())["raw@cost=5"]["unique_photos"] == 72
    else:
        with pytest.raises(ValueError, match="Similarity IDs"):
            main()


def test_cli_failed_training_preserves_approved_model(
    inference_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = (inference_state / "approved.joblib").read_bytes()
    features = Features.load(inference_state / "features.npz")
    features.raw[:] = 1
    features.body[:] = 1
    features.face[:] = 1
    features.save(inference_state / "features.npz")
    monkeypatch.setattr("sys.argv", ["pet-immich", "train", "--state", str(inference_state)])
    with pytest.raises(SystemExit, match="not promoted"):
        main()
    assert (inference_state / "approved.joblib").read_bytes() == previous


@pytest.mark.parametrize("existing", ["new", "changed", "matching"])
def test_cli_imports_confirmed_crops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str) -> None:
    with closing(sqlite3.connect(tmp_path / "crops.db")) as connection:
        connection.execute("CREATE TABLE crops (asset_id TEXT, data BLOB)")
        connection.execute("INSERT INTO crops VALUES (?, ?)", ("asset", pickle.dumps([([0, 0, 1, 1], [])])))
        connection.commit()
    (tmp_path / "labels.json").write_text('{"asset": "pet-a"}')
    (tmp_path / "metadata.csv").write_text("id,fileCreatedAt\nasset,2020-01-01T00:00:00Z\n")
    if existing != "new":
        save_samples(
            tmp_path / "samples.json",
            [Sample("asset:0", "asset", (0, 0, 0.5, 0.5) if existing == "changed" else (0, 0, 1, 1))],
        )
    monkeypatch.setattr(
        "sys.argv",
        [
            "pet-immich",
            "import",
            "--state",
            str(tmp_path),
            "--database",
            str(tmp_path / "crops.db"),
            "--labels",
            str(tmp_path / "labels.json"),
            "--metadata",
            str(tmp_path / "metadata.csv"),
        ],
    )
    if existing == "changed":
        with pytest.raises(ValueError, match="geometry changed"):
            main()
        assert load_samples(tmp_path / "samples.json")[0].box == [0, 0, 0.5, 0.5]
    else:
        main()
        assert load_samples(tmp_path / "samples.json")[0].label == "pet-a"


@pytest.mark.parametrize("labeled_only", [True, False], ids=["confirmed", "all"])
def test_cli_extracts_only_labeled_crops(
    extraction_state: Path, monkeypatch: pytest.MonkeyPatch, *, labeled_only: bool
) -> None:
    save_samples(
        extraction_state / "samples.json",
        [
            Sample("labeled", "one", (0, 0, 1, 1), label="pet-a"),
            Sample("unlabeled", "two", (0, 0, 1, 1)),
        ],
    )
    monkeypatch.setenv("PET_TAGGER_URL", "http://127.0.0.1")
    monkeypatch.setattr(
        "sys.argv",
        ["pet-immich", "extract", "--state", str(extraction_state), *(["--labeled-only"] if labeled_only else [])],
    )
    main()
    assert Features.load(extraction_state / "features.npz").ids == (
        ["labeled"] if labeled_only else ["labeled", "unlabeled"]
    )


def test_cli_plan_requires_approved_model(inference_state: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (inference_state / "approved.joblib").unlink()
    monkeypatch.setattr("sys.argv", ["pet-immich", "plan", "--state", str(inference_state)])
    with pytest.raises(FileNotFoundError, match=r"approved\.joblib"):
        main()


def test_cli_module_entry_point() -> None:
    result = subprocess.run([sys.executable, "-m", "pet_immich.cli", "--help"], check=True, capture_output=True)
    assert b"export-encoder" in result.stdout


@pytest.fixture
def encoder_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    Dinov2Model(
        Dinov2Config(hidden_size=12, num_hidden_layers=1, num_attention_heads=3, image_size=224)
    ).save_pretrained(source)
    BitImageProcessor(image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]).save_pretrained(source)
    return source


def test_cli_exports_compatible_encoder(encoder_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "pet-immich",
            "export-encoder",
            "--state",
            str(tmp_path),
            "--source",
            str(encoder_source),
        ],
    )
    # The legacy exporter traces fixed spatial dimensions; the batch dimension stays dynamic.
    with pytest.warns(
        (DeprecationWarning, TracerWarning),
        match="legacy TorchScript-based ONNX export|feature will be removed|Converting a tensor to a Python boolean",
    ):
        main()
    session = ort.InferenceSession(str(tmp_path / "models" / "encoder.onnx"))
    result = session.run(None, {"input": np.zeros((2, 3, 224, 224), dtype=np.float32)})
    assert isinstance(result[0], np.ndarray)
    assert result[0].shape == (2, 12)
    assert np.isfinite(result[0]).all()


@pytest.mark.parametrize("failure", ["exists", "normalization", "architecture"])
def test_encoder_rejects_incompatible_sources(encoder_source: Path, tmp_path: Path, failure: str) -> None:
    destination = tmp_path / "encoder.onnx"
    if failure == "exists":
        destination.write_bytes(b"keep-model")
        with pytest.raises(FileExistsError):
            export_encoder(str(encoder_source), destination)
        assert destination.read_bytes() == b"keep-model"
    elif failure == "normalization":
        BitImageProcessor(image_mean=[0, 0, 0]).save_pretrained(encoder_source)
        with pytest.raises(ValueError, match="normalization"):
            export_encoder(str(encoder_source), destination)
        assert not destination.exists()
    else:
        BertModel(BertConfig(hidden_size=12, num_hidden_layers=1, num_attention_heads=3)).save_pretrained(
            encoder_source
        )
        with pytest.raises(ValueError, match="DINOv2"):
            export_encoder(str(encoder_source), destination)
        assert not destination.exists()


def test_cli_training_rejects_multiple_operating_points(inference_state: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["pet-immich", "train", "--state", str(inference_state), "--error-cost", "2", "5"])
    with pytest.raises(ValueError, match="exactly one"):
        main()


def test_cli_keeps_each_requested_operating_point(inference_state: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    features = Features.load(inference_state / "features.npz")
    similarities = inference_state / "similarities.npz"
    np.savez(similarities, ids=features.ids, raw=features.raw @ features.raw.T)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pet-immich",
            "compare",
            "--state",
            str(inference_state),
            "--similarities",
            str(similarities),
            "--error-cost",
            "2",
            "20",
        ],
    )
    main()
    reports = json.loads((inference_state / "comparison.json").read_text())
    assert {name: report["error_cost"] for name, report in reports.items()} == {"raw@cost=2": 2, "raw@cost=20": 20}
