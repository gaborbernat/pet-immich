import json
import pickle  # ruff: ignore[suspicious-pickle-import] - creates the upstream cache format for the importer test.
import sqlite3
from contextlib import closing
from typing import TYPE_CHECKING

from pet_immich.data import UNKNOWN
from pet_immich.importer import import_legacy

if TYPE_CHECKING:
    from pathlib import Path


def test_import_keeps_crop_labels_without_guessing_pairs(tmp_path: Path) -> None:
    with closing(sqlite3.connect(tmp_path / "crops.db")) as database:
        database.execute("CREATE TABLE crops (asset_id TEXT, data BLOB)")
        database.executemany(
            "INSERT INTO crops VALUES (?, ?)",
            [
                ("one", pickle.dumps([([0, 0, 1, 1], [])])),
                ("two", pickle.dumps([([0, 0, 0.4, 1], []), ([0.6, 0, 1, 1], [])])),
                ("other", pickle.dumps([([0, 0, 1, 1], [])])),
            ],
        )
        database.commit()
    (tmp_path / "labels.json").write_text(
        json.dumps({"one": "pet-a", "two": "Both", "two:1": "pet-b", "other": "Neither"})
    )
    (tmp_path / "metadata.csv").write_text("id,fileCreatedAt\none,2020-01-01T00:00:00Z\n")
    rows = import_legacy(tmp_path / "crops.db", tmp_path / "labels.json", tmp_path / "metadata.csv")
    assert {sample.id: sample.label for sample in rows} == {
        "one:0": "pet-a",
        "two:0": None,
        "two:1": "pet-b",
        "other:0": UNKNOWN,
    }
