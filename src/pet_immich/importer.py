from __future__ import annotations

import csv
import json
import pickle
import sqlite3
from pathlib import Path

from pet_immich.data import UNCERTAIN, UNKNOWN, Sample


def import_legacy(database: Path, labels_path: Path, metadata_path: Path) -> list[Sample]:
    labels: dict[str, str] = json.loads(labels_path.read_text())
    with metadata_path.open() as stream:
        metadata = {row["id"]: row for row in csv.DictReader(stream)}
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT asset_id, data FROM crops ORDER BY asset_id").fetchall()
    samples = []
    for asset_id, data in rows:
        # The upstream detector stores trusted local crop records as pickles.
        boxes = [box for box, _ in pickle.loads(data)]
        for index, box in enumerate(boxes):
            crop_id = f"{asset_id}:{index}"
            label = labels.get(crop_id)
            if label is None and (len(boxes) == 1 or labels.get(asset_id) in {"Neither", "Unsure"}):
                label = labels.get(asset_id)
            label = {"Neither": UNKNOWN, "Unsure": UNCERTAIN, "Both": None}.get(label, label)
            samples.append(
                Sample(crop_id, asset_id, tuple(box), metadata.get(asset_id, {}).get("fileCreatedAt", ""), label)
            )
    return samples


__all__ = ["import_legacy"]
