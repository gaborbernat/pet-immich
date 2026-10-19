from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import joblib

from pet_immich.data import Features, load_samples, private_write, save_samples
from pet_immich.extract import extract
from pet_immich.importer import import_legacy
from pet_immich.model import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["import", "extract", "train", "plan", "apply", "sync", "export-encoder"])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--labeled-only", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    args.state.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.command == "export-encoder":
        from pet_immich.encoder import export_encoder

        if not args.source:
            parser.error("export-encoder requires --source")
        export_encoder(args.source, args.state / "models" / "encoder.onnx")
        return
    samples_path = args.state / "samples.json"
    if args.command == "import":
        if not all((args.database, args.labels, args.metadata)):
            parser.error("import requires --database, --labels, and --metadata")
        imported = import_legacy(args.database, args.labels, args.metadata)
        if samples_path.exists():
            boxes = {sample.id: tuple(sample.box) for sample in load_samples(samples_path)}
            if any(sample.id in boxes and boxes[sample.id] != tuple(sample.box) for sample in imported):
                raise ValueError("Crop geometry changed; use a new state directory to rebuild features")
        save_samples(samples_path, imported)
        return
    samples = load_samples(samples_path)
    if args.command == "extract":
        if args.labeled_only:
            samples = [sample for sample in samples if sample.label is not None]
        extract(samples, args.state, os.environ["PET_TAGGER_URL"], refresh=args.refresh)
    elif args.command == "train":
        model, report = train(samples, Features.load(args.state / "features.npz"))
        stream = io.BytesIO()
        joblib.dump(model, stream)
        private_write(args.state / "candidate.joblib", stream.getvalue())
        private_write(args.state / "validation.json", json.dumps(report, indent=2).encode())
        print(json.dumps(report, indent=2))
        if model.approved:
            private_write(args.state / "approved.joblib", stream.getvalue())
        else:
            raise SystemExit("Validation failed; candidate was not promoted")
    else:
        from pet_immich.runtime import run

        run(args.state, apply=args.command in {"apply", "sync", "export-encoder"}, sync=args.command == "sync")


__all__ = ["main"]

if __name__ == "__main__":
    main()
