"""Separate model approval from scheduled inference."""

import argparse
import io
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np

from pet_immich.benchmark import compare
from pet_immich.data import Features, load_samples, private_write, save_samples
from pet_immich.encoder import export_encoder
from pet_immich.extract import extract
from pet_immich.importer import import_legacy
from pet_immich.model import train
from pet_immich.runtime import run


def main() -> None:
    """Keep training and application explicit so scheduled runs cannot retrain themselves."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["import", "extract", "train", "compare", "plan", "apply", "sync", "export-encoder"]
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--labeled-only", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--similarities", type=Path)
    parser.add_argument("--duplicates", action="store_true")
    parser.add_argument("--error-cost", type=float, nargs="+", default=[5.0])
    parser.add_argument("--training-fraction", type=float, default=1.0)
    args = parser.parse_args()
    os.umask(0o077)
    args.state.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.command == "compare" and args.similarities is None:
        parser.error("compare requires --similarities")
    if args.command == "export-encoder":
        if not args.source:
            parser.error("export-encoder requires --source")
        export_encoder(args.source, args.state / "models" / "encoder.onnx")
        return
    samples_path = args.state / "samples.json"
    if args.command == "import":
        if not all((args.database, args.labels, args.metadata)):
            parser.error("import requires --database, --labels, and --metadata")
        _import(args)
        return
    samples = load_samples(samples_path)
    if args.command == "extract":
        if args.labeled_only:
            samples = [sample for sample in samples if sample.label is not None]
        extract(samples, args.state, os.environ["PET_TAGGER_URL"], refresh=args.refresh)
    elif args.command == "compare":
        _compare(args)
    elif args.command == "train":
        _train(args.state, args.error_cost, args.training_fraction)
    else:
        run(args.state, apply=args.command in {"apply", "sync", "export-encoder"}, sync=args.command == "sync")


def _train(directory: Path, error_costs: list[float], training_fraction: float) -> None:
    if len(error_costs) != 1:
        msg = "train requires exactly one error cost"
        raise ValueError(msg)
    model, report = train(
        load_samples(directory / "samples.json"),
        Features.load(directory / "features.npz"),
        error_cost=error_costs[0],
        training_fraction=training_fraction,
    )
    stream = io.BytesIO()
    joblib.dump(model, stream)
    private_write(directory / "candidate.joblib", stream.getvalue())
    private_write(directory / "validation.json", json.dumps(report, indent=2).encode())
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    if model.approved:
        private_write(directory / "approved.joblib", stream.getvalue())
    elif training_fraction == 1:
        msg = "Validation failed; candidate was not promoted"
        raise SystemExit(msg)


def _compare(args: argparse.Namespace) -> None:
    features = Features.load(args.state / "features.npz")
    with np.load(args.similarities, allow_pickle=False) as archive:
        if archive["ids"].tolist() != features.ids:
            msg = "Similarity IDs must match the feature cache in order"
            raise ValueError(msg)
        comparison = {
            f"{name}@cost={cost:g}": report
            for cost in args.error_cost
            for name, report in compare(
                load_samples(args.state / "samples.json"),
                features,
                {name: archive[name] for name in archive.files if name != "ids"},
                group_duplicates=args.duplicates,
                error_cost=cost,
            ).items()
        }
    private_write(args.state / "comparison.json", json.dumps(comparison, indent=2).encode())
    sys.stdout.write(json.dumps(comparison, indent=2) + "\n")


def _import(args: argparse.Namespace) -> None:
    samples_path = args.state / "samples.json"
    imported = import_legacy(args.database, args.labels, args.metadata)
    if samples_path.exists():
        boxes = {sample.id: tuple(sample.box) for sample in load_samples(samples_path)}
        if any(sample.id in boxes and boxes[sample.id] != tuple(sample.box) for sample in imported):
            msg = "Crop geometry changed; use a new state directory to rebuild features"
            raise ValueError(msg)
    save_samples(samples_path, imported)


__all__ = ["main"]

if __name__ == "__main__":
    main()
