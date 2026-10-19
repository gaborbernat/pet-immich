# pet-immich

Train a local pet identity model from confirmed crop labels, validate it across capture events, and add confident
assignments to Immich. Training and inference are separate commands: a daily run uses a frozen model and cannot relax
its thresholds or retrain on its own predictions. Each ensemble member keeps the fitted classifier that its threshold
was calibrated against; promotion does not refit it. Inference requires at least two agreeing members and no opposing
vote.

The feature extractor combines whole-animal images, segmented foregrounds, and localized dog faces. A supervised
contrastive projection learns identity features from the labeled examples. Nested grouped validation compares this with
frozen visual features and selects the classifier and rejection thresholds inside each training fold. Capture events and
near-identical images stay in the same fold. Unknown animals participate in training; ambiguous labels do not become
identity examples.

A candidate passes the deployment gate only when each identity reaches 95% precision both per crop and with capture
events weighted equally, across at least ten accepted events. The report includes coverage, recall, unknown false
accepts, and a one-sided precision confidence bound. Passing the point-estimate gate does not establish a 95% population
precision lower bound. Reusing an existing labeled dataset for development also limits how independent its evaluation
can be.

## Local installation

```sh
uv sync --python 3.14 --extra vision --extra dev
```

Keep the state directory outside this checkout. It holds photos if cached by a detector, labels, asset identifiers,
embeddings, model weights, reports, and the application audit. None belongs in Git. The repository ignores common photo,
video, dataset, credential, and model formats. Tests use synthetic data.

Set `IMMICH_URL`, `IMMICH_API_KEY`, and `PET_TAGGER_URL` in your service environment. The API key needs asset search and
face read/create permissions. Set `IMMICH_READ_API_KEY` separately when search permissions live on a read-only key. The
crop provider is the local [immich-pet-tagger](https://github.com/tedornitier/immich-pet-tagger) API. Fetching a crop
uses that provider's read access; this project sends photos only to the configured local services.

Place these weights in the private state directory's `models/` subdirectory:

- `encoder.onnx`: [Open Noodle pet-recognition-base](https://huggingface.co/open-noodle/pet-recognition-base), Apache
  2.0.
- `segment.pt`: [Ultralytics YOLO11s segmentation](https://docs.ultralytics.com/models/yolo11/), AGPL-3.0 or commercial
  license.
- `localizer.tflite`: [DogFLW face localizer](https://huggingface.co/hugocornellier/dog-face-landmarks), CC BY-NC 4.0.

The face localizer's weights permit noncommercial use. Obtain suitable permissions or replace those weights for
commercial use. Model weights are downloaded separately and never committed here.

## Training and backfill

The legacy importer accepts the detector's SQLite crop cache, a labels file, and a metadata CSV with `id` and
`fileCreatedAt` columns. Only import a trusted local cache: its upstream format contains Python pickles.

```sh
pet-immich import --state "$PET_STATE" --database "$CROP_DATABASE" --labels "$LABELS" --metadata "$METADATA"
pet-immich extract --state "$PET_STATE" --labeled-only
pet-immich train --state "$PET_STATE"
pet-immich extract --state "$PET_STATE"
pet-immich plan --state "$PET_STATE"
pet-immich apply --state "$PET_STATE"
```

`Both` labels on a photograph cannot identify the individual crops. The importer keeps explicit crop labels and does not
assign the photograph's identity to multiple detections. `Neither` trains the unknown class; `Unsure` stays untagged.

Create `identities.json` inside the private state directory, mapping training labels to existing Immich person IDs. A
failed training gate saves a candidate and its report but never replaces the approved model. Feature-model hashes
prevent using an approved classifier with different extraction weights.

Application checks existing faces before each write, preserves conflicting assignments, verifies each creation, and
saves a private audit. Repeating a completed run does not create duplicate faces. Conflicts appear in `plan.json` for
diagnosis.

## Daily operation

Run `pet-immich sync --state "$PET_STATE"` from a scheduler with the service environment above. It pages through the
image library, detects assets it has not scanned, extracts missing features, and applies the approved model. Discovery
uses asset IDs rather than capture dates so an old photograph uploaded today is still processed. Failed runs retain
their progress.

For a user systemd service, link the installed entry point to `~/.local/bin/pet-immich`, place the environment values in
`~/.local/state/pet-immich/service.env` with mode `0600`, and install the units in `deploy/` under
`~/.config/systemd/user/`. Enable `immich-pet-autotag.timer` after validating the model. The service skips execution
until an approved model exists. Add a local mount condition if the image service depends on a removable disk.

## Development

```sh
tox -e test,type,lint
```

Tests enforce 100% line and branch coverage across application and test code, including subprocess entry points. Ruff
enables all rules with documented formatter and test conventions; ty treats diagnostics and warnings as errors. CI runs
the same checks on Linux and Windows with Python 3.14. Every direct dependency has a minimum version; `uv.lock` records
the tested resolution.

Inference tests generate images and a small ONNX encoder locally. They mock network calls and native model boundaries;
no personal photos, labels, credentials, or pretrained weights are needed in CI.

Research behind the approach:

- [PetFace](https://arxiv.org/abs/2407.13555): individual animal recognition and verification.
- [Calibrated similarity and graph clustering](https://arxiv.org/abs/2608.02469): segmentation, local/global features,
  and unknown rejection.
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html): fitting confidence estimates on
  predictions from separate folds.
