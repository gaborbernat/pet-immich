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

The objective is to recover more photos of each pet while limiting incorrect assignments. Calibration maximizes
`correct - error_cost * incorrect`, with capture events weighted equally. `--error-cost` expresses how many correct
recoveries one incorrect assignment cancels; the default is five. Compare several costs before choosing a model.
Training selects features and thresholds inside each outer fold. A candidate can replace the approved model only when
both identities have positive event-weighted utility across at least ten accepted events. This is an evidence check, not
a guarantee of future precision.

Reports include correct, incorrect, known, and missed crop counts for each identity, precision, recall, and confidence
bounds. Recall measures recovery of confirmed labeled crops; it cannot measure animals the detector missed or establish
whole-library recall. Reusing labels during development and selecting among many experiments limits independence.

## Local installation

```sh
tox run -e dev
source .tox/dev/bin/activate
```

Keep the state directory outside this checkout. It holds photos if cached by a detector, labels, asset identifiers,
embeddings, model weights, reports, and the application audit. None belongs in Git. The repository ignores common photo,
video, dataset, credential, and model formats. Tests use synthetic data.

Set `IMMICH_URL`, `IMMICH_API_KEY`, and `PET_TAGGER_URL` in your service environment. The API key needs asset search and
face read/create permissions. Set `IMMICH_READ_API_KEY` separately when search permissions live on a read-only key. The
crop provider is the local [immich-pet-tagger](https://github.com/tedornitier/immich-pet-tagger) API. Fetching a crop
uses that provider's read access. Photo API requests accept only literal loopback or private LAN addresses and
`localhost`; public endpoints and other hostnames fail before sending a request. These clients ignore environment
proxies. Photos, crops, embeddings, and labels stay local. External model downloads are allowed; uploading private data
for remote inference or training is not.

Place these weights in the private state directory's `models/` subdirectory:

- `encoder.onnx`: [Open Noodle pet-recognition-base](https://huggingface.co/open-noodle/pet-recognition-base), Apache
  2.0.
- `segment.pt`: [Ultralytics YOLO11s segmentation](https://docs.ultralytics.com/models/yolo11/), AGPL-3.0 or commercial
  license.
- `localizer.tflite`: [DogFLW face localizer](https://huggingface.co/hugocornellier/dog-face-landmarks), CC BY-NC 4.0.

The face localizer's weights permit noncommercial use. Obtain suitable permissions or replace those weights for
commercial use. Model weights are downloaded separately and never committed here.

For encoder comparisons, use separate private state directories with identical samples. The larger Open Noodle model
uses the same ONNX interface. To evaluate [MiewID-msv3](https://huggingface.co/conservationxlabs/miewid-msv3), put its
`model.safetensors` at `models/encoder.safetensors` and omit `encoder.onnx`. The loader reconstructs the EfficientNetV2
backbone, GeM pooling, and batch normalization from the published architecture and requires matching inference weights.
It does not execute model-repository Python. MiewID uses 440-pixel bilinear resizing; Open Noodle uses 224-pixel bicubic
resizing. Both use their published ImageNet channel normalization.

## Training and backfill

The legacy importer accepts the detector's SQLite crop cache, a labels file, and a metadata CSV with `id` and
`fileCreatedAt` columns. Only import a trusted local cache: its upstream format contains Python pickles.

```sh
pet-immich import --state "$PET_STATE" --database "$CROP_DATABASE" --labels "$LABELS" --metadata "$METADATA"
pet-immich extract --state "$PET_STATE" --labeled-only
pet-immich train --state "$PET_STATE" --error-cost 5
pet-immich extract --state "$PET_STATE"
pet-immich plan --state "$PET_STATE"
pet-immich apply --state "$PET_STATE"
```

`Both` labels on a photograph cannot identify the individual crops. The importer keeps explicit crop labels and does not
assign the photograph's identity to multiple detections. `Neither` trains the unknown class; `Unsure` stays untagged.

Create `identities.json` inside the private state directory, mapping training labels to existing Immich person IDs. A
failed training gate saves a candidate and its report but never replaces the approved model. Feature-model hashes and
the preprocessing revision prevent using an approved classifier with different weights or crop handling. After either
changes, extract into a separate state directory and retrain before replacing the production cache and classifier.

Face localization orders the predicted corners and reverses the localizer's integer letterbox padding. A failed second
segmentation pass keeps the original detector crop eligible; multiple segmentation masks still prevent automatic
assignment. Missing segmentation and an ambiguous identity are different failures.

Application checks existing faces before each write, preserves conflicting assignments, verifies each creation, and
saves a private audit. Repeating a completed run does not create duplicate faces. Conflicts appear in `plan.json` for
diagnosis.

## Comparing compositions

`pet-immich compare --state "$EXPERIMENT_STATE" --similarities "$PAIR_SCORES"` evaluates every nonempty combination of
the supplied similarity channels. The private NPZ input contains `ids`, matching `features.npz` in order, and one square
pair-score array per channel. Use cosine similarities for frozen encoder embeddings and match scores for local
descriptors; mark unavailable pairs with NaN. Only frozen, label-independent channels belong in this input. A matcher
trained on these labels must produce its scores separately inside each training fold.

Each combination learns pair-score weights on training events, selects confidence and runner-up margin thresholds on
separate calibration events, and predicts held-out events. Unknown animals supply negative pairs. Reference scores
average the strongest three events per identity, so one long burst cannot occupy all reference positions. The report
counts ineligible extraction results in recall and counts accepted photos once.

Calibration maximizes event-weighted recoveries minus the selected error cost across at least five accepted events. A
small number of calibration mistakes no longer forces rejection of an otherwise useful threshold. Compare operating
points on the same folds:

```sh
pet-immich compare --state "$EXPERIMENT_STATE" --similarities "$PAIR_SCORES" --duplicates --error-cost 2 5 10 20
```

Each result records its error cost. `passed` means positive event-weighted utility for each identity across at least ten
accepted events; compare the actual recovery and error counts rather than ranking configurations by this flag.

`comparison.json` reports each composition on the same folds. It does not promote a model. Choosing the best result
after inspecting this table is exploratory; a deployment claim needs a reserved evaluation set or nested selection of
the composition itself. Increasing the channel count evaluates exponentially more combinations.

Add `--duplicates` to compare each composition with and without direct duplicate propagation. This requires both high
embedding similarity and a close perceptual hash, excludes other crops in the same photo, and rejects conflicting
anchors. Propagated labels cannot become new anchors. Validation hides every evaluation label before propagation;
production callers can supply confirmed labels so corrections take precedence. Capture timestamps only prevent split
leakage; they cannot establish that two crops show the same dog.

`pet_immich.matching.match_candidates` computes a local similarity channel from a global nearest-neighbor shortlist. It
uses pretrained DISK descriptors and LightGlue correspondences, caches descriptors and symmetric pair scores, and binds
the cache to the loaded weights, preprocessing, global encoder signature, candidate limit, and crop order. Uncomputed
pairs remain NaN. Use a separate experiment directory when any of these inputs changes. This implementation uses frozen
pretrained matching weights; it does not claim to reproduce WildMatch's identity-specific matcher training.

For a deployment comparison, build candidate lists using each fold's reference gallery. A shortlist constructed from the
entire evaluation collection is transductive: held-out images can occupy candidate positions even without labels. That
experiment can identify promising combinations, but is not an independent deployment evaluation.

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
tox run -e 3.14,fix,type,pkg_meta
```

Tests enforce 100% line and branch coverage across application and test code, including subprocess entry points. Ruff
enables all rules with documented formatter and test conventions; ty treats diagnostics and warnings as errors. CI runs
the same tox environments on Linux and Windows with Python 3.14. Every direct dependency has a minimum version.

`tox run -e fix` runs the formatting, spelling, workflow, and import-layer checks through prek. The `type` environment
runs ty through a manual prek hook with the installed dependencies. Package checks validate both distributions. Tox uses
tox-uv to install dependencies, including matching CPU builds of torch and torchvision.

Inference tests generate images and a small ONNX encoder locally. They mock network calls and native model boundaries;
no personal photos, labels, credentials, or pretrained weights are needed in CI.

Research behind the approach:

- [PetFace](https://arxiv.org/abs/2407.13555): individual animal recognition and verification.
- [Calibrated similarity and graph clustering](https://arxiv.org/abs/2608.02469): segmentation, local/global features,
  and unknown rejection.
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html): fitting confidence estimates on
  predictions from separate folds.
