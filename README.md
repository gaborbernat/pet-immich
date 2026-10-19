# pet-immich

Local pet identification for Immich. The pipeline detects animals, extracts whole-animal, body, and face features,
learns identities from labeled crops, and adds matching faces to Immich. Scheduled runs process new photos with the
trained model. Existing face assignments and confirmed crop labels take precedence over predictions.

## Setup

Requires Python 3.14 and a local [immich-pet-tagger](https://github.com/tedornitier/immich-pet-tagger) crop provider.

```sh
tox run -e dev
source .tox/dev/bin/activate
```

Set these environment variables:

| Variable              | Purpose                                                    |
| --------------------- | ---------------------------------------------------------- |
| `IMMICH_URL`          | Immich server URL                                          |
| `IMMICH_API_KEY`      | API key with asset search and face read/create permissions |
| `IMMICH_READ_API_KEY` | Optional separate key for asset search                     |
| `PET_TAGGER_URL`      | Crop provider URL                                          |

Photo services accept loopback and private LAN IP addresses, or `localhost`. Requests bypass environment proxies and
reject public destinations. Training and inference run locally.

`PET_STATE` in the examples below is a directory outside the checkout containing samples, labels, embeddings, models,
reports, and the assignment audit. Create `identities.json` there, mapping each training label to its Immich person ID.

Place model weights in `$PET_STATE/models/`:

| File               | Model                                                                                       | License                |
| ------------------ | ------------------------------------------------------------------------------------------- | ---------------------- |
| `encoder.onnx`     | [Open Noodle pet-recognition-base](https://huggingface.co/open-noodle/pet-recognition-base) | Apache 2.0             |
| `segment.pt`       | [YOLO11s segmentation](https://docs.ultralytics.com/models/yolo11/)                         | AGPL-3.0 or commercial |
| `localizer.tflite` | [DogFLW face localizer](https://huggingface.co/hugocornellier/dog-face-landmarks)           | CC BY-NC 4.0           |

Alternative encoders are [Open Noodle large](https://huggingface.co/open-noodle/pet-recognition-large), using the same
ONNX filename, and [MiewID-msv3](https://huggingface.co/conservationxlabs/miewid-msv3), using `encoder.safetensors` with
no `encoder.onnx` present. Each encoder requires its own feature cache and trained classifier.

## Train and apply

Import a trusted crop-provider SQLite cache, crop labels, and a metadata CSV with `id` and `fileCreatedAt` columns:

```sh
pet-immich import --state "$PET_STATE" --database "$CROP_DATABASE" --labels "$LABELS" --metadata "$METADATA"
pet-immich extract --state "$PET_STATE" --labeled-only
pet-immich train --state "$PET_STATE" --error-cost 5
pet-immich extract --state "$PET_STATE"
pet-immich plan --state "$PET_STATE"
pet-immich apply --state "$PET_STATE"
```

`--error-cost` controls the recovery/error trade-off: one incorrect assignment cancels that many correct recoveries when
choosing thresholds. The default is five. Training selects features and thresholds on separate capture groups, with each
event weighted equally. It saves `candidate.joblib` and `validation.json`; positive utility for each identity across at
least ten accepted events also produces `approved.joblib`.

Validation reports correct, incorrect, known, and missed crops per identity. Recall covers labeled crops, excluding
photos missed by the animal detector. `Neither` supplies unknown-animal examples; `Unsure` stays untagged. A photo label
of `Both` needs individual crop labels for training.

`plan.json` lists proposed assignments and conflicts. `apply` verifies each creation and records it in `applied.json`.
Model and preprocessing signatures bind classifiers to their feature caches.

For a learning-curve experiment, `train --training-fraction 0.5` samples capture groups for each known pet within the
training folds. Co-occurring pets stay together; unknown-animal examples and evaluation folds remain complete. The
report records actual training counts. Such runs do not replace the approved model.

## Compare models

```sh
pet-immich compare --state "$EXPERIMENT_STATE" --similarities "$PAIR_SCORES" --duplicates --error-cost 2 5 10 20
```

The NPZ input contains `ids` in feature-cache order and one square similarity matrix per channel. Uncomputed pairs use
NaN. `comparison.json` evaluates channel combinations and error costs on shared capture groups. `--duplicates` also
measures direct propagation between crops with matching visual features and perceptual hashes.

`pet_immich.matching.match_candidates` supplies DISK/LightGlue scores from a nearest-neighbor shortlist. Independent
validation requires reference galleries from each training fold; full-collection shortlists support exploratory
comparisons. Scores from identity-trained matchers likewise require training within each fold.

## Scheduled operation

```sh
pet-immich sync --state "$PET_STATE"
```

`sync` discovers new assets, extracts features, and applies the approved model. It retains progress across restarts. The
user systemd units in `deploy/` use `~/.local/bin/pet-immich`, state in `~/.local/state/pet-immich`, and environment
variables in that directory's `service.env` file with mode `0600`. Install the units in `~/.config/systemd/user/`, then:

```sh
systemctl --user daemon-reload
systemctl --user enable --now immich-pet-autotag.timer
```

## Development

```sh
tox run -e 3.14,fix,type,pkg_meta
```

Tox runs tests, prek formatting and lint checks, ty, and distribution checks. Tests use synthetic data and require 100%
line and branch coverage. CI runs on Linux and Windows.
