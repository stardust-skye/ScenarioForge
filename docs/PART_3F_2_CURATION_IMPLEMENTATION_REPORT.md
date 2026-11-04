# Part 3F.2 — Leakage-Safe Curation Subset Generation

## Executive summary

Implemented a dedicated, split-aware curation workflow and generated immutable
subset artifacts for the approved seed-42 split. The workflow validates the
split against the image inventory, filters scenario features to the training
sample tokens before scoring, verifies every training image and accepted 2D
annotation count, and enforces the same scene quotas across all strategies.

The generated matrix contains 12 reduced subsets (four strategies at 60,
120, and 180 images) and one 242-image full-training reference. No detector
was created or trained, and no test metrics were evaluated. The existing
Faster R-CNN code, split assignments, annotations, and historical experiment
were not modified.

The unrestricted legacy `curate` CLI command is now blocked for training
subset generation. Use `curate-experiment`; the lower-level curation
algorithms remain available for synthetic unit tests and non-training
analysis.

## Architecture and integrity decisions

### Train-only scoring

`curate-experiment` loads the authoritative seed-42 split and validates it
against `image_manifest.csv` and the dataset version. It checks scenario
sample IDs for duplicates and unknown tokens, verifies scenario/image scene
identity, and rejects any missing training scenario. Held-out scenario
features are not passed to rarity analysis, ranking, diversity selection, or
subset selection. Only after selecting the 242 training records does the
workflow compute rarity frequencies, numeric bins, weighted-interest scores,
and diversity gains.

The original generic `curate` CLI path scored its full input before selecting.
It now returns an explicit error directing users to the split-aware workflow.
This avoids accidentally using its unrestricted output as a detector training
selection.

### Image and annotation validation

Each training candidate must have one unique image-manifest row, `image_status`
`ok`, a resolvable image path inside the configured dataset root, a decodable
image with dimensions matching the manifest, and an annotation count matching
the accepted rows in `annotations.csv`. Accepted annotation rows are checked
for sample/image/scene/path/dimension agreement, unique source tokens,
foreground class IDs, and finite in-bounds XYXY boxes.

Zero-annotation images are retained as valid background candidates and
recorded explicitly in the image validation audit. The current 242-image
training partition contained zero such images. Errors are raised with the
affected sample or annotation token; invalid samples are not silently
dropped.

### Equal budgets and scene controls

`balanced_scene_quotas` implements one common, capacity-aware equalization
policy: allocate each next image to the scene with the smallest current quota,
breaking ties by scene token, until the budget is met. This explains and
generalizes the proposed 10/20/30 images per scene for budgets 60/120/180
across six scenes. At budget 242, capacity constraints allocate every
training image (39–41 per scene). Impossible budgets or quotas fail explicitly.

All four methods obey those same quotas:

- **Random:** seeded shuffle within each scene, then take its quota.
- **Frequency:** rank by inverse-frequency rarity, then fill each scene's
  registered quota.
- **Diversity:** greedily maximize Shannon entropy of cumulative object
  category counts while only considering scenes with remaining quota.
- **Weighted interest:** rank with the existing default rarity, composition,
  and complexity weights, then fill the same scene quotas.

Frequency, diversity, and weighted-interest use deterministic sample-token
tie-breaking. Their seed is still recorded, but does not imply a different
ranking when there are no random choices. Random selections are repeatable
for a fixed seed and input.

## Provenance and outputs

Every subset manifest records dataset version, dataset source ID, raw SHA-256
hashes of the split, image, annotation, category-mapping, dataset, and scenario files, the split
manifest's internal digest, strategy, requested/actual budget, seed, selected
sample and scene tokens, per-scene counts and quotas, scores, scoring
configuration, train-only rarity frequencies/bin edges, annotation counts,
empty-target audit, and UTC creation time. The input files are re-hashed before
writing; a source file changed during generation aborts the run. Each manifest
has its own digest.

Each subset also has a JSON string array of selected sample tokens and a CSV
with sample/scene tokens, accepted 2D annotation count, and selection score.
Output is created in a unique timestamp-plus-UUID directory using
`mkdir(exist_ok=False)`; existing subsets are never overwritten.

Verified output directory:

`data/processed/curation_experiments/curation_20261103T205532Z_77d0c39b`

Verification found 13 subset manifests, the expected 242 training samples,
162 held-out scenarios excluded before scoring, exact unique sizes for every
budget, identical quota counts and actual scene counts, and no malformed
subset. This split has zero empty-target training images. The output is under
the repository's ignored `data/processed` tree and does not alter tracked
experiment results.

## Reproduction command

From the repository root in PowerShell, run:

```powershell
.\.venv-ml\Scripts\python.exe -m scenarioforge.scenario_intelligence.cli `
  --output-dir .\data\processed\scenario_intelligence `
  curate-experiment `
  --vision-dir .\data\processed\vision `
  --split-manifest .\data\processed\vision\splits\scene_split_seed42.json `
  --dataset-root .\data\downloaded-dataset `
  --output-root .\data\processed\curation_experiments `
  --seed 42
```

Each invocation intentionally creates a new run directory. To use the token
selection in a later approved training run, pass the corresponding
`*_selected.json` file as the training selection; do not use old unrestricted
curation exports.

## Tests and verification

Focused validation command:

```powershell
.\.venv-ml\Scripts\python.exe -m pytest -q --basetemp .\pytest-f3-focused-final `
  tests\test_experiment_curation.py `
  tests\test_scenario_intelligence.py `
  tests\test_vision_data_splits.py
```

Result: **86 passed**.

The new tests cover held-out feature mutation invariance for all four
strategies, exact budgets, deterministic selection, empty targets,
duplicate/unknown/missing/mismatched scenario tokens, missing image paths,
annotation-count mismatches, equal scene quotas, impossible quotas, and
inventory/file hash changes.

Full-suite command:

```powershell
.\.venv-ml\Scripts\python.exe -m pytest -q --basetemp .\pytest-f3-final-final
```

Result: **157 passed**.

## Limitations

- The generator intentionally requires the approved 242-image seed-42
  training partition. Different splits and dataset sizes require an explicit
  design review rather than silently changing the research protocol.
- Scenario scores remain metadata/3D-annotation proxies; they do not measure
  pixel-level or learned visual diversity.
- Scene quotas constrain scene concentration but do not remove temporal
  correlation among images within a scene.
- The default scoring weights and frequency heuristics are not calibrated for
  detector performance.
- Source manifests are hashed for provenance, but the original dataset and
  split files are not copied into each run.
- This task generated selections only. It did not compare detector outcomes,
  establish a performance benefit, or authorize training.
