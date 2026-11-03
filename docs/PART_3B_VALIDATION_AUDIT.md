# Part 3B Independent Validation Audit

## Executive verdict

**APPROVED WITH FOLLOW-UPS for the leakage remediation; keep model training gated on ML-runtime
validation.**

The root cause was that split-consuming APIs treated the manifest digest and declared split
disjointness as sufficient. `make_split_datasets` did not compare the manifest's sample membership
to the authoritative image-to-scene inventory or the current dataset version. The remediation now
requires the authoritative inventory and dataset version at split-validation boundaries, and binds
schema-version-2 manifests to a canonical inventory digest. The exact audit exploit is rejected by
the authoritative validator; a factory-level regression test was added, but it could not execute in
the active base environment because PyTorch is unavailable there.

The `.venv` still lacks PyTorch and torchvision. Although another installed Python interpreter
advertises both packages, importing torchvision fails because Pillow's native `_imaging` DLL cannot
load (`ImportError: DLL load failed ... Access is denied`). Therefore, torch-backed tests and the
notebook remain unexecuted. Do not treat this remediation as ML-runtime validation.

## Implementation reviewed

Reviewed:

- `src/scenarioforge/vision_data/splits.py`
- `src/scenarioforge/vision_data/dataset.py`
- `src/scenarioforge/vision_data/statistics.py`
- `src/scenarioforge/vision_data/transforms.py`
- `src/scenarioforge/vision_data/dataloader.py`
- `src/scenarioforge/vision_data/cli.py`
- `tests/test_vision_data_splits.py`
- `tests/test_vision_data_torch.py`
- `scripts/prepare_vision_data.py`
- `notebooks/05_PyTorch_Dataset_Validation.ipynb`
- `docs/PART_3B_PYTORCH_DATA_PIPELINE.md`
- `data/processed/vision/splits/scene_split_seed42.json`
- Generated split statistics and leakage reports under
  `data/processed/vision/splits/`

The audit also read the current `image_manifest.csv`, `annotations.csv`, and
`dataset_manifest.json` to reconcile the generated outputs with their source data.

## Exact seed-42 split and statistics

The persisted schema-version-2 manifest digest is
`d93c15fac3e798343ea95fcefea3c03a0b189a0b9036b3f355354550109bf2ab`.
The manifest assigns 10 scenes and 404 samples. Scene names below were joined from the current
image manifest by `scene_token`; image and annotation counts are from the persisted per-scene
statistics and were independently reconciled against a fresh calculation.

| Partition | Scene name | Scene token | Images | Accepted annotations |
|---|---|---|---:|---:|
| Train | scene-0553 | `6f83169d067343658251f72e1dd17dbc` | 41 | 634 |
| Train | scene-0655 | `bebf5f5b2a674631ab5c88fd1aa9e87a` | 41 | 688 |
| Train | scene-0061 | `cc8c0bf57f984915a77078b10eb33198` | 39 | 1,329 |
| Train | scene-1077 | `d25718445d89453381c659b9c8734939` | 41 | 365 |
| Train | scene-1094 | `de7d80a1f5fb4c3e82ce8a4f213b450a` | 40 | 415 |
| Train | scene-1100 | `e233467e827140efa4b42d2b4c435855` | 40 | 205 |
| Validation | scene-0796 | `c5224b9b454b4ded9b5d2d2634bbda8a` | 40 | 209 |
| Validation | scene-0103 | `fcbccedd61424f1b85dcbf8f897f9754` | 40 | 743 |
| Test | scene-0757 | `2fc3753772e241f2ab2cd16a784cc680` | 41 | 159 |
| Test | scene-0916 | `325cef682f064c55a255f2625c533b75` | 41 | 578 |

| Partition | Scenes | Images | Accepted annotations | Empty images |
|---|---:|---:|---:|---:|
| Train | 6 | 242 | 3,636 | 0 |
| Validation | 2 | 80 | 952 | 0 |
| Test | 2 | 82 | 737 | 1 |

### Class distribution

Counts are accepted annotations, not image counts.

| Category | Train | Validation | Test |
|---|---:|---:|---:|
| barrier | 527 | 0 | 0 |
| bicycle | 54 | 15 | 13 |
| bus | 136 | 11 | 35 |
| car | 1,263 | 648 | 440 |
| construction_vehicle | 97 | 0 | 4 |
| motorcycle | 44 | 25 | 78 |
| pedestrian | 814 | 198 | 122 |
| traffic_cone | 580 | 11 | 3 |
| trailer | 17 | 0 | 0 |
| truck | 104 | 44 | 42 |

The split is not balanced. Validation has no `barrier`, `construction_vehicle`, or `trailer`
annotations; test has no `trailer` annotations. Test support is also very sparse for
`construction_vehicle` (4) and `traffic_cone` (3). Per-class held-out conclusions for absent or
sparsely represented categories are not supported by this mini split.

## Leakage and reproducibility findings

### Checks that passed for the persisted manifest

- `validate_split_manifest(manifest, image_rows)` reported zero declared scene overlap and zero
  sample-token overlap.
- Full image-manifest coverage was checked: 6/2/2 scenes and 242/80/82 samples, 404 total.
- The exact seed-42 manifest was rebuilt from the current image manifest; it matched the persisted
  manifest and digest.
- Building with seed 43 produced a different assignment, as expected for this input.
- Current statistics independently reported zero missing image paths and zero invalid annotations
  in every partition.
- The training-selection validator rejects duplicate tokens, validation/test tokens, and unknown
  tokens. `select_training_subset` additionally requires its dataset's samples to be a subset of
  the manifest's training samples and returns a `Subset`; it does not change validation or test
  dataset membership.

### Root cause and remediation

Before remediation, `validate_split_manifest` accepted an optional image inventory and permitted
digest/disjointness-only validation. `make_split_datasets` used that weaker form, while
`select_training_subset` and other consumers also relied on it. The SHA-256 manifest digest only
covered the manifest's own contents; anyone or any process that changed sample memberships and
recomputed the digest could still create inconsistent assignments.

After remediation:

- Split manifests use schema version 2 and include `dataset_version` plus a SHA-256 digest of the
  canonical complete image inventory.
- `validate_split_manifest`, `load_split_manifest`, `write_split_manifest`,
  `validate_training_selection`, and `validate_equal_training_budgets` require authoritative
  inventory rows and dataset version. There is no public digest-only validation mode.
- Validation checks the manifest digest, exact dataset version, inventory provenance digest,
  complete scene and eligible-sample coverage, uniqueness/disjointness, and that each sample's
  actual scene matches the partition declaring it.
- `make_split_datasets` reads the current image inventory and adjacent `dataset_manifest.json`,
  validates the split before constructing any partition dataset, and passes the validated inventory
  snapshot and expected version into every dataset. Construction fails closed if the dataset
  manifest/version is unavailable, invalid, or changed.
- `select_training_subset` validates against the full image inventory and dataset version retained
  by its `CameraDetectionDataset`; it fails closed if that dataset has no authoritative version.
- The statistics API, CLI, and notebook supply the authoritative inventory and dataset version.

The exact exploit was re-created by swapping one train sample and one test sample, preserving the
declared scene lists, and recalculating the manifest digest. Internal structure/digest validation
alone accepts the forged contents, demonstrating why the digest is insufficient; authoritative
validation rejects it with a sample-to-scene membership mismatch. A regression test also asserts
that `make_split_datasets` and `select_training_subset` reject the forged manifest. That factory test
is present in the torch-dependent test module and was skipped in the active `.venv`; it has not been
runtime-executed because the required imports are unavailable.

No scene/sample assignments were changed. The regenerated seed-42 manifest keeps the exact prior
scene memberships and image/annotation counts. Its new manifest digest is
`d93c15fac3e798343ea95fcefea3c03a0b189a0b9036b3f355354550109bf2ab`; its inventory digest is
`0898a28fd93c46ea44eac57b728739aeb90d455431c4a1a210e37cd2ee483a98`.

## Dataset and target validation findings

On code inspection, `CameraDetectionDataset`:

- Converts image pixels to RGB and lazily loads one image per item.
- Groups accepted annotation rows by `sample_token`.
- Reads annotation boxes directly as float32 pixel-coordinate XYXY values.
- Checks accepted-row status, image/sample/scene/path/dimension associations, duplicate source
  annotation tokens, finite coordinates, positive area, image bounds, and known foreground class
  IDs.
- Checks image existence, decodability, and manifest dimensions by default; resolved image paths
  must remain beneath the configured dataset root.
- Remaps positive source IDs to contiguous foreground model labels beginning at 1; background 0 is
  not assigned to an object.
- Emits a deterministic scalar int64 image ID derived from `image_id`, and rejects collisions
  within the image manifest.
- Emits `area` and zero-valued int64 `iscrowd` vectors aligned to the boxes.
- Preserves empty targets as boxes `[0,4]`, labels `[0]`, area `[0]`, and `iscrowd` `[0]`.
- Revalidates transformed boxes against the transformed image dimensions and recomputes area.

These are appropriate target conventions for torchvision detection models. The dataset validates
boxes rather than clipping them; this is suitable for the Part 3A exports, which are expected to
contain already-clipped accepted boxes. Runtime confirmation remains pending because the active
`.venv` lacks PyTorch. The current loader also does not cross-check each annotation's category name
against the category-mapping file's name for its class ID; the current exports were mapped upstream,
but adding that cross-check would make stale or inconsistent mapping inputs fail explicitly.

The split-restricted factory now performs authoritative membership validation before constructing
any partition dataset.

## Transforms and DataLoader findings

- `Resize` uses the image's old/new dimensions to scale XYXY coordinates on their respective axes,
  then recomputes area. The output validation checks boxes remain positive-area and within the
  resized image.
- Horizontal flipping uses `width - x_max` / `width - x_min`, mirrors the image, and recomputes
  area. It is disabled by default.
- Evaluation transforms are deterministic: optional resize followed by RGB float32 tensor
  conversion. Empty boxes survive resize and flip.
- Detection collation returns per-image image/target tuples and does not stack variable-length
  targets.
- `num_workers=0` is a sensible Windows default. With workers enabled, Python and NumPy receive
  worker-specific seeds derived from the DataLoader's seeded generator.
- The generator also seeds shuffled sampling. However, with the default `num_workers=0`, the
  `RandomHorizontalFlip` transform uses the process-global Python `random` generator, which the
  DataLoader's `seed` does not seed. Thus the `seed` parameter alone does not make stochastic
  augmentations reproducible in the default worker configuration; it only controls sampling order.
  Seeded augmentation or a per-transform generator should be addressed before relying on repeatable
  training runs.

## Tests and runtime limitations

After remediation, the focused split tests reported:

```text
17 passed
```

The split-generation CLI recreated the seed-42 manifest with 6/2/2 scenes and 242/80/82 images.
The statistics CLI and a separate recomputation validated zero scene/sample overlap, exact
inventory coverage, matching dataset version and inventory digest, zero unassigned annotations,
and the expected annotation counts (3,636 / 952 / 737). The generated reports were refreshed.

The full current suite was run after remediation:

```text
106 passed, 1 skipped in 148.66s
```

The torch-dependent test module, including the new dataset-factory forged-manifest regression,
remains unexecuted in `.venv` because PyTorch is not installed. An installed Python 3.13 environment
has torch/torchvision module specs, but importing torchvision fails because Pillow's `_imaging` DLL
cannot load with `Access is denied`; it was not used as a runtime-test substitute. No packages were
installed. The notebook JSON and code cells were syntax-checked, not executed.

The runnable split/statistics tests cover deterministic grouped splits, explicit assignments,
digest tampering, the resealed exploit, unknown/mismatched/duplicate/missing/unexpected sample and
scene assignments, dataset-version and inventory-provenance mismatches, training-selection
validation, equal budgets, and statistics. Dataset/transform/DataLoader tests remain runtime
unverified. No torchvision detector forward pass has been run.

## Notebook status

`notebooks/05_PyTorch_Dataset_Validation.ipynb` uses the reusable package APIs to load the split,
build split datasets, inspect many/few/empty samples, retrieve a DataLoader batch, compare split
statistics, check declared overlap, and compare equal training-selection budgets. It does not train
a model. The notebook imports `select_training_subset` but does not call it, so it does not itself
exercise the curation API's held-out guard.

The notebook JSON and all six code cells passed syntax validation during this audit. Every code
cell has `execution_count: null` and an empty output list. It has **not** been executed, and its
visual inspection, package imports, image loading, transformations, DataLoader batches, and
statistics display remain unverified at runtime.

## Generated statistics freshness

The regenerated `split_statistics.json` agrees with a fresh calculation for partition counts,
class counts, per-scene image/annotation counts, empty-image counts, missing paths, invalid
annotations, and unassigned annotation fields. It is bound to the current split manifest digest and
reports zero unassigned annotations.

## Files changed for remediation

- `src/scenarioforge/vision_data/splits.py` — schema 2, inventory provenance, fail-closed
  authoritative validation and required inventory/version parameters for split operations.
- `src/scenarioforge/vision_data/dataset.py` — retain dataset-manifest path/version for curation
  validation and accept the factory's validated inventory snapshot.
- `src/scenarioforge/vision_data/dataloader.py` — validate before split construction and validate
  curation against the complete inventory.
- `src/scenarioforge/vision_data/statistics.py` and `cli.py` — pass current inventory/version.
- `tests/test_vision_data_splits.py` and `tests/test_vision_data_torch.py` — authority and exploit
  regressions.
- `notebooks/05_PyTorch_Dataset_Validation.ipynb` and
  `docs/PART_3B_PYTORCH_DATA_PIPELINE.md` — use and document the authoritative API contract.
- `data/processed/vision/splits/` — regenerated seed-42 manifest and statistics (ignored generated
  data; scene assignments and counts unchanged).

## Remaining issues ranked by severity and recommended follow-ups

1. **MEDIUM — PyTorch and torchvision runtime behavior is unverified.**
   The `.venv` lacks the packages, and the alternate Python 3.13 environment cannot import
   torchvision due to the Pillow native DLL error. Run the new factory regression, torch-backed
   dataset/transform/DataLoader tests, notebook, and a small detector forward pass in a functioning
   `.venv-ml` before training.

2. **MEDIUM — Seed does not control default-process stochastic transforms.**
   With `num_workers=0`, the seeded DataLoader generator does not seed Python's global random
   generator used by `RandomHorizontalFlip`.
   **Recommended fix:** make the augmentation RNG explicitly seeded/owned by the transform or
   provide a documented seeding mechanism that also controls the main-process transform. Test
   repeated runs with the same seed and enabled flipping.

3. **LOW — Mapping name consistency is not checked.**
   The dataset checks that annotation class IDs exist in the mapping but does not verify that each
   annotation's `detection_category` agrees with that mapped ID. Add this consistency check as a
   separate data-integrity improvement.

## Recommendation

The leakage remediation is implemented, the pure-Python forged-manifest regression passes, and the
factory/curation regression is included for a working ML environment. Keep model training gated
until that torch-dependent factory test and the remaining torch-backed suite execute successfully
in a functional `.venv-ml`. The current seed-42 scene assignments remain unchanged and pass
authoritative version, provenance, coverage, and scene-membership validation.
