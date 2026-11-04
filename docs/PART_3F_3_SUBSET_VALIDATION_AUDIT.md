# Part 3F.3 — Independent Curation Artifact Validation Audit

## Executive verdict

**CONDITIONAL GO for controlled training using the individually verified
subset manifests.**

All 13 generated subset manifests passed an independent hash, budget,
membership, split-isolation, scene-quota, image-path, annotation, and
selection-file audit. The 242-image full reference exactly matches the
authoritative seed-42 training tokens. Regeneration in a separate temporary
directory produced the same ordered tokens for all 13 subsets; no existing
curation artifact was overwritten or changed.

The condition is provenance handling: `experiment_manifest.json` has no
embedded self-digest. Its current raw-file SHA-256 is recorded below. Preserve
that digest in downstream training provenance, and treat the individually
hashed subset manifests—not the run summary—as the authoritative selection
records. No artifact was repaired.

The subset statistics show that ranked strategies alter class composition
substantially at smaller budgets, especially by increasing traffic-cone box
share relative to the random subset and full reference. This is a property of
these selected subsets, not evidence that the strategies improve detection.
Metadata-derived category diversity is not visual diversity, and category
entropy is not a detection-performance metric.

## Scope and independent procedure

Audited:

- Design assumptions in [Part 3F.1](./PART_3F_1_CURATION_EXPERIMENT_DESIGN.md)
  and implementation claims in
  [Part 3F.2](./PART_3F_2_CURATION_IMPLEMENTATION_REPORT.md).
- The 13 `*_manifest.json` subset manifests and their matching selected JSON
  and CSV files in
  `data/processed/curation_experiments/curation_20261103T205532Z_77d0c39b/`.
- The authoritative
  `data/processed/vision/splits/scene_split_seed42.json`,
  image/annotation manifests, dataset manifest, category mapping, scenario
  records, and image files.

The audit used independent scripts to canonicalize and hash child manifest
payloads, compare sample IDs against the persisted split, resolve selected
images and accepted annotations, recompute detection-class and scenario
category counts, and calculate entropy. It did not use the curation
implementation's selection or validation functions to verify those
properties. The existing generator was run once into a separate temporary
directory solely for the requested determinism check; its ordered selected
token lists were compared with the recorded lists, then that temporary output
was removed.

Validation/test sample IDs were used only to check split disjointness. No
held-out image files were opened, held-out annotation rows were used in subset
statistics, and no model metrics were evaluated.

## Integrity and membership results

| Check | Result |
|---|---|
| Child subset manifests discovered | 13 |
| Independently recomputed child manifest digests | 13/13 match |
| Source split/image/annotation/category/dataset/scenario file hashes | All match recorded values |
| Dataset version | `v1.0-mini` |
| Authoritative train / validation / test sizes | 242 / 80 / 82 |
| Held-out scenarios excluded before scoring, as recorded | 162 |
| Selected IDs in train split | All 13 subsets |
| Validation or test token overlap | None |
| Exact unique selected counts | 60, 120, 180 for reduced budgets; 242 for full reference |
| Full-reference membership | Exactly the 242 training tokens |
| Selected JSON and CSV order versus child manifest | All match |
| Independent regeneration | All 13 ordered selections match |

The child manifest digests use SHA-256 over the canonical JSON payload with
the `manifest_sha256` field omitted, sorted keys, compact separators, and
ASCII escaping. In addition, the six source file hashes independently match
the bytes present at audit time.

### Run-level summary digest caveat

`experiment_manifest.json` does not contain a `manifest_sha256` field, so it
cannot pass the same embedded-digest check. Its raw SHA-256 at audit time is:

`b7fa8bc6276c626b7a03cc180e1988403dcc8b5b7b091a54495fb07bd92e217d`

This is not evidence that the file is internally self-validating. Record this
external digest in each downstream experiment's provenance. The subset
manifests independently carry the core selection and source provenance.

## Scene-quota comparison

All 12 reduced conditions use
`balanced_capacity_aware_equalization_v1`, and each strategy has exactly the
same per-scene counts at each budget:

| Budget | Per-scene counts for every strategy |
|---:|---|
| 60 | 10, 10, 10, 10, 10, 10 |
| 120 | 20, 20, 20, 20, 20, 20 |
| 180 | 30, 30, 30, 30, 30, 30 |
| 242 full | 41, 41, 39, 41, 40, 40 (all available training images) |

The selected scene-token sets and quota counts match across the four
strategies at each reduced size. Thus none of the strategies has an observed
scene-count concentration advantage in this matrix. Equal scene counts do
not remove within-scene temporal correlation or other scene-specific visual
differences.

## Independent distribution and entropy results

Detection counts below come from accepted 2D annotations and use the exact
column order shown here:

`car, truck, bus, trailer, construction_vehicle, pedestrian, motorcycle,
bicycle, traffic_cone, barrier`

| Strategy | Images | Accepted 2D boxes | Detection-class entropy (bits) | 3D scenario-object entropy (bits) | Mean independent category-rarity indicator | 2D class counts in the stated order |
|---|---:|---:|---:|---:|---:|---|
| Diversity | 60 | 625 | 2.3629 | 2.7605 | 0.03425 | 160, 3, 17, 8, 6, 82, 15, 22, 252, 60 |
| Frequency | 60 | 717 | 2.3483 | 2.6783 | 0.03752 | 211, 3, 23, 9, 5, 128, 13, 22, 251, 52 |
| Random | 60 | 901 | 2.5117 | 2.5643 | 0.01990 | 317, 27, 39, 3, 25, 208, 15, 11, 118, 138 |
| Weighted interest | 60 | 731 | 2.3784 | 2.6787 | 0.03762 | 219, 4, 26, 9, 9, 110, 13, 20, 259, 62 |
| Diversity | 120 | 1,540 | 2.5263 | 2.6949 | 0.02664 | 487, 37, 56, 10, 34, 281, 20, 38, 394, 183 |
| Frequency | 120 | 1,644 | 2.5080 | 2.6400 | 0.02905 | 542, 38, 60, 9, 33, 340, 27, 34, 374, 187 |
| Random | 120 | 1,873 | 2.5092 | 2.5687 | 0.02160 | 632, 58, 72, 9, 51, 446, 24, 22, 252, 307 |
| Weighted interest | 120 | 1,671 | 2.4775 | 2.6471 | 0.02985 | 552, 39, 56, 10, 38, 310, 21, 31, 431, 183 |
| Diversity | 180 | 2,532 | 2.5294 | 2.6430 | 0.02428 | 826, 72, 98, 13, 66, 526, 29, 43, 534, 325 |
| Frequency | 180 | 2,703 | 2.5187 | 2.6065 | 0.02549 | 894, 78, 97, 11, 70, 623, 36, 43, 480, 371 |
| Random | 180 | 2,721 | 2.5114 | 2.5773 | 0.02198 | 924, 80, 103, 13, 72, 625, 33, 37, 418, 416 |
| Weighted interest | 180 | 2,727 | 2.5356 | 2.6122 | 0.02575 | 878, 75, 108, 17, 70, 595, 35, 40, 547, 362 |
| Full reference | 242 | 3,636 | 2.5065 | 2.5736 | 0.02285 | 1,263, 104, 136, 17, 97, 814, 44, 54, 580, 527 |

Entropy is Shannon entropy of the relevant aggregate object-count
distribution. The 3D scenario-object distribution uses metadata category
counts; the detection distribution uses accepted 2D classes. The independent
rarity indicator is the mean over selected samples of the maximum
`1 / (training-sample category frequency + 1)` among that sample's 3D
categories. This transparent diagnostic is not the implementation's combined
rarity score, is not a performance measure, and should not be interpreted as
a quality ranking.

All 13 subsets contain all 16 categories observed in the selected 3D scenario
metadata and all ten mapped 2D detection classes. However, presence alone
hides low support: in the 60-image subsets, several classes have only 3–9
accepted boxes. The full training reference has these 2D class counts:
`1,263, 104, 136, 17, 97, 814, 44, 54, 580, 527`.

### Notable class-composition differences

Traffic-cone boxes are about 16.0% of the full reference. At 60 images they
are 13.1% of the random subset, versus 35.0% (frequency), 35.4% (weighted
interest), and 40.3% (diversity). At 120 images, the corresponding shares are
13.5% for random and 22.7–25.8% for ranked methods. At 180 images, the shares
are 15.4% for random and 17.8–21.1% for ranked methods. Thus the ranked
methods systematically shift this class's share upward relative to the
particular random draw, with the largest difference at the smallest budget.
Other class proportions shift as well; these results motivate reporting
per-class support and AP in the training study rather than interpreting one
aggregate diversity number.

The entropy values do not produce a consistent “more is better” ordering:
for example, at 60 images the random subset has higher 2D detection-class
entropy than the diversity subset, while the diversity subset has higher 3D
scenario-object entropy. At 120 and 180 images the order changes again.
These aggregate distributions describe the selected samples only.

## Image, annotation, and empty-image validation

The audit independently checked all 242 selected training images (the union
of selections is the full reference): each token resolves to one image row,
its path exists under `data/downloaded-dataset`, the image decodes at the
recorded dimensions, and the recorded accepted-annotation count equals the
CSV rows for that sample. Each accepted annotation's sample/image/scene/path
and dimensions match its image row; its class ID maps to the recorded
detection category; and its finite XYXY box lies within image bounds.

The independent check also compared every repeated
`training_image_validation_audit` entry with accepted annotation rows. All
242 images have at least one accepted 2D box; the empty-target count is zero.
No empty image existed in this partition to test positive preservation, but
the output flags and counts correctly represent the observed data. This
audited absence is not evidence that future empty images are handled by this
artifact set.

## Deterministic regeneration

The generator was invoked with seed 42 and the recorded source files, using a
new temporary output root outside the existing artifact directory. All 13
regenerated selected JSON token arrays matched the existing child manifests
exactly, including order. The temporary directory was removed after
comparison. No existing subset, experiment, split, or baseline file was
written or changed by this audit.

This confirms deterministic regeneration for this code, environment, seed,
and data snapshot. It does not prove bitwise reproducibility of later model
training.

## Scientific interpretation and limitations

- **Metadata diversity is not visual diversity.** Scenario statistics are
  derived from 3D annotation and metadata features; they do not measure image
  appearance, camera conditions, or learned embeddings.
- **Category entropy is not detection performance.** The entropy calculations
  summarize class composition. They do not measure localization, precision,
  recall, AP, calibration, or generalization.
- **Selection statistics are not model outcomes.** No detector training or
  validation/test model evaluation was performed. No strategy is shown to be
  superior.
- The fixed six-scene train split and 242-image dataset are small. Scene
  quotas control counts, but not within-scene correlation, geography, or
  environment coverage.
- Strategies intentionally select different images and class distributions.
  Their effects must be estimated with matched training controls and
  repeated seeds as specified in Part 3F.1.

## Decision and conditions

**CONDITIONAL GO for training**, using only the verified selected-token arrays
associated with the audited child manifests, after:

1. Recording the run-level summary file's external SHA-256
   (`b7fa8bc6276c626b7a03cc180e1988403dcc8b5b7b091a54495fb07bd92e217d`) in
   each downstream experiment's provenance.
2. Preserving each chosen child manifest and its own digest with the training
   run, and confirming that the training loader validates the tokens against
   the same seed-42 split.
3. Reporting per-class support and validation metrics for each run, without
   treating subset entropy or rarity as evidence of detector performance.

This decision approves the audited subset artifacts for the planned
controlled training study; it does not approve any claim of model improvement
or authorize test-partition evaluation.
