# Part 3 Experiment Design

**Status: preregistration proposal only.** No model training or performance experiment has been run.
Model architecture and this protocol require review before implementation.

## Research questions

1. At matched training-sample budgets, do random, frequency-, diversity-, or weighted-interest
   curation produce different 2D front-camera detection results on the same held-out scenes?
2. At equal evaluation conditions, how do Faster R-CNN MobileNetV3-Large FPN and a license-approved,
   pinned YOLO-family detector compare?
3. Do any apparent differences persist across scene-grouped folds and independent selection/
   training seeds on a larger labeled dataset?

Higher category entropy is not a detection metric and does not establish improved perception.

## Data and split protocol

- Use nuScenes mini only for data-conversion, visualization, split, and CPU/GPU smoke tests. With
  10 scenes it is insufficient for strong generalization claims.
- For meaningful experiments, use a larger labeled release such as nuScenes `trainval`. Freeze a
  deterministic scene-group split before fitting curation statistics: official training scenes
  are partitioned into train and validation; official validation scenes are reserved as a fixed
  test set. Record exact tokens and release version. Do not report local metrics on the unlabeled
  official test split.
- All keyframes from a scene stay in one partition. Split and annotation manifests are identical
  across all curation strategies, model architectures, and training seeds.
- Curation operates on training IDs only. Validation and test scenes, samples, labels, and
  distributions remain fixed and are never inputs to curation or fit-time statistics.
- Fit any quantile edges, feature normalization, or rarity frequencies on training data only.
- Publish scene and sample counts plus per-class image/box counts by split. If a class has no
  train/validation/test support, state that it is not estimable; do not reassign test scenes after
  seeing results.

For the 10-scene mini dataset, a deterministic grouped split is a smoke-test fixture, not an
evidence-generating benchmark. Report the exact scene IDs and avoid inferential claims.

## Experiment A: curation strategies

Hold the detector architecture, initialization/checkpoint, annotation version, image transforms,
augmentation, optimizer/schedule, number of optimizer updates, batch size, validation/test scenes,
and evaluation code fixed within each comparison. Compare:

| Condition | Training data | Selection |
|---|---|---|
| Random subset | Fixed budgets (e.g. 25%, 50%, 75%) of the eligible training keyframes | Seeded uniform sample |
| Frequency subset | Same budgets and eligible training pool | Highest train-only rarity score |
| Diversity subset | Same budgets and eligible training pool | Greedy category-entropy gain |
| Weighted-interest subset | Same budgets and eligible training pool | Frozen ranking weights |
| Full-data reference | All eligible training keyframes | No curation |

Use exact equal unique-sample budgets for the four subset strategies at each budget. The full-data
reference uses all training records. Record selected sample and scene IDs in each run manifest.
Use multiple predeclared selection seeds per strategy and multiple training seeds; never pick a
seed based on test results.

Use a fixed optimizer-update budget for the primary comparison so wall-clock/update opportunity is
controlled, and report the number of presentations per selected image because smaller subsets are
revisited more often. Add a fixed-epoch or fixed-wall-clock sensitivity analysis if feasible;
these answer different data-efficiency questions and must not be mixed into one result. Evaluate
all conditions on exactly the same validation/test images. Consider a scene-stratified sampling
constraint or report per-scene selection counts, since the current metadata curation algorithms do
not enforce scene quotas.

## Experiment B: architecture comparison

Compare Faster R-CNN MobileNetV3-Large FPN and a license-approved, pinned YOLO-family model using:

- identical scene splits, camera images, projected annotations, class IDs, visibility policy, and
  training-set manifests;
- identical train/validation/test evaluation sets and metric implementation;
- equal curation condition/sample budget and planned optimizer-update budget;
- documented model-specific initialization, optimizer, resizing, augmentation, and inference
  settings where exact equivalence is impossible.

Model-specific heads and optimizer defaults cannot be made mathematically identical. Freeze
validation-based tuning budgets before touching test data and document every difference. Do not
call either detector superior until completed, repeated experiments support that conclusion.

## Evaluation metrics

Primary 2D detection metrics, all from one fixed evaluator:

- mAP@0.50 IoU and mAP@0.50:0.95 IoU;
- per-class average precision and per-class precision/recall;
- overall precision and recall at a validation-selected confidence threshold, frozen before test;
- training time, peak GPU memory, unique training samples, and image presentations/optimizer step.

State the IoU matching, duplicate-detection, ignore/crowd, score-threshold, and empty-class policy.
Report per-scene results and aggregate across held-out scenes/folds; include uncertainty intervals
at the scene grouping level. Do not substitute nuScenes 3D mAP/NDS for these image-plane metrics.
Do not use test results to tune thresholds or choose checkpoints.

## Experiment matrix

| Experiment | Model | Train condition | Selection seeds | Training seeds | Fixed evaluation |
|---|---|---|---|---|---|
| A | Faster R-CNN MobileNetV3-Large FPN | Random / frequency / diversity / weighted-interest at each fixed subset budget; full-data reference | Multiple, predeclared | Multiple, predeclared | Same fixed validation and test scenes |
| B | Faster R-CNN and selected YOLO model | Matched subset conditions and full-data reference | Same selections per budget | Multiple, predeclared | Same fixed validation and test scenes |
| C | Both approved models | Repeat selected conditions with a frozen environment and run manifest | Repeated | Repeated | Same split and evaluator; compare run digests |

The mini dataset may exercise one small condition to verify the pipeline, but it does not fill the
evidence rows in this matrix.

## Reproducibility registry

Each run should have a versioned JSON manifest at minimum:

```json
{
  "schema_version": 1,
  "run_id": "generated-stable-id",
  "dataset": {"name": "nuscenes", "version": "recorded-release"},
  "annotation": {"converter_version": "recorded-version", "config_sha256": "…"},
  "split": {"scene_ids": {"train": [], "validation": [], "test": []}, "sha256": "…"},
  "curation": {
    "strategy": "diversity",
    "seed": 42,
    "requested_unique_samples": 0,
    "selected_sample_ids": [],
    "selected_scene_ids": []
  },
  "model": {"implementation": "recorded-package", "version": "exact-version", "config": {}},
  "training": {"seed": 42, "optimizer_updates": 0, "config": {}},
  "software": {"python": "exact-version", "torch": "exact-version", "torchvision": "exact-version"},
  "hardware": {"gpu": "recorded-device", "driver": "recorded-version"},
  "artifacts": {"checkpoint": "path-and-sha256", "predictions": "path-and-sha256"},
  "metrics": {"evaluation_config_sha256": "…", "results": {}}
}
```

The ellipses and empty arrays above are schema placeholders, not experiment results. Record
dependency lockfile digest, CUDA runtime reported by PyTorch, deterministic-mode caveats, checkpoint
and pretrained-weight hashes, curation parameters, evaluation threshold, timings, peak memory, and
all failures. Keep generated checkpoints/predictions outside raw dataset directories. Preserve the
manifest, split IDs, selected IDs, configs, checkpoints, and evaluation results so a run can be
reconstructed without relying on notebook state.

## Interpretation and limitations

- Curation can change class composition, scene composition, and repeated sample exposure; track all
  three. Distribution entropy and JSD are descriptive covariates, not model outcomes.
- Scene count, not frame count, is the important unit for leakage and much of the uncertainty.
- nuScenes mini contains too few scenes for strong or broad generalization claims. Move to a larger
  labeled release before making research conclusions.
- Front-camera 2D boxes projected from 3D labels inherit geometry, calibration, visibility, and
  projection-policy errors. Manually inspect overlays and retain conversion rejection reports.
- Differences in detector pretraining, optimization, preprocessing, licensing, or hardware can
  limit perfect parity; report them rather than hiding them.
