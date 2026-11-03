# Part 3D.1 First Controlled Baseline

## Outcome

The three-epoch Faster R-CNN baseline completed successfully using the full seed-42
training partition and validation-only model selection. This is an engineering and
baseline-validation run, not a research claim. Three epochs on nuScenes mini do not
establish meaningful model performance.

No test dataset was instantiated or evaluated. The training summary records
`test_evaluated: false`, and the experiment directory contains no `test_metrics.json`.
No split manifest, annotation export, or raw nuScenes data was changed. No YOLO or
curation comparison was started.

## Experiment and exact command

- **Experiment ID:** `20261103T174810Z-9a1fe909`
- **Command:**

  ```powershell
  .\.venv-ml\Scripts\python.exe scripts\train_detection.py train --config configs\object_detection_baseline_part3d1.json
  ```

- **Model:** torchvision Faster R-CNN MobileNetV3 Large 320 FPN,
  `fasterrcnn_mobilenet_v3_large_320_fpn`.
- **Initialization:** torchvision `FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1`.
  The COCO class-specific prediction head was replaced for ten ScenarioForge foreground
  classes plus background.
- **Runtime:** Python 3.12.16; PyTorch 2.15.0+cu132; torchvision 0.29.2+cu132;
  TorchMetrics 1.9.0; Faster-COCO-Eval 1.8.0; Windows 11; CUDA 13.2; cuDNN 92600.
- **Device / precision:** NVIDIA GeForce RTX 3050 Laptop GPU, CUDA, batch size 1,
  mixed precision enabled (the CUDA AMP step had passed the prior Part 3C.2 smoke test).
- **Reproducibility:** seed 42, `best_effort`. This records and seeds supported
  randomness but does not guarantee bitwise-identical CUDA execution. The run emitted
  the expected warning that `roi_align_backward_kernel` has no deterministic
  implementation.
- **Training config:** 3 epochs, no sample limits or token selection file, no external
  `image_size` override, zero data-loader workers, learning rate `0.0001`, weight decay
  `0.0001`, gradient clipping at 1.0, validation after every epoch.

## Dataset and split provenance

- **Dataset:** nuScenes `v1.0-mini`.
- **Split:** existing scene-level seed-42 manifest
  `data/processed/vision/splits/scene_split_seed42.json`; it was read and not modified.
- **Training:** all **242/242** training samples and corresponding **242/242** image
  records. No curation or subset selection was applied.
- **Validation:** all **80/80** validation samples and corresponding image records;
  all 80 were evaluated after each epoch.
- **Training selection audit:** valid, `selected_sample_count: 242`,
  `training_budget: 242`, `heldout_samples_unchanged: true`.
- **Split-manifest digest:** `d93c15fac3e798343ea95fcefea3c03a0b189a0b9036b3f355354550109bf2ab`.
- **Image-inventory digest:** `0898a28fd93c46ea44eac57b728739aeb90d455431c4a1a210e37cd2ee483a98`.
- **Training-selection digest:** `a5d4022728c12970f3a85b0648fc78794d0c9e8574fd5d57703739d08dd754e8`.
- **Configuration digest:** `8f61a856635e1fa00b51149dda3de908f9480cb6e1b8c280034050ffb0a09aa9`.
- **Experiment provenance digest:** `d677858f00a5fa83e51d35d55d71f51d3dc4960870cd4aa2938297e8e0395471`.

The experiment setup constructs only train and validation datasets. The test dataset was
not instantiated, read, evaluated, or used for checkpoint selection.

## Epoch results

Training losses are per-batch means over all 242 training samples. Validation metrics
are COCO-style Faster-COCO-Eval metrics over all 80 validation images. Learning rate is
the rate used during that epoch.

| Epoch | Total train loss | Classifier | Box regression | Objectness | RPN box regression | LR | Duration (s) | Peak CUDA allocated (bytes) | Validation mAP@[.50:.95] | AP50 | AP75 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.080561 | 0.489757 | 0.416406 | 0.078537 | 0.095861 | 0.0001 | 174.023 | 435,321,856 | 0.029374 | 0.092764 | 0.010623 |
| 2 | 0.990161 | 0.385287 | 0.474757 | 0.052410 | 0.077706 | 0.0001 | 154.963 | 435,321,856 | 0.026855 | 0.075586 | 0.012874 |
| 3 | 0.979974 | 0.357931 | 0.509286 | 0.043811 | 0.068946 | 0.0001 | 174.033 | 435,321,856 | 0.034261 | 0.090042 | 0.023811 |

Total measured epoch time, including train and validation work, was **503.019 seconds
(8 minutes 23.019 seconds)**. It excludes model/weight construction and artifact-writing
overhead. The global optimizer step count was 726 (242 batches per epoch).

Peak memory is PyTorch peak allocated CUDA tensor memory, reset and recorded per epoch;
it is not total process or reserved GPU memory. The recorded GPU capacity was
4,294,443,008 bytes. No CUDA out-of-memory error occurred. Device, precision, image
dimensions, and batch size were not changed during the run.

## Per-class validation AP and support

Support is the number of ground-truth objects in the 80-image validation partition.
Per-class AP is COCO AP averaged over IoU 0.50:0.95. `N/A` is undefined because the
validation partition has no ground-truth examples for that class; it is not a zero AP.
The final two columns are the implementation's precision and recall at score threshold
0.50 and IoU 0.50 for epoch 3. Undefined ratios remain `N/A`.

| Class | Support | AP epoch 1 | AP epoch 2 | AP epoch 3 | Epoch 3 precision | Epoch 3 recall |
|---|---:|---:|---:|---:|---:|---:|
| barrier | 0 | N/A | N/A | N/A | N/A | N/A |
| bicycle | 15 | 0.000839 | 0.003465 | 0.062376 | N/A | 0.000000 |
| bus | 11 | 0.000263 | 0.000000 | 0.000239 | 0.000000 | 0.000000 |
| car | 648 | 0.117808 | 0.102038 | 0.114641 | 0.726316 | 0.212963 |
| construction_vehicle | 0 | N/A | N/A | N/A | N/A | N/A |
| motorcycle | 25 | 0.000839 | 0.000000 | 0.000000 | N/A | 0.000000 |
| pedestrian | 198 | 0.021381 | 0.006151 | 0.022254 | 0.636364 | 0.035354 |
| traffic_cone | 11 | 0.022574 | 0.000000 | 0.009901 | N/A | 0.000000 |
| trailer | 0 | N/A | N/A | N/A | N/A | N/A |
| truck | 44 | 0.042754 | 0.076332 | 0.030418 | 0.200000 | 0.022727 |

## Checkpoint and artifact integrity

The complete experiment directory is
`experiments/20261103T174810Z-9a1fe909/` and contains:

- `config.json` and `environment.json`
- `training_history.csv` with epoch losses, validation metrics, learning rate, duration,
  and peak CUDA allocated memory
- `validation_metrics.json` with all three epoch evaluations and per-class support/metrics
- `checkpoint_best.pth` and `checkpoint_last.pth`
- `experiment_summary.json` with selection, environment, configuration and provenance
  digests and the frozen selected-checkpoint hash

Both checkpoints loaded successfully using `torch.load(..., weights_only=True)`. They
are checkpoint version 2, at epoch 3/global step 726, with 284 model-state tensors and
nonempty optimizer, scheduler, scaler, RNG and DataLoader sampling state. Their model
tensors are identical. The summary SHA-256 for `checkpoint_best.pth` matches the file
(`27481566e7524d6c41adc88bd20855c2ccd372b91e6cb517b69c5ff7c79fc438`). Best validation
mAP was 0.0342612453 at epoch 3.

## Validation and limitations

- Focused training tests after adding epoch telemetry: **18 passed**.
- Final full available test suite: **134 passed in 155.46 seconds**.
- Three epochs completed without training or validation failure and without OOM.
- Best-effort CUDA execution emitted the documented nondeterministic ROI Align warning;
  exact bitwise repeatability is not claimed.
- This is a short engineering baseline on the small nuScenes mini dataset. Its validation
  values are descriptive for this run only and do not establish research-level
  performance or statistical conclusions.

**No test metrics were evaluated. No curation comparison or YOLO run was started.**
