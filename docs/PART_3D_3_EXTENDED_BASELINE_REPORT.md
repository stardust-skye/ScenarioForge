# Part 3D.3 — Extended Baseline Training (6 Epochs)

## 1. Objective and hypothesis

This controlled experiment tests whether extending the Faster R-CNN baseline from
three to six epochs improves validation detection. The hypothesis was that additional
optimization on the same 242 training images could improve validation metrics.

The new run was **fresh**, initialized from torchvision COCO_V1 weights; it did not
resume from or load the Part 3D.1 trained checkpoint. The new config differs from the
Part 3D.1 saved config only in `epochs` (3 → 6). No test-partition evaluation, YOLO,
or curation comparison was performed.

## 2. Experimental configuration and provenance

- **New experiment:** `20261103T182854Z-7a21c2a4`
- **Original three-epoch experiment:** `20261103T174810Z-9a1fe909`
- **Command:**

  ```powershell
  .\.venv-ml\Scripts\python.exe scripts\train_detection.py train --config configs\object_detection_extended_baseline_part3d3.json
  ```

- **Model and initialization:** torchvision Faster R-CNN MobileNetV3 Large 320 FPN,
  `fasterrcnn_mobilenet_v3_large_320_fpn`, initialized with
  `FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1`; the COCO class-specific
  predictor is replaced for ten ScenarioForge classes plus background.
- **Dataset:** nuScenes `v1.0-mini`; all 242 training samples and all 80 validation
  samples, with no sample/token file or sample limits. Batch size 1; zero workers;
  no external `image_size` override; validation every epoch.
- **Controls retained:** seed 42, AdamW, learning rate `1e-4`, weight decay `1e-4`,
  gradient clipping 1.0, StepLR step size 3 and gamma 0.1, CUDA, mixed precision,
  best-effort reproducibility.
- **Only configuration change:** `epochs` 3 → 6. This was programmatically checked
  against the original run's `config.json`.
- **Split-manifest digest:** `d93c15fac3e798343ea95fcefea3c03a0b189a0b9036b3f355354550109bf2ab`
  (same in both runs).
- **Image-inventory digest:** `0898a28fd93c46ea44eac57b728739aeb90d455431c4a1a210e37cd2ee483a98`
  (same in both runs).
- **Training-selection digest:** `a5d4022728c12970f3a85b0648fc78794d0c9e8574fd5d57703739d08dd754e8`
  (same in both runs; 242 selected, valid, held-out samples unchanged).
- **New configuration digest:** `e344a05d7753a9a042cce7175dc09375079283484f755b87288a723f25f348b2`.
- **New provenance digest:** `6edd05cbd2b2bcd640e7385b4680fdda16d8167edef8f20e84df6f2ddc8a5f56`.

The saved split has 242 train, 80 validation and 82 test samples and is scene-disjoint.
Training constructed train and validation datasets only. Each validation pass reported
80 images. The summary records `test_evaluated: false`; no `test_metrics.json` exists.
No split manifest, image, annotation, label or prior experiment artifact was modified.

## 3. Training trajectory

Losses are means over the 242 training batches in each epoch. Epoch duration includes
training and validation. The learning rate is the value used during that epoch.

| Epoch | Total loss | Classifier | Box regression | Objectness | RPN box regression | LR | Duration (s) | Peak CUDA allocated (bytes) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.073652 | 0.485235 | 0.412073 | 0.080145 | 0.096199 | 0.0001 | 184.902 | 435,321,856 |
| 2 | 0.990631 | 0.384120 | 0.477353 | 0.051090 | 0.078069 | 0.0001 | 63.795 | 435,321,856 |
| 3 | 0.989559 | 0.365659 | 0.511177 | 0.044115 | 0.068608 | 0.0001 | 135.783 | 435,321,856 |
| 4 | 0.904692 | 0.322226 | 0.488572 | 0.036998 | 0.056897 | 0.00001 | 139.025 | 435,321,856 |
| 5 | 0.875408 | 0.305206 | 0.482587 | 0.034040 | 0.053575 | 0.00001 | 138.559 | 435,321,856 |
| 6 | 0.855494 | 0.298059 | 0.474343 | 0.031352 | 0.051740 | 0.00001 | 147.911 | 435,321,856 |

Total measured epoch time was **809.976 seconds (13 minutes 29.976 seconds)**,
excluding model construction and artifact-writing overhead. The epoch-2 duration is
an unusually short observed value relative to the other epochs; the trainer did not
record batch-level timings, so its cause is unknown. All epochs completed normally.

Total loss decreased from 1.073652 to 0.855494. Classifier, objectness and RPN box
regression losses declined overall; detector box-regression loss rose through epoch 3
and then declined. The scheduler reduced the learning rate to `1e-5` for epoch 4, as
configured. These observations show loss progress but do not establish convergence.

## 4. Validation performance

Metrics use the saved validation partition of 80 images each epoch, TorchMetrics
`MeanAveragePrecision` with Faster-COCO-Eval, XYXY boxes and IoU thresholds 0.50
through 0.95.

| Epoch | Total train loss | LR | mAP@[.50:.95] | AP50 | AP75 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.073652 | 0.0001 | 0.029109 | 0.095447 | 0.010018 |
| 2 | 0.990631 | 0.0001 | 0.025360 | 0.083302 | 0.011421 |
| 3 | 0.989559 | 0.0001 | 0.037169 | 0.095682 | 0.020801 |
| 4 | 0.904692 | 0.00001 | **0.038926** | **0.108495** | **0.029708** |
| 5 | 0.875408 | 0.00001 | 0.036240 | 0.107916 | 0.016442 |
| 6 | 0.855494 | 0.00001 | 0.037744 | 0.102839 | 0.017195 |

The **best checkpoint** is epoch 4 by validation mAP@[.50:.95]. The **final epoch**
is epoch 6. Best-to-final mAP declined by 0.001182, AP50 by 0.005656 and AP75 by
0.012513. The later epochs therefore did not improve every validation metric despite
continuing loss reduction.

## 5. Per-class comparison

Support is the ground-truth object count in the 80 validation images. AP is COCO AP
over IoU 0.50:0.95. `N/A` denotes no validation ground truth, so AP is undefined,
not zero. The three experiments columns are the original three-epoch best (epoch 3),
the new six-epoch best (epoch 4), and the new final checkpoint (epoch 6).

| Class | Support | 3-epoch best AP (e3) | 6-epoch best AP (e4) | 6-epoch final AP (e6) |
|---|---:|---:|---:|---:|
| barrier | 0 | N/A | N/A | N/A |
| bicycle | 15 | 0.062376 | 0.006931 | 0.000000 |
| bus | 11 | 0.000239 | 0.000334 | 0.000000 |
| car | 648 | 0.114641 | 0.130602 | 0.131316 |
| construction_vehicle | 0 | N/A | N/A | N/A |
| motorcycle | 25 | 0.000000 | 0.000000 | 0.000000 |
| pedestrian | 198 | 0.022254 | 0.026415 | 0.026703 |
| traffic_cone | 11 | 0.009901 | 0.059406 | 0.049505 |
| trailer | 0 | N/A | N/A | N/A |
| truck | 44 | 0.030418 | 0.048793 | 0.056683 |

Car, pedestrian, traffic cone and truck improved over the original best at the
six-epoch best checkpoint; bus changed only slightly. Bicycle AP was substantially
lower than the original best, motorcycle remained zero, and the three unsupported
classes remain unmeasurable on this split. At the final epoch, car, pedestrian and
truck AP were higher than the original best; traffic-cone AP remained higher but
below its epoch-4 value. Bicycle and bus AP were zero. Class-level changes are
non-monotonic and especially uncertain for classes with 11–25 validation objects.

## 6. Three-epoch versus six-epoch comparison

### Complete mAP/AP trajectories

| Epoch | Original 3-epoch mAP | Original AP50 | Original AP75 | New 6-epoch mAP | New AP50 | New AP75 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.029374 | 0.092764 | 0.010623 | 0.029109 | 0.095447 | 0.010018 |
| 2 | 0.026855 | 0.075586 | 0.012874 | 0.025360 | 0.083302 | 0.011421 |
| 3 | 0.034261 | 0.090042 | 0.023811 | 0.037169 | 0.095682 | 0.020801 |
| 4 | — | — | — | 0.038926 | 0.108495 | 0.029708 |
| 5 | — | — | — | 0.036240 | 0.107916 | 0.016442 |
| 6 | — | — | — | 0.037744 | 0.102839 | 0.017195 |

### Best checkpoint and final epoch

| Comparison point | mAP@[.50:.95] | AP50 | AP75 |
|---|---:|---:|---:|
| Original best, epoch 3 | 0.034261 | 0.090042 | 0.023811 |
| New best, epoch 4 | **0.038926** | **0.108495** | **0.029708** |
| New final, epoch 6 | 0.037744 | 0.102839 | 0.017195 |
| New-best minus original-best | +0.004665 (+13.6%) | +0.018453 (+20.5%) | +0.005897 (+24.8%) |
| New-final minus original-best | +0.003483 (+10.2%) | +0.012797 (+14.2%) | -0.006616 (-27.8%) |

The extended run's best checkpoint exceeds the original best on all three metrics.
The final checkpoint exceeds the original best on mAP and AP50, but not AP75. The
best-of-six versus best-of-three comparison gives the six-epoch run more validation
selection opportunities; therefore the final-epoch comparison is also reported and
should not be treated as an isolated causal estimate.

The new run started from the same pretrained COCO weights and same seed/configuration
apart from epoch count, but `best_effort` CUDA execution is nondeterministic. Its
epoch-1 metrics are close to, but not identical with, the original run. Small metric
differences cannot be attributed solely to the extra epochs as if the two trajectories
were bitwise paired.

## 7. Runtime, GPU usage, and checkpoint integrity

- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU, 4,294,443,008 reported bytes.
- **Runtime:** Python 3.12.16; PyTorch 2.15.0+cu132; torchvision 0.29.2+cu132;
  TorchMetrics 1.9.0; Faster-COCO-Eval 1.8.0; CUDA 13.2; cuDNN 92600; Windows 11.
- **Peak allocated CUDA memory:** 435,321,856 bytes in every epoch; this is PyTorch
  allocated tensor memory, not total process or reserved memory.
- **Total epoch duration:** 809.976 seconds, compared with 503.019 seconds for the
  original three-epoch run. These are measured epoch durations; the six-epoch total
  is about 1.61x the original measured epoch time, with an unexplained short second
  epoch, not exactly 2x.
- **Checkpoint best:** `checkpoint_best.pth`, epoch 4, global step 968,
  284 model-state tensors; SHA-256
  `f0fa4af671937bd526624cd2ec59b95e4c9b07d0d543441c7ca8552ed76a2d7f`.
- **Checkpoint last:** `checkpoint_last.pth`, epoch 6, global step 1452,
  284 model-state tensors; SHA-256
  `fb2ac9e158f2cf613b5ab3f6b80dbda8d98838777982323a8126e1ead3eead23`.
- Both checkpoints loaded with `weights_only=True` and contain nonempty model,
  optimizer, scheduler and scaler state, Python/NumPy/PyTorch CPU/CUDA RNG state and
  DataLoader generator state. Their metadata configuration/provenance digests match
  the new run's `environment.json`.
- The experiment summary's best-checkpoint hash matches the actual
  `checkpoint_best.pth`. Best metric and epoch are recorded as 0.0389258154 and 4.
- All six history rows are present; each validation result records 80 images.
  Selection audit records 242 selected samples, the same selection digest as the
  original run, and held-out samples unchanged.
- `test_evaluated` is false and no test metrics artifact exists. No CUDA OOM occurred.

The run used `best_effort` reproducibility. Torchvision emitted the known warning
that `roi_align_backward_kernel` has no deterministic implementation. Configuration,
seed and provenance are recorded, but bitwise reproducibility is not guaranteed.

## 8. Limitations

- Each experiment is a single best-effort CUDA run. This is not a multi-seed
  comparison, confidence interval, or statistical-significance test.
- Validation contains only 80 images from two scenes; barrier, construction vehicle
  and trailer have no validation support. Rare-class AP is particularly unstable.
- The best-of-six checkpoint is selected among twice as many epochs as the original
  best-of-three checkpoint. Best-score comparisons have this selection advantage;
  the final-epoch metrics are given separately.
- Training loss declined, but validation metrics fluctuated and AP75 fell after the
  best epoch. Six epochs do not establish convergence or generalization.
- Best-effort CUDA execution emitted the expected nondeterministic ROI Align warning.
- The single unusually short epoch-2 duration has no batch-level profiler evidence
  explaining its cause.
- No test data was accessed or evaluated, and no annotation quality changes or
  human relabeling were attempted.

## 9. Recommendation for the next experiment

The six-epoch result is **directionally encouraging but not conclusive**: best
validation mAP/AP50/AP75 improve over the original three-epoch best, and final mAP
and AP50 also improve, while final AP75 is lower. Loss continues to decline, but
validation performance remains non-monotonic and the best checkpoint is epoch 4.

Before introducing resolution changes, class weighting, or other hyperparameters,
run a second fresh six-epoch replicate with a predeclared different seed while
retaining the same data, model initialization policy, optimizer, scheduler and
validation protocol. This assesses run-to-run variability. Do not select or report
the best seed based on test data. After repeated runs, consider a separately
controlled duration/resolution study; do not claim convergence from the current
six-epoch result.

## Files and validation

- New config: `configs/object_detection_extended_baseline_part3d3.json`.
- New experiment artifacts: `experiments/20261103T182854Z-7a21c2a4/`:
  `config.json`, `environment.json`, `training_history.csv`,
  `validation_metrics.json`, `experiment_summary.json`,
  `checkpoint_best.pth`, and `checkpoint_last.pth`.
- Report: `docs/PART_3D_3_EXTENDED_BASELINE_REPORT.md`.
- Full automated test suite after training: **134 passed in 145.52 seconds**.
- Dependency integrity check before training: `pip check` reported no broken
  requirements.
- Final `git diff --check` passed with no whitespace errors.
