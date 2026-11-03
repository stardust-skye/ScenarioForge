# Part 3D.2 — Baseline Diagnosis

## 1. Executive summary

The three-epoch baseline shows **some optimization progress, but weak and uneven
validation detection**. Mean training loss fell from 1.0806 to 0.9800, while
validation mAP@[.50:.95] moved 0.0294 → 0.0269 → 0.0343. AP75 increased across
epochs, but AP50 did not, and the final recall at the fixed 0.50 score threshold was
low (0.153). Three epochs are insufficient to infer convergence or a stable trend.

The saved-checkpoint inspection found frequent false negatives in the selected sparse,
pedestrian-containing, and dense scenes. It also found a few high-confidence
mislocalized/class-inconsistent detections. No duplicate false positives were found
among score-at-least-0.50 predictions in the five inspected images using one-to-one
same-class matching at IoU 0.50; this small sample cannot establish that duplicates
are absent elsewhere.

**Recommendation:** the smallest useful next experiment is a fresh, controlled
six-epoch run from the same COCO initialization, changing only total epochs while
keeping the seed, split, data budget, resolution, precision, batch size, optimizer,
learning rate and scheduler settings fixed. Continue validation-only selection and do
not access test data. This tests whether losses and validation AP continue improving
without prematurely changing several factors at once.

This diagnosis is an engineering analysis, not a research-performance claim. No
training, fine-tuning, test evaluation, annotation edits, or split changes were made.

## 2. Training behavior

| Epoch | Total loss | Classifier | Box regression | Objectness | RPN box regression | LR used | Epoch time (s) | Peak CUDA allocated (bytes) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.080561 | 0.489757 | 0.416406 | 0.078537 | 0.095861 | 0.0001 | 174.023 | 435,321,856 |
| 2 | 0.990161 | 0.385287 | 0.474757 | 0.052410 | 0.077706 | 0.0001 | 154.963 | 435,321,856 |
| 3 | 0.979974 | 0.357931 | 0.509286 | 0.043811 | 0.068946 | 0.0001 | 174.033 | 435,321,856 |

Total measured epoch time was 503.019 seconds (8 minutes 23 seconds), including
validation. Each epoch processed 242 training batches. The learning rate used was
1e-4 for all three epochs; the StepLR step size is three, so the decay would affect
the next epoch, which was not run.

Total loss decreased about 9.3% from epoch 1 to epoch 3. Classifier, objectness, and
RPN box-regression losses decreased each epoch. The detector box-regression loss
increased from 0.4164 to 0.5093, so the total-loss decrease is not uniform across
components. Epoch durations varied by about 19 seconds, with no error or CUDA OOM.
Peak allocated tensor memory was stable at about 10.1% of the reported 4,294,443,008
GPU bytes; this is allocated tensor memory, not total process or reserved memory.

**Interpretation:** losses show a downward aggregate trend over these three epochs,
but the rising detector box-regression component and non-monotonic validation AP
argue against describing the run as uniformly improving. There is no evidence here
to diagnose numerical instability, but three epochs do not demonstrate convergence.
GPU utilization and reserved memory were not recorded.

## 3. Validation performance

The metrics below were recorded on the same 80 validation images after every epoch.
They use TorchMetrics `MeanAveragePrecision` with Faster-COCO-Eval, XYXY boxes and
IoU thresholds 0.50 through 0.95.

| Epoch | mAP@[.50:.95] | AP50 | AP75 | Precision at score .50 / IoU .50 | Recall at score .50 / IoU .50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.029374 | 0.092764 | 0.010623 | 0.620968 | 0.161765 |
| 2 | 0.026855 | 0.075586 | 0.012874 | 0.523077 | 0.178571 |
| 3 | 0.034261 | 0.090042 | 0.023811 | 0.695238 | 0.153361 |

mAP fell in epoch 2 and recovered in epoch 3. Epoch 3 is the best of these three by
mAP and AP75, but its AP50 is slightly below epoch 1. Precision/recall are separate
fixed-threshold metrics, not COCO AP: at score 0.50 the epoch-3 precision is higher
and recall lower than epoch 2. The low fixed-threshold recall is consistent with the
misses seen in the selected visual samples, but those samples are illustrative, not a
validation-wide estimate.

## 4. Per-class analysis

“Train objects” and “validation support” are counts of accepted 2D annotation boxes,
not counts of images. AP is COCO AP averaged over IoU 0.50:0.95. `N/A` means the
validation partition contains no ground-truth object for that class, so AP is
undefined; it does **not** mean AP is zero. A numeric `0.000000` is a defined,
zero-valued AP because the validation partition does contain ground truth.

| Class | Train objects | Validation support | AP epoch 1 | AP epoch 2 | AP epoch 3 |
|---|---:|---:|---:|---:|---:|
| barrier | 527 | 0 | N/A | N/A | N/A |
| bicycle | 54 | 15 | 0.000000 | 0.003465 | 0.062376 |
| bus | 136 | 11 | 0.000263 | 0.000000 | 0.000239 |
| car | 1,263 | 648 | 0.117808 | 0.102038 | 0.114641 |
| construction_vehicle | 97 | 0 | N/A | N/A | N/A |
| motorcycle | 44 | 25 | 0.000839 | 0.000000 | 0.000000 |
| pedestrian | 814 | 198 | 0.021381 | 0.006151 | 0.022254 |
| traffic_cone | 580 | 11 | 0.022574 | 0.000000 | 0.009901 |
| trailer | 17 | 0 | N/A | N/A | N/A |
| truck | 104 | 44 | 0.042754 | 0.076332 | 0.030418 |

Car is the largest class in both partitions. Training support ranges from 1,263 car
boxes to 17 trailer boxes (about 74:1); bicycle, motorcycle, and truck support is
also modest. Validation support is especially low for bus, traffic cone, and bicycle
(11, 11, and 15 boxes). Barrier, construction vehicle, and trailer are present in
training but absent in validation because of the existing scene-level split, so this
run cannot measure their AP. Low support can make per-class estimates less stable,
but these data do not establish class imbalance as the cause of low AP.

The zero AP values for bicycle (epoch 1), bus (epoch 2), motorcycle (epochs 2–3),
and traffic cone (epoch 2) are defined zero results. These must not be conflated
with the three `N/A` classes that have no validation support.

## 5. Prediction visualizations and observations

The visualizations use the saved best checkpoint, in evaluation/inference mode, on
five selected validation images. Green boxes are ground truth. Predictions with
score >=0.25 are displayed (up to 40 per image); orange/red/blue labels indicate
same-class TP/FP at score >=0.50 and IoU >=0.50, or a displayed prediction below
the 0.50 matching threshold. These five samples were chosen to cover sparse and
dense scenes, pedestrians, vehicles, and multiple categories; they are not a
random sample and are not used to compute new validation metrics.

| Validation index | Selection | Visualization |
|---:|---|---|
| 6 | Sparse, two distant cars | [06 sparse vehicle scene](../experiments/20261103T174810Z-9a1fe909/diagnostics/06_n015-2018-10-02-10-50-40+0800__CAM_FRONT__1538448757012460.png) |
| 2 | Cars, motorcycle, and pedestrians | [02 pedestrians and mixed vehicles](../experiments/20261103T174810Z-9a1fe909/diagnostics/02_n015-2018-10-02-10-50-40+0800__CAM_FRONT__1538448750012460.png) |
| 4 | Bus, truck, and cars | [04 multiple vehicle categories](../experiments/20261103T174810Z-9a1fe909/diagnostics/04_n015-2018-10-02-10-50-40+0800__CAM_FRONT__1538448759012460.png) |
| 42 | Traffic cone, pedestrians, truck, and cars | [42 mixed urban scene](../experiments/20261103T174810Z-9a1fe909/diagnostics/42_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151609512404.png) |
| 73 | Dense multi-category scene, 27 ground-truth objects | [73 dense multi-category scene](../experiments/20261103T174810Z-9a1fe909/diagnostics/73_n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151613912404.png) |

Quantitative image-level matching at score 0.50 and same-class IoU 0.50, from the
same five selected samples:

- **Index 6:** 0/2 car ground truths matched. The only displayed car prediction
  scored 0.281 and had maximum same-class IoU 0.00; it appears over a roadside sign
  rather than either distant ground-truth car. This is a low-confidence apparent
  false positive, not a score-0.50 metric false positive.
- **Index 2:** 1/6 cars matched. A car at score 0.882 had IoU 0.894; another at
  score 0.669 had IoU 0.350 and was a false positive at the matching threshold.
  Five cars, the motorcycle, and both pedestrians were missed at score 0.50. A
  pedestrian prediction at 0.485 had IoU 0.00 and remains below the fixed score
  threshold.
- **Index 4:** None of the two cars, bus, or truck was matched at score 0.50. The
  displayed bus prediction scored 0.336 and had IoU 0.00 with the same-class bus
  target, a clear selected-example localization miss.
- **Index 42:** 3/9 cars were matched, with IoUs 0.802, 0.822, and 0.938 for the
  three highest-scoring matched cars. Six cars, both pedestrians, the traffic cone,
  and the truck were missed at score 0.50. The displayed bus prediction (0.448)
  has no bus ground truth in this image and is a low-confidence class false alarm.
- **Index 73:** In this dense scene, 2/20 cars were matched, 2 car predictions were
  false positives (scores 0.811 and 0.677; same-class IoUs 0.303 and 0.477), and
  18 cars were missed. A pedestrian prediction at 0.513 had IoU 0.00; all four
  pedestrian ground truths were missed. Bicycle, motorcycle, and truck ground
  truths were also missed. No duplicate false positive was identified by the
  one-to-one matching check in these five images.

These are sample-level observations only. Far-away objects, occlusion, overlap and
projected-box extents make some visual judgments ambiguous; the inspection does not
relabel examples or determine annotation correctness.

## 6. Error analysis

### Verified observations

- The run used 242 train images from six scenes and 80 validation images from two
  scenes. The dataset is nuScenes `v1.0-mini`, with 3,636 train boxes and 952
  validation boxes.
- All inspected image records are 1600x900. No external `image_size` override was
  set. The loaded torchvision detector reports transform `min_size=(320,)` and
  `max_size=640`; 1600x900 images are therefore processed internally at about
  640x360 while predicted boxes are mapped back to original coordinates.
- Validation support is highly uneven and three classes have no validation
  positives. The train split has support for all ten classes.
- The initialized COCO detector's class-specific Fast R-CNN predictor is replaced
  by a new predictor for ten ScenarioForge classes plus background. Generic
  pretrained feature/RPN weights are retained, but the COCO class-specific
  classifier and box regressor are not the ScenarioForge head.
- Detector inference has score threshold 0.05, NMS IoU threshold 0.50, and at most
  100 detections per image. COCO AP uses IoU 0.50:0.05:0.95 over returned
  detections; the additional precision/recall values use score 0.50 and IoU 0.50.
  The diagnostic overlays use score 0.25 only for readability, and are not a
  replacement for the saved metrics.
- The Part 3A audit verified the projection geometry against the nuScenes devkit.
  Those targets are image-plane bounds of projected 3D cuboids, not manually
  traced visible silhouettes; occlusion, truncation, and image-edge clipping can
  make a projected box include background or differ from the visible object.

### Plausible explanations — hypotheses, not established causes

- **Short training:** three epochs may be insufficient, but the observed AP trend is
  too short and non-monotonic to establish undertraining or convergence.
- **Dataset and class support:** only six train scenes are available. Several classes
  have few train/validation boxes and three have no validation examples. This may
  contribute to weak or variable class-specific AP; it is not proven to be causal.
- **Resolution and small objects:** after internal resizing to about 640x360,
  median original validation boxes include pedestrians around 37x79 pixels and
  bicycles around 32x61 pixels. Their approximate resized medians are only about
  15x32 and 13x25 pixels. Limited spatial detail could hurt localization and
  recognition, but this experiment does not isolate resolution as a cause.
- **Projected 3D targets:** the distinction between cuboid projection and visible
  silhouette could contribute to localization disagreement, especially for
  occluded/truncated objects. The geometry audit found no projection mismatch, and
  no human-redrawn comparison was performed here.
- **Pretraining/domain gap:** COCO provides useful generic representations but has
  different camera viewpoints, scene density, and label distribution; moreover,
  the class-specific predictor was reinitialized. These differences are plausible
  contributors, not measured attribution.
- **Confidence and evaluation settings:** low recall at score 0.50 suggests many
  detections are either absent, low-confidence, or mismatched. However, the COCO AP
  metric ranks returned scores and does not apply the separate 0.50 precision/
  recall cutoff. Threshold tuning alone is therefore not a demonstrated remedy
  for low AP.
- **Annotation quality:** selected examples show misses and some bounding-box
  mismatch, but visual inspection cannot separate model error from the known
  projected-cuboid/visibility differences. Existing projection checks support
  geometric correctness; targeted human review may still be useful.

## 7. Limitations

- This is a five-image, purposively selected visual inspection, not a blinded or
  random validation review. It cannot characterize the frequency of each error.
- The validation partition contains only two scenes and has no support for barrier,
  construction vehicle, or trailer. Per-class AP for those classes is undefined.
- `best_effort` CUDA execution is not bitwise reproducible; torchvision emitted the
  expected nondeterministic ROI Align backward warning during training. This
  diagnosis itself performs inference only, without AMP or gradients.
- The baseline's three epochs cannot establish convergence, generalization, or
  statistically meaningful performance.
- No test data was accessed or evaluated, and no new full-validation metric run was
  performed.

## 8. Recommended next experiment

Run a **fresh six-epoch training experiment from the same COCO_V1 initialization**.
Keep the existing seed-42 split, all 242 training and 80 validation images, seed 42,
batch size 1, CUDA, AMP, best-effort mode, no external resize, optimizer, 1e-4
initial learning rate, and scheduler step size/gamma unchanged. Change only total
epochs from 3 to 6. Validate every epoch and retain per-class support/AP, losses,
learning rate, duration, and peak memory. Under the existing step size of three,
the learning rate should remain 1e-4 for epochs 1–3 and decay for epochs 4–6.

Use the run to assess whether loss components and validation metrics continue to
improve or plateau; do not infer convergence from six epochs alone. Start fresh
rather than attempting to resume the completed, immutable Part 3D.1 experiment.
Do not tune the score threshold or add class weighting in this next run, because
neither is isolated by current evidence. If extended training remains weak, the next
separate experiments should investigate detector resolution and targeted annotation
quality/class-support issues one factor at a time. Keep test evaluation disabled.

## 9. Reproducibility and provenance

- **Experiment:** `20261103T174810Z-9a1fe909`
- **Checkpoint:** `checkpoint_best.pth`, SHA-256
  `27481566e7524d6c41adc88bd20855c2ccd372b91e6cb517b69c5ff7c79fc438`
- **Split-manifest digest:** `d93c15fac3e798343ea95fcefea3c03a0b189a0b9036b3f355354550109bf2ab`
- **Image-inventory digest:** `0898a28fd93c46ea44eac57b728739aeb90d455431c4a1a210e37cd2ee483a98`
- **Experiment provenance digest:** `d677858f00a5fa83e51d35d55d71f51d3dc4960870cd4aa2938297e8e0395471`
- **Environment:** Python 3.12.16; PyTorch 2.15.0+cu132; torchvision 0.29.2+cu132;
  TorchMetrics 1.9.0; Faster-COCO-Eval 1.8.0; CUDA 13.2; cuDNN 92600; Windows 11;
  NVIDIA GeForce RTX 3050 Laptop GPU.
- **Inspection procedure:** `.venv-ml` loaded the persisted configuration, constructed
  a validation-only dataset using the existing split manifest, loaded the best
  checkpoint with `weights_only=True`, set the model to eval mode, and used
  `torch.inference_mode()` on indices 6, 2, 4, 42, and 73. Inference used CUDA
  without autocast. No optimizer, training loader, or test dataset was used.
- **Reproduction command:** from the repository root,
  `.venv-ml\Scripts\python.exe experiments\20261103T174810Z-9a1fe909\diagnostics\reproduce_diagnosis.py`.
  The script checks the checkpoint and provenance digests, requires the full
  validation split, and writes only files under this experiment's `diagnostics/`
  directory.
- `diagnostics/selection_and_inspection.json` records sample selection, checkpoint
  hash, thresholds, device, and sample-level match counts.
  `diagnostics/prediction_error_details.json` records displayed predictions and
  their maximum same-class ground-truth IoUs. The validation-only reproduction
  script is `diagnostics/reproduce_diagnosis.py`.
- Existing configuration, environment, training history, validation metrics,
  summary, and checkpoints were read only. New files are limited to this report
  and the requested `diagnostics/` outputs.
