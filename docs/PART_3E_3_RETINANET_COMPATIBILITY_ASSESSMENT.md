# Part 3E.3 — Torchvision RetinaNet Compatibility Assessment

## Executive summary

**Decision: CONDITIONAL GO.** Torchvision RetinaNet is available in the
existing ML environment and its list-of-images / target-dictionary training
API can fit the current ScenarioForge adapter and evaluator design. The
recommended candidate is **RetinaNet ResNet-50 FPN V2**, using
`RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1` only after its checkpoint-specific
license is clarified.

The code license is confirmed as BSD 3-Clause, but the official Torchvision
documentation does not grant an explicit license for this exact binary
checkpoint; it cautions users that pretrained weights may have separate
terms. Thus, the weight license remains **unresolved** and must not be
inferred from the source-code license. No weights were downloaded.

Technically, the current `.venv-ml` imports the exact RetinaNet builder and
weight enum. The interface and ten-class adaptation appear compatible with
the existing framework, but no model using pretrained weights, training
batch, empty target, or GPU-memory smoke test was run. A 4 GiB fit is
plausible only with a conservative batch-one, 320/640 resize and AMP
configuration; it is **not verified**. Do not implement or train until this
assessment is reviewed and the checkpoint-license and smoke-test gates are
approved.

## 1. Environment compatibility

### Verified runtime facts

The `.venv-ml` environment was inspected without installing or upgrading
packages:

| Component | Observed value |
|---|---|
| OS | Windows |
| Python | 3.12.16 |
| PyTorch | 2.15.0+cu132 |
| TorchVision | 0.29.2+cu132 |
| CUDA runtime reported by PyTorch | 13.2 |
| CUDA available | Yes |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU memory | 4,294,443,008 bytes (4 GiB) |
| ML requirements | Torch 2.15.0 and TorchVision 0.29.2 are pinned |

`torchvision.models.detection.retinanet_resnet50_fpn_v2` and
`RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1` import successfully. The runtime
reported this exact weight URL:

`https://download.pytorch.org/models/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth`

The installed weight metadata reports 38,198,935 parameters and 146.0 MB
checkpoint size. These are metadata/API facts, not a local download or
training-memory measurement.

For API inspection only, the RetinaNet builder was also instantiated on CPU
with `weights=None` and ten output classes; its transform defaults and head
attributes were read. No pretrained model was loaded, no forward/backward
pass was run, and no CUDA memory measurement was made.

### Official references

- [Torchvision 0.29 RetinaNet ResNet-50 FPN V2 API and weights](https://docs.pytorch.org/vision/0.29/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html)
- [Torchvision 0.29 detection model source](https://github.com/pytorch/vision/blob/v0.29.2/torchvision/models/detection/retinanet.py)
- [Torchvision 0.29.2 source license](https://github.com/pytorch/vision/blob/v0.29.2/LICENSE)
- [Torchvision models and pretrained weights guidance](https://docs.pytorch.org/vision/0.29/models.html)
- [Exact COCO_V1 weight file](https://download.pytorch.org/models/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth)

## 2. Architecture and pretrained-weight licensing

### Candidate

**RetinaNet ResNet-50 FPN V2** is a one-stage, dense, anchor-based detector.
Torchvision's versioned model documentation identifies the `COCO_V1` weights,
reports COCO-val2017 box mAP 41.5, 38,198,935 parameters, and a 146.0 MB file.
The installed enum metadata confirms the exact official checkpoint URL and
the COCO category list.

The V2 model is a reasonable architecture candidate for the framework. Its
ResNet-50/FPN and large default image scale make it materially heavier than
the current Faster R-CNN MobileNetV3 320 FPN candidate.

### Source-code license: confirmed

The versioned Torchvision 0.29.2 repository `LICENSE` is BSD 3-Clause. This
applies to the Torchvision source code under that license; it does not, by
itself, establish the terms for a separately distributed pretrained
checkpoint.

### Exact checkpoint license: unresolved

- The exact official checkpoint is hosted by `download.pytorch.org` and
  selected by `RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1`.
- The versioned model page and installed weight metadata identify the weight
  enum, COCO performance/category metadata, recipe, URL, and file size, but do
  not provide an explicit license for this checkpoint.
- Torchvision's pretrained-model guidance warns that provided weights may
  have their own licenses or terms based on the training dataset and places
  responsibility on the user to determine permission for their use.
- Therefore, do not claim that the checkpoint is BSD-3, Apache-2.0, or
  otherwise permissively licensed solely because Torchvision source code is
  BSD-3. The checkpoint license and any applicable dataset-derived terms
  need authoritative clarification before download or use.

**Licensing gate:** obtain checkpoint-specific written terms from an
authoritative source (for example, Torchvision maintainers or the checkpoint
publisher) and record them. Until then, technical feasibility does not equal
permission to use the pretrained weights.

## 3. Model and dataset interface compatibility

### Existing ScenarioForge contracts

The current dataset returns float32 RGB images in `[0, 1]`, XYXY pixel boxes,
and contiguous positive foreground labels beginning at 1. Its category
mapping is authoritative, and its output validation checks class range,
finite coordinates, box dimensions, and image bounds. The persisted split
factory supports explicit split selection; the experiment constructs only
training and validation datasets for training.

The existing model-adapter protocol asks each model to implement
`compute_loss(images, targets)` and `predict(images)`. The trainer accepts a
generic scalar loss dictionary, applies the configured autocast/GradScaler
and optimizer, and the evaluator consumes standard Torchvision prediction
dictionaries. These abstractions provide a direct integration path without
changing Faster R-CNN behavior.

Relevant implementation references:

- [model_factory.py](../src/scenarioforge/training/model_factory.py#L15) —
  adapter contract, Faster R-CNN adapter, constructor, and output validation.
- [trainer.py](../src/scenarioforge/training/trainer.py#L42) — optimizer,
  scheduler, AMP scaler, generic training loop and model calls.
- [evaluator.py](../src/scenarioforge/training/evaluator.py#L43) — COCO-style
  metrics, prediction validation, and evaluator label conversion.
- [config.py](../src/scenarioforge/training/config.py#L12) — currently only
  permits the Faster R-CNN architecture.
- [experiment.py](../src/scenarioforge/training/experiment.py#L64) — current
  transforms and explicit training/validation dataset construction.
- [dataloader.py](../src/scenarioforge/vision_data/dataloader.py#L66) —
  split-restricted dataset factory.
- [dataset.py](../src/scenarioforge/vision_data/dataset.py#L235) — target and
  box validation; label mapping follows at line 272.
- [transforms.py](../src/scenarioforge/vision_data/transforms.py#L27) —
  synchronized image/box resizing and float-tensor conversion.
- [requirements-ml.txt](../requirements-ml.txt) — pinned Torch and Torchvision
  versions.

### Torchvision RetinaNet differences and adapter requirements

| Concern | Current Faster R-CNN path | RetinaNet path and required adapter behavior |
|---|---|---|
| Training call | Detector receives lists of images and targets, returns a loss dictionary in train mode. | Same high-level list API; training returns classification and box-regression losses. Preserve generic loss validation and do not assume Faster R-CNN loss names. |
| Evaluation call | Returns one prediction dictionary per image. | Same standard fields: `boxes`, `labels`, and `scores`; convert/validate through the existing prediction contract. |
| Boxes | XYXY absolute pixel coordinates. | Same XYXY pixel convention; RetinaNet's model transform resizes images and boxes together and postprocesses boxes back to original input dimensions. |
| Class count | Faster R-CNN uses one background class plus ten foreground classes (11 outputs); foreground labels are 1–10. | Torchvision RetinaNet documents `num_classes` including background. Use `num_classes=11` and preserve foreground labels 1–10; its focal-loss targets use class indices for matched anchors, while unmatched/background anchors have all-zero class targets. Verify this in a bounded loss smoke test before training. |
| Pretrained head | Existing builder replaces the COCO box predictor with a fresh ScenarioForge head. | Load the COCO checkpoint only after license approval, then replace the RetinaNet classification head with a newly initialized 11-output head. Keep compatible backbone, FPN, and box-regression weights. Do not retain or describe the COCO classifier as the ten-class ScenarioForge classifier. |
| Loss names | The adapter currently passes the Faster R-CNN loss dictionary through. | RetinaNet's keys differ; the generic trainer can sum scalar components, but history must preserve RetinaNet component names. |
| Empty targets | Existing dataset represents empty boxes as `[0, 4]` and labels as `[0]`. | Preserve that shape and dtype. Source/API inspection is not an empirical empty-target test; exercise it in the later smoke suite. |

The evaluator maps model labels 1–10 to metric labels 0–9 for TorchMetrics.
RetinaNet predictions should be adapted to the same positive ScenarioForge
labels before they reach `validate_predictions`; then the existing metric
conversion can remain shared. This avoids an off-by-one class error.

### Required framework changes (assessment only)

No implementation changes were made. A future approved integration would
need to:

1. Extend `ModelConfig` validation and experiment serialization for a
   RetinaNet architecture and explicit RetinaNet preprocessing settings.
2. Add a modular RetinaNet adapter and architecture-specific builder without
   changing the existing Faster R-CNN adapter's behavior.
3. Replace only the pretrained classification head after loading the
   approved COCO checkpoint, and validate output count, labels, and head
   initialization.
4. Record architecture, Torch/Torchvision versions, exact weight enum and
   checkpoint URL, checkpoint SHA-256, head replacement, image-size settings,
   class mapping, split/inventory digests, and training selection in
   experiment/checkpoint provenance.
5. Add focused tests for head adaptation, class mapping, target/prediction
   conversion, empty images, loss validation, checkpoint round-trip, and
   validation-only evaluation.

## 4. GPU memory feasibility and conservative configuration

### Verified versus estimated

- **Verified:** the GPU has 4 GiB; the candidate has about 38.2 million
  parameters; Torchvision's default RetinaNet transform is min-size 800 and
  max-size 1333.
- **Verified comparison baseline:** the preserved six-epoch Faster R-CNN
  configuration uses batch size 1, CUDA, mixed precision enabled, no explicit
  dataset resize, seed 42, six epochs, and best-effort reproducibility. The
  Faster R-CNN MobileNetV3 320 FPN model transform is min-size 320 and
  max-size 640.
- **Not verified:** RetinaNet training peak memory, AMP numerical stability,
  or whether any configuration fits the RTX 3050's 4 GiB.

The default 800/1333 RetinaNet training resolution is not an appropriate
starting assumption for this GPU or a close preprocessing match to the
existing 320-FPN baseline.

### Recommended smoke-test configuration

For a future, explicitly approved one-batch smoke test:

- Batch size: **1**.
- RetinaNet transform: **min-size 320, max-size 640**, preserving aspect ratio.
  The current 900×1600 images would be approximately 320×569 after internal
  resize, matching the existing Faster R-CNN transform.
- AMP: **enabled**, matching the preserved baseline configuration, with the
  existing CUDA GradScaler path. Require finite component losses and finite
  gradients; do not silently disable AMP or alter precision if the test fails.
- Reset and record CUDA peak allocated memory around the actual forward,
  backward, optimizer step, and checkpoint reload.

These are conservative **recommendations**, not validated guarantees. If
batch-one at the matched transform and AMP fails or exceeds memory, stop and
seek approval before changing the resolution, precision, or architecture.

## 5. Comparison fairness

A useful controlled comparison is possible in principle if the six-epoch
RetinaNet run uses:

- the exact same seed-42 scene-level manifest, annotation version, category
  mapping, all 242 training images, and all 80 validation images;
- the same pretrained COCO initialization policy, subject to license
  clearance;
- the same six-epoch budget and validation after every epoch;
- the matched 320/640 image transform, batch size 1, AMP setting, and
  training-selection rules;
- the existing COCO-style TorchMetrics/Faster-COCO-Eval implementation and
  identical category IDs, IoU thresholds, and checkpoint-selection metric.

The existing six-epoch Faster R-CNN configuration also uses AdamW with
learning rate `1e-4`, weight decay `1e-4`, StepLR step size 3 / gamma 0.1,
gradient clipping at 1.0, seed 42, and best-effort reproducibility. Reusing
these settings would control the written configuration, but would not prove
they are equally optimal for both architectures. RetinaNet's anchor design,
classification loss, head, and convergence behavior differ. Report each
model's loss components separately, and do not present equal epoch counts as
equal compute or a fully controlled optimizer response.

The comparison would be **matched on data, metrics, nominal pixel scale,
batch, precision, and epochs**, but remain limited by architecture-specific
optimization and a single run. Do not use the test split during training or
model selection, and do not make a universal superiority or significance
claim.

## 6. Risks and limitations

1. **Checkpoint-license ambiguity:** the central pretrained artifact has no
   explicit checkpoint-specific license verified in this assessment.
2. **Unmeasured GPU fit:** 38.2M parameters and the 146 MB checkpoint do not
   account for FPN activations, anchors, gradients, or optimizer state.
3. **Default scale mismatch:** RetinaNet defaults to 800/1333; use the
   explicitly recorded 320/640 setting for the first candidate comparison.
4. **Head adaptation is untested:** the ten-class head must be freshly
   initialized, with compatible COCO backbone/FPN and box-regression weights
   retained.
5. **Empty images are untested:** verify actual zero-target training behavior
   before longer training.
6. **Framework config is Faster R-CNN-only:** adding RetinaNet will require a
   model configuration/adapter extension and tests; no such code exists yet.
7. **One six-epoch run cannot establish convergence, statistical significance,
   or general architecture superiority.**

## 7. Recommended next steps

1. Obtain authoritative, checkpoint-specific licensing terms for
   `RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1`; otherwise do not download or use
   it.
2. After report review and explicit implementation approval, add a separate
   RetinaNet config and adapter while keeping Faster R-CNN unchanged.
3. Add unit tests for class count/labels, new-head initialization, prediction
   conversion, empty targets, and safe checkpoint provenance.
4. Run only a batch-one GPU smoke test at min-size 320 / max-size 640 with AMP
   enabled; validate finite losses/gradients, save/reload, and record peak
   memory.
5. If that passes and checkpoint terms are cleared, run the matched six-epoch
   train/validation experiment. Keep test data untouched.

## 8. Final decision

**CONDITIONAL GO — technical candidate only.** The installed Torchvision API,
dataset contract, trainer, and evaluator make RetinaNet a plausible second
architecture without replacing the existing ML stack. This is **not** approval
to implement, download weights, or train. Proceed only after (1) this
assessment is reviewed and implementation is explicitly approved,
(2) checkpoint-specific licensing is clarified, and (3) the adapted model
passes the prescribed batch-one CUDA smoke test without unapproved
configuration changes.
