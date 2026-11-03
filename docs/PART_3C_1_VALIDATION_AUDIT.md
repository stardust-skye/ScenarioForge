# Part 3C.1 Training Infrastructure Validation Audit

## Executive verdict

**Reject longer training for now.** The detector construction, box pipeline, training loss path, and COCO-style evaluation are substantially correct. The focused CUDA check completed a real train-split forward/backward pass and found nonzero gradients on all 192 trainable parameters. The current infrastructure is not yet a sufficiently reproducible and isolated basis for a longer research run: checkpoint resume does not restore random/data-loader state or verify the full saved experiment provenance, and deterministic mode explicitly permits nondeterministic operations.

The split manifest and training-selection audit are recorded, and the trainer receives only training and validation loaders. However, each split dataset instance still validates the full image inventory and loads annotations for all samples into memory. This does not demonstrate leakage into optimization, but it is weaker than a strict held-out access boundary.

No test-partition metrics were computed in this audit. No annotations, split assignments, or prior experiment results were changed. No implementation code or tests were modified.

The retained smoke run is a systems check only: one selected training sample, two validation samples, one epoch, validation mAP/AP50/AP75 of 0.0, and no test metrics. These values are not evidence of research-level model performance.

## Findings

### Major

#### M1. Checkpoint resume does not preserve the full experiment trajectory

Checkpoints contain model, optimizer, scheduler, scaler, epoch, best validation AP, and metadata. They do not contain Python, NumPy, CPU/CUDA PyTorch, or DataLoader-generator states, nor a global step or early-stopping counter. Resume restores the epoch and CSV history but starts with freshly seeded process and loader state. A resumed shuffled run therefore is not guaranteed to continue the same sample order or stochastic augmentation trajectory.

The experiment metadata records dataset version, image-inventory digest, model configuration, and training-selection audit, but resume/evaluation compatibility checks compare only experiment ID, split-manifest digest, and label mapping. A changed training selection or other configuration can therefore escape those checks if the split and label map remain unchanged.

**Remediation:** persist and restore all relevant RNG and DataLoader-generator states and training counters; compare a canonical configuration/provenance digest covering model, optimizer/scheduler, dataset/inventory, split, and training selection before loading state. Reject incompatible resumes before mutating the model or optimizer.

Evidence: `src/scenarioforge/training/checkpointing.py:24-80`; `src/scenarioforge/training/trainer.py:79-113`; `src/scenarioforge/training/seed.py:14-26`; `src/scenarioforge/vision_data/dataloader.py:28-55`; `src/scenarioforge/training/experiment.py:106-122, 264-269`.

#### M2. “Deterministic” configuration allows known nondeterministic operations

`initialize_seed` seeds Python, NumPy, PyTorch, and CUDA, and configures cuDNN determinism. However, it calls `torch.use_deterministic_algorithms(..., warn_only=True)`, so unsupported nondeterministic operations continue rather than fail. The executed notebook records a warning that `roi_align_backward_kernel` has no deterministic implementation. The saved config’s `deterministic: true` therefore represents a best-effort setting, not a guarantee of repeatable or bitwise-identical training.

**Remediation:** either make strict determinism fail on unsupported operations when requested, or explicitly record the run as nondeterministic/best-effort and document the affected operations. Validate repeatability separately from configuration serialization.

Evidence: `src/scenarioforge/training/seed.py:14-26`; `notebooks/06_Object_Detection_Training_Smoke_Test.ipynb:190-210`.

### Minor

#### m1. Split datasets expose held-out metadata and annotations during setup

`make_split_datasets` creates a dataset for each split, and each dataset receives the full image inventory and annotations CSV. Dataset construction opens/verifies every inventory image and loads all annotation rows into `_annotations`; only `image_records` and `__getitem__` are restricted to that instance’s requested sample tokens. The training DataLoader itself is built from the audited training subset, and `DetectionTrainer.fit` accepts only train and validation loaders, so this audit found no path by which test labels enter optimization or validation metrics. Nonetheless, test assets and labels are accessed during experiment setup, which weakens the promised held-out boundary.

**Remediation:** construct each split dataset from only its own image records and annotations, and avoid opening/verifying test files during training or validation setup. Keep class mapping fixed from the authoritative mapping file rather than deriving any statistics from held-out labels.

Evidence: `src/scenarioforge/vision_data/dataloader.py:66-96`; `src/scenarioforge/vision_data/dataset.py:55-116, 169-208`; `src/scenarioforge/training/experiment.py:54-81, 127-146`.

#### m2. The post-selection test gate is a mutable marker, not a checkpoint seal

Test evaluation is blocked until `experiment_summary.json` exists, its `selection_and_checkpoint_frozen` field is truthy, and the requested checkpoint filename is `checkpoint_best.pth`. This is a useful workflow guard, but the marker and filename do not bind evaluation to an immutable checkpoint digest. The checkpoint metadata check also omits training-selection and full configuration digests.

**Remediation:** record and validate a content hash for the selected checkpoint and its selection/configuration provenance before permitting test evaluation; fail closed if the selected artifact changes after validation-based selection.

Evidence: `src/scenarioforge/training/experiment.py:244-269, 294-299`; `src/scenarioforge/training/trainer.py:180-188`.

#### m3. Backbone freezing leaves BatchNorm running state trainable

The freeze helper disables parameter gradients, but `fit` calls `model.train()` before applying that flag. A runtime inspection found all 46 `BatchNorm2d` modules in the MobileNetV3 backbone still in training mode after `set_backbone_trainable(False)`; their running statistics can therefore update during nominally frozen epochs.

**Remediation:** define whether “freeze” includes normalization state. If it does, keep the frozen backbone’s BatchNorm modules in evaluation mode during frozen epochs and test that running buffers remain unchanged.

Evidence: `src/scenarioforge/training/model_factory.py:75-77`; `src/scenarioforge/training/trainer.py:123-126, 259-262`.

#### m4. The notebook smoke optimizer step is not isolated from the reported epoch

The executed notebook calls `run_one_batch_smoke_test()` and then `experiment.train()` on the same experiment/trainer. The smoke method performs an optimizer step; trainer creation is cached, so the subsequent fit continues from the updated model and optimizer state. That extra update is not represented as an epoch or global step in the training history.

**Remediation:** run the one-batch check on a disposable model/optimizer or create a fresh experiment for the epoch smoke run; alternatively, explicitly record the preliminary step and its optimizer state.

Evidence: `src/scenarioforge/training/experiment.py:188-230`; `src/scenarioforge/training/trainer.py:115-127`; `notebooks/06_Object_Detection_Training_Smoke_Test.ipynb:181-228, 243-268`.

#### m5. Training history records the learning rate after scheduler advancement

The epoch loop advances the scheduler and then writes `optimizer.param_groups[0]["lr"]` into the record. At a scheduler boundary, that value is the next epoch’s learning rate, not the rate used for the epoch whose losses are recorded.

**Remediation:** capture the effective learning rate before stepping the scheduler, or rename the field to make its post-step meaning explicit.

Evidence: `src/scenarioforge/training/trainer.py:153-160`.

#### m6. Important detector and metric edge paths lack focused regression tests

The inspected training tests cover model construction with `pretrained_weights=None`, a simple perfect synthetic detection metric case, one tiny-model training/resume path, and training-selection checks. They do not assert pretrained COCO loading followed by replacement of the COCO-specific predictor, frozen BatchNorm behavior, CUDA AMP, empty-target training, empty detections, or deterministic resume continuity. The test model’s prediction method always returns one detection.

**Remediation:** add focused tests for the pretrained head replacement, empty targets and predictions, undefined precision/AP semantics, frozen normalization state, AMP-enabled CUDA (where available), and checkpoint RNG/provenance compatibility. Keep hardware-dependent checks marked/skipped with explicit reasons when CUDA is unavailable.

Evidence: `tests/test_training.py:30-45, 99-115, 128-190, 198-231`.

## Engineering and methodology assessment

### Model construction and label mapping

- The configured architecture is `fasterrcnn_mobilenet_v3_large_320_fpn`.
- For `COCO_V1`, torchvision constructs the pretrained detector; the code then replaces only `roi_heads.box_predictor` with a predictor sized to ten foreground classes plus background. Thus the original COCO class-specific predictor is not used as if it were the ten ScenarioForge classes. The replacement head is newly initialized; pretrained feature extraction and non-class-specific detector components remain from the COCO model.
- Model labels are validated as contiguous foreground IDs `1..10`; background remains `0`. The adapter retains the reversible model-label-to-source-ID mapping. Inference returns model labels, not remapped source IDs; downstream consumers must use the exposed mapping.
- The test suite verifies an 11-output predictor and mapping for a non-pretrained construction, but does not exercise the `COCO_V1` initialization path.

Evidence: `src/scenarioforge/training/model_factory.py:45-58, 80-123`; `tests/test_training.py:99-115`.

### Bounding boxes and image transforms

- Annotations are interpreted as XYXY, checked for finite coordinates, positive area, and image bounds before use.
- Resize applies the same width/height scale factors to XYXY coordinates; optional horizontal flip transforms the box and image together. The default flip probability is zero.
- Images become RGB float32 CHW tensors in `[0,1]` without external normalization, consistent with torchvision Faster R-CNN’s internal normalization.
- Empty boxes are represented with shape `[0,4]`; output validation permits that shape and recomputes area. The audit’s sampled training split contained no empty-target sample, so an actual empty-target training step was not verified here.

Evidence: `src/scenarioforge/vision_data/dataset.py:136-165, 187-208, 220-253`; `src/scenarioforge/vision_data/transforms.py:30-48, 52-78`.

### Training and GPU behavior

- The trainer enters training mode, sums the scalar loss dictionary, transfers images and tensor targets to the configured device, supports gradient accumulation, unscales before gradient clipping, and uses `GradScaler` only when mixed precision is enabled on CUDA.
- A focused CUDA train-split batch using the detector with `pretrained_weights=None` produced finite losses and nonzero gradients for all 192/192 trainable parameters. This proves gradient flow for that sample/configuration only, not for every class, batch, or the pretrained configuration.
- Explicit `device="cuda"` fails if CUDA is unavailable; it does not silently fall back. OOM handling clears gradients/cache and raises an actionable error without silently changing batch size or configuration. OOM recovery and AMP behavior were not stress-tested.
- The notebook’s reported 386,103,808 bytes is `torch.cuda.max_memory_allocated` for that one smoke step, not total process/GPU memory. The saved environment reports a 4,294,443,008-byte RTX 3050 Laptop GPU. These figures support the one-sample smoke case only and do not establish capacity for longer or larger-batch jobs.

Evidence: `src/scenarioforge/training/trainer.py:22-76, 191-262`; `src/scenarioforge/training/experiment.py:210-230`; notebook output at `notebooks/06_Object_Detection_Training_Smoke_Test.ipynb:202-222`; retained `experiments/20261103T163030Z-e652390e/environment.json`.

### Evaluation and metric correctness

- Evaluation runs under `torch.inference_mode()`, switches the model to evaluation mode, and restores training mode when appropriate.
- Predictions and targets are supplied to TorchMetrics as XYXY boxes; model labels are converted from `1..10` to `0..9`. `MeanAveragePrecision` uses the Faster-COCO-Eval backend, class metrics, and the standard 0.50:0.05:0.95 IoU thresholds; AP50/AP75 are read from the same COCO-style metric result.
- Per-class AP is `null` when the validation set has no ground truth for that class. Precision and recall are `null` when their denominator is zero rather than being reported as zero.
- The fixed-threshold precision/recall implementation uses score threshold 0.5 and IoU 0.5, greedily matches predictions in descending score order, and reports aggregate micro counts as well as per-class counts.
- Focused synthetic checks found: no GT and no predictions yields undefined mAP/precision/recall; a GT with no predictions yields AP 0, undefined precision, and recall 0. This preserves the distinction between zero performance and an undefined ratio.
- The retained smoke validation result has mAP/AP50/AP75 0.0 on two images. For car, no predictions reached the 0.5 score threshold, so precision is correctly undefined while recall is 0.0. No test metrics were evaluated.

Evidence: `src/scenarioforge/training/evaluator.py:34-106, 110-128, 154-249`; `tests/test_training.py:128-145`; retained `experiments/20261103T163030Z-e652390e/validation_metrics.json`.

### Data isolation and provenance

- The training selection validator rejects non-training tokens and records a selection digest and unchanged-held-out assertion. The split manifest binds scene assignments and image inventory; the experiment summary records both split and inventory digests.
- The fit API receives only train and validation loaders; test evaluation is a separate explicit method and was not called. No test metrics were produced.
- No normalization statistics are computed from the dataset. The fixed category mapping is loaded as class metadata.
- The limitation is setup-time access described in finding m1: split dataset instances validate all image files and parse the full annotations CSV before split filtering of `image_records`.

Evidence: `src/scenarioforge/vision_data/splits.py:239-267`; `src/scenarioforge/vision_data/dataloader.py:66-127`; `src/scenarioforge/training/experiment.py:59-81, 112-122, 244-302`.

### Reproducibility and checkpoint integrity

- Configuration and environment artifacts record package versions, Python/platform, device/GPU, dataset/split digests, model config, label mapping, and training-selection audit.
- Checkpoint loading uses `torch.load(..., weights_only=True)`. Atomic replacement is used for checkpoint writes.
- These are useful provenance features, but they do not make a resume trajectory reproducible. Missing RNG/DataLoader state and incomplete compatibility checks are described in M1. `deterministic: true` is best-effort because unsupported operations are allowed to continue, as described in M2.

Evidence: `src/scenarioforge/training/checkpointing.py:24-80`; `src/scenarioforge/training/seed.py:31-58`; `src/scenarioforge/training/experiment.py:106-141`.

### Test and notebook coverage

The project reports 125 passing tests and the smoke notebook has six executed code cells. This audit inspected the relevant tests and retained outputs rather than relying on those counts. The notebook demonstrates CUDA availability, one real optimizer step, checkpoint reload, and a one-epoch train/validation smoke run. It does not evaluate the test partition. The coverage gaps listed in m6 remain important before a longer research run.

Audit-only runtime checks performed: one real CUDA training batch on the training split, a synthetic empty-metric check on CPU, and a model-mode inspection for frozen-backbone BatchNorm. The full test suite was not rerun as part of this audit.

## Required remediation before longer training

1. Make checkpoint resume provenance-complete: persist/restore RNG and loader-generator state, global step, and early-stopping state; validate a canonical configuration and data-selection digest before loading.
2. Resolve and accurately label determinism: enforce strict deterministic algorithms when promised, or explicitly mark runs as best-effort/non-bitwise reproducible and record unsupported operations.
3. Restrict dataset construction to each requested split so test images and annotations are not opened or materialized during training/validation setup.
4. Bind test evaluation to an immutable selected checkpoint hash and its validated configuration/selection provenance.
5. Decide and test the intended BatchNorm semantics for frozen-backbone epochs.
6. Add focused tests for COCO weight/head replacement, empty targets and predictions, undefined metrics, resume continuity/provenance, and AMP-enabled CUDA behavior.
7. Isolate the notebook’s one-batch optimizer smoke step from its subsequent epoch run and correct the learning-rate history semantics.

After these remediations, rerun the focused test suite and a short train/validation-only integration smoke test. Keep test evaluation disabled until a final model-selection protocol is frozen; any eventual test evaluation should be a single, documented final evaluation and must not feed subsequent training or model selection.

## Longer-training decision

**Not approved at this time.** The 1-train/2-validation smoke run and successful CUDA backward pass establish basic execution only. The unresolved resume and determinism limitations prevent a defensible reproducibility claim, and the split-construction access boundary should be tightened before research-scale use. No research-level performance claim is supported by the smoke metrics.
