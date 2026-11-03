# Part 3C: Object-detection training infrastructure

## Scope and limitations

Part 3C provides reusable experiment setup, one torchvision detector adapter, a training loop,
validation-based checkpoint selection, COCO-style evaluation, and artifact tracking. The
nuScenes v1.0-mini data is only a smoke-test dataset: its ten scenes are insufficient for
statistical performance claims or curation comparisons. YOLO training, long training runs, and
curation-strategy comparisons are not included.

The Part 3B split manifest is read and validated but never rewritten. Training constructs only
the train and validation datasets; the test dataset is constructed only for an explicitly
authorized final test evaluation. Fitting and model selection use validation only. The test
loader is never passed to the trainer. Test evaluation requires a JSON authorization bound to
the frozen best checkpoint hash and records that authorization in the experiment directory.
Validation and test metrics are written separately.

## Architecture and category mapping

Common experiment setup lives under `src/scenarioforge/training/`. `DetectionModelAdapter`
isolates the trainer/evaluator from detector-specific loss and prediction calls. The initial
adapter is torchvision
`fasterrcnn_mobilenet_v3_large_320_fpn` with `FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1`.
The COCO predictor is replaced with `FastRCNNPredictor` sized for the ten ScenarioForge
foreground classes plus background. Dataset labels remain positive foreground IDs 1–10;
background remains label 0. The persisted reversible source/model category mapping is recorded
with each run and is not modified.

`freeze_backbone_epochs` controls initial freezing: zero trains the backbone from the first
epoch, while a positive value freezes it for that many epochs before unfreezing. The optimizer
contains all parameters so the backbone can be unfrozen without rebuilding optimizer state.
The torchvision model's 320-pixel transform preserves aspect ratio internally; the input image
resize can be configured independently through `image_size` (height, width), or left unset to
preserve the exported image dimensions.

## Environment and installation

Use the isolated `.venv-ml` environment and install PyTorch/torchvision builds appropriate for
the machine's CPU or CUDA driver. The verified environment for this implementation is Python
3.12.16, PyTorch 2.15.0+cu132, torchvision 0.29.2+cu132, and CUDA 13.2 on an RTX 3050 Laptop GPU.
The pinned metric dependencies are TorchMetrics 1.9.0 and Faster-COCO-Eval 1.8.0. Faster-COCO-Eval
publishes a CPython 3.12 Windows x64 wheel; TorchMetrics is platform-independent and declares
Python 3.10+ support. TorchMetrics documents both the COCO mAP API and Faster-COCO-Eval backend.
See the [TorchMetrics mAP documentation](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html)
and [Faster-COCO-Eval on PyPI](https://pypi.org/project/faster-coco-eval/).

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv-ml
.\.venv-ml\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements-ml.txt
python scripts/train_detection.py inspect-environment
```

The ML dependency file pins torch, torchvision, torchmetrics, and faster-coco-eval. For CUDA,
install a PyTorch wheel matching the local driver if the package index does not provide the
appropriate CUDA build; do not install these dependencies into the base `.venv`.

## Memory-conscious training

Defaults use batch size 1, zero workers (safe on Windows), no mixed precision, and no gradient
accumulation beyond one step. CUDA is selected when available; `device: "cuda"` fails explicitly
if CUDA is unavailable, rather than silently falling back to CPU. Mixed precision is opt-in and
uses CUDA float16 with a gradient scaler. A CUDA out-of-memory error is reported with suggested
configuration changes; the code does not silently change device, batch size, or image size.
Gradient accumulation, clipping, worker count, and image dimensions are configurable. Images
and targets are transferred per batch; the full image dataset is never preloaded.

Before longer training, the one-batch smoke test performs a real forward/backward optimizer step
and checkpoint save/load. Peak allocated CUDA memory is measured after that step. Memory
measurements vary with the installed CUDA build and background GPU use.

## Commands

All commands below use the ML environment from the repository root:

```powershell
# Verify runtime/hardware and instantiate the configured detector.
python scripts/train_detection.py inspect-environment
python scripts/train_detection.py create-model --config configs/object_detection_smoke.json

# Run a single real forward/backward step and checkpoint round-trip.
python scripts/train_detection.py smoke-test --config configs/object_detection_smoke.json

# Run the configured short training job. Use a production experiment config for real training.
python scripts/train_detection.py train --config configs/object_detection_smoke.json

# Resume an interrupted run to the configured total epoch count.
python scripts/train_detection.py resume --experiment-dir experiments/<experiment-id>

# Evaluate validation during iteration.
python scripts/train_detection.py evaluate --experiment-dir experiments/<experiment-id> `
  --checkpoint experiments/<experiment-id>/checkpoint_best.pth --split validation

# Test evaluation requires an explicitly prepared authorization JSON bound to the checkpoint.
python scripts/train_detection.py evaluate --experiment-dir experiments/<experiment-id> `
  --checkpoint experiments/<experiment-id>/checkpoint_best.pth --split test `
  --test-authorization path\to\test-authorization.json
```

The authorization JSON must contain `experiment_id`, `checkpoint_sha256` (matching
`best_checkpoint_sha256` in `experiment_summary.json`), `provenance_digest`,
`split_manifest_digest`, `purpose`, `authorized_by`, and an ISO-8601 `authorized_at` timestamp.
Keep this file under the study's access-control process;
the CLI intentionally does not generate authorization on the user's behalf.

JSON configurations are validated and copied into the experiment directory. The supplied
`configs/object_detection_smoke.json` uses one training sample and two validation samples to
exercise the pipeline quickly; its metrics are not a benchmark. Production curation runs should
provide the sample tokens selected by the existing training-selection API in
`training_sample_tokens_file`; the selection audit and digest are persisted. A `training_sample_limit`
is available for smoke tests only.

## Evaluation methodology

Mean AP, AP@0.50, AP@0.75, and per-class AP use
`torchmetrics.detection.MeanAveragePrecision` with `backend="faster_coco_eval"`, XYXY boxes,
and COCO IoU thresholds 0.50 through 0.95. Metric labels are translated from positive
ScenarioForge labels 1–10 to zero-based metric IDs, then associated with the saved reversible
class mapping.

Precision and recall are additionally reported at IoU 0.50 and score threshold 0.50. Per-image,
per-class predictions are processed in descending score order and greedily matched one-to-one
to the highest-IoU unmatched ground-truth box; unmatched predictions are false positives and
unmatched ground truth is false negative. The output names this threshold explicitly. Raw
prediction and ground-truth object counts are also reported per class. For a class with no
ground-truth objects, per-class AP and undefined precision/recall are JSON `null`, not fabricated
zero or perfect values. Overall AP follows the documented TorchMetrics/Faster-COCO-Eval result.

## Reproducibility and artifacts

Set `reproducibility_mode` to `strict` or `best_effort`. Both seed Python, NumPy, CPU/CUDA
PyTorch, and the per-epoch DataLoader generator, and disable TF32. `strict` enables deterministic
algorithms without warning-only fallback and sets CUDA's `CUBLAS_WORKSPACE_CONFIG` before CUDA
initialization when possible; if CUDA is already initialized, the variable must already be set.
Unsupported operations (including the observed
torchvision `roi_align` backward kernel on some builds) stop the run with an explicit error.
`best_effort` requests deterministic algorithms in warning-only mode; it records settings but
does not guarantee bitwise results. Even strict mode does not promise identical results across
different hardware, drivers, CUDA/cuDNN/PyTorch versions, or operations without deterministic
kernels.

Windows CUDA runs require a PyTorch/torchvision wheel pair compatible with the installed NVIDIA
driver and CUDA runtime. Use `num_workers: 0` as the reproducible Windows baseline; if workers are
enabled, the implementation derives worker seeds from a deterministic per-epoch DataLoader
generator. The environment metadata records PyTorch, CUDA, cuDNN, deterministic-algorithm,
cuDNN, TF32, and cuBLAS workspace settings.

Every checkpoint stores model, optimizer, scheduler, scaler, completed epoch, global step, best
validation score and epoch, early-stopping count, Python/NumPy/PyTorch CPU/CUDA RNG states, and
DataLoader generator state. Epoch-seeded sampling is used so a resumed epoch starts with the
same shuffle and worker seed stream. Resume verifies configuration, data, split, label-map, and
training-selection provenance and refuses to overwrite completed or inconsistent history.
The integration test verifies exact continuation for a lightweight CPU model in the same
runtime. That does not establish bitwise equality for the full detector across platforms;
floating-point kernels and platform/runtime differences can still prevent bitwise equality.

Environment metadata records package versions, Python/platform, device/CUDA details, dataset
version, split-manifest digest, image-inventory digest, training-selection digest, architecture,
class mapping, seed, and configuration/provenance digests. Every new run gets a UTC timestamp
plus a unique suffix.

Each run writes under `experiments/<experiment-id>/`:

- `config.json` and `environment.json`
- `training_history.csv` and `validation_metrics.json`
- `checkpoint_best.pth` and `checkpoint_last.pth`
- `experiment_summary.json`
- `test_metrics.json`, only when requested after checkpoint selection is frozen

`checkpoint_best.pth` is selected by validation mAP@0.50:0.95. `checkpoint_last.pth` supports
resume and is never substituted for the best checkpoint in test evaluation. Checkpoints include
model, optimizer, scheduler, scaler, resumable random state, and experiment/split/selection
metadata. The
`experiments/` directory is ignored by Git.

## Notebook and tests

`notebooks/06_Object_Detection_Training_Smoke_Test.ipynb` calls the package API to inspect a real
batch, construct the detector, run a training step, perform a short training run, and verify
checkpoint reload and validation evaluation. Synthetic tests cover configuration, seeding and reproducibility modes, model construction and
freeze behavior, metric edge cases, training, deterministic interruption/resume equivalence,
checkpoint integrity, and selection/data-isolation boundaries.
GPU-specific execution is performed as an optional integration smoke test, not a prerequisite
for the unit suite.

## Future YOLO integration

Add a separate adapter implementing the `DetectionModelAdapter` loss/prediction contract and
model-specific checkpoint metadata. The shared experiment setup, dataset split checks,
selection digest, seeding, and output structure should remain unchanged. No YOLO package or
training code is currently installed or implemented.
