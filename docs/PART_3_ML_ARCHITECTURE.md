# Part 3 ML Architecture Proposal

**Status: design only.** No detector, annotation converter, split generator, training code, or ML
dependency is implemented by this proposal. Do not start training until the architecture and
experiment design have been reviewed and approved.

## Goal and scope

Build a reusable 2D object-detection pipeline for the nuScenes **front camera**, compare one
Torchvision Faster R-CNN baseline with one YOLO-family detector, and measure how training-set
curation changes detection results. The first implementation should be small, testable on CPU, and
able to run on the RTX 3050 laptop GPU or a compatible Kaggle/Colab GPU.

This is not nuScenes 3D detection. Report 2D box metrics; do not report or imply nuScenes 3D mAP or
NDS. The current Part 1 schema already retains the needed source information, so begin with an
adapter over the existing schema and raw nuScenes tables rather than a database migration.

## Proposed architecture

Keep the existing package dependency layers and add an optional ML area only after approval:

```text
dataset access
  -> scene split manifest
  -> front-camera projection / annotation records
  -> validated COCO-style dataset export
  -> train-only curation manifest
  -> detector-specific adapters
  -> common 2D evaluator
  -> experiment registry + immutable run artifacts
```

Suggested future modules (names are proposals, not created files):

- `scenarioforge.ml.annotations`: nuScenes projection, category mapping, and validation report.
- `scenarioforge.ml.splits`: deterministic scene-group assignment and split manifest.
- `scenarioforge.ml.datasets`: front-camera image/target loading and transforms.
- `scenarioforge.ml.models`: Faster R-CNN and YOLO wrappers behind a common train/predict boundary.
- `scenarioforge.ml.training`: seed setup, train/validation loop, checkpointing, and resource tracking.
- `scenarioforge.ml.evaluation`: common prediction format and fixed 2D metrics.
- `scenarioforge.ml.experiments`: configuration, manifest, and artifact registry.

Do not import PyTorch in Part 1 or Part 2 modules. CPU-only data and projection tests must not need
PyTorch. Keep ML dependencies in a separate environment/optional requirements set rather than
expanding the stable ingestion environment.

## 2D annotation conversion

### Inputs and provenance

For each `sample` choose its `CAM_FRONT` sample-data record, image path, dimensions, timestamp,
calibrated-sensor token, ego-pose token, and scene token. Read the corresponding
`calibrated_sensor.camera_intrinsic`, calibration translation/rotation, ego-pose translation/
rotation, and sample-annotation translation/size/rotation from the existing normalized schema and
nuScenes source metadata. The current adapter already preserves calibration JSON, image dimensions,
pose source IDs, 3D annotation geometry, and source annotation IDs.

Export one image record per valid front-camera keyframe and one 2D annotation record per accepted
visible 3D box. Preserve at least:

- image ID, sample token, scene token, sample-data token, relative image path, width, height, and
  camera channel;
- 2D `xyxy` box in image pixels and/or COCO `xywh` on export, class ID/name, source annotation
  token, source category, and source instance token;
- projection status/rejection reason and source visibility metadata for the validation report.

Use dataset-relative paths in manifests; never persist machine-specific absolute paths. Verify that
the image exists, is decodable, and its dimensions agree with metadata before exporting labels.

### Projection and visibility

Use the official nuScenes `Box`, quaternion, and camera-geometry utilities where practical. The
coordinate path is:

1. Construct each 3D annotation cuboid from its global center, `[width, length, height]` size, and
   `(w, x, y, z)` rotation.
2. Transform global coordinates into ego coordinates using the inverse ego pose.
3. Transform ego coordinates into the calibrated camera frame using the inverse camera extrinsic.
4. Project camera-frame points with the intrinsic matrix `K`; in the nuScenes camera convention,
   positive camera depth is in front of the lens.
5. Clip 3D box edges against a small positive near plane before projection. Projecting only the
   eight original corners is insufficient for a box that crosses the camera plane.
6. Form the 2D bounds from projected visible corners/intersections, clip to `[0, width] x
   [0, height]`, then reject non-finite, fully invisible, or zero/negative-area boxes.

Make the visibility policy explicit and version it. Initial policy: retain a geometrically
intersecting box when its clipped 2D rectangle has positive area and meets a configurable minimum
pixel-area threshold; record source visibility level but do not treat it as a pixel-accurate
occlusion judgment. Report projected, retained, and rejected counts by reason and class. Do not
silently clamp a box that is entirely behind the camera into a valid label.

### Class mapping

Start with the familiar 10 nuScenes detection classes, with a documented source-category mapping:
car, truck, bus, trailer, construction vehicle, pedestrian, motorcycle, bicycle, traffic cone, and
barrier. Map only explicit nuScenes leaf categories. Ignore unsupported categories such as bicycle
rack, debris, animals, or pushable objects in the initial detection target, and count them in the
conversion report. Do not map the broad Part 2 `other` group into a detector class. Keep class IDs
stable and identical across splits and model implementations.

### Visual annotation inspection

Before any training, provide a small inspection command/notebook that overlays accepted boxes,
class names, source annotation IDs, and projection/rejection diagnostics on actual front-camera
images. Include examples sampled across scenes, classes, image boundaries, and source visibility
levels; include rejected-box cases in a separate diagnostic view. The reviewer must be able to
check camera orientation, box extent, clipping, and mapping against raw images. The converter is
not ready for training until these overlays and its validation report pass manual review.

## Scene-level splits and leakage controls

- Assign **complete scenes** to train, validation, or test before fitting bin edges, rarity
  frequencies, feature transforms, or curation. Never split frames randomly.
- Persist sorted scene IDs and sample IDs, dataset version, split seed/algorithm, and a manifest
  digest. Reuse exactly the same held-out scene IDs for every strategy, model, and training seed.
- Apply curation only to the training portion. Validation/test IDs and labels are fixed, uncurated,
  and inaccessible to curation scoring, fitting, early stopping, and hyperparameter search.
- Fit any data-dependent normalization or rarity/binning rules on training records only.
- Report image and box counts by split and per-class image/box counts. Check that the split has
  usable class support before training; disclose absent classes rather than moving test scenes to
  improve a result.
- nuScenes mini has 10 scenes, so a scene-held-out smoke-test split has very few independent
  groups and unstable class coverage. It can validate code paths but cannot support strong
  generalization or statistical claims.
- For meaningful experiments, use a larger labeled release such as nuScenes `trainval`. A
  practical proposal is to reserve the official validation scenes as the fixed test set and make a
  deterministic scene-grouped train/validation split within official training scenes. Do not use
  the unlabeled official test split for local metric claims; use it only through the supported
  benchmark submission process. Record the exact release and scene IDs.

## Model candidates and recommendation

### Model A: Faster R-CNN MobileNetV3-Large FPN

Use Torchvision's `fasterrcnn_mobilenet_v3_large_fpn` as a two-stage, lightweight baseline. Its
Torchvision implementation is a direct fit for a PyTorch dataset/target pipeline and provides
official pretrained weights. Replace the COCO classifier/box predictor with the fixed nuScenes
class set. Freeze or partially freeze the backbone during the first smoke runs; then compare
fine-tuning under a predeclared schedule. Keep the model's internal resize limits explicit.

Trade-off: it is a useful, interpretable two-stage baseline, but region proposals and multi-scale
features can still be memory- and latency-intensive, and its preprocessing/resizing must be matched
carefully with the YOLO path.

### Model B: YOLO-family detector

The most practical initial integration candidate is a small Ultralytics YOLO detection model behind
the same dataset/evaluation boundary. The Python package supports common Windows/PyTorch workflows,
but its exact release and model/checkpoint should be selected only after the environment smoke test.
Pin the package version, architecture name, checkpoint identifier/hash, input resolution, and
pre/postprocessing configuration when experiments are approved.

**License gate:** Ultralytics' repository is AGPL-3.0; review its current license and the specific
pretrained checkpoint's terms before adopting it. A research/portfolio project still needs to
respect applicable source-distribution and network-use obligations. If AGPL is not acceptable,
select a YOLO-family implementation and weights with a license compatible with the project's
intended use; do not assume the architecture name determines the implementation or weight license.
Have a qualified reviewer resolve licensing questions.

Trade-off: a single-stage YOLO detector may provide a practical speed/accuracy comparison, while
the package brings additional dependencies and a distinct training/inference pipeline. Keep its
wrapper narrow and convert outputs to the common evaluator format.

### Recommendation

Begin implementation with the Torchvision Faster R-CNN baseline and validate labels, CPU behavior,
and CUDA memory before selecting exact releases. Treat Ultralytics YOLO as the preferred second
candidate conditional on license review and a successful Windows/Python 3.12 compatibility smoke
test. Pin a Torch/Torchvision pair from the official compatibility matrix; do not choose a CUDA
wheel or YOLO release based on a guessed driver/toolkit version. If the license gate fails, pause
and choose a reviewed alternative before adding dependencies.

## Training, evaluation, and artifacts

Use the same immutable dataset/split/annotation manifests and class mapping for both detectors.
Detector-specific model heads and optimizers may differ; document those differences and freeze
the comparison before test evaluation. Save checkpoints and validation-selected settings outside
the raw dataset. The test set is evaluated once per preregistered run configuration.

Use a common evaluator over image IDs, class IDs, scores, and pixel-space boxes. The primary metrics
are 2D mAP@0.50 IoU and mAP@0.50:0.95 IoU, plus fixed-operating-point precision/recall, per-class
AP, training wall time, peak GPU memory, and number of unique training samples. Use one evaluator
and one IoU/ignore policy for both architectures. Record per-scene metrics; do not conflate these
2D metrics with nuScenes 3D metrics.

Each run should capture:

- source dataset release and annotation converter version/configuration;
- split manifest and digest; curation strategy/configuration/seed and selected IDs;
- model implementation/version, architecture, checkpoint source/hash, and license reference;
- training configuration, all random seeds, software lockfile, Python/PyTorch/Torchvision/CUDA
  runtime versions, and deterministic-mode settings;
- checkpoint path/hash, evaluation configuration/results, per-class and per-scene outputs;
- hardware, GPU name/VRAM, driver version, peak allocated memory, elapsed time, and failures.

Write a machine-readable JSON manifest per run and immutable prediction/metric artifacts. The
manifest should reference checkpoints by path and digest, not embed weights. Include a schema
version so the registry can evolve.

## Hardware and dependency plan

Keep the existing Python 3.11/3.12 Part 1/2 environment unchanged. Create a separate Python 3.12 ML
environment only after architecture approval. Select compatible `torch` and `torchvision` versions
as a pair using the official PyTorch installation selector; record the exact command and resolved
versions in the ML lockfile. Do not install a system CUDA Toolkit merely because a CUDA PyTorch
wheel is used. Check the installed NVIDIA driver against the selected wheel's stated requirements,
then verify `torch.cuda.is_available()` and a small detector forward/backward pass.

For the RTX 3050 Laptop GPU with 4 GB VRAM, start with batch size 1, conservative aspect-ratio-
preserving image resizing (for example, maximum side around 640 pixels), mixed precision where
supported, and a frozen or partially frozen backbone. Profile peak allocated memory before
increasing resolution or batch size. Gradient accumulation can increase effective optimizer batch
size but does not reduce a single-image activation peak. Keep model/data smoke tests executable on
CPU; gate optional CUDA tests on device availability. Kaggle/Colab runs must record their own GPU,
driver/runtime, and package versions rather than inheriting local assumptions.

Official references:

- [PyTorch installation selector](https://pytorch.org/get-started/locally/) and
  [version compatibility/install commands](https://docs.pytorch.org/get-started/previous-versions/).
- [Torchvision Faster R-CNN MobileNetV3-Large FPN](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html).
- [Ultralytics license information](https://www.ultralytics.com/license) and
  [repository license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
- [NVIDIA CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/).
- [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit) and
  [dataset format](https://www.nuscenes.org/nuscenes#data-format).
