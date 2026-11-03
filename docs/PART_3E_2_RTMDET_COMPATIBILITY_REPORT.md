# Part 3E.2 — RTMDet Compatibility and GPU Feasibility Report

## Executive summary

**Decision: no-go for RTMDet-Ins-tiny integration and training in the current
environment.** The current Python/PyTorch/CUDA combination is outside the
documented compatibility range reviewed for MMCV 2.1.0, and no matching
OpenMMLab CUDA wheel was found for CUDA 13.2 / PyTorch 2.15. A controlled CUDA
source build is not currently possible because the CUDA Toolkit and `nvcc`
are absent. Installing `mmcv-lite` would not provide the CUDA operators needed
to validate the requested GPU path.

No packages were installed, no checkpoint was downloaded, and no model was
constructed. Consequently, checkpoint loading, head adaptation, empty-image
behavior, inference conversion, and GPU memory use remain untested. Do not
begin Part 3E.3 until a compatible, isolated environment is established and
the requested smoke tests pass.

A bounded data-only check did succeed on one annotated **training** sample
from the existing seed-42 split. The training dataset contained 242 samples;
neither the validation nor test dataset was instantiated. No empty training
sample was available for that check.

## 1. Environment compatibility

| Component | Observed environment |
|---|---|
| OS | Windows |
| Python | 3.12.16 |
| PyTorch | 2.15.0+cu132 |
| TorchVision | 0.29.2+cu132 |
| CUDA runtime reported by PyTorch | 13.2 |
| CUDA availability | Available |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| Reported GPU memory | 4,294,443,008 bytes (4 GiB) |
| Relevant existing packages | NumPy 1.26.4, opencv-python-headless 4.11.0.86, pycocotools 2.0.11 |
| MMDetection / MMEngine / MMCV / OpenMIM | Not installed |

The existing PyTorch and TorchVision installation was preserved. The project
ML requirements also pin Torch 2.15.0 and TorchVision 0.29.2; those pins were
not changed.

### Compatibility findings

- MMDetection 3.3.0 declares `mmcv>=2.0.0rc4,<2.2.0` and
  `mmengine>=0.7.1,<1.0.0`.
- MMDetection's general installation documentation lists Windows and broad
  minimum versions for Python, CUDA, and PyTorch. Those minimums do **not**
  certify this exact Python 3.12 / PyTorch 2.15 / CUDA 13.2 combination.
- MMCV 2.1.0's documented prebuilt compatibility range reviewed here covers
  PyTorch 1.8–2.0 and CUDA 10.1–11.8. This does not include the current stack.
- Probes of the OpenMMLab wheel-index paths for `cu132/torch2.15` and
  `cu132/torch2.15.0` returned HTTP 404. No compatible prebuilt CUDA MMCV
  wheel was identified.
- Visual Studio C++ Build Tools / MSVC 14.51.36231 were present, but no CUDA
  Toolkit directory or `nvcc` command was found. A Windows MMCV CUDA-ops
  source build therefore lacks a required CUDA compilation toolchain and
  would be uncontrolled. It was not attempted.
- `mmcv-lite` omits MMCV's compiled CUDA operators, so using it would not
  satisfy the requested CUDA model validation.

**Exact compatible versions:** none were verified for this environment.
Although MMDetection's version bounds define an allowed range for MMEngine
and MMCV, choosing versions within those bounds would not resolve the missing
wheel / PyTorch-CUDA ABI compatibility. No package installation is recommended
until an exact supported combination is documented and available.

References:

- [MMDetection v3.3.0 installation documentation](https://mmdetection.readthedocs.io/en/v3.3.0/get_started.html)
- [MMDetection v3.3.0 dependency declarations](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/requirements/mminstall.txt)
- [MMCV 2.1.0 README and compatibility notes](https://github.com/open-mmlab/mmcv/blob/v2.1.0/README.md)
- [MMCV installation documentation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
- [OpenMMLab prebuilt package index](https://download.openmmlab.com/mmcv/dist/)

## 2. Dependency installation plan and outcome

### Outcome

- Nothing was installed or upgraded.
- `.venv-ml` and the separate `.venv` were left unchanged.
- No dependency resolver or source build was run.
- No model checkpoint was downloaded.

### Rollback and isolation plan for a future attempt

1. Keep both existing environments untouched; do not install OpenMMLab packages
   into `.venv` or `.venv-ml`.
2. Before any approved attempt, capture `python --version`, `pip freeze`, and
   the PyTorch/TorchVision/CUDA/GPU details from `.venv-ml`.
3. Use a disposable, separately named experimental environment only after
   confirming an exact official Windows wheel set or obtaining approval for a
   matching CUDA Toolkit, compiler, and source-build procedure.
4. If the experiment fails, remove only that dedicated environment. Do not
   roll back by replacing packages in either existing environment.

This plan has not been executed because a compatible dependency set has not
been established.

## 3. Model construction and pretrained checkpoint

No MMDetection config was loaded and no model was constructed. Therefore:

- Missing and unexpected checkpoint keys: **not measured**.
- Compatible pretrained layers loaded: **not verified**.
- Ten-class detection-head adaptation: **not implemented or tested**.
- Head initialization and class/background conventions: **not verified**.
- Instance-segmentation mask branch removal/disablement: **not tested**.
- Detection-only RTMDet-tiny substitution: **not made**.

The candidate remains the exact model and checkpoint documented in Part 3E.1:
RTMDet-Ins-tiny, using the official
[configuration](https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py)
and
[COCO checkpoint](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth).
The instance-segmentation checkpoint was not silently changed to a detection
checkpoint or stripped of components.

## 4. Dataset compatibility check

The existing ScenarioForge dataset factory was run with `splits=("train",)`
and the canonical seed-42 manifest, annotations, category mapping, and image
manifest. The factory validated the manifest but instantiated only the
training dataset.

Observed results:

- Training dataset size: **242** samples.
- Validation dataset instantiated: **no**.
- Test dataset instantiated: **no**.
- Source-to-model label mapping: source IDs 1–10 map directly to model labels
  1–10.
- One annotated training image loaded successfully as RGB float32 tensor with
  shape `[3, 900, 1600]` and values in `[0, 1]`.
- Its target contained 17 XYXY pixel boxes with matching labels; box
  dimensions were positive and valid.
- No zero-annotation sample was present in the training dataset, so empty
  image behavior was **not empirically tested**.
- No annotation, image manifest, split assignment, or dataset file was
  modified. No validation or test image was loaded.

This verifies the existing ScenarioForge input path for one annotated training
sample only. It does not demonstrate that MMDetection accepts that sample or
that RTMDet preprocessing preserves coordinates.

## 5. GPU smoke test and resource results

The requested batch-one CUDA forward/backward smoke test was **not run** because
the compatible CUDA-enabled MMDetection/MMCV stack could not be installed in
a controlled manner.

| Check | Result |
|---|---|
| RTMDet CUDA model construction | Not run |
| Finite forward/backward losses | Not run |
| Gradient presence/validity | Not run |
| Peak allocated GPU memory | Not measured |
| Checkpoint save/reload | Not run |
| Inference output and box validity | Not run |
| Empty-image forward behavior | Not run |
| Dataset prediction conversion | Not run |
| Multi-epoch training | Not run |

The GPU's 4 GiB capacity makes a batch-one run a plausible future smoke-test
starting point, but there is no measured memory result and no claim that the
candidate fits. Do not reduce input resolution, change precision, alter batch
size, or substitute an architecture without explicitly recording and
approving the configuration change.

## 6. Licensing notes

Part 3E.1 documented the MMDetection repository as Apache-2.0 and reported a
checkpoint-specific maintainer response concerning the exact RTMDet-Ins-tiny
checkpoint. The source and checkpoint links are retained above. This is
affirmative evidence, not a legal opinion or a guarantee about every use.

No OpenMMLab dependencies were installed, so the exact resolved dependency
set and its third-party licenses were not audited. Before distributing an
integrated environment or model, record the installed package versions and
review the applicable licenses and notices for each dependency and artifact.

## 7. Limitations

1. No compatible MMCV CUDA wheel was found for the current stack; the
   documented MMCV 2.1.0 compatibility range does not cover it.
2. No CUDA Toolkit / `nvcc` is available for a controlled Windows CUDA
   extension build.
3. The checkpoint, class-head adaptation, mask-branch handling, and evaluator
   bridge are entirely unvalidated.
4. The data-only check covered one annotated training image; no empty training
   image was available.
5. GPU fit and memory use are unknown. Model size or nominal FLOPs cannot
   establish training feasibility on a 4 GiB GPU.
6. This was not a training run and produced no experiment artifacts or
   performance results.

## 8. Go/no-go recommendation

**NO-GO** for full integration, GPU smoke testing, or Part 3E.3 in the current
environment. No package was installed and the PyTorch/CUDA stack remains
unchanged.

The next step should be to identify and approve an exact supported Windows
dependency combination (or a separately managed environment with a supported
PyTorch/CUDA toolchain), then repeat this bounded assessment. Only proceed to
integration after model loading, head adaptation, training-only sample
conversion, empty-target behavior, checkpoint reload, and the batch-one
forward/backward smoke test all pass.
