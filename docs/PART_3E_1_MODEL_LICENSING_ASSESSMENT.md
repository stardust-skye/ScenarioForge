# Part 3E.1 — Pretrained Model Licensing Assessment

## Executive summary

One suitable candidate now has checkpoint-specific licensing evidence:
**RTMDet-Ins-tiny**, a compact one-stage detector distributed through
MMDetection's official model zoo. In an open issue specifically naming its
official COCO checkpoint and asking about Apache-2.0 licensing, fine-tuning,
and use, a repository collaborator replied, “Yes. They are all permitted.”
This is materially stronger than inferring the checkpoint license from the
repository license alone.

YOLOX-Nano remains a promising, smaller technical candidate, but its
checkpoint terms remain unresolved: the official issue asking about the
release weights has no maintainer answer. It should not be used unless that
question is answered separately.

RTMDet-Ins-tiny's model-zoo size (5.6M parameters, 11.8 GFLOPs, 640-pixel
input) is plausible for the RTX 3050 at batch one, but not a memory guarantee.
MMDetection/MMCV installation and runtime compatibility with this Windows
Python 3.12 / PyTorch 2.15.0+cu132 / TorchVision 0.29.2+cu132 environment is
also not established. Therefore, the recommendation is **A — Proceed with a
verified permissively licensed detector**, specifically to a bounded
compatibility assessment and smoke-test phase for RTMDet-Ins-tiny. This is
not approval to begin the full six-epoch comparison: installation,
checkpoint-head adaptation, empty-image handling, and the 4 GB GPU smoke
test must pass first.

## Candidate comparison

| Candidate | Source license | Exact pretrained source and dataset | Checkpoint-license evidence | Compatibility / size | Assessment |
|---|---|---|---|---|---|
| **YOLOX-Nano** | Apache-2.0 in the official [YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE). | The official [model zoo](https://github.com/Megvii-BaseDetection/YOLOX#benchmark) links [`yolox_nano.pth` from release 0.1.1rc0](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth), trained on COCO. | **Unresolved.** The release asset does not declare its own license. [Issue #1865](https://github.com/Megvii-BaseDetection/YOLOX/issues/1865) asks the maintainers whether the official release weights inherit Apache-2.0; the reviewed issue and comments do not provide a maintainer answer. Do not infer the checkpoint license from the source repository license. | Official [Nano config](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/default/yolox_nano.py) sets 416×416 input. Official model-zoo table reports 0.91M parameters and 1.08 GFLOPs. `setup.py` declares Python `>=3.6`, requirements declare `torch>=1.7`, and the custom-op build is skipped on Windows. Those declarations do not certify Python 3.12, PyTorch 2.15, or CUDA 13.2. No local runtime test has been performed. | Best technical candidate, but **not license-cleared**. Likely feasible on 4 GB at Nano/batch-one scale, but that is an unverified hypothesis until a smoke test. |
| **RTMDet-Ins-tiny (MMDetection)** | Apache-2.0 in the official [MMDetection repository](https://github.com/open-mmlab/mmdetection/blob/main/LICENSE). | Exact official COCO checkpoint: [`rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth`](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth); official [config](https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py). | **Affirmative checkpoint-specific evidence.** In [MMDetection issue #11484](https://github.com/open-mmlab/mmdetection/issues/11484), a user asked about this exact checkpoint's Apache-2.0 status, fine-tuning, modifications, and private use. A repository collaborator replied, “Yes. They are all permitted.” The answer is brief but directly responds to that exact checkpoint and the listed use cases. | Official [model-zoo table](https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/README.md) reports COCO, 640 input, 5.6M parameters, and 11.8 GFLOPs. MMDetection documents Windows, Python 3.7+, CUDA 9.2+, and PyTorch 1.8+, but these general minimums do not certify this current Windows/Python 3.12 / Torch 2.15 / CUDA 13.2 stack. MMEngine/MMCV are required; 4 GB training fit is plausible, not proven. | **Best license-cleared candidate.** Proceed to compatibility and smoke testing only; no full training approval yet. |
| **TorchVision RetinaNet ResNet-50 FPN V2** | TorchVision source repository is BSD-3-Clause; see its [LICENSE](https://github.com/pytorch/vision/blob/v0.29.2/LICENSE). | Exact official COCO checkpoint is specified by the versioned [RetinaNet V2 documentation](https://docs.pytorch.org/vision/0.29/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html) and its source weight enum. | **Not explicitly verified for this checkpoint.** TorchVision's [pretrained-model guidance](https://docs.pytorch.org/vision/stable/models.html) specifically cautions that weights may have their own licenses or terms derived from training datasets and instructs users to check each model. The BSD source-code license is not an explicit license declaration for this exact binary. | Native PyTorch/TorchVision detection API and the least disruptive engineering integration. ResNet-50 FPN is considerably larger than YOLOX-Nano; the published compatibility with the current runtime is better, but training memory on 4 GB is untested and may require a batch-one smoke test. | Useful technical fallback only if checkpoint-specific terms are clarified; it does not meet the current licensing evidence requirement. |
| **PP-YOLOE (PaddleDetection)** | The official [PaddleDetection repository](https://github.com/PaddlePaddle/PaddleDetection) is Apache-2.0. | The official [PP-YOLOE model documentation](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8/configs/ppyoloe) describes COCO-pretrained variants and associated model sources. | **Not verified for an exact checkpoint.** The evidence reviewed establishes the repository license, not a separate explicit license for a selected binary checkpoint. | Implemented in PaddlePaddle, not PyTorch. Adopting it would require a separate framework and training/evaluation path, making it a poor fit for the existing ScenarioForge Torch trainer and constrained environment. | Not recommended for this comparison. |

### Excluded license classes

- **Ultralytics YOLO (including current YOLO11 releases):** excluded because
  the official project uses AGPL-3.0 or a commercial Enterprise license; the
  user explicitly requested avoiding AGPL.
  See the [official licensing page](https://www.ultralytics.com/license).
- **YOLOv6:** excluded because the official [repository license](https://github.com/meituan/YOLOv6/blob/main/LICENSE)
  is GPL-3.0. GPL is not an acceptable substitute for avoiding AGPL.
- **YOLO-NAS:** not advanced as a candidate because its pretrained model terms
  are separate from the implementation license and the reviewed sources
  describe additional restrictions. It does not provide a simpler,
  checkpoint-cleared path than YOLOX-Nano.

## Source-code licensing

### YOLOX

The official repository's root license is Apache-2.0. The official README
identifies YOLOX as a one-stage, anchor-free YOLO detector, and the Nano
configuration supplies a compact 416×416 model. These are confirmed source
and architecture facts. Apache-2.0 source licensing does not automatically
prove that a separately distributed release asset has the same license.

### RTMDet-Ins-tiny: checkpoint-specific clarification

The official model-zoo config and checkpoint URL identify the exact
RTMDet-Ins-tiny COCO checkpoint. MMDetection issue #11484 asks about that
checkpoint's Apache-2.0 status, fine-tuning, architecture changes, and
private use. A repository collaborator answered, “Yes. They are all
permitted.” The answer is brief, but it is a direct, attributable response
to the explicit checkpoint-licensing questions. This assessment treats that
as affirmative evidence for the named checkpoint and the requested uses;
it does not generalize the answer to every MMDetection checkpoint.

### Other candidates

MMDetection's code is Apache-2.0; PaddleDetection's code is Apache-2.0;
TorchVision's code is BSD-3-Clause. These source licenses are permissive, but
the exact checkpoint requirement must be assessed separately for each weight
file. No source-license inference is made here about those binaries.

## Pretrained checkpoint licensing

### YOLOX-Nano

Confirmed:

1. The official model-zoo table links the Nano checkpoint in the official
   GitHub release.
2. The model-zoo table identifies the COCO benchmark and the Nano model
   footprint.
3. The official YOLOX source repository includes Apache-2.0 terms.
4. An official-repository issue asks whether release checkpoints are covered
   by those terms.

Not confirmed:

1. The release asset itself does not display an explicit license declaration
   in the inspected release materials.
2. No maintainer response resolving the issue was found in the issue/comments
   reviewed.
3. A repository-level license is not being treated as proof that the
   checkpoint asset has the same license.

Thus, checkpoint rights remain **unresolved**, including whether the weights
may be fine-tuned, redistributed, or included in a downstream project under
Apache-2.0.

### Alternative checkpoints

The MMDetection issue clarifies the named RTMDet-Ins-tiny checkpoint, but
should not be generalized to RTMDet-tiny or unrelated model-zoo weights
without confirming those exact assets. TorchVision explicitly warns that
pretrained models may carry separate or dataset-derived terms. No exact
RetinaNet checkpoint license notice was identified in its versioned
docs/source. The inspected PP-YOLOE materials also did not establish
licensing for an exact chosen checkpoint.

## Compatibility assessment

### YOLOX-Nano

- **Python/PyTorch:** Official setup metadata states Python `>=3.6`; the
  requirements state `torch>=1.7` and depend on TorchVision. There are no
  upper version bounds, but this is not proof of support for Python 3.12,
  PyTorch 2.15, TorchVision 0.29.2, or CUDA 13.2. No package installation or
  runtime validation was performed.
- **Windows:** The official setup skips the custom fast COCO evaluation
  extension build on Windows. The repository has Torch/TorchVision-based
  model code; this suggests a possible Windows path, not a verified
  end-to-end installation or training path.
- **Integration:** A ScenarioForge adapter would need to translate the
  current 1-based labels and XYXY pixel boxes into YOLOX's zero-based,
  center/width/height padded target format; preserve empty images; adapt
  YOLOX's training loss outputs so only the optimization loss is summed;
  convert inference outputs to the shared evaluator's expected label and
  XYXY format; and replace/reinitialize the 80-class output head for the
  ten ScenarioForge classes. The official augmentation and letterbox
  preprocessing would also need to be recorded as model-specific protocol.
- **Data isolation:** Existing ScenarioForge experiment setup already
  constructs train and validation datasets for training without creating
  the test dataset. A future adapter can reuse that boundary, but this has
  not been implemented or tested.

### RTMDet-Ins-tiny

- **Python/PyTorch/Windows:** MMDetection's official getting-started guide
  states general support for Windows, Python 3.7+, CUDA 9.2+, and PyTorch
  1.8+. Those broad minimums do not certify the current Python 3.12,
  PyTorch 2.15.0+cu132, TorchVision 0.29.2+cu132 combination. MMCV's
  official installation guide says a source build may be needed when no
  matching wheel exists. Exact installation/build success was not tested.
- **Integration:** MMDetection is PyTorch-based and exposes detection
  predictions, but its `DetDataSample`, config, and runner interfaces differ
  from ScenarioForge's adapter. A thin adapter is feasible in principle,
  but must bridge datasets, transforms, loss dictionaries, prediction
  labels/boxes, checkpoints, and evaluator calls without using its default
  training runner to bypass ScenarioForge provenance or split controls.
- **Memory and protocol:** The published model is 640×640 and lists a
  distributed 8-GPU, batch-32 training recipe. That is not evidence that a
  single 4 GB GPU can train it at batch one. Any single-GPU batch/optimizer
  adjustment and the instance-mask branch must be recorded as
  model-specific comparison differences. A CUDA smoke test is required
  before training.

### Runtime and hardware

The RTX 3050 Laptop GPU has 4 GB VRAM. RTMDet-Ins-tiny's official
5.6M-parameter, 11.8-GFLOP, 640×640 model makes a batch-one run plausible,
but model-zoo complexity figures are not training-memory measurements. No
weights were downloaded, no CUDA smoke test was run, and no claim is made
that the requested training fits.

## Recommended model

**RTMDet-Ins-tiny with the exact official COCO checkpoint named above** is
the recommended permissively licensed candidate for a one-stage architecture
comparison. It has an Apache-2.0 implementation, official COCO model-zoo
configuration and checkpoint source, and checkpoint-specific affirmative
maintainer/collaborator confirmation in the licensing issue.

The model is an instance-segmentation variant that also predicts bounding
boxes. For this comparison, use its box detections and do not introduce a
mask task. Its extra mask branch is a model-specific difference that should
be recorded. Because the checkpoint's COCO class head differs from
ScenarioForge's ten categories, transfer learning will require replacing or
adapting that head and loading compatible pretrained parameters.

YOLOX-Nano is smaller and closer to the desired YOLO-family implementation,
but remains blocked by its unanswered checkpoint-licensing question.

## Remaining legal or technical uncertainties

1. The RTMDet-Ins-tiny reply is affirmative but brief; retain the exact
   checkpoint name and issue permalink in the provenance, and obtain legal
   review if a formal opinion is needed.
2. Verify the pinned MMDetection/MMCV implementation against Python 3.12,
   PyTorch 2.15.0+cu132, TorchVision 0.29.2+cu132, and CUDA on Windows
   without replacing the existing Torch stack. OpenMMLab's MMCV installation
   guide explains that a source build may be necessary when a matching
   prebuilt wheel is unavailable. MMDetection's generic minimum-version
   statement is not treated as proof for these exact versions. No exact
   wheel/build was validated here.
3. After compatibility checks, run the prescribed one-batch CUDA
   forward/backward, finite-loss/gradient, empty-image, peak-memory, and
   checkpoint round-trip smoke tests. Stop if it OOMs; do not silently
   change batch size, image size, precision, or architecture.
4. Before any full comparison, document and test class-head adaptation,
   model-specific preprocessing, the shared COCO evaluator protocol, and
   train/validation/test isolation.

## Go/no-go decision

**A — Proceed with a verified permissively licensed detector: RTMDet-Ins-tiny
using the exact checkpoint identified in this report.**

This is a **conditional go for dependency/compatibility assessment,
implementation, and the required smoke test only**. It is not approval for
the six-epoch training run. Do not start that run unless the existing
environment remains compatible, batch-one CUDA smoke tests succeed within
4 GB, class mapping and data isolation are verified, and the shared
evaluation path is tested. YOLOX-Nano remains a no-go until its own
checkpoint-license question is answered.

This decision does not change any experiment artifacts, source code,
annotations, or split assignments.

## Sources

### Primary sources

- [YOLOX source repository and Apache-2.0 LICENSE](https://github.com/Megvii-BaseDetection/YOLOX)
- [YOLOX Apache-2.0 LICENSE](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE)
- [YOLOX official model zoo / checkpoint table](https://github.com/Megvii-BaseDetection/YOLOX#benchmark)
- [YOLOX-Nano configuration](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/default/yolox_nano.py)
- [Official YOLOX-Nano release asset](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth)
- [YOLOX issue #1865 — checkpoint license clarification request](https://github.com/Megvii-BaseDetection/YOLOX/issues/1865)
- [YOLOX setup metadata](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/setup.py)
- [YOLOX dependencies](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/requirements.txt)
- [MMDetection model zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html)
- [MMDetection issue #11484 — RTMDet checkpoint license clarification request](https://github.com/open-mmlab/mmdetection/issues/11484)
- [MMDetection collaborator response to the exact RTMDet-Ins-tiny checkpoint question](https://github.com/open-mmlab/mmdetection/issues/11484#issuecomment-6407704720)
- [MMDetection source repository and LICENSE](https://github.com/open-mmlab/mmdetection)
- [RTMDet config/model-zoo results and exact checkpoint link](https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/README.md)
- [MMCV installation guide and source-build fallback](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
- [MMDetection getting-started platform and version requirements](https://mmdetection.readthedocs.io/en/latest/get_started.html)
- [TorchVision 0.29.2 RetinaNet V2 documentation](https://docs.pytorch.org/vision/0.29/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html)
- [TorchVision pretrained-model licensing guidance](https://docs.pytorch.org/vision/stable/models.html)
- [TorchVision 0.29.2 BSD-3-Clause LICENSE](https://github.com/pytorch/vision/blob/v0.29.2/LICENSE)
- [PaddleDetection PP-YOLOE configuration and model documentation](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8/configs/ppyoloe)
- [PaddleDetection Apache-2.0 LICENSE](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8/LICENSE)
- [Official Ultralytics license options](https://www.ultralytics.com/license)
- [Official YOLOv6 GPL-3.0 LICENSE](https://github.com/meituan/YOLOv6/blob/main/LICENSE)

### Evidence standard

Search summaries and third-party mirrors were not accepted as proof of
checkpoint licensing. For RTMDet-Ins-tiny, the affirmative answer is a
repository-collaborator response in the official issue thread, not an
inference from the repository LICENSE. Where official sources did not
explicitly state or answer for the exact checkpoint, the status is reported
as unresolved.
