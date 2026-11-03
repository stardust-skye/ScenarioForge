# Part 3E.4 — RetinaNet Pretrained Weight Licensing Resolution

## Executive summary

**Recommendation: CONDITIONAL GO — do not use the checkpoint until
checkpoint-specific clarification is obtained.**

The exact candidate is Torchvision's
`RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1`, hosted at:

`https://download.pytorch.org/models/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth`

The publisher/source is the PyTorch/Torchvision project; the official versioned
Torchvision API page and model enum identify this URL and the COCO-trained
weights. However, the materials reviewed do **not** state an explicit license
for this exact checkpoint. Torchvision's general model guidance expressly
warns that pretrained weights can have separate terms and puts responsibility
on the user to determine permission. The Torchvision BSD-3-Clause source
license cannot be treated as a checkpoint license.

The COCO terms distinguish CC BY 4.0 annotations from images whose copyrights
remain with their individual owners and whose use must follow Flickr terms.
Those dataset terms do not themselves state a license for this trained
checkpoint or settle whether fine-tuned model parameters are permitted for a
particular use.

No evidence found here is sufficient to conclude that the exact weights may
be used for academic research, modified, fine-tuned, redistributed, or
publicly demonstrated. Ask Torchvision/the checkpoint publisher for an
authoritative answer covering the intended uses before downloading or using
the checkpoint. Do not infer permission from its public URL or repository
license.

## 1. Exact checkpoint and provenance

| Item | Evidence |
|---|---|
| Architecture | RetinaNet ResNet-50 FPN V2 |
| Weight identifier | `RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1` |
| Exact official URL | [`retinanet_resnet50_fpn_v2_coco-5905b1c5.pth`](https://download.pytorch.org/models/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth) |
| Host/publisher | `download.pytorch.org`, referenced by Torchvision's official weight enum and documentation |
| Pretraining provenance | Torchvision's versioned model documentation presents COCO categories, COCO-val2017 evaluation metadata, and the official training recipe reference. The checkpoint is identified as COCO_V1. |
| Recipe/source history | [Torchvision PR #5756, “Add RetinaNet improved weights”](https://github.com/pytorch/vision/pull/5756) |
| Local artifact status | No local checkpoint file was intentionally saved or used. See the operational note at the end of this report. |

The exact hash prefix appears in the official filename. No independent full
SHA-256 was obtained because downloading or analyzing the checkpoint file was
out of scope.

## 2. Separate licensing layers

### Torchvision implementation source — confirmed BSD-3-Clause

The [Torchvision v0.29.2 LICENSE](https://github.com/pytorch/vision/blob/v0.29.2/LICENSE)
states BSD 3-Clause. This is a source-code license. It does not explicitly
identify the license for the RetinaNet COCO_V1 binary.

### RetinaNet architecture — no separate grant identified

The reviewed Torchvision API, implementation, and recipe identify the model
architecture and its code, but do not give a separate patent or architecture
license for the RetinaNet design. This assessment does not infer a grant or
restriction beyond the explicit Torchvision source-code license and does not
provide a legal conclusion about architecture-related rights.

### Exact pretrained checkpoint — unresolved

The [Torchvision 0.29 RetinaNet API page](https://docs.pytorch.org/vision/0.29/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html)
documents the `COCO_V1` weight enum, model metrics and the official model
builder. It does not state a checkpoint-specific license in the reviewed
material.

Torchvision's [0.29 model and pretrained-weight guidance](https://docs.pytorch.org/vision/0.29/models.html)
states:

> The pre-trained models provided in this library may have their own licenses
> or terms and conditions derived from the dataset used for training. It is
> your responsibility to determine whether you have permission to use the
> models for your use case.

Accordingly, the checkpoint's license is **not established** by the available
official metadata or by the repository's BSD-3-Clause license.

### COCO training data — mixed rights; does not decide the checkpoint license

The [official COCO terms of use](https://cocodataset.org/dataset/termsofuse.htm)
distinguish:

- **Annotations and website:** licensed under
  [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/legalcode).
- **Images:** COCO Consortium does not own their copyright; use must abide by
  the [Flickr terms](https://www.flickr.com/creativecommons/), and users
  accept responsibility for their dataset use.
- **COCO software:** has its own BSD-style terms, separate from both model
  weights and the dataset images.

These are terms for COCO components, not an explicit license grant for the
Torchvision RetinaNet checkpoint. They establish neither that the trained
weights are BSD-3 nor that all downstream uses or redistribution of the
weights are permitted. Whether and how image or annotation terms affect
trained parameters and derivatives remains a question for the rights holders
and qualified legal counsel.

## 3. Official maintainer and repository evidence

1. **Current general notice:** Torchvision's model documentation explicitly
   warns that pretrained weights may carry terms separate from the code
   license. This is the most directly applicable published guidance, but it
   does not answer which terms apply to this checkpoint.
2. **Exact RetinaNet checkpoint:** No official checkpoint-specific license
   statement or maintainer clarification was found in the versioned model
   page, weight metadata, linked recipe, or reviewed repository issues.
3. **General Torchvision weight-license inquiry:** [Issue #8017](https://github.com/pytorch/vision/issues/8017)
   asks what terms apply to binaries at `download.pytorch.org`; it does not
   identify this RetinaNet file or provide a maintainer answer establishing
   its terms.
4. **Other checkpoint inquiries:** [Issue #9726](https://github.com/pytorch/vision/issues/9726)
   asks about different, ImageNet-classification artifacts and does not
   establish RetinaNet COCO_V1 terms. It is not evidence that the RetinaNet
   checkpoint has the same licensing status.
5. **Historical generic discussion:** [Issue #160](https://github.com/pytorch/vision/issues/160)
   contains a 2019 Torchvision member's “probably yes” reply to a general
   question about older VGG weights. It predates the RetinaNet V2 checkpoint,
   is qualified rather than definitive, and is not checkpoint-specific. It
   does not resolve this assessment.

No release note, model card, or exact-artifact maintainer response was found
that answers permission for this checkpoint's research use, modification,
fine-tuning, or redistribution.

## 4. Use-case status

| Intended use | Status for this exact checkpoint | Reason |
|---|---|---|
| Academic/research experiments | **Unresolved** | Public availability and BSD-licensed code do not establish checkpoint permission; no explicit weight terms were found. |
| Replacing the detection head and fine-tuning on ScenarioForge data | **Unresolved** | Torchvision provides technical APIs, but no checkpoint-specific terms were found for modification or fine-tuning. |
| Retaining the original checkpoint privately | **Unresolved** | No authoritative use grant was found for this binary. |
| Public portfolio demonstration using the model | **Unresolved** | Inference use still depends on permission to use the weights; demonstrating output does not resolve that prerequisite. |
| Redistributing the original or fine-tuned checkpoint | **Unresolved; do not proceed** | No explicit terms address redistribution or derivatives. |
| Publishing source code that downloads/uses the checkpoint | **Unresolved** | A public URL is not itself a license grant, and no checkpoint terms are supplied by the reviewed documentation. |

This table records evidence status; it is not legal advice.

## 5. Alternatives considered

- **RTMDet-Ins-tiny:** Part 3E.1 records checkpoint-specific affirmative
  licensing evidence from a repository collaborator concerning the exact
  OpenMMLab checkpoint. However, Part 3E.2 found the current Windows / Python
  3.12 / PyTorch 2.15 / CUDA 13.2 stack incompatible with the reviewed MMCV
  wheel/build path. It is a licensing lead, not a currently viable technical
  replacement in this environment.
- **YOLOX-Nano:** Part 3E.1 found the source repository license but no
  maintainer answer establishing terms for the official Nano checkpoint.
  Its checkpoint is therefore not a licensing-cleared alternative.
- **Other Torchvision pretrained detectors:** The general Torchvision weight
  notice applies; no exact COCO detector checkpoint with sufficiently
  explicit, reviewed checkpoint terms was established in this investigation.

Thus, no alternative currently meets both the desired technical constraints
and the requirement for explicit checkpoint licensing evidence. An
unpretrained/random-initialized RetinaNet could avoid this checkpoint question
but would not satisfy the stated pretrained-COCO initialization objective and
would be a different experiment requiring approval.

## 6. Questions for authoritative clarification

Ask the Torchvision maintainers or checkpoint publisher to answer, with
reference to this exact enum and file:

1. What license or other governing terms apply to
   `RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1` and the file
   `retinanet_resnet50_fpn_v2_coco-5905b1c5.pth`?
2. Does that grant permit academic research, modification of the
   classification head, fine-tuning on ScenarioForge data, and private
   inference?
3. Does it permit public portfolio demonstration, distribution of derived
   fine-tuned weights, or redistribution of the original checkpoint? What
   attribution, notice, or other conditions apply?
4. Are any additional obligations or restrictions applicable because the
   model was trained using COCO images/annotations? If so, which source terms
   govern those uses?
5. Can the maintainer point to a public license file, model card, or written
   policy that covers this exact checkpoint and its derivatives?

Until an authoritative response is recorded, do not download or use the
checkpoint.

## 7. Decision

**CONDITIONAL GO — only after obtaining checkpoint-specific clarification.**

The available evidence does not support a GO. It also does not establish that
the checkpoint is prohibited, so a categorical NO-GO is not supported by
these sources alone. Keep the candidate blocked for implementation and
training until the exact terms and intended-use permissions are documented.
If clarification is unavailable, choose a different detector whose exact
pretrained checkpoint has explicit terms and passes the separate technical
compatibility assessment.

## Operational note

During research, web retrieval calls were inadvertently made against the
checkpoint's direct URL. The tool responses contained raw checkpoint bytes.
No checkpoint file was saved into the repository or intentionally written to
local storage, and the bytes were not used for inspection or model loading.
Nevertheless, the remote artifact content was accessed; therefore the strict
“no checkpoint downloads” constraint was not fully observed. This deviation
is disclosed here rather than represented as full compliance.
