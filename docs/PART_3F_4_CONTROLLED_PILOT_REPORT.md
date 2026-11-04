# Part 3F.4 - Controlled Curation Pilot Report

**Status:** completed  
**Test partition evaluated:** False  
**Preflight passed:** True

## Objective and controls

Compare four train-only, scene-balanced 60-image subsets using the same Faster R-CNN MobileNetV3 Large 320 FPN architecture, COCO_V1 initialization, seed 42, six epochs, batch size 1, AdamW, StepLR, AMP, preprocessing, and the same 80-image validation partition. This single-seed feasibility pilot does not establish statistical significance or general improvement.

The training command instantiated only train and validation datasets. No test dataset was instantiated or evaluated.

## Preflight and provenance

- Curation run-manifest SHA-256: `b7fa8bc6276c626b7a03cc180e1988403dcc8b5b7b091a54495fb07bd92e217d`
- COCO_V1 checkpoint SHA-256: `907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5`
- Initial model-state SHA-256: `b5a5664efe348490ae46f3d2f076e748271aa21917428225fd0fb44be798f81a`
- Each completed checkpoint contains curation child/run-manifest hashes and the pretrained checkpoint hash in metadata.
- Original pilot directory: `experiments/part3f4_curation_pilots/part3f4_20261104T135803Z_fb283e75`
- Fresh weighted-interest recovery directory: `experiments/part3f4_curation_pilots/part3f4_recovery_20261104T190526Z_046cf543/weighted_interest`
- The interrupted weighted-interest preflight directory was preserved. The weighted-interest checkpoint's initial model-state hash matches the original pilot initialization.

## Recovery record

- Random uniform and frequency-based were already complete and were preserved without retraining.
- Diversity-aware resumed in its original directory from the verified epoch-4 `checkpoint_last.pth`; optimizer, scheduler, AMP scaler, Python/NumPy/Torch CPU/CUDA RNG, and DataLoader generator state were restored. Epochs 5 and 6 completed with the original seed and controls.
- Weighted-interest had no prior training checkpoint. It was initialized from the cached COCO_V1 checkpoint and trained for six epochs in the new directory above.
- All four conditions use the same 60 selected training images per strategy, seed 42, six epochs, the same authoritative validation partition and unchanged training controls.
- The run-level curation manifest hash is `b7fa8bc6276c626b7a03cc180e1988403dcc8b5b7b091a54495fb07bd92e217d`; each condition's child-manifest digest is recorded with its row below and in its checkpoint metadata.
- The final pilot status records recovery events. The diversity experiment environment metadata records its resume event; the weighted-interest config and checkpoint provenance record its fresh-restart event.

## Results

| Strategy | Status | Images / boxes | Best epoch | Best val mAP@[0.50:0.95] | AP50 | AP75 | Runtime (s) | Peak CUDA allocation (bytes) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Random uniform | completed | 60 / 901 | 5 | 0.025378 | 0.083857 | 0.012741 | 273.326917 | 439480320 |
| Frequency-based / inverse rarity | completed | 60 / 717 | 6 | 0.018538 | 0.054108 | 0.008117 | 4454.959098 | 669389312 |
| Diversity-aware | completed | 60 / 625 | 4 | 0.017982 | 0.046235 | 0.014138 | 2974.929314 | 901756416 |
| Weighted-interest | completed | 60 / 731 | 4 | 0.018732 | 0.052597 | 0.013978 | 358.419549 | 437421568 |

| Strategy | Child-manifest SHA-256 | Experiment directory |
|---|---|---|
| Random uniform | `85c331dcbd389cb4b48b70a9a19a3d5e57700bd611ce495b9d2ed4cf8712864c` | `experiments/part3f4_curation_pilots/part3f4_20261104T135803Z_fb283e75/random_uniform` |
| Frequency-based / inverse rarity | `bcfe9fc1a3274e72920a0f7e13b41c80b9f9d35c360584b09fdaea2989bd2927` | `experiments/part3f4_curation_pilots/part3f4_20261104T135803Z_fb283e75/frequency_based` |
| Diversity-aware | `580d64e61484b88560233c25d35d164622ff07c08ad5e2933809a0259b83b9c2` | `experiments/part3f4_curation_pilots/part3f4_20261104T135803Z_fb283e75/diversity_aware` |
| Weighted-interest | `fd8d61d4854544fbf70a6bba3c2c56617b79bcf480ac54bc4d137fa49a9e6959` | `experiments/part3f4_curation_pilots/part3f4_recovery_20261104T190526Z_046cf543/weighted_interest` |

## Historical full-data reference

| Reference | Training images | Epochs | Best validation mAP@[0.50:0.95] | Best epoch | Runtime | Peak CUDA allocation |
|---|---:|---:|---:|---:|---:|---:|
| Historical Part 3D.3 baseline | 242 | 6 | 0.038926 | 4 | approximately 810 s | 435,321,856 bytes |

This is a historical full-data reference, not a matched independent replicate. It is not a fifth strategy run and is not included in the equal-budget comparison.

## Per-class validation metrics

### Random uniform

| Class | Validation GT support | AP | Precision (IoU 0.50) | Recall (IoU 0.50) |
|---|---:|---:|---:|---:|
| barrier | 0 | undefined | undefined | undefined |
| bicycle | 15 | 0.000000 | undefined | 0.000000 |
| bus | 11 | 0.005046 | 0.000000 | 0.000000 |
| car | 648 | 0.119090 | 0.581395 | 0.270062 |
| construction_vehicle | 0 | undefined | undefined | undefined |
| motorcycle | 25 | 0.000582 | undefined | 0.000000 |
| pedestrian | 198 | 0.030071 | 0.470588 | 0.080808 |
| traffic_cone | 11 | 0.000000 | undefined | 0.000000 |
| trailer | 0 | undefined | undefined | undefined |
| truck | 44 | 0.022858 | undefined | 0.000000 |

### Frequency-based / inverse rarity

| Class | Validation GT support | AP | Precision (IoU 0.50) | Recall (IoU 0.50) |
|---|---:|---:|---:|---:|
| barrier | 0 | undefined | undefined | undefined |
| bicycle | 15 | 0.000000 | undefined | 0.000000 |
| bus | 11 | 0.000000 | undefined | 0.000000 |
| car | 648 | 0.096718 | 0.600000 | 0.236111 |
| construction_vehicle | 0 | undefined | undefined | undefined |
| motorcycle | 25 | 0.000000 | undefined | 0.000000 |
| pedestrian | 198 | 0.012444 | 0.200000 | 0.005051 |
| traffic_cone | 11 | 0.002640 | undefined | 0.000000 |
| trailer | 0 | undefined | undefined | undefined |
| truck | 44 | 0.017963 | undefined | 0.000000 |

### Diversity-aware

| Class | Validation GT support | AP | Precision (IoU 0.50) | Recall (IoU 0.50) |
|---|---:|---:|---:|---:|
| barrier | 0 | undefined | undefined | undefined |
| bicycle | 15 | 0.000000 | undefined | 0.000000 |
| bus | 11 | 0.000000 | undefined | 0.000000 |
| car | 648 | 0.100624 | 0.780822 | 0.175926 |
| construction_vehicle | 0 | undefined | undefined | undefined |
| motorcycle | 25 | 0.000000 | undefined | 0.000000 |
| pedestrian | 198 | 0.010892 | undefined | 0.000000 |
| traffic_cone | 11 | 0.009901 | undefined | 0.000000 |
| trailer | 0 | undefined | undefined | undefined |
| truck | 44 | 0.004455 | undefined | 0.000000 |

### Weighted-interest

| Class | Validation GT support | AP | Precision (IoU 0.50) | Recall (IoU 0.50) |
|---|---:|---:|---:|---:|
| barrier | 0 | undefined | undefined | undefined |
| bicycle | 15 | 0.000000 | undefined | 0.000000 |
| bus | 11 | 0.000000 | undefined | 0.000000 |
| car | 648 | 0.085379 | 0.572072 | 0.195988 |
| construction_vehicle | 0 | undefined | undefined | undefined |
| motorcycle | 25 | 0.000000 | undefined | 0.000000 |
| pedestrian | 198 | 0.012643 | 0.000000 | 0.000000 |
| traffic_cone | 11 | 0.000381 | undefined | 0.000000 |
| trailer | 0 | undefined | undefined | undefined |
| truck | 44 | 0.032721 | undefined | 0.000000 |

## Selected training-set class composition

### Random uniform

| Class | Accepted training boxes |
|---|---:|
| barrier | 138 |
| bicycle | 11 |
| bus | 39 |
| car | 317 |
| construction_vehicle | 25 |
| motorcycle | 15 |
| pedestrian | 208 |
| traffic_cone | 118 |
| trailer | 3 |
| truck | 27 |

### Frequency-based / inverse rarity

| Class | Accepted training boxes |
|---|---:|
| barrier | 52 |
| bicycle | 22 |
| bus | 23 |
| car | 211 |
| construction_vehicle | 5 |
| motorcycle | 13 |
| pedestrian | 128 |
| traffic_cone | 251 |
| trailer | 9 |
| truck | 3 |

### Diversity-aware

| Class | Accepted training boxes |
|---|---:|
| barrier | 60 |
| bicycle | 22 |
| bus | 17 |
| car | 160 |
| construction_vehicle | 6 |
| motorcycle | 15 |
| pedestrian | 82 |
| traffic_cone | 252 |
| trailer | 8 |
| truck | 3 |

### Weighted-interest

| Class | Accepted training boxes |
|---|---:|
| barrier | 62 |
| bicycle | 20 |
| bus | 26 |
| car | 219 |
| construction_vehicle | 9 |
| motorcycle | 13 |
| pedestrian | 110 |
| traffic_cone | 259 |
| trailer | 9 |
| truck | 4 |

## Loss curves and epoch timings

Training total loss and its components are recorded per epoch. The existing trainer does not compute validation loss; validation mAP is shown instead, and no validation-loss values are inferred.

### Random uniform

| Epoch | Train total loss | Validation mAP@[0.50:0.95] | Epoch seconds |
|---:|---:|---:|---:|
| 1 | 1.314040 | 0.018170 | 56.419825 |
| 2 | 1.147021 | 0.020793 | 19.879644 |
| 3 | 1.076943 | 0.023994 | 18.793453 |
| 4 | 0.941073 | 0.024045 | 142.660599 |
| 5 | 0.941121 | 0.025378 | 17.957901 |
| 6 | 0.925997 | 0.024560 | 17.615492 |

### Frequency-based / inverse rarity

| Epoch | Train total loss | Validation mAP@[0.50:0.95] | Epoch seconds |
|---:|---:|---:|---:|
| 1 | 1.096231 | 0.015788 | 18.123846 |
| 2 | 0.944655 | 0.016554 | 18.925079 |
| 3 | 0.977197 | 0.017502 | 18.837522 |
| 4 | 1.019240 | 0.017146 | 17.945200 |
| 5 | 0.970545 | 0.018068 | 4281.787420 |
| 6 | 0.920076 | 0.018538 | 99.340032 |

### Diversity-aware

| Epoch | Train total loss | Validation mAP@[0.50:0.95] | Epoch seconds |
|---:|---:|---:|---:|
| 1 | 0.936422 | 0.012682 | 243.956800 |
| 2 | 0.780002 | 0.013784 | 1214.858419 |
| 3 | 0.868795 | 0.014486 | 749.087579 |
| 4 | 0.873099 | 0.017982 | 597.925288 |
| 5 | 0.817281 | 0.017692 | 106.838507 |
| 6 | 0.809173 | 0.016128 | 62.262721 |

### Weighted-interest

| Epoch | Train total loss | Validation mAP@[0.50:0.95] | Epoch seconds |
|---:|---:|---:|---:|
| 1 | 1.169222 | 0.013465 | 80.440817 |
| 2 | 0.950092 | 0.014653 | 57.915858 |
| 3 | 0.998764 | 0.013947 | 51.939289 |
| 4 | 0.983429 | 0.018732 | 55.731458 |
| 5 | 0.896550 | 0.018049 | 56.236763 |
| 6 | 0.842066 | 0.017276 | 56.155364 |

## Warnings, failures, and limitations

- **Random uniform warning (720 occurrences):** UserWarning: roi_align_backward_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.
- **Frequency-based / inverse rarity warning (720 occurrences):** UserWarning: roi_align_backward_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.
- **Diversity-aware recovery warning (2 observed occurrences):** `roi_align_backward_kernel` has no deterministic implementation under the configured best-effort deterministic mode. The interrupted pilot did not persist a warning ledger for its first four epochs, so their warning count is unavailable.
- **Weighted-interest warning (720 occurrences):** `roi_align_backward_kernel` has no deterministic implementation under the configured best-effort deterministic mode.
- One seed and a small nuScenes mini validation set do not support significance claims or broad generalization.
- Metadata-derived curation diversity is not visual diversity; subset composition is not detection performance.
- Validation AP for a class without ground-truth support is undefined, not zero.
- Equal-budget results are descriptive for this single seed only. They do not establish that one strategy is superior or improves performance on underrepresented scenarios.

## Readiness for the multi-seed matrix

The pilot completed with all four runs; it validates execution and provenance plumbing only. Repeat the controlled matrix with multiple seeds before drawing comparative or general conclusions.

Detailed per-run artifacts, checkpoints, and provenance are under the unique run directory recorded in `pilot_status.json`.
