# Part 3C.2 Training Infrastructure Remediation Report

## Executive verdict

The Part 3C.1 remediation checklist is addressed, and no known critical blocker remains. The training pipeline now has provenance-checked resume state, explicit strict and best-effort reproducibility modes, split-limited dataset construction, and an auditable final-test authorization gate. Focused regression tests cover the previously identified gaps.

**Recommendation: controlled longer training is approved with conditions.** Use `best_effort` on the current CUDA detector path and describe results as non-bitwise-reproducible. `strict` mode is available and fails clearly when the detector encounters a CUDA operation without a deterministic implementation; it is not currently a viable mode for this detector/GPU combination. Keep test evaluation disabled for training and model selection, and use the explicit one-time authorization workflow only for a separately frozen final evaluation. This approval is for controlled experiments and infrastructure validation, not a claim of research-level model performance.

No long training job was run. No test-partition metrics were evaluated. No raw nuScenes data, annotation exports, scene split assignments, or prior experiment results were modified.

## Original audit findings and resolution

| Finding | Severity | Resolution |
|---|---|---|
| M1. Checkpoint resume did not preserve the training trajectory or validate complete experiment provenance. | Major | **Resolved.** Checkpoints now preserve model, optimizer, scheduler, enabled GradScaler, completed epoch, global step, best metric/epoch, stale-epoch count, Python/NumPy/PyTorch CPU/CUDA RNG state, loader-generator state, configuration and data provenance. Resume validates history and provenance before restoring state. Training samples use reproducible per-epoch loader seeding; incompatible or incomplete legacy checkpoints are rejected. The interrupted/resumed integration test matches uninterrupted lightweight CPU training state and history. |
| M2. “Deterministic” configuration permitted nondeterministic operations to continue. | Major | **Resolved with documented limitation.** `strict` requests deterministic algorithms and fails on unsupported operations; `best_effort` seeds supported sources and records runtime settings/limitations. The current detector’s CUDA ROI Align backward path is not deterministic, so strict CUDA execution fails as intended. Best-effort does not promise bitwise reproducibility. |
| m1. Split setup accessed held-out metadata and annotations. | Minor | **Resolved.** Training experiment setup constructs only train and validation datasets. Split-local filtering avoids validating test images or retaining their annotation rows; full inventory metadata remains available separately for split validation. Regression coverage uses malformed held-out annotations and a missing test image to verify that training setup does not instantiate/access the test dataset. |
| m2. Test gate was a mutable marker and did not seal the selected checkpoint. | Minor | **Resolved.** Final evaluation requires a one-time authorization record binding experiment, checkpoint SHA-256, configuration/data provenance and split-manifest digest, plus purpose, author and timestamp. The authorization is written before test data access. |
| m3. Frozen backbone BatchNorm running state could still update. | Minor | **Resolved.** Frozen-backbone BatchNorm modules remain in evaluation mode and return to training mode when the backbone is unfrozen. Tests verify freeze/unfreeze behavior. |
| m4. Notebook smoke optimizer step affected the subsequent epoch run. | Minor | **Resolved.** The one-batch smoke step uses a disposable model/trainer and restores random/loader state; it does not mutate the actual experiment’s training trajectory. The notebook documents this behavior and that test data is not used. |
| m5. Training history recorded the post-scheduler learning rate. | Minor | **Resolved.** The learning rate associated with each epoch is captured before scheduler advancement. |
| m6. Detector and metric edge paths lacked focused regression tests. | Minor | **Resolved.** Tests cover pretrained COCO construction with replacement of the class-specific predictor, empty-target training, empty detections/targets and undefined metrics, resume/provenance compatibility, frozen BatchNorm, and CUDA AMP when available. Hardware-specific coverage is conditional on CUDA availability. |

## Files changed

- `src/scenarioforge/training/config.py` — explicit reproducibility mode and legacy deterministic-config migration.
- `src/scenarioforge/training/seed.py` — strict/best-effort setup, RNG capture/restore, and runtime determinism metadata.
- `src/scenarioforge/training/checkpointing.py` — versioned full-state checkpoint serialization, validation and atomic writes.
- `src/scenarioforge/training/trainer.py` — provenance-checked resume, counters, epoch-seeded sampling, learning-rate history semantics and frozen-backbone behavior.
- `src/scenarioforge/training/experiment.py` — train/validation-only setup, provenance metadata, isolated smoke step and test authorization.
- `src/scenarioforge/training/model_factory.py` — frozen-backbone BatchNorm mode handling.
- `src/scenarioforge/training/cli.py` — explicit test-authorization argument.
- `src/scenarioforge/vision_data/dataloader.py` and `src/scenarioforge/vision_data/dataset.py` — split-selective access and annotation streaming/filtering while retaining inventory metadata for split validation.
- `tests/test_training.py` — remediation regression and integration coverage.
- `configs/object_detection_smoke.json` — explicit best-effort mode.
- `notebooks/06_Object_Detection_Training_Smoke_Test.ipynb` — test-isolation and disposable smoke-step guidance.
- `docs/PART_3C_TRAINING_INFRASTRUCTURE.md` — updated resume, determinism, Windows/CUDA and test-authorization documentation.
- `docs/PART_3C_2_REMEDIATION_REPORT.md` — this report.

## Validation results

- Focused training tests: **18 passed**.
- Part 3B vision tests: **49 passed**.
- Final full available test suite, run after the runtime-provenance update: **134 passed in 42.76s**.
- Dependency integrity: `python -m pip check` — **No broken requirements found**.
- No long training jobs were run.

### Checkpoint and resume evidence

The deterministic lightweight CPU integration test compares uninterrupted training against an interrupted run resumed from checkpoint. Both produce matching history, model weights, optimizer/scheduler state, global RNG state, loader-generator state and final global step (16). The interruption checkpoint records epoch 1, global step 4 and best epoch 1. Resume rejects incompatible provenance and inconsistent history rather than silently restarting or overwriting prior results.

Exact trajectory replay is supported for the tested lightweight CPU configuration in the same runtime. It is not a universal bitwise guarantee across hardware, software versions or operations whose implementation is nondeterministic.

### Determinism assessment

`strict` enables deterministic algorithms without warning-only fallback. Unsupported operations raise a clear error. A strict CUDA detector check encountered the expected unsupported deterministic operation in ROI Align backward. `best_effort` seeds Python, NumPy, PyTorch CPU/CUDA and the DataLoader’s epoch sampling mechanism, and records relevant PyTorch, CUDA, cuDNN, deterministic-algorithm and hardware settings in experiment metadata and provenance.

The CUDA detector path must therefore be run in best-effort mode on the tested setup; repeated runs may differ numerically. A recorded configuration and environment make a run auditable, not necessarily bitwise reproducible. Windows/CUDA requirements and this limitation are documented in the training guide.

### Held-out-data isolation assessment

Training setup creates only train and validation datasets. The fit interface consumes only those loaders; test construction is deferred to the separate final evaluation path. A regression test verifies training setup succeeds without constructing or reading the test dataset even when the test image is missing and held-out annotation fields are malformed. Training selections reject held-out sample tokens; existing scene-level assignments and split provenance are preserved. No held-out-derived normalization statistics are computed, and category IDs come from the authoritative mapping file.

Final test evaluation requires a one-time authorization tied to the selected checkpoint hash and experiment/split provenance. No test dataset was instantiated for this remediation validation, and no test metrics were evaluated.

### GPU behavior

A short CUDA AMP Faster R-CNN batch completed in best-effort mode with finite losses; all **192/192** trainable parameters received gradients. Peak allocated memory was **386,788,352 bytes** for that batch. This is allocated tensor memory for a smoke step, not total process/GPU memory or a capacity guarantee. Strict mode produced the intended unsupported-operation failure rather than silently continuing. No long or large-batch GPU training was attempted.

### Metric and model behavior

Regression tests cover COCO pretrained initialization followed by replacement of the class-specific predictor with the ScenarioForge class count, empty-target handling, empty predictions/targets, and undefined metric denominators. Undefined precision/recall/AP cases remain `null` rather than being coerced to zero. Evaluation continues to use the Faster-COCO-Eval-backed TorchMetrics COCO metric path, with validation only for selection.

## Remaining risks and limitations

- Strict deterministic execution is unavailable for the current CUDA detector path because ROI Align backward lacks a deterministic implementation in this environment. Best-effort runs are not bitwise reproducible.
- Resume equivalence was validated with a lightweight deterministic CPU integration configuration, not as a guarantee for every detector operation or hardware stack.
- The CUDA AMP evidence is a short smoke batch only; OOM recovery, extended training stability, and larger batch behavior are not established.
- A controlled longer run must pin and preserve its configuration, environment and provenance artifacts, use validation only for selection, and avoid test evaluation until the final selection protocol is frozen.
- The smoke metrics remain systems checks and do not support research-level performance claims.

## Final recommendation

**Controlled longer training: approved conditionally in `best_effort` mode**, with experiment artifacts retained and validation-only model selection. Do not describe the run as strictly or bitwise reproducible. Strict mode should be used only where its fail-fast behavior is acceptable or after the unsupported detector operation is replaced by a deterministic implementation. Final test evaluation remains a separate, explicitly authorized, one-time action after model selection is frozen.
