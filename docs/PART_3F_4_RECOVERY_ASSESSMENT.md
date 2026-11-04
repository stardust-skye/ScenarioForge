# Part 3F.4 - Interrupted Pilot Recovery Assessment

## Executive summary

The pilot process is no longer active. The saved state shows:

| Strategy | Saved state | Classification | Remaining work |
|---|---|---|---|
| Random uniform | Six epochs completed; best checkpoint at epoch 5 | **COMPLETE** | None |
| Frequency-based | Six epochs completed; best checkpoint at epoch 6 | **COMPLETE** | None |
| Diversity-aware | Four epochs completed; valid epoch-4 resume checkpoint | **RESUMABLE** | Resume for epochs 5-6 |
| Weighted-interest | Preflight artifacts only; no history or checkpoint | **RESTART REQUIRED** | Start a new six-epoch run after approval |

No run was resumed or restarted during this assessment. The test partition was not evaluated. The original pilot artifacts remain unchanged.

## Inspection scope and interruption state

Inspected the pilot controller record and each strategy output directory under
[`part3f4_20261104T135803Z_fb283e75`](../experiments/part3f4_curation_pilots/part3f4_20261104T135803Z_fb283e75/).
The controller record reports `training_status: running`, `completed_conditions: 2`,
and no `failed_strategy` or `failure`. Its condition statuses are `completed` for
random and frequency, and `preflight_passed` for diversity and weighted-interest.
The controller state is therefore stale relative to the stopped process.

No active `scenarioforge.training.cli curation-pilot`, `train`, or `resume` process
was found. The Python processes present were Jupyter, editor language-server, and
Unity MCP processes, not the ScenarioForge pilot. No explicit error, interruption,
or final pilot report file exists in the pilot directory. The user-reported manual
interruption is consistent with the checkpoint state, but the precise interruption
time and cause are not recorded in the artifacts.

## Per-strategy findings

### Random uniform - COMPLETE

- `training_history.csv` contains contiguous epochs 1-6.
- `checkpoint_last.pth` is checkpoint version 2 at epoch 6 / global step 360.
- `checkpoint_best.pth` is version 2 at epoch 5.
- The best epoch, saved best score, history row, and best-checkpoint marker agree.
- `experiment_summary.json` records six completed epochs and `test_evaluated: false`.
- No resume or restart is required.

### Frequency-based - COMPLETE

- `training_history.csv` contains contiguous epochs 1-6.
- `checkpoint_last.pth` and `checkpoint_best.pth` are version 2 at epoch 6 /
  global step 360.
- The best epoch, saved best score, history row, and best-checkpoint marker agree.
- `experiment_summary.json` records six completed epochs and `test_evaluated: false`.
- No resume or restart is required.

### Diversity-aware - RESUMABLE

- `training_history.csv` contains contiguous epochs 1-4.
- `checkpoint_last.pth` and `checkpoint_best.pth` are version 2 at epoch 4 /
  global step 240.
- The best epoch, saved best score, history row, and best-checkpoint marker agree.
- The run has no `experiment_summary.json`, consistent with an incomplete run.
- Two epochs remain under the recorded six-epoch configuration.

The checkpoint contains model, optimizer, scheduler, and nonempty AMP scaler
state; RNG state for Python, NumPy, CPU Torch, and CUDA; and a train-loader
generator state. The saved config and checkpoint metadata agree on the
experiment ID, configuration digest, provenance digest, and split digest.
History and checkpoint epoch alignment, best-checkpoint consistency, and
provenance checks required by the resume path all pass. This run can be resumed
in its existing directory without changing the saved controls.

### Weighted-interest - RESTART REQUIRED

- The directory contains `config.json` and `environment.json` from preflight.
- It has no `training_history.csv`, `checkpoint_last.pth`,
  `checkpoint_best.pth`, `experiment_summary.json`, or validation metrics.
- The six-epoch training run has not started; there is no resumable checkpoint.

If approved for execution, create a new uniquely named experiment directory and
preserve the existing preflight-only directory. Use the same approved weighted
interest selection and controls; do not overwrite or repurpose the interrupted
pilot artifacts.

## Checkpoint and provenance integrity

Checkpoint files were loaded read-only on CPU with `weights_only=True`.
All three existing checkpointed runs contain checkpoint version 2 and the
expected state components. Their latest and best checkpoint metadata agree.

| Provenance item | Verified value | Result |
|---|---|---|
| Run-level curation manifest SHA-256 | `b7fa8bc6276c626b7a03cc180e1988403dcc8b5b7b091a54495fb07bd92e217d` | Current manifest bytes and pilot record match |
| Random child manifest digest | `85c331dcbd389cb4b48b70a9a19a3d5e57700bd611ce495b9d2ed4cf8712864c` | Canonical digest matches the manifest and checkpoint |
| Frequency child manifest digest | `bcfe9fc1a3274e72920a0f7e13b41c80b9f9d35c360584b09fdaea2989bd2927` | Canonical digest matches the manifest and checkpoint |
| Diversity child manifest digest | `580d64e61484b88560233c25d35d164622ff07c08ad5e2933809a0259b83b9c2` | Canonical digest matches the manifest and checkpoint |
| Weighted-interest child manifest digest | `fd8d61d4854544fbf70a6bba3c2c56617b79bcf480ac54bc4d137fa49a9e6959` | Canonical digest matches the manifest and pilot record |
| Pretrained COCO checkpoint SHA-256 | `907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5` | Matches all existing checkpoint metadata and the cached checkpoint |

Child manifest digests are computed using the project's canonical JSON payload
digest (excluding the embedded `manifest_sha256` field), not by hashing the
serialized JSON file bytes. Recomputing this canonical digest matched all four
approved child digests. The pilot record stores the run-level digest at the
controller level; the three trained checkpoints independently carry both their
child and run-level digests.

## Controls and resume behavior

All four saved configs specify the same training controls: Faster R-CNN
MobileNetV3 Large 320 FPN, COCO_V1 initialization, seed 42, six epochs, CUDA,
batch size 1, learning rate `1e-4`, weight decay `1e-4`, StepLR (step size 3,
gamma 0.1), gradient clipping at 1.0, AMP enabled, validation every epoch, and
best-effort reproducibility. No image-size override is configured. The saved
environment metadata is internally consistent with each checkpoint's
configuration and provenance digests.

The resume implementation rejects completed runs, requires version-2 state,
requires history to end exactly at the checkpoint epoch, checks the best metric
against history and the best checkpoint, and validates expected experiment
metadata before restoring optimizer/scheduler/scaler and RNG state. It restores
the DataLoader generator state before continuing. These checks are implemented
in `src/scenarioforge/training/trainer.py` (`resume` and `fit`) and
`src/scenarioforge/training/checkpointing.py` (`load_checkpoint`).

## Safe recovery recommendation

1. Keep the random and frequency runs unchanged and mark them complete.
2. After explicit approval, resume diversity-aware from its existing
   `checkpoint_last.pth` in the same experiment directory for exactly two
   remaining epochs. Preserve seed, data selection, validation set, and all
   saved controls.
3. After explicit approval, launch weighted-interest as a fresh run in a new,
   unique output directory, using the existing audited manifest and six-epoch
   controls. Do not overwrite its preflight-only directory.
4. Update or regenerate the pilot-level status/report only after approved
   execution. Do not treat the stale `training_status: running` field as proof
   that a process remains active.

Exact remaining work is **two diversity-aware epochs plus six weighted-interest
epochs**. No work is required for random or frequency. A multi-strategy pilot
report should be finalized only after those remaining runs complete; no
comparative or statistical conclusion is supported by this single seed.

## Validation performed

- Read-only inventory of pilot status, all strategy outputs, histories,
  summaries, environment/config records, and checkpoint files.
- Process-table inspection for active ScenarioForge training commands.
- Safe CPU checkpoint deserialization and checks of state sections, epoch/global
  step, history continuity, best metric consistency, and metadata digests.
- Independent recomputation of the run-manifest file hash and canonical child
  manifest digests.
- No training, resume, restart, test evaluation, artifact modification, or
  package installation was performed.
