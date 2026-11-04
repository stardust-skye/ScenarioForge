# Part 3F.5 - Multi-Seed Replication Plan

## Decision

**NO-GO for execution with the current pilot orchestration.** The seed-42 block
is suitable to retain as replicate 42, and the curation generator supports
additional selection seeds. However, the training pilot runner currently pins
the approved seed-42 run and child-manifest digests and validates only that
specific condition set. Before training seeds 123, 2027, 31415, and 271828,
the runner must be extended and tested to accept independently generated,
immutable manifests for those seeds without weakening the current leakage
checks. No training was launched for this plan.

After that prerequisite is implemented, reviewed, and passes its preflight
tests, the design is ready to proceed. No changes to the detector or
hyperparameters are proposed.

## Objective and experimental unit

Estimate whether validation performance differences among random uniform,
frequency-based / inverse-rarity, diversity-aware, and weighted-interest
curation are consistent across five predetermined paired seed blocks, at a
fixed 60-image training budget and six epochs.

One seed block comprises four independent training runs, one per strategy.
Within each block, all strategies use the same training seed, model
initialization, validation set, and training controls. The five blocks are
paired by seed; the seed block, not an individual validation image, is the
unit for across-seed comparisons.

## Seed-42 audit and eligibility

The completed Part 3F.4 results qualify as the seed-42 block, with execution
history disclosed:

- All four conditions completed six contiguous epochs. Their histories,
  latest/best checkpoints, best-epoch metrics, and summaries were independently
  checked for consistency.
- All four use seed 42, 60 unique training images, the same full 80-image
  validation partition, and the same six-epoch Faster R-CNN MobileNetV3 Large
  320 FPN controls.
- Every checkpoint records the same COCO_V1 checkpoint SHA-256
  (`907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5`).
  All four runs record the same initial full-model-state fingerprint
  (`b5a5664efe348490ae46f3d2f076e748271aa21917428225fd0fb44be798f81a`).
- The split-manifest digest, inventory digest, package versions, hardware,
  scene-quota policy, validation interval, and core controls match. All four
  selections contain 10 images from each of the same six training scenes.
- Each run has its own child-manifest digest; all point to the same audited
  run-level curation manifest. Selection and checkpoint provenance are
  recorded in the artifacts.
- Random and frequency completed before interruption. Diversity resumed from
  its verified epoch-4 checkpoint, restoring optimizer, scheduler, AMP scaler,
  Python/NumPy/Torch RNG, and loader-generator state. Weighted-interest had no
  prior training checkpoint and was initialized from the same pretrained
  checkpoint in a new directory. Neither operation changed the controls.
- Test evaluation is recorded as false. No test dataset was instantiated for
  the pilot.

The per-condition configuration and provenance digests are not equal. These
are expected to differ because each condition has different selected tokens,
child-manifest and annotation-count provenance, and output directory.
Execution-event records are also present in diversity and weighted-interest
environment metadata. These differences do not represent differences in
optimizer, scheduler, model architecture, augmentation, batch size, AMP, epoch
count, or validation protocol. The historical seed-42 block should
nevertheless be labeled as a recovered feasibility block, not as four
uninterrupted runs.

Pilot runtime is an important limitation: frequency took about 4,455 seconds
and diversity about 2,975 seconds, with unusually long individual epochs.
The cause is not established by the saved records. This does not invalidate
their completed validation results, but runtime comparisons are exploratory
and the anomalies should be investigated and recorded during replication.

### Seed-42 results to carry forward

| Strategy | Best validation mAP@[0.50:0.95] | AP50 | AP75 | Best epoch | Training images |
|---|---:|---:|---:|---:|---:|
| Random uniform | 0.025378 | 0.083857 | 0.012741 | 5 | 60 |
| Frequency-based / inverse rarity | 0.018538 | 0.054108 | 0.008117 | 6 | 60 |
| Diversity-aware | 0.017982 | 0.046235 | 0.014138 | 4 | 60 |
| Weighted-interest | 0.018732 | 0.052597 | 0.013978 | 4 | 60 |

These are descriptive pilot measurements only. They are not evidence of
statistical significance or strategy superiority.

## Predetermined seeds and selection behavior

Use exactly these five seeds:

| Block | Curation seed | Training seed | Status |
|---:|---:|---:|---|
| 1 | 42 | 42 | Existing completed pilot block |
| 2 | 123 | 123 | To generate and execute |
| 3 | 2027 | 2027 | To generate and execute |
| 4 | 31415 | 31415 | To generate and execute |
| 5 | 271828 | 271828 | To generate and execute |

The authoritative dataset split remains the existing seed-42, scene-grouped
split for every block. The curation/training seed listed above is not a request
to regenerate or alter train/validation/test assignments.

The current curation implementation uses the supplied selection seed in the
random-uniform selector. Frequency ranking uses scenario score with sample ID
as a stable tie-break; diversity-aware selection uses deterministic input
ordering and deterministic greedy tie resolution; weighted-interest ranking
uses score with sample ID as a stable tie-break. Consequently, with unchanged
training-only scenario inputs and scoring configuration, the three ranked
strategies are expected to select the same tokens across these seed values.
Their model training, augmentation, and loader randomness still vary by
training seed. Random uniform selection varies both its selected subset and
training randomness.

This design estimates the combined repeatability of each implemented strategy
under training-seed variation; it does not estimate subset-sampling variance
for the deterministic ranked strategies. If that scientific question is
desired later, it requires a separately designed randomized ranking/tie policy,
not an unrecorded change to this experiment.

## Curation and leakage-safe preflight

For each of seeds 123, 2027, 31415, and 271828:

1. Generate a new unique curation output using `curate-experiment --seed
   <seed>` and the existing authoritative seed-42 split, source files, and
   scoring configuration. The current CLI generates 60-, 120-, and 180-image
   subsets plus a full reference; use only the generated 60-image manifests
   for training. Preserve every prior curation run.
2. Record the generated run-manifest file hash and each strategy child
   manifest's canonical hash, source hashes, seed, split digest, and selected
   tokens. Do not use validation/test metadata to calculate rarity, frequency,
   bins, rankings, or selected subsets.
3. Independently validate that each child manifest selects exactly 60 unique
   sample tokens from the 242 training tokens, with zero validation/test
   overlap; image/annotation mapping must be valid and empty images retained.
4. Require the exact common
   `balanced_capacity_aware_equalization_v1` policy. On this fixed dataset and
   60-image budget, verify 10 selected images in each of the same six training
   scenes for all four strategies. Fail instead of relaxing quotas if a seed
   or source change makes a quota infeasible.
5. Recompute all manifest/source hashes; independently check selected tokens,
   annotation counts, split provenance, and per-scene counts before model
   construction.
6. During training preflight, instantiate only train and validation datasets.
   Verify exactly 60 selected train samples and the same 80 validation tokens
   for each condition. Assert that no test dataset is constructed.
7. Initialize each strategy from the same COCO_V1 checkpoint SHA within a
   seed block. Verify a matching initial model-state fingerprint across the
   four strategies in that block. Never initialize from another run's trained
   checkpoint.

Expected curation artifacts are immutable and uniquely located per seed. The
current generator defaults to 60-, 120-, and 180-image subsets plus a full-data
reference; use only the four 60-image manifests for this matrix and preserve
the other generated artifacts without training them. The full-data reference
is not trained as part of this matrix.

## Fixed training and evaluation controls

Use these controls unchanged for all 20 strategy-by-seed conditions:

| Control | Required value |
|---|---|
| Architecture | Faster R-CNN MobileNetV3 Large 320 FPN |
| Pretrained initialization | Torchvision COCO_V1; verify SHA-256 per run |
| Foreground classes | Existing ten-category mapping plus background |
| Train budget | Exactly 60 images per condition |
| Epochs | 6; no early stopping |
| Batch size / workers | 1 / 0 |
| Optimizer | AdamW, learning rate `1e-4`, weight decay `1e-4` |
| Scheduler | StepLR, step size 3, gamma 0.1 |
| Gradient handling | Accumulation 1; clip norm 1.0 |
| AMP | Enabled |
| Image size / transforms | Existing configuration; no external image-size override |
| Validation | Full fixed 80-image validation split; every epoch |
| Reproducibility | Existing `best_effort` mode; record nondeterministic warnings |
| Device | CUDA only; no CPU fallback |
| Test partition | Not instantiated or evaluated |

Within each seed block, all four strategies use the same training seed. Across
blocks, use the listed seed for both selection and training. Store the seed,
configuration, run/child hashes, initial model fingerprint, checkpoint
provenance, split/inventory hashes, environment, exact selected tokens, scene
counts, and recovery/restart events in each run's metadata.

## Execution order and GPU safety

Run one experiment process at a time; do not parallelize on the 4 GB RTX 3050.
Complete preflight for all four strategies in a seed block before starting
that block's first training run. Use unique output directories and stop the
block on any hash, leakage, initialization, OOM, checkpoint, or validation
failure. Do not alter hyperparameters or retry into an existing directory.

To reduce systematic order/time effects, retain the already-completed seed-42
order as historical and rotate strategy order across the four new blocks:

| Seed | Suggested serial order |
|---:|---|
| 123 | Frequency, diversity, weighted-interest, random |
| 2027 | Diversity, weighted-interest, random, frequency |
| 31415 | Weighted-interest, random, frequency, diversity |
| 271828 | Random, frequency, diversity, weighted-interest |

Keep the machine on stable power and avoid other GPU workloads. Record GPU
utilization/available memory before runs, per-epoch duration and peak allocated
memory, warnings, failures, and any interruption. Investigate the prior extreme
epoch-time outliers before interpreting runtime as an outcome; do not change
the experimental controls to compensate.

## Runtime and storage estimates

Measured six-epoch training durations in seed 42 were:

| Strategy | Observed seconds |
|---|---:|
| Random uniform | 273 |
| Frequency-based / inverse rarity | 4,455 |
| Diversity-aware | 2,975 |
| Weighted-interest | 358 |
| **Four-run block** | **8,062 (2.24 GPU-hours)** |

At the observed block total, the 16 remaining runs are approximately **8.96
GPU-hours**. Given the large unexplained timing outliers, reserve at least
**18 GPU-hours of active GPU time** for those runs (roughly a 2x buffer), plus
time for preflight, curation, environment checks, review, and any approved
recovery. This is a planning estimate, not a runtime guarantee. The all-five-
block training total at the observed rate is about 11.2 GPU-hours.

Each current best/last checkpoint is about 228 MB; both together are about
456 MB per completed run. The 16 new conditions are therefore expected to
require about **6.8 GiB** for two checkpoints per run. All 20 conditions would
occupy about **8.5 GiB** for these checkpoint pairs, before histories, reports,
curation manifests, temporary files, or backups. Reserve at least **10 GiB**
of free storage; more is prudent if retaining duplicate backups.

## Evaluation and statistical analysis

### Outcomes and aggregation

- Primary outcome: best validation mAP@[0.50:0.95], selected by the existing
  validation-only best-checkpoint rule.
- Secondary outcomes: AP50, AP75, per-class AP, class ground-truth support,
  and precision/recall at the existing IoU protocol where defined.
- Report the complete per-seed result matrix. For each strategy and metric,
  report the arithmetic mean and sample standard deviation (`n=5`, `ddof=1`);
  also show the individual seed values and range.
- For each non-random strategy, calculate the within-seed paired difference
  from random uniform. Report the five differences, their mean and sample
  standard deviation, and an uncertainty interval. The paired seed block must
  remain intact in resampling or testing.
- Any confidence interval or paired test is exploratory with only five seed
  blocks. If inferential p-values are presented, predeclare the test and
  correct the three primary strategy-vs-random contrasts for multiplicity
  (e.g. Holm). Do not claim significance or broad generalization solely from
  this five-seed study; emphasize effect estimates and uncertainty.
- Report training duration and peak allocated CUDA memory separately from
  detection quality. They are operational outcomes and can be affected by
  environment noise.

### Per-class missingness

Ground-truth support is fixed by the same validation partition. The pilot
validation set has zero ground-truth instances for barrier,
construction_vehicle, and trailer; their AP is **undefined** for every seed,
not zero. Report support and `NA/undefined` for each such class, exclude these
classes from any mean over defined per-class AP, and state the denominator.
For classes with support, report AP per seed and aggregate across seeds only
when defined. A class with zero selected-training annotations but positive
validation support remains measurable; report its training support as zero,
not its AP as missing. Precision with no predicted positives is undefined;
do not replace undefined precision with zero. Recall is undefined when
ground-truth support is absent.

The official aggregate mAP remains the existing evaluator's output; do not
silently recompute it using a different class denominator. Report support next
to class metrics so missingness is visible.

## Limitations and interpretation

- Five seeds provide a useful replication check but limited power, with only
  five matched seed-block differences for each non-random-versus-random
  contrast. Treat the results as exploratory.
- The validation set has fixed class absences and is small; it cannot support
  claims for categories with no validation ground truth.
- The six-scene quota prevents strategies from gaining an advantage through
  scene concentration, but each scene contributes only ten selected images.
- Frequency, diversity, and weighted-interest selections are deterministic on
  fixed inputs; their repeated blocks mainly replicate model-training
  stochasticity, unlike random uniform which also varies its selected images.
- `best_effort` determinism is not bitwise reproducibility. The pilot observed
  warnings for a nondeterministic `roi_align_backward_kernel` operation.
- Very large observed epoch-duration outliers make runtime comparisons noisy;
  no cause is established in the current artifacts.
- Metadata-derived diversity is not visual diversity, and subset class or
  scenario composition is not evidence of downstream detection improvement.
- The test partition remains sealed until the multi-seed analysis and model/
  strategy selection are finalized under a separately documented protocol.

## Execution prerequisites

Before any run:

1. Extend the pilot runner's fixed seed-42 manifest checks to a parameterized,
   source-hash-verified multi-seed interface. It must still reject any
   held-out token, altered source, incorrect budget, or quota mismatch.
2. Add focused tests for non-42 curation manifests, training-only filtering,
   seed-specific provenance, exact 60-image selection, same-per-seed
   validation membership, identical within-block initialization, unique
   output paths, and no test dataset construction.
3. Freeze and publish the five seed values and the strategy execution order
   before launching training.
4. Generate and independently validate the four new seed-specific curation
   artifacts. Confirm the shared split digest and source hashes.
5. Check GPU availability, storage headroom, cached pretrained checkpoint
   SHA-256, and no active training process.

No model training, test evaluation, or experiment-artifact modification was
performed while preparing this plan.
