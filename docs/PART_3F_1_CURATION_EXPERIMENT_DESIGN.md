# Part 3F.1 — Data Curation Experiment Design

## Executive summary

This design tests whether metadata-guided selection can improve detection on
predefined underrepresented scenario groups at a smaller training-image
budget. It is a proposed experiment, not evidence that any selection method
improves model performance.

The current infrastructure contains four deterministic selection methods
(random, frequency, diversity, and weighted-interest) and the detector trainer
can validate an explicit training-token selection. The baseline uses
Faster R-CNN MobileNetV3 Large 320 FPN, six epochs, 242 training images and
80 validation images. The existing six-epoch experiment's best validation
mAP@[.50:.95] is 0.038926 at epoch 4; final-epoch mAP is 0.037744 at epoch 6.
Keep that run unchanged as the historical full-data reference.

**Blocking limitation before any curation experiment:** the current scenario
intelligence CLI runs rarity analysis and curation on all records in
`scenarios.json`; it has no split-manifest restriction. A train-only selection
must be assembled and verified **before** fitting frequency counts, numeric
bin edges, normalizations, rankings, or selecting records. Otherwise
validation/test metadata can influence the curation policy even if selected
IDs are later passed through the trainer's training-selection validator.

Second, scenario records are derived from nuScenes 3D annotations and sensor
metadata, while the detector consumes accepted 2D camera annotations.
The experiment must validate the exact sample-token join and record the
accepted 2D target counts for each selected image. A training-only audit of
the current seed-42 manifest found 242 images across six scenes (39–41 images
per scene), 3,636 accepted 2D boxes, and no training image with zero accepted
2D boxes. This does not yet verify that a particular curation export joins
one-to-one to those image records.

**No implementation, training, or test evaluation was performed for this
design task.** Only the existing training-partition manifest and annotations
were counted for the data-path audit.

## 1. Research question and hypotheses

### Research question

Can diversity-aware curation of multimodal autonomous-driving data improve
perception performance on underrepresented driving scenarios while reducing
the number of training images?

### Hypotheses to pre-register

- **H1 — Target-group performance:** at a fixed subset size, diversity-aware
  and weighted-interest selections will have higher validation performance
  on predeclared underrepresented scenario groups than the random-selection
  baseline.
- **H2 — Data efficiency:** at least one curated subset size will approach
  the full-data reference on the primary validation metric while using fewer
  training images.
- **H3 — Coverage mechanism:** diversity-aware selection will increase
  metadata-defined scenario/category coverage relative to random selection
  at the same image count.

These are hypotheses only. They are not supported or rejected until the
experiments are run. Set a practically meaningful non-inferiority margin for
H2 before observing results; do not choose it after seeing the curves.

## 2. Existing curation infrastructure audit

### 2.1 Feature source and meaning

`SqlScenarioDataSource` creates one `SampleInput` per database keyframe.
It reads sample/scene metadata, 3D annotations, sensor references, and
optionally resolved nuScenes lookups for ego pose, attributes, visibility,
and location. `extract_scenarios` converts these into sample-level
`ScenarioRecord`s.

Current features include:

- nuScenes object/category/group counts and category combinations;
- object count, category-mix entropy inputs, unique-instance counts;
- visibility and attribute counts when lookup data is available;
- ego position, heading, coarse keyframe-to-keyframe speed/displacement;
- sensor reference counts, file availability, and completeness metadata;
- scene sample count, duration and log location.

These are primarily **metadata and annotation-derived proxies**. The
curation feature path does not consume camera pixels, LiDAR point clouds,
radar signals, learned visual embeddings, or detector errors. “Multimodal”
therefore describes some available metadata/sensor-reference fields; it does
not mean the current ranking measures multimodal visual diversity. In
addition, `expected_channels` is currently computed across the entire
database dataset and used in `sensor_completeness`; a train-only curation
path should recompute that denominator from training records to avoid even
metadata-level held-out influence.

### 2.2 Strategy definitions and current limitations

| Strategy | Current implementation | Limitations relevant to this study |
|---|---|---|
| Random | Sort records by sample ID, then shuffle with `random.Random(seed)`. | Reproducible, but current implementation samples from its entire input, not a split-aware pool. |
| Frequency | Sort by the combined inverse-frequency rarity score. Category frequency counts scenarios containing a category, not object instances; combination and binned-feature frequencies also contribute. | A heuristic sensitive to the input population, quantile bins and rare/noisy values. No split restriction in current CLI. |
| Diversity | Greedily add a scenario that maximizes Shannon entropy of the cumulative object-category counts; ties resolve by sample ID. | Optimizes category-count entropy only, not a joint multimodal feature space or visual diversity. `CurationConfig.seed` does not alter this deterministic tie rule. |
| Weighted-interest | Rank by weighted rarity, within-scenario composition entropy, and normalized complexity; preferences are optional. Defaults give equal weight to rarity, composition diversity and complexity, and zero weight to preference. | Scores are transparent but heuristic and uncalibrated. Min-max scaling depends on the ranking population; missing components are excluded and remaining weights renormalized. |

Rarity uses inverse frequency `1 / (frequency + epsilon)`, followed by
min-max normalization. Its components are category presence, category/group
combination frequency, and mean rarity of binned features. Numeric bins are
fitted on the records passed to the analyzer. The current CLI passes all
loaded scenario records to rarity analysis before curation. Do not retain
those whole-dataset frequencies or bin edges for the experiment.

### 2.3 Reproducibility and ranking

The existing implementation has useful deterministic properties:

- input records are sorted by sample ID;
- duplicate sample IDs are rejected;
- random selection uses a local seeded RNG;
- frequency and weighted-interest ties are ordered by sample ID;
- diversity ties are resolved by sample ID;
- manifest JSON is canonicalized for a SHA-256 digest and contains no
  wall-clock timestamp;
- synthetic tests verify repeatability and that changing the random seed
  changes random selection.

This proves reproducibility for identical records/configuration, not
robustness to a new selection seed or data version. For non-random strategies,
the current seed usually has no effect because tie-breaking is deterministic.
Persist the complete score inputs, train-only input digest, frequency/binning
configuration, selection seed, scene quotas, selected tokens, and manifest
hash for every experiment.

### 2.4 Mapping to images and empty targets

Scenario IDs originate from source sample IDs; the vision dataset keys images
by `sample_token`. They should identify the same nuScenes keyframe, but the
curation CLI does not join a manifest against `image_manifest.csv`, verify
image paths, or validate 2D annotation counts. The training token file loader
reads a JSON string array; downstream `select_training_subset` validates
uniqueness and that IDs belong to train, but it does not prove that the
curation manifest's provenance and feature records came only from train.

The training-only inventory audit for the current seed-42 split found:

- 242 distinct train sample tokens and six train scenes;
- 39–41 camera images per train scene;
- every train sample token had an image-manifest row;
- 3,636 accepted 2D boxes total and no train image with zero accepted boxes.

This is a current-data observation, not a permanent guarantee. Future
annotation versions and any selected-token file must be revalidated. If an
empty 2D target is encountered later, retain it as a valid background image
only after confirming it is supported by the trainer/evaluator and report its
count. Do not infer a 2D empty target from a 3D category count: projection,
conversion acceptance, and camera visibility can differ.

### 2.5 Equal selection sizes and scene isolation

With no `require_groups`, feature filters, or near-duplicate exclusion and
with enough eligible records, current selection methods can return the
requested number. Constraints can reduce eligibility and produce a
shortfall; the implementation reports rather than hides it. Enforce exact
size in the experiment launcher: abort a run if selected count differs from
the registered budget.

The existing split manifest assigns complete scenes to one split and
validates no scene/sample overlap. However, the curation CLI has no split
argument, so it does not enforce that property itself. It must filter records
to the six training scenes/sample tokens **before** any fit/score step, and
then validate all selected IDs and scene IDs against the authoritative
training manifest. Validation and test records must not influence feature
scaling, binning, rarity, preferences, selection, or quotas.

### 2.6 Test coverage gaps

Current scenario-intelligence tests use synthetic records. They exercise
score helpers, deterministic selection, random-seed changes, constraints,
and manifest output, but do not test:

- curation against the real split manifest or train-only input;
- exclusion of validation/test metadata from score fitting;
- sample-token/image-manifest/2D-annotation joins;
- scene quota feasibility or exact equal sizes across all strategies;
- cross-strategy overlap and selected sample distributions;
- reproducibility across separate processes or input ordering for every
  end-to-end CLI configuration.

Training selection/split tests do not close these curation-to-vision joins.
Add those tests before running the matrix; do not change the detector.

## 3. Experiment matrix and subset sizes

### Proposed budgets

| Budget | Images | Share of 242 train images | Scene-stratified allocation across six train scenes |
|---|---:|---:|---:|
| Small | 60 | 24.8% | 10 per scene |
| Medium | 120 | 49.6% | 20 per scene |
| Large | 180 | 74.4% | 30 per scene |
| Full reference | 242 | 100% | Use every train image (39–41 per scene) |

The three reduced budgets span a meaningful range while remaining feasible
for repeated six-epoch runs. The exact per-scene quotas are recommended to
control the mix of the six training scenes across methods and budgets. They
are a proposed common constraint, not currently supported by `curate`.
Implement quotas uniformly for every strategy (including random) and abort
if any strategy cannot meet them. For the diversity method, keep its global
entropy objective while only considering candidates from scenes with quota
capacity remaining. Do not add a class-presence constraint that could make
the random baseline infeasible or silently alter the requested N.

### Factor matrix

| Data selection strategy | 60 | 120 | 180 | 242 |
|---|:---:|:---:|:---:|:---:|
| Random uniform, train-only | Yes | Yes | Yes | — |
| Frequency / inverse rarity | Yes | Yes | Yes | — |
| Diversity / greedy category-count entropy | Yes | Yes | Yes | — |
| Weighted-interest / fixed default score | Yes | Yes | Yes | — |
| Full training-data reference (historical/new repeats) | — | — | — | Yes |

The primary factorial has 13 unique strategy-budget conditions: four
selection strategies at each of three reduced budgets (12), plus the full
242-image reference (1). The full-data reference is common to all strategies,
not four separately selected full-data subsets. Keep the existing six-epoch
baseline as an immutable historical reference; use fresh repeated full-data
runs for the matched-seed estimate.

## 4. Data-selection protocol

1. Load and validate the existing seed-42 manifest against the authoritative
   dataset version and image inventory. Do not rebuild or edit it.
2. Derive the set of 242 training sample tokens and six training scene
   tokens. Restrict scenario records to those train IDs **before** fitting
   rarity frequencies, quantile edges, min-max normalizations, similarity
   filters, weighted scores, or diversity selection.
3. Recompute any population-level metadata, including expected sensor
   channels, using training records only. Use train-only quantile edges and
   category frequencies and persist them in the curation manifest.
4. Require an exact one-to-one mapping from every candidate scenario
   `sample_id` to a training `sample_token` and image-manifest row. Verify
   scene ID agreement, image status/path, accepted 2D annotation counts,
   class mapping, and inventory/split digests before selection.
5. Apply the same exact scene quota for every strategy and N: 10, 20, or 30
   samples per training scene. Random selection must be seeded and uniform
   within each scene; ranked strategies use the same candidates and quota
   eligibility rule. Keep all other filters disabled for the primary study.
6. Require the selected set to have exactly N unique image tokens, all
   present in the train manifest, no validation/test tokens, and no overlap
   between scene assignments. Store full ordered IDs, sample/scene tokens,
   annotation counts, ranking scores/components, seeds, eligibility,
   shortfall status, and a digest.
7. Write a JSON string array of selected sample tokens for the existing
   experiment config's `training_sample_tokens_file`. Preserve selection
   file and curation manifest as experiment artifacts; do not regenerate or
   overwrite them during training.
8. Use the full 80-image validation partition only for epoch-wise model
   selection and reporting. Never instantiate/evaluate the test partition
   during this phase.

### Strategy definitions to freeze

- **Random:** uniform without replacement, separately within each scene, with
  independent selection seed.
- **Frequency:** current combined rarity score using train-only frequencies
  and train-only quantile bins, with exact component weights/configuration
  recorded.
- **Diversity:** current greedy entropy of cumulative nuScenes object-category
  counts, with per-scene quota capacity enforced. This tests that specific
  algorithm, not a claim of general multimodal diversity.
- **Weighted-interest:** current default ranking definition:
  equal positive weights for rarity, within-scenario composition diversity,
  and complexity; preference weight zero. Freeze this configuration for the
  primary comparison. Any custom preferences require a separately
  preregistered experiment, not tuning against validation/test outcomes.

## 5. Training controls

Use the approved Faster R-CNN MobileNetV3 Large 320 FPN architecture for all
conditions. Each run must be initialized fresh from the same
Torchvision COCO_V1 checkpoint; never resume a prior trained checkpoint or
initialize from another curation run.

Hold constant across conditions:

- nuScenes `v1.0-mini`, source annotation version, category mapping,
  image inventory and existing seed-42 scene split;
- six epochs, validation after every epoch, full 242/80 split provenance;
- batch size 1, zero workers, CUDA, mixed precision enabled, best-effort
  reproducibility mode;
- no external resize override (`image_size: null`), so the detector's
  existing 320/640 transform remains in use;
- AdamW, learning rate `1e-4`, weight decay `1e-4`, StepLR step size 3 /
  gamma 0.1, gradient clipping norm 1.0, gradient accumulation 1;
- same training and evaluation transforms; no strategy-specific
  augmentation;
- identical loss aggregation, checkpoint-selection rule (best validation
  mAP@[.50:.95]), metric implementation and validation schedule.

Only the selected training token set and predeclared random seeds vary.
Record per-epoch total/component losses, LR, validation metrics, epoch/wall
time, peak allocated CUDA memory, checkpoint hash, config hash, curation
manifest hash, split/inventory digests, and training-selection digest. Record
both best-checkpoint and final-epoch metrics; do not compare only best
scores.

The historical six-epoch baseline records 242 train / 80 validation images,
seed 42, the controls above, best mAP 0.038926 (epoch 4), final mAP 0.037744
(epoch 6), and 809.976 seconds of measured epoch time. It is a useful
reference, not a multi-seed estimate. Its artifacts must remain untouched.

## 6. Evaluation methodology

### Primary and secondary metrics

For each run, report:

- primary: validation mAP@[0.50:0.95];
- secondary: AP50, AP75, per-class AP and validation ground-truth object
  support;
- precision and recall at a declared score threshold of 0.50 and IoU
  threshold 0.50 (same semantics as the existing evaluator);
- predeclared underrepresented-group metrics;
- train image count, epochs, total image presentations (`N × 6`), training
  time and peak allocated CUDA memory.

Use the existing TorchMetrics `MeanAveragePrecision` with Faster-COCO-Eval,
XYXY boxes, and IoU thresholds 0.50:0.05:0.95 for overall metrics. Retain
the existing model-label-to-metric-label conversion. The evaluator's
precision/recall values are fixed-threshold statistics, not COCO AP.

### Underrepresented scenario groups

Before any detector results are examined, define the candidate scenario
groups from **training data only**. Use only group definitions that can be
joined to the validation keyframes consistently, for example:

- rare 2D/3D object-category groups (pedestrian, two-wheeler,
  traffic-related, and vehicle subgroups);
- low-visibility and object-density strata where metadata coverage permits;
- sensor-completeness or ego-motion strata only when feature coverage is
  sufficient and their definitions are frozen.

Derive category rarity thresholds and numeric bin edges using only the 242
training samples. Apply those fixed thresholds to validation group
assignment; do not fit bins or choose groups from validation performance.
Validation metadata/annotations may be used for group evaluation only, never
for selection or training.

For each fixed group, evaluate COCO-style metrics on the validation images
assigned to that group and report image and object support. Report both
group-level overall AP and relevant class AP where defined. Group-conditioned
AP is secondary/exploratory on this small validation set and must not replace
the primary full-validation mAP selection metric.

### Undefined metrics

- A class with zero validation ground-truth objects has undefined AP and
  recall: report `N/A`, not zero. The existing baseline diagnosis already
  documents three classes with no validation support.
- If a class/group has no ground truth in the selected evaluation subset,
  its AP/recall is undefined, even if predictions exist.
- Precision is undefined when there are no predictions at the selected score
  threshold (zero denominator). If there are predictions but no true
  positives, precision can be defined as zero. Preserve the evaluator's
  `null`/`N/A` semantics rather than replacing missing values by zero.
- Report the number of supported classes used in any mAP aggregate and do
  not silently treat unsupported classes as zero AP.

## 7. Reproducibility and statistical analysis

### Repeated runs

Use **five paired end-to-end replicate seeds** as the recommended minimum:
for each replicate, use the same model seed for every one of the 13
conditions, and an independent curation-selection seed for the random
strategy. Record both. The existing full-data seed-42 baseline can serve as
the first full-data replicate only after verifying its config, initialization,
split and selection digests match this protocol; never edit it. Run four
additional fresh full-data references for the other paired seeds.

The current non-random selectors use stable sample-ID tie-breaking, so their
selected subsets may be identical across curation seeds. That is acceptable
for estimating training-seed variability but does not estimate subset
sampling uncertainty for these strategies. To quantify that separately,
add a preregistered nested design with several independent selection draws
and multiple model seeds per draw if compute permits. Do not make a single
random subset appear to represent the distribution of random selection.

Randomness in initialization, data-loader ordering and stochastic CUDA
operations means same seed does not guarantee bitwise-identical GPU training.
Use the repository's best-effort mode and state this limitation.

### Comparisons and reporting

- Match replicate seeds across strategies and compare per-seed paired
  differences at each N. Report each seed, mean and standard deviation, and
  median/range; do not report only pooled means.
- Report best-validation-checkpoint and final-epoch outcomes separately.
  Best checkpoint selection is validation-based and may be optimistically
  biased; use one identical predeclared rule for all runs.
- Plot metric-versus-image-count learning curves, include the full-data
  reference, and present class/group support alongside AP.
- For the data-efficiency claim, identify the smallest N whose primary
  metric is within the **predeclared** practical margin of the full-data
  reference; if uncertainty does not support that conclusion, report
  indeterminate.
- Five seeds on one fixed, small validation split provide limited
  uncertainty information, not population-level evidence. If confidence
  intervals are reported, explain their resampling unit and limitations;
  scene clustering matters, and the number of validation scenes is small.
  Avoid unpaired object-level bootstrap as if boxes were independent.
- Do not claim statistical significance from a single run, overlapping
  confidence intervals, or a handful of scenes. Treat the initial results
  as exploratory and seek external data/repeated split validation for
  stronger claims.

### Compute estimate

The full six-epoch run measured about 810 seconds (13.5 minutes), including
validation. Thirteen conditions at five seeds would be 65 total runs, or
about 14.6 hours if every run cost as much as that full-data baseline. If the
existing seed-42 full-data experiment is verified and counted as one
replicate, that leaves 64 new runs (about 14.4 full-run-equivalent hours).
Reduced subsets should reduce training batches but still require the same
80-image validation each epoch, so runtime will not necessarily scale
linearly with N. This is a planning estimate, not a benchmark. Record actual
measured durations; do not change the schedule to fit a time estimate.

## 8. Provenance and artifact requirements

Every curation selection and training run should preserve:

- dataset version, annotation/conversion and category-mapping digests;
- canonical split-manifest and inventory digests;
- train-only scenario input digest and curation configuration;
- exact strategy, scoring weights, bin edges/frequencies, selection seed,
  model seed and per-scene quota;
- full ordered selected sample tokens, scene tokens and accepted 2D
  annotation counts;
- requested and actual selected sizes, shortfall/error state and selection
  digest;
- model architecture, exact COCO initialization identifier, training config
  and environment metadata;
- epoch histories, validation metrics, best and last checkpoints, checkpoint
  hashes, training-selection metadata and test-evaluated=false.

Store new artifacts only in new uniquely identified experiment directories.
Do not overwrite or resume the six-epoch baseline. No test metrics artifact
should be created during this phase.

## 9. Risks and research limitations

1. **Small dataset:** nuScenes mini has only 404 camera keyframes overall,
   with 242 training images and 80 validation images. Rare categories and
   groups have low support; per-class estimates can be unstable.
2. **Scene-level correlation:** adjacent keyframes are correlated.
   The official split is scene-disjoint, but sample-level subsets can still
   overrepresent nearby frames. The proposed common per-scene quotas control
   scene counts but do not remove within-scene temporal correlation.
3. **Class imbalance:** training box frequencies are uneven; a curation
   method may alter class support substantially. Report selected image-level
   and object-level distributions, not just N.
4. **Metadata versus visual diversity:** current entropy/rarity scores use
   annotations and metadata, not visual embeddings or raw multimodal signal.
   Higher metadata entropy does not prove more visual coverage.
5. **3D/2D target mismatch:** scenario object counts and detector targets
   have different provenance. Validate selected tokens against accepted 2D
   annotations, and show both metadata and 2D class distributions.
6. **Geographic/environmental coverage:** nuScenes mini is geographically
   and environmentally limited. Scene/location metadata may be confounded
   with the few source scenes and cannot establish broad driving coverage.
7. **Overfitting and validation reuse:** repeated epoch-wise validation and
   many strategy/budget comparisons can overfit decisions to the same 80
   validation images. Keep the test partition untouched, freeze the design,
   and treat validation comparisons as exploratory.
8. **Optimizer fairness:** identical optimizer and six-epoch budget match
   the existing baseline but may not be optimal for every subset or curation
   strategy. Do not retune per strategy in the primary controlled study.
9. **Ranking sensitivity:** min-max normalization, inverse-frequency
   epsilon, quantile bins, missing metadata, and sample-ID tie-breaking can
   materially affect selections. Freeze and publish all these choices.
10. **Limited statistical power:** five random seeds quantify only part of
    training variability; the validation scenes remain fixed and few.

## 10. Recommended implementation sequence

1. Add a split-aware curation entry point that loads the authoritative
   manifest and filters to training sample/scene tokens before computing
   rarity, diversity, bins, normalization, or ranks. Recompute global
   metadata denominators from train only.
2. Add and test the scenario-ID to image-manifest/annotation join; reject
   unknown IDs, split mismatches, scene mismatches, duplicates, invalid paths
   and wrong annotation versions. Verify empty-target semantics explicitly.
3. Implement identical exact per-scene quotas for all four selection
   strategies and fail on any shortfall. Add tests proving equal sizes,
   train-only records/statistics, no scene leakage, and deterministic
   manifests.
4. Freeze score weights, binning, underrepresented-group definitions,
   selection quotas, subset sizes, non-inferiority margin and primary/secondary
   metrics in versioned configs before any model result is observed.
5. Generate and review curation manifests for 60/120/180, checking overlap,
   class/scene distributions, train-only provenance, and annotation joins.
   Do not inspect or score validation/test samples in this step.
6. Run an approved one-batch infrastructure smoke test if needed; then run
   fresh, isolated six-epoch Faster R-CNN experiments for the preregistered
   matrix. Do not change the existing baseline.
7. Compare per-seed paired results, full trajectories, per-class/group AP,
   support, time and memory. Report limitations and no unsupported
   performance or significance claims.
8. Only after design and implementation validation should any separate,
   authorized final test evaluation be considered. It is out of scope here.

## 11. Audit conclusion

The existing algorithms are deterministic and useful as research prototypes,
but their current CLI is **not split-safe** and its metadata-ranking scores
are not visual-diversity measures. The trainer's train-only selection checks
are a valuable downstream safeguard, but they do not prevent held-out
statistics from influencing the selection policy. Add train-first filtering,
verified image/annotation joins, and common scene quotas before initiating
the curation experiment matrix.
