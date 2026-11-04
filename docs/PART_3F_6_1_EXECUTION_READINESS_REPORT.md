# Part 3F.6.1 — Multi-Seed Training Execution Readiness

## Verdict

**GO, with execution gates.** All 20 strategy/seed conditions have verified
curation manifests and successful runner preflight results. The fixed
validation set, split provenance, training-only membership, quotas, model
initialization, and core controls match the approved design. The seed-42
recovery artifacts are complete and eligible as the historical seed-42 block.

Run the 16 new experiments serially, with a fresh preflight immediately before
each seed block, and stop on any provenance, checkpoint, resource, or
validation failure. Reserve at least 18 hours of uninterrupted GPU time; a
24-hour operational window is safer given the unexplained pilot timing spikes.
Recheck free disk space before starting. This GO authorizes training only; it
does not authorize test-set evaluation or a claim of statistical significance.

## Scope and preflight evidence

The matrix consists of five paired seed blocks, four strategies per block,
60 selected training images per run, and six epochs per run. Selection and
training seeds are numerically equal within each block, but they remain
separate provenance fields for new runs.

All manifests select 60 unique tokens from the authoritative 242-image
training partition, with 10 images from each of the same six training scenes.
All conditions use the authoritative split digest
`d93c15fac3e798343ea95fcefea3c03a0b189a0b9036b3f355354550109bf2ab`,
inventory digest
`0898a28fd93c46ea44eac57b728739aeb90d455431c4a1a210e37cd2ee483a98`,
dataset version `v1.0-mini`, and the same 80 validation tokens
(`7c591a8498e2aa67ec6e021a6e1ca2f77a5e7952eee9abb53db7d981c9d4961c`).
Validation/test leakage, unknown or missing samples, missing image paths,
and annotation-count mismatches were absent.

Runner preflight passed for the four seed-42 conditions and all 16 newly
generated conditions. It instantiated only training and validation datasets,
loaded the exact selected training tokens and common validation tokens, and
verified fresh model initialization. All conditions use COCO_V1 checkpoint
SHA-256
`907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5`.
There is no training or test-dataset construction in preflight.

## Fixed controls and provenance

The runner validates the approved configuration before creating a run:

- Faster R-CNN MobileNetV3 Large 320 FPN; COCO_V1 initialization.
- CUDA only; batch size 1; zero DataLoader workers.
- Six epochs; AdamW with learning rate `0.0001` and weight decay `0.0001`.
- StepLR with step size 3 and gamma `0.1`; gradient accumulation 1 and
  clipping norm 1.0.
- AMP enabled; best-effort determinism; validation every epoch; no early stop.
- Same preprocessing, augmentation, fixed validation partition, and evaluation
  protocol in all conditions.

The config-control digest appropriately differs across seed blocks because
the seed is part of the serialized configuration. Within each block, it is
the same across strategies. Per-run provenance additionally varies as
expected for selected tokens, annotations, manifests, and output paths.

Seed 42 is a legacy pilot: its records persist a single `seed: 42` rather than
explicit `selection_seed` and `training_seed` keys. Both values are
unambiguously 42 from the audited curation manifest and experiment
configuration; the historical artifacts are not to be rewritten. New runs
must preserve both explicit fields.

While cross-checking this matrix against the source files, the seed-123 parent
manifest hash printed in the Part 3F.5.2 report was found to contain a
transcription error. The source `experiment_manifest.json` currently hashes
to `512eaab63b1509717b1af7ccb8da13694b151602b3f4f6f58f24899247299566`;
this corrected raw-file hash is used in the matrix. The manifest and its
children were not changed, and the runner preflight continues to validate the
source and parent-child provenance.

The runner allocates a unique timestamp/identifier output root for each seed
command and uses non-overwriting child directories. The 16 new output
directories are not allocated yet; the matrix below gives their intended
strategy subdirectories under the unique root recorded by each execution.
Do not reuse an interrupted run root. Preserve it and recover only from a
verified checkpoint or a newly allocated output directory.

## Complete execution matrix

`Run-manifest SHA-256` is the raw parent experiment-manifest hash. Child
hashes are canonical manifest hashes. Every listed manifest is validated and
preflight-passed. Rows using `<new-unique-run-root>` must be allocated at
execution; seed-42 output locations already exist.

| Selection / training seed | Strategy | Child manifest SHA-256 | Run-manifest SHA-256 | Output directory |
|---:|---|---|---|---|
| 42 | random | `85c331dcbd389cb4b48b70a9a19a3d5e57700bd611ce495b9d2ed4cf8712864c` | `b7fa8bc6276c626b7a03cc180e1988403dcc8b5b7b091a54495fb07bd92e217d` | `experiments/part3f4_curation_pilots/part3f4_20261104T135803Z_fb283e75/random_uniform` |
| 42 | frequency | `bcfe9fc1a3274e72920a0f7e13b41c80b9f9d35c360584b09fdaea2989bd2927` | `b7fa8bc6276c626b7a03cc180e1988403dcc8b5b7b091a54495fb07bd92e217d` | `experiments/part3f4_curation_pilots/part3f4_20261104T135803Z_fb283e75/frequency_based` |
| 42 | diversity | `580d64e61484b88560233c25d35d164622ff07c08ad5e2933809a0259b83b9c2` | `b7fa8bc6276c626b7a03cc180e1988403dcc8b5b7b091a54495fb07bd92e217d` | `experiments/part3f4_curation_pilots/part3f4_20261104T135803Z_fb283e75/diversity_aware` |
| 42 | weighted_interest | `fd8d61d4854544fbf70a6bba3c2c56617b79bcf480ac54bc4d137fa49a9e6959` | `b7fa8bc6276c626b7a03cc180e1988403dcc8b5b7b091a54495fb07bd92e217d` | `experiments/part3f4_curation_pilots/part3f4_recovery_20261104T190526Z_046cf543/weighted_interest` |
| 123 | random | `d8b922a6c92c2cf4f6bb41ba4120b235f020a89b150b99828f6f804b61be0dd8` | `512eaab63b1509717b1af7ccb8da13694b151602b3f4f6f58f24899247299566` | `<new-unique-run-root>/random_uniform` |
| 123 | frequency | `fc5d9d1e5f9371813526457642d5844d25dc614c114c27f3aaabb1213a7f6068` | `512eaab63b1509717b1af7ccb8da13694b151602b3f4f6f58f24899247299566` | `<new-unique-run-root>/frequency_based` |
| 123 | diversity | `6e71dcb25264d25d4cf996643f6d5e6cba5fb1ae2850b98547c5749a52b800d7` | `512eaab63b1509717b1af7ccb8da13694b151602b3f4f6f58f24899247299566` | `<new-unique-run-root>/diversity_aware` |
| 123 | weighted_interest | `d03566d9a5ceb864048e8eeb016e4771c7410c6f8c17d8134ed95ef681f3bbf9` | `512eaab63b1509717b1af7ccb8da13694b151602b3f4f6f58f24899247299566` | `<new-unique-run-root>/weighted_interest` |
| 2027 | random | `8e4e1c5b6c499f0f3e82f36646f3345f59e4690284dbc6720759398e9b674601` | `ae779b1abeb9c82b8fe9d950ef207d371202032f22021230fb08c30378cde32e` | `<new-unique-run-root>/random_uniform` |
| 2027 | frequency | `8397354724743aede21d55eafccf51af6b1df9c02b881565139f4fd8dd1ca271` | `ae779b1abeb9c82b8fe9d950ef207d371202032f22021230fb08c30378cde32e` | `<new-unique-run-root>/frequency_based` |
| 2027 | diversity | `16a7396f7c4682c144d7456a6f8511aad5fd184e05649ab7ac8a6ba8d8d04456` | `ae779b1abeb9c82b8fe9d950ef207d371202032f22021230fb08c30378cde32e` | `<new-unique-run-root>/diversity_aware` |
| 2027 | weighted_interest | `ff5afb914cc7f99e0bb7f93397997b79c9593dc77d2db2ec76f47383e26477b5` | `ae779b1abeb9c82b8fe9d950ef207d371202032f22021230fb08c30378cde32e` | `<new-unique-run-root>/weighted_interest` |
| 31415 | random | `7bef29401c3fc122b94eb5b81466c95ab64fd14adf35927d1d65dff4ff6eb410` | `5ecf6ec427199f3c413a5722a01c462d23283b6d9774d70040d23827fc485009` | `<new-unique-run-root>/random_uniform` |
| 31415 | frequency | `0ebfd887d649b15077969df67f50eb314ca573ad2698c567f53746e7ea844d60` | `5ecf6ec427199f3c413a5722a01c462d23283b6d9774d70040d23827fc485009` | `<new-unique-run-root>/frequency_based` |
| 31415 | diversity | `4dc5223c1491f7d08147e6e31626551f43c60c6d0aec71a7964d38e83a9c9ca6` | `5ecf6ec427199f3c413a5722a01c462d23283b6d9774d70040d23827fc485009` | `<new-unique-run-root>/diversity_aware` |
| 31415 | weighted_interest | `7e7f213fe473d50b522e5927106b10121d55ead2a961127ea76bcaab67bde578` | `5ecf6ec427199f3c413a5722a01c462d23283b6d9774d70040d23827fc485009` | `<new-unique-run-root>/weighted_interest` |
| 271828 | random | `8b458160b2241dd05018ff55b4d5a827b1e32adfdcb85093d350374e79ec60e7` | `51cd6ab14a7e7f17cb23555a7ea3456342faffa59a78523b472db94f6e9630b2` | `<new-unique-run-root>/random_uniform` |
| 271828 | frequency | `6597e08cdd196dc9458e701af091c8e59c3e83b9a536c0bbe46759b07b0c2931` | `51cd6ab14a7e7f17cb23555a7ea3456342faffa59a78523b472db94f6e9630b2` | `<new-unique-run-root>/frequency_based` |
| 271828 | diversity | `a9849c3829bb52709da455c761f3794b8d06e0b84695fe7073baeb8335d69e26` | `51cd6ab14a7e7f17cb23555a7ea3456342faffa59a78523b472db94f6e9630b2` | `<new-unique-run-root>/diversity_aware` |
| 271828 | weighted_interest | `24f5a1301d392fa07571c95b22114c5a1d3a7d2bf0de79ee58825efe6cf48095` | `51cd6ab14a7e7f17cb23555a7ea3456342faffa59a78523b472db94f6e9630b2` | `<new-unique-run-root>/weighted_interest` |

Seed bundle directories for the new blocks are:

- 123: `data/processed/curation_experiments/curation_20261104T200039Z_74947b58`
- 2027: `data/processed/curation_experiments/curation_20261104T200101Z_a5d55ce0`
- 31415: `data/processed/curation_experiments/curation_20261104T200122Z_680b340b`
- 271828: `data/processed/curation_experiments/curation_20261104T200147Z_a71ac412`

Frequency, diversity, and weighted-interest deterministically select the same
60 tokens across seeds for these fixed ranking inputs. This is expected, not a
preflight failure. Random-uniform selections vary by selection seed.

## Seed-42 checkpoint and recovery status

All four seed-42 conditions have six contiguous completed epochs and valid
latest/best checkpoints. The latest checkpoints were verified to contain
model, optimizer, scheduler, AMP scaler, Python/NumPy/PyTorch RNG,
DataLoader-generator, and provenance state. Random and frequency were
preserved; diversity resumed from its verified epoch-4 checkpoint; the
interrupted weighted-interest attempt was preserved and the completed run
was created in a separate recovery directory. The recovery did not change
the seed, selection, initialization, or controls.

The pilot reports `test_evaluated: false`. All seed-42 runs have the same COCO
checkpoint hash and initial model-state fingerprint. They are eligible as the
seed-42 block, with the restart/resume events and legacy seed-field format
disclosed.

## GPU runtime and storage

Measured six-epoch seed-42 runtimes from the saved training histories:

| Strategy | Epoch times (seconds) | Total (seconds) |
|---|---|---:|
| random | 56.4, 19.9, 18.8, 142.7, 18.0, 17.6 | 273.3 |
| frequency | 18.1, 18.9, 18.8, 17.9, 4281.8, 99.3 | 4455.0 |
| diversity | 244.0, 1214.9, 749.1, 597.9, 106.8, 62.3 | 2974.9 |
| weighted_interest | 80.4, 57.9, 51.9, 55.7, 56.2, 56.2 | 358.4 |

The four-run seed-42 total is about 8,062 seconds (2.24 hours). Four new
blocks at the same measured total are about 8.96 hours. A 2x allowance is
about 17.9 hours; reserve **at least 18 hours**, preferably a 24-hour window,
because the cause of the unusually slow frequency and diversity epochs is
unknown. Do not treat that outlier-sensitive pilot average as a service-level
runtime guarantee.

Peak allocated CUDA memory in the pilot was 439,480,320 bytes (random),
669,389,312 (frequency), 901,756,416 (diversity), and 437,421,568
(weighted-interest). The maximum observed allocation was about 0.84 GiB.
This is below the 4 GB GPU capacity, but it does not include all other GPU
process usage or guarantee future runs will not OOM. Run one experiment at a
time, monitor device memory and temperature, and stop on OOM rather than
falling back to CPU or changing controls.

At inspection, D: had 26,268,205,056 bytes free (about 24.46 GiB). The
estimated checkpoint storage for the 16 new runs is about 6.79 GiB, leaving
about 17.67 GiB before histories and reports. Retain at least 10 GiB of free
space as an operational margin, and recheck before execution. The estimate
does not justify deleting any historical artifacts.

## Serial execution order and operational checklist

The runner now accepts `--strategy-order`; its default remains the existing
order. The order validator requires every approved strategy exactly once,
and output allocation remains non-overwriting. Use this rotation to avoid
always placing a strategy first or last:

| Block | Seed | Serial strategy order |
|---:|---:|---|
| 1 | 123 | frequency → diversity → weighted_interest → random |
| 2 | 2027 | diversity → weighted_interest → random → frequency |
| 3 | 31415 | weighted_interest → random → frequency → diversity |
| 4 | 271828 | random → frequency → diversity → weighted_interest |

Before each block:

1. Recheck disk and GPU availability; ensure no other training process is
   using the GPU.
2. Run the runner's validation-only preflight for that exact curation bundle
   with the selection and training seed explicitly set to the block seed.
3. Confirm the preflight hashes, 60 selected tokens, six 10-image quotas,
   validation-token digest, COCO checkpoint SHA, and `test_dataset_instantiated:
   false`.
4. Execute the block serially using the order above. Each run must be created
   under a new unique root; never point an execution at a prior run directory.
5. After every run, check contiguous epoch history, validation outputs,
   `checkpoint_last.pth`, best checkpoint, and explicit seed/provenance
   metadata before continuing.

If interrupted, preserve the directory. Resume only if the latest checkpoint
has complete model/optimizer/scheduler/scaler/RNG/loader-generator state and
matching manifest, config, split, and pretrained-checkpoint provenance.
Otherwise stop, document the failure, and request approval before restarting
in a new unique directory. Do not evaluate the test split.

## Validation performed for this readiness assessment

- Re-ran focused curation-pilot tests: **35 passed**.
- Full test suite: **193 passed**.
- `git diff --check`: passed.
- No packages were installed.
- No model training or test-set evaluation was run.
- No existing curation, pilot, checkpoint, split, or dataset artifacts were
  modified.

The only implementation adjustment for this readiness phase is the
non-overwriting runner's explicit strategy-order option and its validation
tests, needed to follow the approved rotated serial schedule.
