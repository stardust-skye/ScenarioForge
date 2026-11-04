# Part 3F.5.2 — Multi-Seed Manifest Generation and Preflight

## Decision

**GO for the multi-seed training phase**, subject to the existing execution
controls in the Part 3F.5 plan. All 16 new 60-image child manifests passed
independent artifact checks and the runner's validation-only preflight. No
training or test-set evaluation was performed.

The generator was invoked once for each remaining seed using the existing
`curate-experiment` pipeline, the authoritative seed-42 split, and a 60-image
reduced budget. Each unique bundle also contains the generator's mandatory
full-training reference manifest; only the four 60-image strategy manifests
per bundle were preflighted for this matrix. No 120- or 180-image subsets
were generated.

## Generated immutable bundles

| Selection / training seed | Directory | Parent manifest SHA-256 |
|---:|---|---|
| 123 | `data/processed/curation_experiments/curation_20261104T200039Z_74947b58` | `512eaab63b1509717b1af7ccb8da13694b151602b3b4f6f58f24899247299566` |
| 2027 | `data/processed/curation_experiments/curation_20261104T200101Z_a5d55ce0` | `ae779b1abeb9c82b8fe9d950ef207d371202032f22021230fb08c30378cde32e` |
| 31415 | `data/processed/curation_experiments/curation_20261104T200122Z_680b340b` | `5ecf6ec427199f3c413a5722a01c462d23283b6d9774d70040d23827fc485009` |
| 271828 | `data/processed/curation_experiments/curation_20261104T200147Z_a71ac412` | `51cd6ab14a7e7f17cb23555a7ea3456342faffa59a78523b472db94f6e9630b2` |

Each parent and child records `selection_seed` and `training_seed` separately;
both values match the seed shown. The generator's existing `seed` field is
preserved for compatibility.

## Child-manifest hashes and budgets

The table gives the canonical child-manifest SHA-256, independently recomputed
from the manifest JSON excluding its `manifest_sha256` field. All 16 computed
hashes matched their stored values.

| Seed | Strategy | Canonical child-manifest SHA-256 | Selected / unique |
|---:|---|---|---:|
| 123 | random | `d8b922a6c92c2cf4f6bb41ba4120b235f020a89b150b99828f6f804b61be0dd8` | 60 / 60 |
| 123 | frequency | `fc5d9d1e5f9371813526457642d5844d25dc614c114c27f3aaabb1213a7f6068` | 60 / 60 |
| 123 | diversity | `6e71dcb25264d25d4cf996643f6d5e6cba5fb1ae2850b98547c5749a52b800d7` | 60 / 60 |
| 123 | weighted_interest | `d03566d9a5ceb864048e8eeb016e4771c7410c6f8c17d8134ed95ef681f3bbf9` | 60 / 60 |
| 2027 | random | `8e4e1c5b6c499f0f3e82f36646f3345f59e4690284dbc6720759398e9b674601` | 60 / 60 |
| 2027 | frequency | `8397354724743aede21d55eafccf51af6b1df9c02b881565139f4fd8dd1ca271` | 60 / 60 |
| 2027 | diversity | `16a7396f7c4682c144d7456a6f8511aad5fd184e05649ab7ac8a6ba8d8d04456` | 60 / 60 |
| 2027 | weighted_interest | `ff5afb914cc7f99e0bb7f93397997b79c9593dc77d2db2ec76f47383e26477b5` | 60 / 60 |
| 31415 | random | `7bef29401c3fc122b94eb5b81466c95ab64fd14adf35927d1d65dff4ff6eb410` | 60 / 60 |
| 31415 | frequency | `0ebfd887d649b15077969df67f50eb314ca573ad2698c567f53746e7ea844d60` | 60 / 60 |
| 31415 | diversity | `4dc5223c1491f7d08147e6e31626551f43c60c6d0aec71a7964d38e83a9c9ca6` | 60 / 60 |
| 31415 | weighted_interest | `7e7f213fe473d50b522e5927106b10121d55ead2a961127ea76bcaab67bde578` | 60 / 60 |
| 271828 | random | `8b458160b2241dd05018ff55b4d5a827b1e32adfdcb85093d350374e79ec60e7` | 60 / 60 |
| 271828 | frequency | `6597e08cdd196dc9458e701af091c8e59c3e83b9a536c0bbe46759b07b0c2931` | 60 / 60 |
| 271828 | diversity | `a9849c3829bb52709da455c761f3794b8d06e0b84695fe7073baeb8335d69e26` | 60 / 60 |
| 271828 | weighted_interest | `24f5a1301d392fa07571c95b22114c5a1d3a7d2bf0de79ee58825efe6cf48095` | 60 / 60 |

## Leakage, provenance, and scene-quota validation

For all 16 manifests:

- The selected set has exactly 60 distinct sample tokens, all present in the
  authoritative 242-sample training split. Validation overlap: zero; test
  overlap: zero; unknown or missing selected samples: zero.
- Selected image paths resolve under the dataset root and exist. The generator
  validated the training-image inventory and annotation mappings; runner
  preflight loaded each selected train sample and confirmed its box count
  against the manifest. Empty targets are retained.
- Parent and child source hashes, dataset version, split digest, scenario
  digest, scoring configuration, seed fields, strategy, and quota policy
  matched. Child canonical hashes matched independently recomputed hashes;
  parent raw-file hashes are recorded above.
- All four conditions in each block use the same split digest:
  `d93c15fac3e798343ea95fcefea3c03a0b189a0b9036b3f355354550109bf2ab`.
  The split inventory digest is
  `0898a28fd93c46ea44eac57b728739aeb90d455431c4a1a210e37cd2ee483a98`;
  dataset version is `v1.0-mini`.
- The same six training scenes appear in each subset, with exactly 10 samples
  from each:
  `6f83169d067343658251f72e1dd17dbc`,
  `bebf5f5b2a674631ab5c88fd1aa9e87a`,
  `cc8c0bf57f984915a77078b10eb33198`,
  `d25718445d89453381c659b9c8734939`,
  `de7d80a1f5fb4c3e82ce8a4f213b450a`,
  `e233467e827140efa4b42d2b4c435855`.

The runner's preflight-only command constructed only train and validation
datasets. For every strategy, it loaded 60 selected training samples and the
same 80 validation samples. The validation token digest was identical across
all 16 conditions:
`7c591a8498e2aa67ec6e021a6e1ca2f77a5e7952eee9abb53db7d981c9d4961c`.
No test dataset was instantiated.

The preflight also constructed fresh COCO_V1-initialized models on CPU (without
training) and verified the same initial-state fingerprint across the four
strategies within each block. The pretrained checkpoint SHA-256 was
`907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5` for all
blocks.

| Seed | Control digest (same for all four strategies in block) | Initial model-state SHA-256 (same for all four strategies) |
|---:|---|---|
| 123 | `18cfb9f049c88b770205d56fa375a4de9696ad2cbfb17cf56ca34a5b046dfd63` | `43f4cd94e782c7a68042f45725d604e5241fe7a7de5d91509fb9502ffa98ddaa` |
| 2027 | `291af82b462134542c26ecb30e8eeb738c2d3ebc2fb44ed95921b082b4ce638b` | `c14bed83ddb3180aeb776a67525e691a450e716b23163611fd6bcb536c5f4612` |
| 31415 | `dbebdde2ba4f77abf8e77b9a1ff5cd11dc3b390f459b4e37712847e56e969ce0` | `e39217c6f4114b531b98be09064ac46f6771fc19eadd22fece50e9e5d6a43157` |
| 271828 | `ee027ffb8e6d8af4c7206a45d1b9224844526044c77f2891ca4c8b58490713be` | `b00e0c386f72f93be030f0f2ca863f59c36ed74daff0934818fbbd5cac54ce42` |

The control digest includes seed-specific config values, so it is expected to
differ between seed blocks. Architecture, optimizer, learning-rate schedule,
batch size, transforms, augmentation, AMP, epoch count, device setting, and
validation protocol are the same in all blocks.

## Selected-image overlap

Intersection counts below compare 60-image subsets (not Jaccard similarity).
Within each block, columns show pairwise overlap among strategies:

| Seed | random/frequency | random/diversity | random/weighted | frequency/diversity | frequency/weighted | diversity/weighted |
|---:|---:|---:|---:|---:|---:|---:|
| 123 | 11 | 14 | 13 | 30 | 44 | 28 |
| 2027 | 11 | 19 | 12 | 30 | 44 | 28 |
| 31415 | 17 | 12 | 15 | 30 | 44 | 28 |
| 271828 | 12 | 19 | 14 | 30 | 44 | 28 |

Across random-uniform seeds, pairwise intersections were:

| Seeds | Shared images |
|---|---:|
| 42 / 123 | 14 |
| 42 / 2027 | 14 |
| 42 / 31415 | 17 |
| 42 / 271828 | 15 |
| 123 / 2027 | 15 |
| 123 / 31415 | 16 |
| 123 / 271828 | 13 |
| 2027 / 31415 | 13 |
| 2027 / 271828 | 15 |
| 31415 / 271828 | 18 |

Frequency, diversity, and weighted-interest each selected exactly the same 60
tokens across all five seeds (including the existing seed-42 bundle). This is
consistent with their deterministic ranking/tie-breaking behavior on fixed
training-only inputs; it is not a validation failure. Random-uniform subsets
vary across seeds. These metadata selection statistics do not establish that
any strategy is superior or that its selected data is visually more diverse.

## Runner preflight results and limitations

All four commands completed successfully:

```text
python -m scenarioforge.training.cli curation-pilot --selection-seed <seed> --training-seed <seed> --curation-dir <bundle> --preflight-only
```

Each reported four conditions, 60 selected training images per condition,
the common validation digest, and the expected pretrained-checkpoint hash.
Every preflight reported `test_dataset_instantiated: false`. Models were
constructed only to verify identical fresh initialization within each block;
no optimizer step, model training, validation evaluation, or test evaluation
occurred.

The generator does not embed a self-hash in its parent manifest. The report
records the parent's raw SHA-256; parent-child linkage and common source,
split, dataset, scoring, and quota provenance were validated. The seed-42
bundle remains unchanged and is not included among the new generated outputs.

## Validation and recommendation

- Runner preflight: **16/16 child manifests passed**.
- Budgets, uniqueness, scene quotas, source/split provenance, image/annotation
  loading, and training-only membership: **all passed**.
- Validation partition: identical 80 tokens in all conditions.
- Test-set evaluation and model training: **not performed**.

**GO for multi-seed training**, following the plan's serial GPU execution
order, fixed six-epoch controls, per-block preflight, and stop-on-failure
requirements. This approval is limited to training the already-preflighted
manifests; it does not authorize test-set evaluation.
