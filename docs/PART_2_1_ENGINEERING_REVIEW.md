# Part 2.1 Engineering Review

## Scope inspected

- Scenario extraction, feature engineering, rarity, diversity, duplicate detection, ranking,
  curation, comparison, CLI configuration, and their tests under `src/scenarioforge/` and `tests/`.
- The SQLAlchemy schema and nuScenes adapter, including calibration, sample-data metadata, and
  source annotation geometry.
- Generated statistics, scenario reports, curation manifests/comparisons, and the two Part 2
  notebooks.
- Project configuration, README, and existing architecture/data-dictionary documentation.

The review is metadata-only. It did not modify raw nuScenes data, change the Part 1 schema, install
packages, or train models.

## Verified dataset snapshot

The generated Part 1 statistics report records nuScenes v1.0-mini with 10 scenes, 404 samples,
18,538 annotations, 4,848 keyframe sensor references, and zero missing sensor files or unresolved
sample/scene/sensor references. The object-category distribution has 18 observed categories and
normalized entropy 0.5941. These are counts and distribution summaries, not evidence of perception
quality or real-world event prevalence.

The schema can support the proposed front-camera pipeline without a migration: calibrated sensor
translation, rotation, and camera intrinsics are retained in `sensors.calibration`; sensor-data
metadata includes image dimensions and ego-pose source IDs; annotation JSON retains 3D translation,
size, rotation, and source annotation tokens. Existing access patterns should be reused rather than
adding duplicate geometry columns.

## Findings and changes

### Duplicate detection

The detector compares metadata only. Exact fingerprints use category, attribute, and visibility
counts. Near-duplicate similarity uses category-presence Jaccard, category-count cosine, cosine on
min-max-scaled numeric features (`n_objects`, `n_unique_instances`, `ego_speed_mps`, and
`n_objects_near_ego` by default), and same-known-location agreement. Missing numeric/context groups
are omitted and the remaining configured weights are renormalized. No image or point-cloud
similarity is computed.

The saved 404-sample report at threshold 0.9 had 21 samples in exact-fingerprint groups and 4,623
threshold-passing near-duplicate candidates: 381 consecutive pairs, 206 other temporally adjacent
pairs within one second, 4,022 non-adjacent same-scene pairs, and 14 cross-scene pairs. The
4,609 same-scene total is therefore not primarily immediate temporal neighbors. The 4,623 count is
a metadata-similarity candidate count, not a count of redundant training examples.

Two usability gaps were fixed:

- Timestamp separation alone did not exclude every ordinally consecutive frame pair. The new
  `exclude_temporally_adjacent` setting and CLI flag exclude both consecutive pairs and pairs within
  the configurable `temporal_adjacency_s` window. The independent minimum-time-separation filter
  remains available.
- Pair reports previously exposed only the aggregate score. They now include each component score,
  its configured weight, and its weighted contribution, plus the component definitions and numeric
  feature list at report level. Both JSON and CSV pair exports preserve this explanation. This keeps
  the score auditable without implying visual redundancy.

### Rarity and diversity

Category groups are explicit: pedestrian, vehicle, two-wheeler, traffic-related, static object,
animal, and other. Unknown category names remain present as source categories and map to `other`.
Category combinations are sorted string sets; default rarity uses group combinations because full
category combinations are too sparse in mini.

Rarity uses inverse frequency `1 / (frequency + epsilon)`, followed by per-component min-max
normalization and a weighted mean over available components. The default `rare_max_share=0.05`,
epsilon, binning, and weights are configurable. “Rare” means infrequent in the supplied records;
mini's 10 scenes cannot support real-world prevalence estimates. Continuous features use explicit
quantile or uniform bins; missing values are kept distinct and are not imputed.

Missing category combinations were previously counted as an empty-string combination and could be
marked rare. They are now excluded from combination frequency and combination scoring, with a
separate missing count. Diversity reports likewise expose missing-combination count and coverage.
Unknown-category and empty-category tests cover these semantics.

The diversity implementation uses Shannon entropy, normalized by `log2(K)`, Simpson diversity
`1 - sum(p_i^2)`, Herfindahl concentration `sum(p_i^2)`, JSD, and coverage. Category entropy in the
curation comparisons is computed over pooled annotated-object counts. It measures evenness, not
distributional preservation; JSD and coverage answer separate questions. Neither establishes a
model-performance benefit.

### Curation

All four strategies share `curate(records, CurationConfig, ...) -> CurationResult`. Inputs are
ordered by sample ID; random selection uses a local seeded generator; ties are deterministic.
Constraints are applied consistently and shortfalls are reported rather than silently padded.
The versioned manifest records the configuration, dataset source ID, input digest, selected sample
and scene IDs, category totals, selection scores, counts, and manifest digest. Same-budget saved
comparisons exist for random, frequency, diversity, and weighted-interest strategies (60 of 404
samples, seed 42).

Selected sample IDs are unique, but metadata-similar candidates are not suppressed by default.
The near-duplicate similarity constraint is opt-in; use the same threshold and temporal policy
across strategies if it is enabled. This avoids silently changing the random baseline or claiming
that adjacent metadata-similar frames are redundant.

The original and subset distributions are compared by the same object-count definition and the
comparison uses the original category support for entropy normalization. The current curation API
does not implement per-scene quotas; any model experiment should add or predeclare scene-level
controls rather than allowing all strategies to concentrate in different scenes.

### Notebooks and reports

`02_Scenario_Discovery.ipynb` and `03_Data_Curation.ipynb` call package/report functions instead of
reimplementing the algorithms. Both locate the project from the notebook location, include
interpretation/limitations, and contain executed outputs. The discovery notebook was updated to
show the new pair explanations and explicit adjacency exclusion, then executed and saved. The
curation notebook was executed in-memory to validate it without changing its saved output.

The saved curation comparisons show normalized object-category entropy of 0.5912 for random,
0.6363 for frequency, 0.6866 for diversity, and 0.6337 for weighted interest, versus 0.5941 for
the original. These are dataset-distribution observations only; no model was trained and no
perception conclusion follows.

The existing dataset/database schema and Part 1 ingestion code were not changed. The previous
3D-detection-oriented Part 3 proposal in the scenario documentation was replaced with a pointer to
the new, explicitly 2D-camera design.

## Validation

- `.venv\Scripts\python.exe -m pytest -q`: **67 passed**.
- Both notebooks executed successfully with the project virtual environment. The discovery
  notebook's refreshed outputs were saved; the curation notebook was executed in-memory.
- The project virtual environment is Python 3.12.16. The unactivated system `python` resolves to
  Python 3.13, outside the declared project range, and its Matplotlib/Pillow import failed during
  test collection; no packages were installed or modified to work around that environment.

## Remaining limitations and risks

- Near-duplicate similarity is a metadata heuristic and its threshold is not calibrated for
  redundancy. Pairwise comparison is quadratic in sample count.
- Rarity and diversity reflect the supplied dataset and binning. NuScenes mini is too small and
  scene-correlated for strong prevalence or generalization claims.
- Curation has no built-in scene quotas or split-aware training boundary; those controls belong in
  the later ML data pipeline.
- There is no 2D annotation converter, scene split registry, detector, evaluator, or training loop
  yet. These are design deliverables only in this phase.
- The newly added component-level pair fields are present in fresh detector outputs. Older saved
  standalone duplicate reports are historical outputs and were not overwritten.
