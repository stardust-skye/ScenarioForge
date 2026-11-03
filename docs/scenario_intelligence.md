# Scenario Intelligence (Part 2)

Metadata-only scenario analysis and dataset curation for ingested nuScenes mini **keyframes**.
It does not read images or point clouds, does not analyse sweeps, does not train models, and
produces no safety or risk assessment.

## Pipeline

```
Part 1 SQLite/Postgres DB + raw nuScenes JSON (visibility, attribute, log, ego_pose; read-only)
        -> extract -> data/processed/scenario_intelligence/scenarios.json
        -> stats | rarity | diversity | duplicates | rank | curate | compare
```

`extract` is the only command that touches the database. Everything else reads `scenarios.json`,
so analyses are reproducible offline. Only `data_access.py` imports ORM models; the analysis
modules operate on `ScenarioRecord` objects, so another dataset adapter can feed them.

Visibility, attribute names, location and ego pose are not stored in the Part 1 schema, so they are
read from the raw nuScenes JSON through `datasets/nuscenes_lookups.py` (no schema change).

## Commands

```powershell
python scripts/scenario_intelligence.py extract
python scripts/scenario_intelligence.py stats
python scripts/scenario_intelligence.py rarity
python scripts/scenario_intelligence.py diversity
python scripts/scenario_intelligence.py duplicates
python scripts/scenario_intelligence.py rank [--weight rarity=2] [--weight preference=1 --prefer n_pedestrians=1]
python scripts/scenario_intelligence.py curate --strategy diversity --n 60 --seed 42 [--require-group pedestrian] [--min n_objects=10] [--max-similarity 0.9]
python scripts/scenario_intelligence.py compare --manifest data/processed/scenario_intelligence/curation_diversity_n60_seed42_manifest.json
```

Global options: `--output-dir`, `--config <json>` (overrides analysis defaults; unknown keys are an
error), `--no-plots`. Exit codes: 0 success, 2 bad input/arguments, 1 unexpected failure.

## Metrics

- Shannon entropy `H = -sum p_i log2 p_i`; normalized entropy `H / log2(K)` (K = support size; 0 if K<=1).
- Simpson diversity `1 - sum p_i^2`; Herfindahl `sum p_i^2`.
- Jensen-Shannon divergence in bits, bounded in [0, 1].
- Inverse frequency `1 / (count + epsilon)`-style rarity, min-max normalized per component.
- The default rarity threshold (`rare_max_share=0.05`), epsilon, binning strategy/edges, rarity
  component weights, ranking weights, and preference weights can be overridden in a JSON config.
  Reports include raw and normalized rarity components and the weights used. These are configurable
  heuristics, not thresholds or weights validated on held-out data.
- Missing category combinations are excluded from combination frequency and scoring, and their
  count is reported separately. Unknown category names remain visible and map to the `other`
  category group. A category combination occurring rarely in this tiny sample is not evidence that
  the corresponding event is rare in the real world.
- Similarity between scenarios: weighted mean of category-set Jaccard, category-count cosine,
  numeric-feature cosine and same-location; unavailable components are skipped and weights renormalized.
- Missing values (`None`) are never imputed; missing components are excluded and stated in explanations.

## Scores

**Rarity** = weighted mean of category, combination and binned-feature rarity. "Rare" means infrequent
in this dataset. Combination level defaults to category *groups*: on the real mini data, full-category
combinations flagged 322 of 404 scenarios as rare (uninformative), group-level flagged 60.

**Scenario-interest score** = weighted mean of rarity, composition diversity, complexity and an
optional preference component. Default positive weights are equal, but are not calibrated or
empirically validated. Components, their weighted contributions, and unavailable components appear
in the ranking output. Composite complexity is unavailable if any of its three constituent features
is missing. Scores are min-max normalized relative to the supplied set, so scores from different
datasets or subsets should not be compared as if calibrated.

## Curation

Strategies: `random` (seeded baseline), `frequency`, `diversity` (greedy entropy gain), `weighted_interest`.
Constraints: required groups, min/max feature values, near-duplicate similarity limit. If constraints
cannot be met the result is short and a `shortfall_reason` is recorded rather than padding. Manifests
contain no timestamps and carry a SHA-256, so identical inputs give identical manifests.

The reported category entropy is computed over annotated-object counts pooled across the selected
keyframes. Higher entropy means a more even object-category mix; it does not mean the subset
preserves the original distribution. JSD measures the distribution shift, and seeded random
sampling is a useful same-budget preservation baseline. Neither entropy nor distributional similarity
establishes improved model training performance; that requires controlled model experiments.

## Near-duplicate interpretation

Similarity scores are unchanged by temporal classification or filtering. Threshold-passing pairs are
classified as consecutive, temporally adjacent (within `temporal_adjacency_s`), non-adjacent within
the same scene, or cross-scene. Each reported pair also includes component similarity scores,
weights, and weighted contributions, so the metadata signal behind the combined score is
inspectable. The CLI accepts `--scene-scope all|same|cross`, `--min-temporal-separation-s`, and
`--temporal-adjacency-s`. Use `--exclude-temporally-adjacent` to omit both consecutive pairs and
pairs within the configured adjacency interval from the reported candidates. The minimum-separation
filter is an independent timestamp-based filter for same-scene pairs. Reports retain unfiltered
class counts as well as counts after the requested filters. Similarity scores do not change when
filtering. These are metadata-similarity observations, not proof of redundant training data.

Examples: `python scripts/scenario_intelligence.py duplicates --scene-scope cross` reports only
cross-scene pairs; `python scripts/scenario_intelligence.py duplicates
--exclude-temporally-adjacent --temporal-adjacency-s 1.0` excludes consecutive pairs and other
within-scene pairs no more than one second apart.

On the inspected 404-scenario mini output at similarity threshold 0.9, the 4,623 candidate pairs
partitioned into 381 consecutive, 206 temporally adjacent (within 1 second), 4,022 non-adjacent
same-scene, and 14 cross-scene pairs. Thus the 4,609 same-scene candidates were mostly *not*
temporally adjacent. Since the score includes category, category-count, numeric, and location
similarity, scene membership and repeated metadata may account for many high-scoring pairs; the
result alone does not establish duplicate or redundant training examples. Filtering same-scene pairs
to a minimum 1-second separation retained 4,036 candidates (4,022 non-adjacent same-scene plus 14
cross-scene); the similarity values themselves were unchanged.

## Part 3 preparation (design only)

No annotation conversion, split generator, model, or training code is implemented. The proposed
Part 3 task is 2D front-camera object detection; its metrics and split protocol therefore differ
from nuScenes 3D detection and its NDS. See [PART_3_ML_ARCHITECTURE.md](PART_3_ML_ARCHITECTURE.md)
and [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) for the design and leakage controls. The 10-scene
mini release is appropriate for pipeline smoke tests only, not statistically strong generalization
claims.

## Limitations

- Keyframes only (about 2 Hz); no sweeps. Ego speed is a coarse keyframe-to-keyframe estimate.
- Metadata similarity only; no visual or LiDAR similarity.
- 404 samples from 10 scenes (3 share one log): rarity is relative to a tiny dataset.
- Curated-subset benefit for model performance is not tested.
- Only nuScenes mini has been tested.
