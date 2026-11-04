# Part 3F.5.1 — Multi-Seed Runner Implementation Report

## Summary

The controlled curation runner now accepts the five approved seed values and an
explicit curation-artifact directory. Preflight validates the run and child
manifests, dataset and split provenance, exact 60-sample budgets, train-only
membership, image-file presence, annotation counts, and the common 10-image
quota for each of the six training scenes. Each run still uses the same fixed
validation split and must initialize from the same COCO_V1 checkpoint.

**Recommendation: GO for seed-specific manifest generation and preflight only.**
The generated manifests for seeds 123, 2027, 31415, and 271828 must pass this
runner's preflight before any training is approved. No training was launched
and no test data was evaluated in this implementation phase.

## Files changed

- `src/scenarioforge/training/curation_pilot.py`
- `src/scenarioforge/training/cli.py`
- `src/scenarioforge/training/config.py`
- `src/scenarioforge/training/experiment.py`
- `tests/test_curation_pilot.py`
- `tests/test_training.py`

No historical reports, datasets, split assignments, curation artifacts, or
experiment outputs were changed.

## Seed-42 assumptions identified and addressed

The runner previously pinned the run-manifest SHA-256 and four seed-42 child
manifest hashes, built only `*_n60_seed42` paths, required the manifest seed
and training seed to be 42, and initialized every run with seed 42. The CLI
also defaulted to a shared historical report path, which could overwrite an
existing report.

The runner now accepts only selection and training seeds from
`42, 123, 2027, 31415, 271828`. The approved matrix requires the two seed
values to match within a seed block. Seed 42 retains its default audited
curation directory and pinned hashes; any other seed must supply
`--curation-dir`. The existing curation selection algorithms and the
authoritative seed-42 split are unchanged.

## Input and provenance safeguards

Before creating experiment output directories, preflight:

- Checks the requested seed against the run manifest and checks the requested
  child-manifest filenames are listed by that parent manifest.
- Retains the exact pinned run and child hashes for the original audited
  seed-42 directory; verifies canonical child-manifest hashes for other
  directories.
- Verifies the active dataset version, source-file hashes, authoritative
  seed-42 split digest and 242/80/82 split sizes.
- Requires four approved strategies, 60 distinct selected tokens per strategy,
  and validates each token against training-only membership. Held-out,
  duplicate, and unknown tokens fail preflight.
- Verifies all selected image files are present under the dataset root,
  selected scene assignments and counts match the image inventory, annotation
  counts match the accepted training annotations, and the approved scene
  quota is exactly 10 samples per training scene.
- Checks the child manifests share their parent run's scenario, split, source,
  dataset-version, and quota provenance.
- Keeps the prior exact dataset-instantiation check: only train and validation
  datasets may be constructed, with all 80 validation samples used.
- Checks the cached pretrained checkpoint provenance before model construction
  and before each run, and verifies fresh initial model-state fingerprints
  match across strategies.

The runner calculates the parent manifest's raw SHA-256 and persists it with
the child hash in downstream curation provenance. The existing curation
generator does not embed a self-hash in its parent manifest; for non-original
bundles, parent provenance is therefore checked through its declared child
list and the shared source, split, dataset, and scenario values rather than
against an independently pinned parent digest.

## Seed persistence and ranked strategies

`selection_seed` and `training_seed` are explicit config and experiment
metadata fields. They are included in run status, curation provenance,
environment metadata, checkpoint metadata, and experiment summaries. The
legacy `seed` continues to drive the training/DataLoader RNG state and must
equal `training_seed`. Configurations without the new optional fields retain
their previous serialized configuration shape.

Random selection is recorded as randomized. Frequency, diversity, and
weighted-interest selection are recorded as deterministic rankings for fixed
inputs. Identical ranked selections across seed-specific manifests are
accepted; the runner does not claim that these strategies necessarily choose
different samples when their seed changes.

Each invocation creates a unique output directory. Reports default to a file
inside that directory; an explicitly supplied report path is rejected if it
already exists.

## Tests and validation

New tests cover:

- Seed-specific preflight for all five planned seed values.
- Same-sample results for deterministic ranked strategies across different
  selection seeds.
- Seed mismatch, wrong budget or strategy, changed child hash, and incorrect
  parent, split, and source provenance.
- Validation/test leakage, duplicate and unknown samples, missing image files,
  wrong scene assignments, and incorrect quotas.
- Mismatched pretrained-checkpoint hashes, output-directory non-reuse, and
  selection/training seed persistence.

The seed-specific preflight fixtures are copies of the audited seed-42 bundle
with seed metadata and canonical child hashes updated in temporary test
directories. They exercise the runner's validation paths but do not validate
that the curation generator produces those selections for the new seeds;
that is the next phase's required generation/preflight check.

Validation results:

- Focused curation-pilot and training tests: **48 passed**.
- Full test suite: **187 passed**.
- Python 3.12.16 `py_compile`: passed for all changed Python files.
- `git diff --check`: passed.
- Ruff was not available in the active project environment; no linting package
  was installed.
- Original audited manifests retain their pinned hashes; tests confirmed the
  original bundle still passes its exact-hash preflight.
- No packages were installed. No training or test-set evaluation occurred.

## Remaining limitations

- Seed-specific manifests for the four additional seeds have not yet been
  generated or independently preflighted.
- The tests validate seed-specific manifest handling using temporary
  checksum-consistent fixtures; they do not prove selector outputs are
  reproducible for those seeds.
- The parent-manifest format has no embedded self-hash. Its raw hash is
  computed and recorded downstream, while authenticity is checked through
  associated child and source provenance.
- The runner does not train or compare model performance; this report makes no
  claim about curation effectiveness.

## Next-phase gate

Proceed to generate immutable seed-specific curation bundles for
`123, 2027, 31415, 271828`, then run preflight on each bundle. Confirm that
all four strategy manifests within each seed block agree on dataset version,
authoritative split digest, source hashes, and the six-scene quota policy.
Stop on any discrepancy. This GO applies only to manifest generation and
preflight; it does not authorize training or test evaluation.
