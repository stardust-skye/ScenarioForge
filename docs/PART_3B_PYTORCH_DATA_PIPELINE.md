# Part 3B: PyTorch-ready detection data pipeline

## Scope and architecture

Part 3B prepares reusable, scene-disjoint CAM_FRONT data inputs. It does not instantiate detector
architectures, train/evaluate models, or perform the curation experiments. The base ingestion and
Part 3A environment does not depend on PyTorch; model-input code is isolated in
`scenarioforge.vision_data` and should be used from a separate `.venv-ml`.

- `vision_data.splits` creates deterministic scene-grouped manifests, validates full scene/sample
  coverage and manifest digests, and checks training-only/equal-budget selections.
- `vision_data.dataset.CameraDetectionDataset` reads the Part 3A accepted-annotation CSV and image
  manifest, validates paths/provenance/boxes at construction, and lazily decodes one image per
  `__getitem__`.
- `vision_data.transforms` applies paired image/XYXY operations. The default/evaluation path is
  deterministic; optional horizontal flipping is explicit and off by default.
- `vision_data.dataloader` provides detection-compatible collation, seeded DataLoaders with
  `num_workers=0` by default, split-restricted datasets, and a training-only subset guard.
- `vision_data.statistics` reports scene/image/box/class/empty-image/bounding-box distributions,
  missing paths, invalid rows, and leakage checks to JSON and CSV.
- `scripts/prepare_vision_data.py` exposes split creation and report generation.

## Input schema and target contract

The loader uses the generated `image_manifest.csv`, `annotations.csv`, and
`category_mapping.json` under `data/processed/vision`. The manifest's `sample_token` is the sample
key, `scene_token` is the leakage-prevention group, `image_path` is dataset-root-relative, and
`width`/`height` are source image dimensions. Accepted annotation rows reference that same
`sample_token`, `scene_token`, `image_id`, image path, and dimensions; `x_min`, `y_min`, `x_max`,
`y_max` are pixel `xyxy`; and `class_id`/`detection_category` are the Part 3A category mapping.
Rejected rows are not loaded as training targets.

Each item returns a floating RGB `[C,H,W]` tensor in `[0,1]` and a target with:

- `boxes`: finite float32 `[N,4]` XYXY coordinates;
- `labels`: int64 `[N]`;
- `image_id`: a deterministic, unique scalar int64 tensor derived from the image ID;
- `area`: float32 `[N]` computed from the current transformed boxes;
- `iscrowd`: int64 `[N]`, initialized to zero.

Zero-box images return correctly shaped `[0,4]` and `[0]` tensors. The existing mapping uses
foreground category IDs 1–10 and background ID 0. Torchvision detection models reserve 0 for
background, so positive class IDs are remapped in sorted source-ID order to contiguous model labels
1–K. The dataset exposes both `source_id_to_model_label` and
`model_label_to_source_id`; the current 1–10 mapping remains 1–10. RGB conversion and scaling are
performed without normalization by default. Torchvision detection models generally apply their
configured normalization internally; custom normalization is available as an explicit transform.

Paths must resolve beneath the configured dataset root. Image files are checked for existence,
decodability, and metadata dimensions at dataset construction. Annotation/image associations,
class IDs, finite coordinates, positive area, and image bounds are checked before loading.
Transforms receive and return `(image, target)` together; custom transforms must preserve this
contract.

## Scene-level splits and leakage controls

Random image-level splitting is not supported. The split API requires complete, disjoint scene
groups, assigns every eligible image-manifest sample exactly once, stores scene and sample tokens,
and calculates SHA-256 digests over both the canonical manifest and the complete canonical image
inventory. The manifest also records the authoritative dataset version. Loading, writing, validating,
computing training selections, and constructing split datasets require the current inventory and
dataset version; a manifest digest alone is never treated as proof that its assignments match the
dataset. Split dataset construction fails closed if the adjacent `dataset_manifest.json` is missing
or has no dataset version. A fixed seed and input manifest reproduce the same assignment; explicit
train/validation/test scene lists are also supported and must cover every scene exactly once. Too
few scenes for all requested non-empty splits is a clear error, not a fallback.

### Mini smoke-test split used here

Scene counts and class distributions were inspected before choosing an explicit **60/20/20
scene-level** smoke-test ratio: with ten groups this yields six training, two validation, and two
test scenes. This is a practical code-path smoke test, not a claim that the resulting partitions are
representative or balanced. Use seed 42:

```powershell
python scripts/prepare_vision_data.py split --seed 42
python scripts/prepare_vision_data.py stats --split-manifest data/processed/vision/splits/scene_split_seed42.json
```

The generated manifest is saved at
`data/processed/vision/splits/scene_split_seed42.json`; statistics and leakage reports are saved
next to it. `data/processed` is intentionally ignored by Git. The manifest contains the complete
scene/sample assignments, dataset version, image-inventory provenance digest, and manifest digest.
On the local mini exports the current manifest digest is
`d93c15fac3e798343ea95fcefea3c03a0b189a0b9036b3f355354550109bf2ab`.

| Split | Scenes | Images | Accepted boxes | Empty images |
|---|---:|---:|---:|---:|
| Train | 6 | 242 | 3,636 | 0 |
| Validation | 2 | 80 | 952 | 0 |
| Test | 2 | 82 | 737 | 1 |

Seed-42 scene assignments:

- Train: `6f83169d067343658251f72e1dd17dbc`,
  `bebf5f5b2a674631ab5c88fd1aa9e87a`,
  `cc8c0bf57f984915a77078b10eb33198`,
  `d25718445d89453381c659b9c8734939`,
  `de7d80a1f5fb4c3e82ce8a4f213b450a`,
  `e233467e827140efa4b42d2b4c435855`
- Validation: `c5224b9b454b4ded9b5d2d2634bbda8a`,
  `fcbccedd61424f1b85dcbf8f897f9754`
- Test: `2fc3753772e241f2ab2cd16a784cc680`,
  `325cef682f064c55a255f2625c533b75`

The local per-split category counts are:

| Category | Train | Validation | Test |
|---|---:|---:|---:|
| barrier | 527 | 0 | 0 |
| bicycle | 54 | 15 | 13 |
| bus | 136 | 11 | 35 |
| car | 1,263 | 648 | 440 |
| construction_vehicle | 97 | 0 | 4 |
| motorcycle | 44 | 25 | 78 |
| pedestrian | 814 | 198 | 122 |
| traffic_cone | 580 | 11 | 3 |
| trailer | 17 | 0 | 0 |
| truck | 104 | 44 | 42 |

The validation split has no barrier, construction-vehicle, or trailer examples; test has no
trailer examples. This split is not balanced, and per-class validation/test conclusions for absent
or sparse classes are not supported. Fix and reuse the exact held-out scene lists for any future
strategy/model comparisons on a larger release.

`validate_split_manifest` checks manifest digest, scene/sample disjointness, and exact image
coverage. `validate_training_selection` rejects duplicate, validation/test, or unknown sample
tokens. `select_training_subset` accepts only a training-split dataset, and
`validate_equal_training_budgets` provides a guard for equal future strategy budgets. Give
frequency/diversity scoring only the training dataset; the helpers prevent held-out IDs entering a
training selection, but cannot prove that arbitrary external scoring code never inspected held-out
labels. Reuse the same persisted validation/test memberships and do not recompute or rebalance them
after curation.

## Transforms and DataLoader

`build_eval_transforms()` is deterministic and converts RGB pixels to float `[0,1]`. An optional
size is `(height,width)`; paired resize scales all four XYXY coordinates and area. The training
builder has the same safe defaults. A horizontal flip can be explicitly enabled and updates boxes
with the image, but is off by default because mirroring can change road-side/driving semantics.
Empty targets remain valid through resize and flip. Optional `Normalize(mean,std)` is explicit.

`detection_collate_fn` returns tuples of per-image tensors/targets so boxes of different counts are
not stacked. `make_detection_dataloader` supports batch size, shuffle, worker count, seed, and
optional pinned memory. Its default `num_workers=0` is chosen for Windows portability; workers
receive deterministic Python/NumPy seeds when enabled. Shuffle uses a seeded PyTorch generator.

## Separate ML environment (Windows PowerShell)

The verified version pair is **torch 2.5.1 + torchvision 0.20.1**, which is a matched release pair
and supports Python 3.12. The base `.venv` is not changed. The `requirements-ml.txt` file pins the
pair; use one of the following wheel indexes:

```powershell
# Create/activate an isolated environment (Python 3.12 is already used by the base project).
py -3.12 -m venv .venv-ml
.\.venv-ml\Scripts\Activate.ps1
python -m pip install --upgrade pip

# CPU-only installation:
python -m pip install -r requirements-ml.txt --index-url https://download.pytorch.org/whl/cpu

# Then install ScenarioForge and notebook/test tooling in this environment:
python -m pip install -e ".[dev,notebooks]"

# Verify package versions and CUDA availability:
python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA runtime:', torch.version.cuda)"
```

For CUDA 12.4 wheels instead of CPU-only wheels, replace the requirements installation command
with:

```powershell
python -m pip install -r requirements-ml.txt --index-url https://download.pytorch.org/whl/cu124
```

CUDA availability still depends on a compatible NVIDIA driver/device; CPU execution remains
supported. Confirm current wheel/platform requirements on the
[official PyTorch previous-versions page](https://pytorch.org/get-started/previous-versions/) and
the [torchvision compatibility table](https://pypi.org/project/torchvision/). Do not install the
`requirements-ml.txt` packages into `.venv`.

## Commands, tests, and notebook

From the repository root, create a split and review its reports in the base environment:

```powershell
python scripts/prepare_vision_data.py split --seed 42
python scripts/prepare_vision_data.py stats --split-manifest data/processed/vision/splits/scene_split_seed42.json
```

Run all tests in `.venv-ml` to include the torch-backed dataset, transforms, DataLoader, and
nuScenes-mini smoke tests:

```powershell
python -m pytest -q
```

Execute [notebooks/05_PyTorch_Dataset_Validation.ipynb](../notebooks/05_PyTorch_Dataset_Validation.ipynb)
from the repository root using the `.venv-ml` kernel. It loads the persisted split, builds the
three datasets, visualizes samples with many/few/zero boxes, fetches actual batches, prints split
class/scene statistics, and validates no overlap. Figures and reports stay under the generated
vision output directory; source images remain untouched.

## Limitations and scaling to full nuScenes

The mini release has only ten scenes and sparse/absent class support in held-out groups. These
partitions validate input/output and leakage-control code paths only; they cannot support reliable
generalization or class-balanced model claims.

For full `trainval`, regenerate statistics, define and review complete scene-level assignments,
then persist and freeze the exact split before any curation/scoring. Check all class and empty-image
support, keep validation/test scenes identical across strategies and seeds, and ensure selection
budgets are equal. Validate image availability, label counts, runtime, storage, and worker settings
at full scale before training. No full-release split or training experiment is included here.
