# Part 3A: nuScenes CAM_FRONT annotation projection

## Scope

Part 3A converts nuScenes `v1.0-mini` global 3D sample annotations into model-independent 2D
`xyxy` pixel boxes for the `CAM_FRONT` keyframe. It writes deterministic, dataset-relative
manifests, CSV tables, a category map, a validation report, and optional visual-inspection
overlays. Source images remain in place; no dataset files or database schema are changed.

This is annotation preparation only. It does not create train/validation/test splits, curate
training data, install or import PyTorch, or train/evaluate a detector. Review actual overlays and
the conversion report before using the exported labels in later work.

## Inputs and geometry

The adapter reads each sample's front-camera sample-data reference, dimensions and image path,
camera calibrated-sensor record, ego pose, scene, and sample annotations through the official
nuScenes devkit. Annotations retain their source token, category, instance, 3D center, dimensions,
visibility token, and mapped visibility label. Missing camera references and incomplete source
records are preserved as structured rejections where an annotation exists.

The projection uses nuScenes' `Box`, `Quaternion`, and `view_points` implementations:

1. Construct the oriented cuboid from the global center, `[width, length, height]`, and
   `(w, x, y, z)` quaternion.
2. Transform global points to ego coordinates by subtracting ego translation and applying the
   inverse ego rotation.
3. Transform ego points to camera coordinates by subtracting calibrated-sensor translation and
   applying the inverse camera rotation.
4. Clip all 12 cuboid edges to the configurable positive camera near plane (default `0.1 m`), then
   project visible corners/intersections with the camera intrinsic matrix.
5. Bound the projected points, clip the rectangle to image bounds, and reject fully invisible,
   degenerate, too-small, or excessively large boxes.

The default minimum projected point count is 3, minimum box area is 1 pixel, and maximum accepted
clipped area is 98% of the image. All thresholds and the camera channel are in
`ProjectionConfig`; the CLI can override them with a JSON file. Source visibility is retained as
metadata, not used as a pixel-accurate occlusion filter.

## Category mapping

The versioned default mapping uses background ID 0 and stable foreground IDs:

| ID | Detection category | nuScenes leaf categories |
|---:|---|---|
| 1 | car | `vehicle.car` |
| 2 | truck | `vehicle.truck` |
| 3 | bus | `vehicle.bus.bendy`, `vehicle.bus.rigid` |
| 4 | trailer | `vehicle.trailer` |
| 5 | construction_vehicle | `vehicle.construction` |
| 6 | pedestrian | `human.pedestrian.adult`, `human.pedestrian.child`, `human.pedestrian.construction_worker`, `human.pedestrian.police_officer` |
| 7 | motorcycle | `vehicle.motorcycle` |
| 8 | bicycle | `vehicle.bicycle` |
| 9 | traffic_cone | `movable_object.trafficcone` |
| 10 | barrier | `movable_object.barrier` |

Only these explicit leaf categories map to a detection class. Other categories are retained in the
rejection table as `unsupported_category`; they are not silently folded into a broad class. Supply
a versioned JSON mapping with `--mapping` to use another set of classes.

## Reproducible run

Install the project's regular dependencies (no ML extra is needed), configure `NUSCENES_ROOT`, and
run these commands from the repository root:

```powershell
python scripts/prepare_vision_dataset.py convert
python scripts/prepare_vision_dataset.py inspect --mode many --count 4
python scripts/prepare_vision_dataset.py inspect --mode partial --count 4
python scripts/prepare_vision_dataset.py inspect --mode empty --count 2
python scripts/prepare_vision_dataset.py inspect --mode rejected --count 4
```

The output directory defaults to `data/processed/vision`; `--output-dir` changes it. If the
configured dataset directory is unavailable, the CLI can use the valid root recorded by the
already-ingested dataset metadata. `--dataset-root` overrides both. A configuration JSON can
override projection thresholds:

```json
{
  "camera_channel": "CAM_FRONT",
  "near_plane_m": 0.1,
  "min_projected_points": 3,
  "min_box_area_px": 1.0,
  "max_box_area_fraction": 0.98
}
```

The `inspect` command supports `random`, `many`, `partial`, `empty`, and `rejected` selection
modes, with a reproducible seed; `--sample-token` targets one sample. Overlays show category,
source annotation token prefix, visibility, and per-image rejection counts.
[`notebooks/04_Annotation_Projection_Inspection.ipynb`](../notebooks/04_Annotation_Projection_Inspection.ipynb)
demonstrates package-level conversion and displays examples from real images.

## Exports and quality checks

The output directory contains:

- `annotations.json` and `annotations.csv`: accepted annotations with `xyxy` coordinates, class
  IDs/names, image/sample/scene/camera provenance, source category and annotation token,
  visibility, 3D geometry, clipping status, and projected point count.
- `rejected_annotations.csv`: the same provenance with a reason and no fabricated box.
- `image_manifest.csv`: one record per camera image with relative path, dimensions, image status,
  and accepted/rejected annotation counts.
- `category_mapping.json`: the versioned mapping used for the run.
- `dataset_manifest.json`: output file references and dataset/configuration counts.
- `conversion_report.json`: accepted counts by category, rejected counts by reason and mapped
  category, visibility and scene; box-size distributions; image-status totals; and geometry/ID
  sanity checks.

Paths in the exports are relative to the dataset root; images are not copied. The report's UTC
processing timestamp changes per run. Given identical input data and configuration, the remaining
records and exports are sorted deterministically.

## Local `v1.0-mini` run

On the repository's available mini dataset, conversion processed 404 samples and 404 decodable
CAM_FRONT images, considering 18,538 source annotations. It accepted 5,325 boxes and rejected
13,213: 8,682 behind-camera, 4,357 outside-image, and 174 unsupported-category annotations. The
accepted class counts were car 2,351; pedestrian 1,134; traffic cone 594; barrier 527; truck 190;
bus 182; motorcycle 147; construction vehicle 101; bicycle 82; and trailer 17.

The report found zero accepted boxes out of image bounds, zero zero/negative-area boxes, zero
unexpected class IDs, and zero duplicate source annotation IDs. All ten scenes had camera images;
one image had no accepted annotations after applying the geometry/category rules. The
`many`, `partial`, `empty`, and `rejected` overlay modes were rendered and visually inspected. The
sample overlays show projected boxes in expected image locations, including edge-clipped boxes;
the rejected examples expose behind-camera, outside-image, and unsupported-category counts.

These are conversion diagnostics, not model-performance results. Mini data is small, and a
successful geometric projection does not establish label suitability for any detector or replace
manual review across more samples, classes, and scenes.

## Tests

The focused unit/integration tests cover coordinate transforms, near-plane clipping, projection
and image-bound policies, class mapping, rejection reasons, deterministic exports, reporting, and
real-image inspection plumbing. Run:

```powershell
python -m pytest -q tests/test_vision_projection.py tests/test_vision_conversion.py tests/test_vision_integration.py
```
