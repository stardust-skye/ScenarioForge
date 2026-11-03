# Part 3A Final Validation Audit

**Audit result: APPROVE proceeding to Part 3B planning and implementation.** The audit did not
find a transform, projection, provenance, bounds, or reproducibility defect. This approves the
annotation-projection stage as a foundation; it is not approval to train a detector or evidence
that these projected cuboids are equivalent to human-drawn visible-object boxes.

**Audit conducted:** 2026-11-03.

No implementation changes were made during this audit. No scene splits or training were run.

## 1. Rejection analysis

Counts below were read from `data/processed/vision/conversion_report.json` and independently
aggregated from every row of `rejected_annotations.csv`. Percentages use all 18,538 source
annotations.

- Accepted: **5,325 / 18,538 (28.72%)**
- Rejected: **13,213 / 18,538 (71.28%)**
- All rejection rows have one of the recorded geometry/category reasons below; no reason was
  inferred from an aggregate difference.

### Rejections by reason

| Recorded reason | Count | Share of source annotations |
|---|---:|---:|
| `object_behind_camera` | 8,682 | 46.84% |
| `outside_image` | 4,357 | 23.50% |
| `unsupported_category` | 174 | 0.94% |
| **Total rejected** | **13,213** | **71.28%** |

The two geometric field-of-view reasons account for **13,039 records (70.34% of all source
annotations; 98.68% of rejections)**. In addition to reading the reason on each rejected CSV row,
the audit independently transformed every sample annotation with the devkit's
`get_sample_data(..., box_vis_level=NONE)` camera-frame boxes and recomputed projection
outcome/rejection reason from corner depths and projected bounds. All **18,538** recomputed outcomes matched the
exported labels exactly: 8,682 behind-camera, 4,357 outside-image, 174 unsupported, and 5,325
accepted. The maximum camera-frame corner depth among `object_behind_camera` records was
**0.09513 m**, below the configured 0.1 m near plane. Thus the high geometric rejection rate is
explained by each box's actual position relative to CAM_FRONT and the image bounds, not inferred
from aggregate totals. This is expected when projecting nuScenes' scene-wide 360-degree annotation
set into a single forward-facing camera. The remaining 174 are exactly the unsupported categories
in the explicit initial mapping. No accepted record was removed for visibility, minimum-area,
maximum-area, or missing-input rules in this dataset run.

### Rejections by original nuScenes category and recorded reason

Zero counts are shown explicitly. These are source categories from the CSV, not mapped class names.

| Original category | Behind camera | Outside image | Unsupported | Total rejected |
|---|---:|---:|---:|---:|
| `human.pedestrian.adult` | 2,402 | 1,303 | 0 | 3,705 |
| `human.pedestrian.child` | 22 | 17 | 0 | 39 |
| `human.pedestrian.construction_worker` | 100 | 27 | 0 | 127 |
| `human.pedestrian.personal_mobility` | 0 | 0 | 25 | 25 |
| `human.pedestrian.police_officer` | 6 | 4 | 0 | 10 |
| `movable_object.barrier` | 1,498 | 298 | 0 | 1,796 |
| `movable_object.debris` | 0 | 0 | 13 | 13 |
| `movable_object.pushable_pullable` | 0 | 0 | 82 | 82 |
| `movable_object.trafficcone` | 518 | 266 | 0 | 784 |
| `static_object.bicycle_rack` | 0 | 0 | 54 | 54 |
| `vehicle.bicycle` | 104 | 57 | 0 | 161 |
| `vehicle.bus.bendy` | 14 | 3 | 0 | 17 |
| `vehicle.bus.rigid` | 112 | 99 | 0 | 211 |
| `vehicle.car` | 3,256 | 2,012 | 0 | 5,268 |
| `vehicle.construction` | 75 | 20 | 0 | 95 |
| `vehicle.motorcycle` | 205 | 119 | 0 | 324 |
| `vehicle.trailer` | 41 | 2 | 0 | 43 |
| `vehicle.truck` | 329 | 130 | 0 | 459 |
| **Total** | **8,682** | **4,357** | **174** | **13,213** |

The largest rejected groups are cars, adult pedestrians, barriers, and traffic cones. Their
breakdown follows the many source annotations in those categories and is consistent with the
single-camera field-of-view restriction; it does not show one class being silently discarded by
the mapping or a visibility threshold.

### Accepted counts by mapped category

| Mapped category | Class ID | Accepted |
|---|---:|---:|
| `car` | 1 | 2,351 |
| `truck` | 2 | 190 |
| `bus` | 3 | 182 |
| `trailer` | 4 | 17 |
| `construction_vehicle` | 5 | 101 |
| `pedestrian` | 6 | 1,134 |
| `motorcycle` | 7 | 147 |
| `bicycle` | 8 | 82 |
| `traffic_cone` | 9 | 594 |
| `barrier` | 10 | 527 |
| **Total** |  | **5,325** |

### Visibility distribution

The report distribution covers all considered source annotations, accepted and rejected; source
visibility is metadata, not a projection filter.

| Visibility level | Count |
|---|---:|
| `v0-40` | 5,480 |
| `v40-60` | 1,636 |
| `v60-80` | 1,947 |
| `v80-100` | 9,475 |
| **Total** | **18,538** |

Accepted/rejected counts by visibility were respectively 1,389/4,091 (`v0-40`), 525/1,111
(`v40-60`), 590/1,357 (`v60-80`), and 2,821/6,654 (`v80-100`). The continued rejections at all
visibility levels support the report's stated geometric/category policy rather than an accidental
visibility filter.

## 2. Geometry verification

Reviewed [projection.py](../src/scenarioforge/vision/projection.py),
[converter.py](../src/scenarioforge/vision/converter.py), and the installed nuScenes devkit's
`NuScenes.get_sample_data` implementation.

- The source cuboid is built with the devkit `Box` from the annotation's global center, nuScenes
  `[width, length, height]` dimensions, and `(w, x, y, z)` quaternion.
- The transform sequence is correct and in inverse order: subtract ego translation, rotate by
  inverse ego rotation; then subtract camera-calibration translation, rotate by inverse camera
  rotation. This matches the devkit's `get_sample_data` source code for sensor-frame boxes.
- The output uses the calibrated camera intrinsic matrix and nuScenes `view_points(...,
  normalize=True)`. Positive camera depth is enforced; boxes with no cuboid points at/after the
  positive near plane are rejected as `object_behind_camera`.
- The 12 cuboid edges are intersected with the near plane where they cross it. In the actual data,
  one accepted annotation has such a crossing: six corners lie at/after 0.1 m and four edges
  intersect the plane. Independently using the devkit-transformed cuboid and `view_points` on those
  ten retained corners/intersections produces the same clipped saved box:
  `[1315.261155, 480.670871, 1600, 900]`. Its raw projected extrema exceed the image because the
  cuboid is very close to the camera; image clipping bounds it. The matching real-image overlay is
  [`01_4e169e93acb346d59de1af7f2b4882e1.png`](../data/processed/vision/inspection/audit_near_plane/01_4e169e93acb346d59de1af7f2b4882e1.png).
- Image-space bounds are `xyxy` in pixels, clipped to `[0, width] x [0, height]`; fully off-image,
  degenerate, too-small, and excessive-area boxes have explicit rejection paths.
- Eight real annotations from eight mapped categories were independently transformed with
  `NuScenes.get_sample_data(..., box_vis_level=NONE, selected_anntokens=[...])`, then projected
  with the devkit `view_points`. For boxes whose corners are all in front of the near plane, all
  four saved coordinates matched the independently projected/clipped coordinates exactly
  (maximum absolute difference **0.0 px**). Comparisons included an image-edge car and a
  bottom-clipped traffic cone. The other comparisons covered pedestrian, truck, bus, bicycle,
  motorcycle, and barrier.
- `get_sample_data` uses the sample's annotations directly for keyframes, so using the selected
  CAM_FRONT keyframe sample annotations does not omit the devkit's intermediate-frame annotation
  interpolation.

No geometry discrepancy was found. The near-plane case confirms edge clipping is exercised on
real input; the focused projection tests also cover synthetic near-plane and field-of-view
boundaries.

## 3. Visual inspection

Viewed the actual source images with saved overlays in all four requested output folders:

- `many/`: three dense scenes with cars, pedestrians, bus/truck and other mapped classes.
- `partial/`: three images with image-edge-clipped objects, including vehicles cut by the left or
  right frame boundary.
- `empty/`: one real image with no accepted boxes; the source image is in fact an empty roadway.
- `rejected/`: three images with accepted boxes and rejected source-record reason summaries.
- Also inspected the real near-plane-crossing car overlay cited above.

Across these examples, vehicle and pedestrian rectangles align with the corresponding visible
objects; bus/truck labels and traffic-cone/motorcycle labels in the viewed scenes are plausible.
Frame-edge rectangles terminate at the image edge rather than extending beyond it. The empty
example has no visible traffic object that should have an accepted CAM_FRONT annotation.

The audit also found quality limitations that are inherent to this projection policy, not
transform mismatches. Dense scenes contain overlapping rectangles, and some projected 3D cuboids
extend beyond visible silhouettes or include occluded/background area. The one very-near car box
spans a substantial lower-right image region after clipping. These are 2D bounds of 3D cuboids,
not human-traced visible-object boundaries. Of the accepted labels, 743 touch an image edge; three
have width under 1 px, and one has area under 25 px (the configured minimum area is 1 px). These
are worth reviewing or filtering for a downstream detector, but they pass the explicit current
policy and are not outside bounds or zero-area.

This is a qualitative review of ten selected overlays, not a frame-by-frame human relabeling of all
404 images.

## 4. Dataset integrity and reproducibility

Checks were run against the actual JSON/CSV exports and source nuScenes metadata:

| Check | Result |
|---|---:|
| CAM_FRONT images / image-manifest records | 404 / 404 |
| Source annotations considered | 18,538 |
| Accepted / rejected annotations | 5,325 / 13,213 |
| Accepted image paths that exist under the dataset root | 5,325 / 5,325 |
| Accepted boxes with positive area | 5,325 / 5,325 |
| Accepted boxes within their declared image bounds | 5,325 / 5,325 |
| Accepted class/category mismatches against mapping | 0 |
| Duplicate source annotation tokens across accepted and rejected rows | 0 |
| Accepted or rejected sample/scene/camera/image-path mismatches against nuScenes records | 0 |

Reproducibility was checked by rerunning the adapter and converter over the same source dataset
with the saved configuration and fixed processing timestamp, writing to a temporary directory,
and comparing each output file byte-for-byte. All seven files matched: `annotations.json`,
`annotations.csv`, `conversion_report.json`, `category_mapping.json`, `image_manifest.csv`,
`rejected_annotations.csv`, and `dataset_manifest.json`.

The saved report also records zero accepted boxes out of bounds, zero zero/negative-area boxes,
zero unexpected class IDs, and zero duplicate source IDs.

## 5. Issues discovered and fixes

No concrete correctness defect was discovered. No implementation fix was needed or made. The
reported totals reconcile exactly, and the reason-by-category CSV aggregation sums to the saved
reason totals.

Suspicious-looking high rejection counts are attributable to projecting 360-degree annotations
into one forward camera: recorded behind-camera and outside-image reasons account for 98.68% of
rejections. Unsupported categories are explicit and total 174. Near-plane crossing produces one
large but correctly image-clipped box; this is a low-frequency review case rather than a numerical
projection mismatch.

## 6. Remaining limitations

- The source is nuScenes mini (10 scenes) and CAM_FRONT only; this does not establish full-release
  coverage or generalization.
- Projected cuboid bounds can differ from visible silhouettes due to occlusion, truncation,
  object shape, and source annotation/camera timing. Ten overlays do not replace broader manual
  sampling or human-drawn-label comparison.
- Very thin edge slivers satisfy the configured 1 px area threshold. Downstream use may warrant a
  stricter minimum width/height/area policy, but such a change would alter labels and was not made
  without an approved requirement.
- Visibility categories are coarse metadata and are not a pixel-accurate occlusion measure.
- No detector, scene split, training, or model-quality metric was part of this audit.

## 7. Final recommendation

**APPROVE proceeding to Part 3B planning and implementation.** The data conversion has passed
this geometry, visual-sample, provenance, bounds, mapping, and determinism audit. Keep model
training gated on a separate review of split/leakage controls and a broader label-quality sample,
including the rare large near-camera and thin edge-clipped boxes. This audit did not create splits
or start training.
