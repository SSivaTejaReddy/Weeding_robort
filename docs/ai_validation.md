# AI Validation Implementation

This workspace implements the AI validation stage described for the rover:

| Goal | Implementation |
|---|---|
| Setup environment | `requirements-ai.txt` |
| Collect dataset | `scripts/capture_dataset.py` with capture metadata |
| Annotate dataset | YOLO dataset config and label audit |
| Train initial model | `scripts/train_yolo.py` for YOLO26 selected-crop detect or segment |
| Improve detection | quality tags and overlay tests for sunlight/shadows |

## Scope

The initial model is deliberately small. It learns one field label:

- `crop`: the crop selected for the current field run, such as onion or groundnut

Runtime targeting uses that crop detector as a protection mask. Points in the
working region that are not protected by a crop detection are emitted as
`not_crop` validation points, meaning `~crop`. That avoids trying to train a
fixed detector for every possible item outside the selected crop.

`not_crop` is a complement decision, not an object name. If the robot later
needs to distinguish stones from tools, people, or irrigation pipes, add a
safety detector before extraction.

## Phase Acceptance

Use the AI validation phase to reach these gates before hardware timing work:

| Deliverable | Gate |
|---|---|
| Development environment | Capture, audit, train, and inference commands start successfully |
| Initial dataset | 100 or more usable images for the selected crop |
| Annotated dataset | Selected-crop labels are consistent and pass the audit |
| Initial model | Field validation protects at least 90% of held-out selected-crop views |
| Robustness pass | Sunlight and shadow overlays show fewer avoidable false positives after dataset improvement |

## Dataset Workflow

Collect at least 100 usable images before the first model run. Include close,
mid-height, sunlit, shadowed, damp-soil, dry-soil, and cluttered-row views for
both supported crops.

Capture images from a USB camera, Pi camera exposed as a camera device, or a
saved video:

```powershell
python -m scripts.capture_dataset --crop onion --condition sunlight --source 0
python -m scripts.capture_dataset --crop groundnut --condition shadow --source field_walk.mp4 --interval 1.5
```

The capture script stores raw images below `datasets/raw` and appends
`datasets/raw/captures.csv` with brightness, shadow ratio, highlight ratio, and
blur score. Use the metadata to keep poor images visible during review instead
of silently training on them.

Annotate the selected images in LabelImg, CVAT, or another YOLO exporter.
Export them into:

```text
datasets/crop_non_crop/
  images/train  images/val  images/test
  labels/train  labels/val  labels/test
```

Follow these annotation rules:

1. Label visible selected-crop foliage as `crop`, including partial crops near
   the implement working region.
2. Use one dataset per selected crop during early validation. For the current
   onion set, every positive label is `crop`.
3. Keep empty YOLO label files for frames with no selected crop if you collect
   background negatives.
4. Do not label soil shadows, weeds, stones, or tools for this crop-only run.
   Keep difficult background examples in validation.
5. Keep train, validation, and test views separated by capture sequence where
   possible so near-duplicate frames do not inflate accuracy.

Audit the export before training:

```powershell
python -m scripts.audit_yolo_dataset --dataset datasets/crop_non_crop --classes 1
```

For an early safety-biased onion baseline, the repo also includes a weak-label
preparation step. It segments green vegetation pixels into YOLO polygons from
the raw selected-crop images:

```powershell
python -m scripts.prepare_selected_crop_dataset
```

These labels are only a starting point. They can over-protect other green
regions and miss dry or heavily shadowed leaf tissue, so review or replace them
with human crop masks before field extraction.

## Training

Run YOLO26 nano segmentation first when using polygon masks. Use a larger image
size and longer patience for the recall-first crop-protection run:

```powershell
python -m scripts.train_yolo --task segment --epochs 100 --imgsz 768 --batch 8 --name selected_crop_recall
```

If bounding-box labels are exported instead, train the detector:

```powershell
python -m scripts.train_yolo --task detect --epochs 100 --imgsz 768 --batch 8 --name selected_crop_recall
```

Training uses `configs/selected_crop_dataset.yaml` and writes runs below
`runs/ai_validation`. Ultralytics reports validation precision, recall, and
mAP; use field extraction tests as the final gate because a missed crop
protection near a tool point matters more than a neat desktop metric.

## Field Robustness Run

Validate the same model on sunlit and shadowed sequences:

```powershell
python -m scripts.run_detection --model runs/ai_validation/selected_crop_recall/weights/best.pt --source sunlight.mp4 --conf 0.25 --guard-px 32 --save-dir runs/sunlight_overlay
python -m scripts.run_detection --model runs/ai_validation/selected_crop_recall/weights/best.pt --source shadow.mp4 --conf 0.25 --guard-px 32 --save-dir runs/shadow_overlay
```

The default inference policy is intentionally recall-first: low-confidence crop
detections still protect the crop, and a 32-pixel guard margin suppresses nearby
`not_crop` working points. Raise `--conf` only after reviewing saved overlays and
confirming that missed crops are no longer present in the field camera views.

The overlay shows:

- model boxes or masks from YOLO
- the lower camera working region
- four extractor zones
- red `~crop` working-point marks
- brightness, shadow, highlight, and blur-derived lighting tags

JSON lines printed to the terminal expose complement label, pixel center, and
extraction zone for later controller integration.

## Targeting Boundary

`ai_validation/targeting.py` is the software boundary between desktop AI
validation and actuator timing. It accepts selected-crop detections and emits
one `not_crop` validation point per extractor zone when that working point is
outside crop bounding boxes and the crop guard margin.

It does not actuate servos yet. Camera calibration, vehicle speed, camera-to-tool
offset, and extraction delay must be measured on the single-module MVP before a
target event is sent to Arduino control code.
