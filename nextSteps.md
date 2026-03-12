# Next Steps: Multi-Camera Dart Detection

## Current State

The Dart Sense application supports **IP webcams** (via HTTP stream) and **USB webcams** (via OpenCV device index).  The GUI configures up to **3 cameras** (each can be IP or USB).

**Multi-camera detection is now implemented** with the following features:

- **`CameraThread`** — a background daemon thread that continuously captures frames from a `CameraSource` via `cv2.VideoCapture`, keeping the main loop decoupled from slow cameras.
- **Batch YOLO inference** — all camera frames are passed together to `model.predict(frames)` in a single call for GPU efficiency.
- **Confidence-weighted fusion** — each camera's detections are transformed to the board plane via per-camera homography.  Detections within 5 % of the board diameter are clustered; each cluster's final position is the YOLO-confidence-weighted average of its members.
- **Occlusion recovery** — darts seen by only one camera are automatically included (natural consequence of the fusion strategy).
- **Auto-detect USB camera resolution** — `CameraSource.get_actual_resolution()` reads `CAP_PROP_FRAME_WIDTH/HEIGHT` from the device and falls back to the stored default for unavailable devices.
- **Single-camera backward compatibility** — when only one camera is configured, the original YOLO streaming path (`model(source, stream=True)`) is used unchanged.

---

## Implemented: Architecture Overview

```
                 ┌────────────┐
                 │  Camera 1  │──── CameraThread 1 ──┐
                 └────────────┘                      │
                 ┌────────────┐                      ├── model.predict(batch) ── YOLO Results
                 │  Camera 2  │──── CameraThread 2 ──┤                                │
                 └────────────┘                      │                                ▼
                 ┌────────────┐                      │              Per-camera homography + transform
                 │  Camera 3  │──── CameraThread 3 ──┘                                │
                 └────────────┘                                                       ▼
                                                            Confidence-weighted fusion (clustering)
                                                                                       │
                                                                                       ▼
                                                                         Score (existing pipeline)
```

---

## Remaining Roadmap

### 1. Per-Camera User Calibration

Currently, manual calibration-point corrections in the GUI (`user_calibration`) apply only to the **primary** camera.  For maximum accuracy with 2–3 cameras, each camera should have independent user-correctable calibration:

- Extend the GUI with a camera selector to choose which camera's calibration is being adjusted.
- Store `user_calibration` as a dict keyed by camera index in `VideoProcessing`.

### 2. Synchronized Frame Processing

Frames from different cameras are collected at the time of the main-loop iteration; they are not guaranteed to be perfectly synchronised in time.

- **Timestamp-based matching:** attach a `time.time()` timestamp to each frame in `CameraThread` and pair frames across cameras by closest timestamp.
- **Frame rate normalisation:** cameras may run at different FPS; nearest-frame or interpolation matching handles this.

### 3. GUI Enhancements for Multi-Camera

- **Camera switcher:** button or radio button in the game screen to select which camera's feed is shown in the canvas.
- **Camera health indicators:** show connection status and per-camera FPS (available from `CameraThread`).
- **Per-camera calibration controls:** see §1 above.

### 4. Advanced Fusion Strategies

- **Consensus voting:** require 2-of-3 cameras to agree before committing a detection (useful in noisy environments).
- **Triangulation:** with known camera poses (stereo calibration), compute 3-D dart positions and project back to the board plane — eliminates per-camera homography entirely.

### 5. Batch YOLO Inference Optimisation

- **GPU memory:** YOLOv8n is ~6 MB, so running 3 streams is feasible, but batching (already implemented) is more efficient.
- Consider upgrading to YOLOv11 or RT-DETR for better accuracy with similar speed.

### 6. Testing Multi-Camera Setup

- **Unit tests:** the core fusion logic (`_fuse_detections_confidence_weighted`, `CameraThread`) is already tested with mock data.
- **Integration tests:** verify USB camera detection and streaming on real hardware.
- **Accuracy tests:** compare single-camera vs. multi-camera accuracy on identical dart throws.
- **Latency tests:** measure end-to-end latency with 1, 2, and 3 cameras.

### 7. Recommended Physical Setup for 3 Cameras

```
        Camera 1 (top centre, ~30° downward)
              │
   Camera 2 ──┼── Camera 3
   (left,       (right,
    ~45° angle)  ~45° angle)
              │
         Dartboard
```

- **120° spacing** around the board provides maximum angular coverage and minimises mutual occlusion.
- **Height variation** (one camera above, two at board level) reduces vertical occlusion.
- **Distance:** 1–2 metres from the board works well for most USB webcams.

---

## Data Collection & Dataset Setup for Training

This section describes how to obtain, organise, and prepare the training data so that the training scripts (`training/train_docker.ps1`, `training/train_docker.sh`, `training/train_optimal.sh`) work out of the box.

### 1. Obtain the Data

The project was trained on ~24,000 images from two sources:

1. **McNally et al. (DeepDarts) — ~16,000 images (D1, D2)**
   - Paper: [DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping in Darts using a Single Camera](https://arxiv.org/abs/2105.09880)
   - Dataset download: [IEEE Dataport — DeepDarts Dataset](https://ieee-dataport.org/open-access/deepdarts-dataset)
   - Download **`images.zip`** (original) or **`cropped_images.zip`** (already 800×800).
   - These images form datasets **d1** and **d2** in the YAML configs.

2. **Self-collected data — ~8,000 images (D3–D7)**
   - Collected from YouTube videos and personal playing setups.
   - Various camera angles, dart types, board types, lighting conditions.

### 2. Required Directory Structure

The training YAML configs (`data/d1_to_d7.yaml`, `data/d1_to_d7.docker.yaml`) expect the following layout under `data/darts/`:

```
data/darts/
├── classes.txt
├── images/
│   ├── d1/
│   │   ├── train/        # ~75% of d1 images
│   │   ├── val/          # ~10% of d1 images
│   │   └── test/         # ~15% of d1 images
│   ├── d2/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── d3_sharpened/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── d4_sharpened/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── d5/
│   │   ├── val/          # D5 is only used for val/test (unseen data)
│   │   └── test/
│   ├── d6_resized/
│   │   └── completed/
│   │       ├── train/
│   │       └── test/
│   └── d7_resized/
│       └── completed/
│           ├── train/
│           └── test/
└── labels/
    ├── d1/
    │   ├── train/        # One .txt per image, same filename stem
    │   ├── val/
    │   └── test/
    ├── d2/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── d3_sharpened/
    │   ├── ...
    ...  (mirrors images/ structure exactly)
```

Each `labels/<dataset>/<split>/` directory must mirror the corresponding `images/` directory with one `.txt` label file per image.

### 3. YOLO Label Format

Each `.txt` label file contains one line per object:

```
<class_id> <center_x> <center_y> <width> <height>
```

- Coordinates are **normalised** (0–1) relative to image width/height.
- `class_id` mapping (7 classes):
  | ID | Name |
  |----|------|
  | 0  | 20 (calibration point) |
  | 1  | 3 (calibration point) |
  | 2  | 11 (calibration point) |
  | 3  | 6 (calibration point) |
  | 4  | dart |
  | 5  | 9 (calibration point) |
  | 6  | 15 (calibration point) |

- Bounding box size for calibration points and darts: `0.025 0.025` (2.5% of image).

Example label (`d1_IMG_1093.txt`):
```
4 0.456789 0.234567 0.025 0.025
4 0.678901 0.543210 0.025 0.025
0 0.312345 0.123456 0.025 0.025
1 0.698765 0.876543 0.025 0.025
2 0.234567 0.765432 0.025 0.025
3 0.789012 0.345678 0.025 0.025
```

### 4. Pre-Processing Steps

Use the functions in `prepare_data.py`:

```python
from prepare_data import resize_images, sharpen_images, split_dataset, change_bb_size

# 1. Resize all images to 800x800
resize_images("data/darts/images/d3", size=(800, 800))

# 2. Sharpen lower-resolution images (e.g. from video captures)
sharpen_images("d3")  # creates d3_sharpened in data/darts/images/

# 3. Standardise bounding box sizes
change_bb_size("d3_sharpened", bb_size=0.025)

# 4. Split into train/val/test (75/10/15)
split_dataset("d3_sharpened", val_frac=0.1, test_frac=0.15)
```

### 5. Labelling New Images

1. Install **LabelImg**: `pip install labelimg`
2. Open: `labelimg data/darts/images/<your_dataset> data/darts/labels/<your_dataset> data/darts/classes.txt`
3. Set save format to **YOLO**.
4. Draw bounding boxes for each dart tip and each visible calibration point.
5. After labelling, run `change_bb_size("<your_dataset>", bb_size=0.025)` to standardise box sizes.

**Tip:** Once you have a trained model, use it to pre-label new images and then manually correct the predictions. This is much faster than labelling from scratch.

### 6. Verify and Train

```powershell
# Check that all referenced directories exist
Get-ChildItem data/darts/images -Recurse -Directory | Select-Object FullName

# Quick Docker GPU check
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# Start training
.\training\train_docker.ps1
```

---

## Priority Order (Remaining Work)

1. **Dataset acquisition** (download McNally et al. from IEEE Dataport, collect additional data)
2. **Per-camera calibration UI** (accuracy improvement for multi-camera)
3. **Timestamp-based frame synchronisation** (correctness)
4. **Camera health / switcher UI** (usability)
5. **Triangulation** (optional: removes homography dependency)
6. **Advanced fusion options** (consensus voting)
7. **Model upgrade** (YOLOv11, RT-DETR)
