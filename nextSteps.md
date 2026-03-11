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

## Priority Order (Remaining Work)

1. **Per-camera calibration UI** (accuracy improvement for multi-camera)
2. **Timestamp-based frame synchronisation** (correctness)
3. **Camera health / switcher UI** (usability)
4. **Triangulation** (optional: removes homography dependency)
5. **Advanced fusion options** (consensus voting)
6. **Model upgrade** (YOLOv11, RT-DETR)
