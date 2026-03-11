# Project Overview: Dart Sense

## Project Description

Dart Sense is an automatic dart scoring application built with Python. It uses a YOLOv8 object detection model to track dart landing positions and board calibration points from a live camera feed, then computes scores in real time. The system supports full game logic (x01, 121 rulesets), multi-player games, and provides a Tkinter-based graphical user interface.

---

## Architecture & File Structure

```
dart-sense/
├── GUI.py                      # Main Tkinter GUI application
├── GUI_colour_scores.py        # Alternative GUI using colour-based scoring
├── video_processing.py         # Video stream handling & YOLO inference loop
├── get_scores.py               # Dart scoring logic & board transformation (homography)
├── game_logic.py               # Game rules, turn management, score tracking
├── camera_config.py            # Camera source abstraction (IP + USB support)
├── prepare_data.py             # Data preprocessing (resize, sharpen, split)
├── accuracy.py                 # Accuracy evaluation metrics
├── requirements.txt            # Python dependencies (conda format)
├── data/
│   ├── darts/classes.txt       # YOLO class definitions
│   ├── d1_to_d7.yaml           # YOLO dataset config (7 classes)
│   └── d1234_sharp.yaml        # Alternative dataset config
├── training/
│   └── train_optimal.sh        # Training script with tuned hyperparameters
├── images/                     # Documentation images & sample data
└── other_version/
    └── predict_v2.py           # Alternative prediction approach
```

### Core Processing Pipeline

**Single camera (1 camera configured):**
```
Camera Stream → YOLO Streaming → Coordinate Extraction → Homography → Score Calculation
```

**Multi-camera (2–3 cameras configured):**
```
CameraThread 1 ──┐
CameraThread 2 ──┼── model.predict(batch) ── Per-camera homography + transform
CameraThread 3 ──┘                                          │
                                          Confidence-weighted fusion (clustering)
                                                            │
                                                    Score Calculation
```

1. **Camera Input** (`video_processing.py`, `camera_config.py`): Captures frames from IP or USB cameras. USB resolution is auto-detected via `CameraSource.get_actual_resolution()`. Multi-camera uses `CameraThread` background threads.
2. **Object Detection** (`video_processing.py` + YOLO): Detects darts (class 4) and 6 calibration points (classes 0–3, 5–6). Multi-camera: batch inference via `model.predict(frames)`.
3. **Coordinate Extraction** (`get_scores.py:process_yolo_output`): Extracts normalized bounding box centers.
4. **Homography Transform** (`get_scores.py:find_homography`): Per-camera homography maps each camera's image coordinates to the board plane.
5. **Fusion** (`video_processing.py:_fuse_detections_confidence_weighted`): Clusters board-plane detections across cameras; final position is confidence-weighted average. Darts seen by only one camera are included (occlusion recovery).
6. **Score Calculation** (`get_scores.py:score`): Computes angle and distance from board centre to determine segment and scoring region.
7. **Prediction Smoothing** (`video_processing.py:_process_predictions`): 5-frame FIFO queue with repeat threshold of 3 for stable detections.

### GUI Architecture

- **Menu Screen**: Game setup (type, players, scores, camera configuration)
- **Game Screen**: Live camera feed, scorecard, interactive dart/calibration point manipulation
- **Display Modes**: Original image, transformed (board plane), and live (with YOLO annotations)

---

## Strengths

### 1. Robust Scoring via Homography
The calibration-point-based homography approach is a major strength. By detecting fixed board reference points and computing a perspective transform, the system works **regardless of camera angle, distance, or distortion**. This is a much more robust approach than fixed camera setups.

### 2. Well-Tuned Object Detection
- YOLOv8n model tuned over **134 genetic algorithm iterations** (25 epochs each)
- Custom hyperparameters for learning rate, momentum, augmentation
- Achieves strong metrics: **97.7% precision**, **94.2% recall**, **88.4% image correct score**
- Data augmentation (rotation, scaling, brightness) improves generalization

### 3. Prediction Smoothing
The 5-frame FIFO queue with repeat threshold prevents flickering detections. A dart must be consistently detected across 3+ frames before being committed, reducing false positives effectively.

### 4. Interactive User Corrections
Users can manually move dart and calibration point positions via click-and-drag on the canvas. This addresses the inevitable detection errors and provides a fallback.

### 5. Clean Separation of Concerns
- `get_scores.py` handles pure scoring math (board geometry, homography, angles)
- `video_processing.py` handles streaming and prediction logic
- `game_logic.py` handles rules and turn management
- `GUI.py` handles display and user interaction

### 6. Data Collection Built In
The "Save data" feature exports images and labels in YOLO format, enabling continuous model improvement from real gameplay data.

---

## Weaknesses

### 1. Multi-Camera Architecture (partially addressed)
Multi-camera support is now implemented: `CameraThread` provides per-camera background capture, batch YOLO inference processes all frames together, and confidence-weighted fusion combines detections from all cameras in the board plane.  Remaining gaps: per-camera user calibration in the GUI, timestamp-based frame synchronisation, and a camera-switcher UI.

### 2. No Automated Tests
There are **no unit tests, integration tests, or test framework** for the main application.  Core multi-camera logic (`_fuse_detections_confidence_weighted`, `CameraSource.get_actual_resolution`) is covered by standalone tests.  A full test framework (pytest) would make refactoring safer.

### 3. USB Camera Resolution Auto-Detected
~~The camera resolution is hardcoded to `(1200, 1600)`.~~  USB cameras now auto-detect their actual resolution via `CameraSource.get_actual_resolution()` using `cv2.CAP_PROP_FRAME_WIDTH/HEIGHT`.  IP cameras still default to `(1200, 1600)`.

### 4. Tight Coupling Between GUI and Video Processing
`video_processing.py:start()` directly calls `GUI._display_graphics()` inside the processing loop. This makes it impossible to use the video processing module independently (e.g., for headless testing, multi-camera processing, or alternative UIs).

### 5. Blocking Main Thread
The video processing loop runs on the main thread, which blocks the Tkinter event loop. The `root.update()` call inside `_display_graphics` is a workaround but can cause UI freezing and responsiveness issues.

### 6. Limited Error Handling
- No handling for camera disconnection or stream interruption
- No validation of user inputs (e.g., invalid IP address, negative device index)
- `exit(0)` calls in `game_logic.py` abruptly terminate the application

### 7. Windows-Specific Dependencies
The requirements include `pypiwin32`, `pyreadline3`, and `comtypes` which are Windows-only. The path separator in `get_scores.py` uses `\\`. Cross-platform support is limited.

### 8. Outdated Python Version
The project targets Python 3.8, which reached end-of-life in October 2024. Modern YOLO versions and dependencies may require Python 3.9+.

### 9. Calibration Point Limitations
The system detects 6 calibration points (corners of segments 20, 3, 11, 6, 9, 15). If the camera cannot see at least 4 of these, the homography fails. With steep camera angles, some calibration points may be occluded.

---

## Optimization Opportunities for Dart Detection

### Model Improvements

#### 1. Upgrade to YOLOv8s or YOLOv8m
- Current: YOLOv8n (nano) — fastest but least accurate
- YOLOv8s (small) offers ~15% better mAP with acceptable speed (~50 FPS on modern GPUs)
- Worth testing if hardware supports it and real-time FPS remains above 15–20

#### 2. Increase Training Data Diversity
- Current: ~24K images from limited setups
- Add more lighting conditions, camera angles, dart types, and board styles
- Synthetic data generation could augment the dataset cheaply

#### 3. Keypoint Detection Instead of Bounding Boxes
- Instead of detecting bounding boxes and taking their center, train a keypoint detection model to directly predict dart tip locations
- YOLOv8 supports pose/keypoint estimation — this could be repurposed for dart tip detection
- More precise than box-center approximation, especially for angled darts

#### 4. Improve Small Object Detection
- Darts tips are very small objects relative to the full frame
- **SAHI (Slicing Aided Hyper Inference)**: Slice the image into overlapping tiles, run YOLO on each tile, and merge results. This significantly improves small object detection
- Available via the `sahi` Python library, compatible with ultralytics

### Processing Pipeline Improvements

#### 5. Temporal Consistency
- Use tracking algorithms (e.g., ByteTrack, BoT-SORT) to track darts across frames
- YOLOv8 has built-in tracking: `model.track(source, tracker='bytetrack.yaml')`
- This provides stable IDs for each dart across frames, preventing duplicate counts

#### 6. Region of Interest (ROI)
- Detect the dartboard area first, then run dart detection only within that region
- Reduces false positives from background objects and speeds up inference

#### 7. Better Confidence Calibration
- The current confidence threshold of 0.85 for calibration points may be too aggressive in poor lighting
- Implement adaptive thresholding based on detection statistics per session
- Log confidence distributions to identify optimal thresholds

---

## Current Trends in YOLO and Computer Vision (Relevant to This Project)

### YOLOv8 & Beyond

- **YOLOv9 (Feb 2024):** Introduces Programmable Gradient Information (PGI) and GELAN architecture. Better gradient flow leads to improved accuracy, especially on small objects
- **YOLOv10 (May 2024):** Eliminates NMS (Non-Maximum Suppression) post-processing with one-to-one assignment during training. Reduces latency by ~30% while maintaining accuracy
- **YOLOv11 (Sep 2024):** Built on ultralytics framework (same as v8). Improved C3k2 blocks and SPPF. Drop-in replacement for YOLOv8 with better accuracy/speed trade-offs
- **RT-DETR:** Real-time transformer-based detector from Baidu. Competes with YOLO on accuracy and can be better for scenes with many small objects. Available in ultralytics

### Relevant Computer Vision Concepts

#### Multi-View Geometry
- With multiple cameras, **triangulation** can provide 3D dart positions
- **Stereo calibration** between camera pairs allows depth estimation
- 3D detection eliminates the need for per-camera homography — compute once in 3D space

#### Self-Supervised Pre-Training
- Models pre-trained with self-supervised methods (DINOv2, MAE) on large unlabeled datasets show better fine-tuning performance on small domain-specific datasets
- Could improve dart detection with limited labeled data

#### Test-Time Augmentation (TTA)
- Apply augmentations (flips, scales) at inference time and merge predictions
- Available in YOLOv8: `model.predict(source, augment=True)`
- Can improve detection of occluded darts at the cost of ~2x inference time

#### Knowledge Distillation
- Train a larger teacher model (YOLOv8l) and distill its knowledge into the nano model
- Achieves teacher-level accuracy with student-level speed
- Supported in ultralytics via `model.train(distill=True)` (experimental)

#### Edge Deployment
- **ONNX/TensorRT export:** Convert the YOLOv8 model for faster inference on edge devices
- **NVIDIA Jetson:** Affordable GPU-enabled edge device ideal for real-time dart detection
- Already partially supported (the project includes onnx-related packages)

---

## Summary

Dart Sense is a well-conceived project with a solid foundation in computer vision and dart board geometry. The homography-based scoring approach is robust and elegant. The main areas for improvement are:

1. **Multi-camera support** — implemented: `CameraThread` (threaded capture), batch YOLO inference, and confidence-weighted fusion across up to 3 USB/IP cameras.  Remaining: per-camera calibration UI, frame synchronisation, camera switcher.
2. **Automated testing** (to enable confident refactoring)
3. **Decoupling video processing from the GUI** (for flexibility)
4. **Model upgrades** (newer YOLO versions, keypoint detection, SAHI)
5. **Tracking integration** (ByteTrack for temporal consistency)

The multi-camera implementation uses **confidence-weighted fusion**: YOLO detections from all cameras are transformed to the board plane via per-camera homography, clustered by proximity, and merged using confidence-score weights.  This naturally handles occlusion recovery (darts visible to only one camera are included) and noise rejection (cameras with blurry or partially-occluded views contribute less).  Single-camera usage retains the original YOLO streaming path unchanged.
