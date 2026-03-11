# Next Steps: Multi-Camera Dart Detection

## Current State

The Dart Sense application now supports both **IP webcams** (via HTTP stream) and **USB webcams** (via OpenCV device index). The GUI allows configuring up to **3 cameras** (each can be IP or USB). Currently, the **primary camera** (Camera 1) is used for dart detection.

The `CameraManager` class stores all configured camera sources, making it straightforward to extend the system to use multiple cameras simultaneously.

---

## Roadmap for Multi-Camera Support

### 1. Multi-Threaded Camera Capture

Currently, frame processing is single-threaded and sequential. To use multiple cameras simultaneously, each camera needs its own capture thread.

**Implementation approach:**
- Create a `CameraThread` class that inherits from `threading.Thread`
- Each thread continuously reads frames from its assigned `CameraSource` via `cv2.VideoCapture`
- Frames are placed into a thread-safe `queue.Queue` for the main processing loop to consume
- This decouples capture from processing and prevents slow cameras from blocking the pipeline

```python
import threading
import queue

class CameraThread(threading.Thread):
    def __init__(self, camera_source, frame_queue):
        super().__init__(daemon=True)
        self.camera_source = camera_source
        self.frame_queue = frame_queue
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_source.get_cv2_source())
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_queue.put((self.camera_source.label, frame))
        cap.release()
```

### 2. Multi-View Fusion Strategies

With 2–3 cameras viewing the dartboard from different angles, detection accuracy can be significantly improved. Below are key strategies:

#### a) Consensus Voting
- Run YOLO independently on each camera's frame
- Transform all dart coordinates to the board plane (using per-camera homography)
- If 2 out of 3 cameras agree on a dart position (within a tolerance), commit the detection
- This reduces false positives and compensates for occlusion

#### b) Confidence Weighting
- Weight each camera's detections by YOLO confidence scores
- Cameras with a clearer view (higher confidence) contribute more to the final prediction
- Weighted average of board-plane coordinates gives the final dart position

#### c) Occlusion Recovery
- The biggest failure case is dart occlusion (one dart blocking the view of another)
- With multiple cameras at different angles, each camera can "see around" different occlusions
- Union of detections across cameras: if camera A misses a dart but camera B detects it, include it

### 3. Camera Calibration

Each camera will have different viewing angles, distortions, and resolutions:

- **Per-camera homography:** Each camera needs its own set of calibration points and homography matrix
- **Cross-camera alignment:** After transforming to the board plane, coordinates from all cameras should align. Verify this during setup
- **Resolution handling:** USB webcams may have different resolutions than IP webcams. Ensure the `resolution` parameter is correctly set per camera (currently `1200×1600` is the default)
- **Auto-detect resolution:** For USB cameras, read the actual resolution from `cv2.VideoCapture` properties:
  ```python
  cap = cv2.VideoCapture(device_index)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  ```

### 4. Synchronized Frame Processing

For accurate multi-view fusion, frames from different cameras need to be approximately synchronized:

- **Timestamp-based matching:** Pair frames across cameras by closest timestamp
- **Frame rate normalization:** Cameras may run at different FPS; use interpolation or nearest-frame matching
- **Software sync:** Use threading barriers or events to align capture cycles

### 5. GUI Enhancements for Multi-Camera

- **Multi-view display:** Show each camera's feed in a separate panel or allow toggling between cameras
- **Per-camera calibration controls:** Allow adjusting calibration points independently per camera
- **Camera health indicators:** Show connection status (connected/disconnected) and FPS for each camera
- **Primary camera selector:** Let users switch which camera is the active primary

### 6. Architecture Changes

The main processing loop in `video_processing.py` would need restructuring:

```
                 ┌────────────┐
                 │  Camera 1  │──── Thread 1 ──── YOLO ──── Detections 1 ─┐
                 └────────────┘                                            │
                 ┌────────────┐                                            │
                 │  Camera 2  │──── Thread 2 ──── YOLO ──── Detections 2 ──┼── Fusion ── Score
                 └────────────┘                                            │
                 ┌────────────┐                                            │
                 │  Camera 3  │──── Thread 3 ──── YOLO ──── Detections 3 ─┘
                 └────────────┘
```

**Key decisions:**
- **One YOLO model vs. multiple:** Running 3 separate YOLO inference streams may be GPU-limited. Consider batching frames from all cameras and running YOLO once on the batch
- **YOLO batch inference:** `model.predict(batch_of_frames)` can process multiple images more efficiently than sequential calls
- **GPU memory:** YOLOv8n is small (~6MB), so 3 instances are feasible, but batching is more efficient

### 7. Testing Multi-Camera Setup

- **Unit tests:** Test `CameraSource` and `CameraManager` classes with mock data
- **Integration tests:** Verify that USB cameras are correctly detected and can stream frames
- **Accuracy tests:** Compare single-camera vs. multi-camera accuracy on the same dart throws
- **Latency tests:** Measure end-to-end latency with 1, 2, and 3 cameras to ensure real-time performance

### 8. Recommended Physical Setup for 3 Cameras

For optimal coverage with minimal occlusion:

```
        Camera 1 (top center, ~30° downward)
              │
   Camera 2 ──┼── Camera 3
   (left,       (right,
    ~45° angle)  ~45° angle)
              │
         Dartboard
```

- **120° spacing** around the board provides maximum angular coverage
- **Height variation** (one camera above, two at board level) reduces vertical occlusion
- **Distance:** 1–2 meters from the board works well for most webcams

---

## Priority Order

1. **Multi-threaded capture** (essential for multi-camera)
2. **Per-camera calibration and homography**
3. **Consensus voting fusion** (simplest, most effective improvement)
4. **Auto-detect USB camera resolution**
5. **GUI multi-view display**
6. **Advanced fusion strategies** (confidence weighting, occlusion recovery)
7. **Batch YOLO inference optimization**
