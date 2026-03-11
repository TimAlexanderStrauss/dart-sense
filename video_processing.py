from ultralytics import YOLO
from get_scores import GetScores
from camera_config import CameraSource, CameraManager

import numpy as np
import cv2
import time
import threading
import queue


class CameraThread(threading.Thread):
    """Background thread that continuously captures frames from a camera source.

    Frames are placed into a bounded :class:`queue.Queue`.  When the queue is
    full the oldest frame is dropped so that consumers always receive the most
    recent frame, keeping end-to-end latency low.
    """

    def __init__(self, camera_source, max_queue_size=2):
        super().__init__(daemon=True)
        self.camera_source = camera_source
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_source.get_cv2_source())
        while self.running:
            ret, frame = cap.read()
            if ret:
                # Drop the oldest frame when the queue is full to minimise latency
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
        cap.release()

    def stop(self):
        self.running = False

    def get_latest_frame(self, timeout=0.1):
        """Return the latest captured frame, or *None* if none is available."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class VideoProcessing:
    def __init__(self, model_dir="weights.pt"):
        self.model = YOLO(model_dir)
        self.predict = GetScores(model_dir)
    
    def _distance(self, coord1, coord2):
        return np.sqrt(np.sum((coord1 - coord2) ** 2))

    def _assess_visit(self, darts):
        darts = [dart for dart in darts if dart != '']
        score=0
        for dart in darts:
            score += self.scorer.get_score_for_dart(dart)
        
        remaining = self.scorer.scores[self.scorer.current_player] - score

        if remaining <= 1 or len(darts) == 3:
            if self.wait_for_dart_removal == False:
                self.scorer.read_score(score)

            self.wait_for_dart_removal = True
        else:
            self.wait_for_dart_removal = False

        if (remaining == 0 and darts[-1][0] != 'D') or remaining == 1 or remaining < 0:
            remaining = 'BUST'
        
        return score, remaining


    def _commit_score(self):
        self.scorer.commit_score([dart for dart in self.darts_in_visit if dart != ''])
        self.dart_coords_in_visit, self.darts_in_visit = [], ['']*3
        self.user_calibration = -np.ones((6, 2))
        self.wait_for_dart_removal = False
        self.pred_queue = -np.ones((5,3,2))
        self.pred_queue_count = 0


    def _adjust_coords(self, calibration_coords, dart_coords, resolution, crop_start, crop_size):
        # needed in order to adjust the coords for the square crop
        calibration_coords *= resolution # get pixel coords
        calibration_coords -= crop_start # adjust pixel coords for square crop
        calibration_coords /= crop_size # convert back to normalised coords
        if dart_coords.shape != (0,): # do same for darts
            dart_coords *= resolution
            dart_coords -= crop_start
            dart_coords /= crop_size
            dart_coords = dart_coords[np.all(np.logical_and(dart_coords>=0, dart_coords<=1), axis=1)] # remove any dart points detected outside of square crop
        
        return calibration_coords, dart_coords

    def _process_predictions(self, transformed_dart_coords, repeat_threshold):
        if len(transformed_dart_coords) == 0:
            self.pred_queue[self.pred_queue_count % 5] = -np.ones((3, 2))
        else:
            self.pred_queue[self.pred_queue_count % 5] = np.vstack((transformed_dart_coords, -np.ones((3-len(transformed_dart_coords), 2)))) # add [-1, -1] to fill any spaces when < 3 darts
        self.pred_queue_count += 1

        if self.wait_for_dart_removal:
            count = 0
            for frame in self.pred_queue:
                if np.all(frame == -1):
                    count += 1
            if count >= repeat_threshold:
                self._commit_score()
        
        elif self.darts_in_visit.count('') > 0:
            # check based on number of darts in visit and if the dart has been scored before
            unique_predictions = np.unique(self.pred_queue[self.pred_queue != -1].reshape(-1,2), axis=0)
            matches = {tuple(pred): [] for pred in unique_predictions} # for grouping together all similar predictions
            
            for frame in self.pred_queue:
                for pred in frame:
                    if np.any(pred == -1):
                        continue
                    for unique_pred in unique_predictions:
                        if self._distance(pred, unique_pred) < 0.01: # assume same prediction if distance < 0.01
                            matches[tuple(unique_pred)].append(pred)
                            break
            # sort dictionary based on length of values lists
            matches = {k: v for k, v in sorted(matches.items(), key=lambda item: len(item[1]), reverse=True) if len(v) >= repeat_threshold}
            best_predictions = []
            for _, match_ in matches.items():
                best_predictions.append(np.mean(match_, axis=0))
            
            if len(self.dart_coords_in_visit) == 0:
                self.dart_coords_in_visit = [pred for pred in best_predictions[:3]]
            
            else:
                for best_pred in best_predictions:
                    if all([self._distance(coords, best_pred) > 0.01 for coords in self.dart_coords_in_visit]):
                        if len(self.dart_coords_in_visit) == 3:
                            break
                        self.dart_coords_in_visit.append(best_pred)


    def _get_results_stream(self, camera_source):
        """Get YOLO results stream from a camera source.

        Supports both IP webcams (HTTP streams) and USB webcams (device indices).
        """
        if isinstance(camera_source, CameraSource):
            source = camera_source.get_yolo_source()
        elif isinstance(camera_source, int):
            # Legacy: direct device index for USB webcam
            source = camera_source
        elif isinstance(camera_source, str):
            # Legacy: IP address string
            source = 'http://' + camera_source + '/video'
        else:
            raise ValueError(f"Unsupported source type: {type(camera_source)}")

        return self.model(source, stream=True, verbose=False)


    def _process_camera_frame(self, result, resolution, crop_start, crop_size, user_calibration=None):
        """Process a single YOLO result for one camera frame.

        Returns a tuple ``(calibration_coords, dart_coords, dart_confs,
        H_matrix, transformed_darts)`` or *None* when not enough calibration
        points are detected.

        *dart_confs* is a float32 array of YOLO confidence scores, one per
        detected dart (up to 3), used for confidence-weighted fusion across
        cameras.
        """
        calibration_coords, dart_coords = self.predict.process_yolo_output(result)
        if np.count_nonzero(calibration_coords == -1) / 2 > 2:
            return None
        calibration_coords, dart_coords = self._adjust_coords(
            calibration_coords, dart_coords, resolution, crop_start, crop_size)
        if user_calibration is not None:
            calibration_coords = np.where(user_calibration == -1, calibration_coords, user_calibration)
        H_matrix = self.predict.find_homography(calibration_coords, crop_size)
        transformed_darts = self.predict.transform_to_boardplane(H_matrix[0], dart_coords, crop_size)

        # Collect YOLO confidence scores for the detected darts
        dart_confs = []
        for i, cls in enumerate(result.boxes.cls):
            if int(cls) == 4 and len(dart_confs) < 3:
                dart_confs.append(float(result.boxes.conf[i]))

        return calibration_coords, dart_coords, np.array(dart_confs, dtype=np.float32), H_matrix, transformed_darts


    def _fuse_detections_confidence_weighted(self, detections_per_camera, cluster_tolerance=0.05):
        """Fuse dart detections from multiple cameras using confidence weighting.

        Detections that are spatially close (within *cluster_tolerance* in
        normalised board-plane coordinates) are grouped together.  The final
        position for each group is the confidence-weighted average of all
        member detections.  Darts seen by only a single camera are still
        included, providing natural occlusion recovery.

        Args:
            detections_per_camera: list of ``(transformed_darts, dart_confs)``
                tuples, one entry per camera.  *transformed_darts* is a numpy
                array of shape ``(N, 2)`` and *dart_confs* is a float array of
                length ``N``.
            cluster_tolerance: maximum Euclidean distance (normalised) between
                two detections to be treated as the same dart.

        Returns:
            numpy array of up to 3 fused dart positions in board-plane
            coordinates, shape ``(M, 2)`` where ``M <= 3``.
        """
        all_darts = []
        for transformed_darts, dart_confs in detections_per_camera:
            if len(transformed_darts) > 0:
                # Use available confidence values; pad with 1.0 for any missing entries
                n = len(transformed_darts)
                if len(dart_confs) >= n:
                    confs = dart_confs[:n]
                else:
                    confs = np.ones(n)
                    confs[:len(dart_confs)] = dart_confs
                for coord, conf in zip(transformed_darts, confs):
                    all_darts.append((np.asarray(coord, dtype=np.float64), float(conf)))

        if not all_darts:
            return np.array([])

        coords = np.array([d[0] for d in all_darts])
        confs = np.array([d[1] for d in all_darts])

        # Greedy clustering: group nearby detections together
        used = np.zeros(len(coords), dtype=bool)
        clusters = []
        for i in range(len(coords)):
            if used[i]:
                continue
            cluster_idx = [i]
            used[i] = True
            for j in range(i + 1, len(coords)):
                if not used[j] and self._distance(coords[i], coords[j]) < cluster_tolerance:
                    cluster_idx.append(j)
                    used[j] = True
            clusters.append(cluster_idx)

        # Compute confidence-weighted average position for each cluster
        fused = []
        for idx_list in clusters:
            c = coords[idx_list]
            w = confs[idx_list]
            total = w.sum()
            # total is always > 0 (YOLO confidences are in (0, 1] and fallbacks are 1.0)
            pos = (c * w[:, np.newaxis]).sum(axis=0) / total
            fused.append(pos)

        return np.array(fused[:3])


    def _start_multi_camera_loop(self, GUI, camera_manager, repeat_threshold):
        """Processing loop for multiple cameras with confidence-weighted fusion.

        One :class:`CameraThread` is started per configured camera.  On each
        iteration the latest frame from every camera is collected, batch YOLO
        inference is run for efficiency, and detections from all cameras are
        fused via :meth:`_fuse_detections_confidence_weighted`.  The primary
        camera (index 0) supplies the frame shown in the GUI.
        """
        # Prepare per-camera crop metadata
        cam_meta = []
        for cam in camera_manager.cameras:
            res = cam.resolution
            crop_size = min(res)
            crop_start = res / 2 - crop_size / 2
            cam_meta.append({'camera': cam, 'resolution': res,
                              'crop_size': crop_size, 'crop_start': crop_start})

        # Start one capture thread per camera
        threads = [CameraThread(m['camera']) for m in cam_meta]
        for t in threads:
            t.start()

        prev_frame_time = 0
        try:
            while not self.game_over:
                # Gather the latest frame from every camera
                frames_with_meta = []
                for i, (t, meta) in enumerate(zip(threads, cam_meta)):
                    frame = t.get_latest_frame(timeout=0.05)
                    if frame is not None:
                        frames_with_meta.append((frame, meta, i))

                if not frames_with_meta:
                    continue

                # Batch YOLO inference for efficiency
                frames = [item[0] for item in frames_with_meta]
                all_results = self.model.predict(frames, verbose=False)

                camera_detections = []  # (transformed_darts, dart_confs) per camera
                primary_result = None
                primary_H = primary_crop_start = primary_crop_size = None
                primary_calibration = primary_dart_coords = None

                for result, (frame, meta, cam_idx) in zip(all_results, frames_with_meta):
                    # User calibration corrections apply to the primary camera only
                    user_cal = self.user_calibration if cam_idx == 0 else -np.ones((6, 2))
                    processed = self._process_camera_frame(
                        result, meta['resolution'], meta['crop_start'],
                        meta['crop_size'], user_cal)

                    if processed is None:
                        camera_detections.append((np.array([]), np.array([])))
                        continue

                    cal_coords, d_coords, d_confs, H_matrix, transformed = processed
                    camera_detections.append((transformed, d_confs))

                    if cam_idx == 0:
                        primary_result = result
                        primary_H = H_matrix
                        primary_crop_start = meta['crop_start']
                        primary_crop_size = meta['crop_size']
                        primary_calibration = cal_coords
                        primary_dart_coords = d_coords

                if primary_result is None:
                    continue

                # Fuse detections from all cameras using confidence weighting
                fused = self._fuse_detections_confidence_weighted(camera_detections)
                self._process_predictions(fused, repeat_threshold)

                self.darts_in_visit, score = self.predict.score(np.array(self.dart_coords_in_visit))
                while len(self.darts_in_visit) < 3:
                    self.darts_in_visit.append('')
                score, remaining = self._assess_visit(self.darts_in_visit)

                new_frame_time = time.time()
                fps = round(1 / (new_frame_time - prev_frame_time), 1) if prev_frame_time > 0 else 0.0
                prev_frame_time = new_frame_time

                GUI._display_graphics(primary_result, primary_H, primary_crop_start,
                                      primary_crop_size, primary_calibration,
                                      primary_dart_coords, score, remaining, fps)
        finally:
            for t in threads:
                t.stop()


    def start(self, GUI, source, scorer, resolution: np.array = None):
        """Start the dart detection and scoring loop.

        *source* may be a :class:`~camera_config.CameraManager` (preferred),
        a :class:`~camera_config.CameraSource`, an ``int`` device index, or a
        legacy IP-address ``str``.  When a ``CameraManager`` with more than one
        camera is supplied the multi-camera confidence-weighted fusion path is
        used; otherwise the original single-camera YOLO streaming path is kept.
        """
        self.scorer = scorer
        self.num_corrections = 0

        # Normalise *source* to a CameraManager for uniform handling
        if isinstance(source, CameraManager):
            camera_manager = source
        else:
            camera_manager = CameraManager()
            if isinstance(source, CameraSource):
                camera_manager.add_camera(source)
            elif isinstance(source, int):
                cam = CameraSource(source_type='usb', device_index=source,
                                   resolution=resolution or np.array((1200, 1600)))
                camera_manager.add_camera(cam)
            else:
                cam = CameraSource(source_type='ip', address=source,
                                   resolution=resolution or np.array((1200, 1600)))
                camera_manager.add_camera(cam)

        self.dart_coords_in_visit, self.darts_in_visit = [], [''] * 3
        self.user_calibration = -np.ones((6, 2))
        self.wait_for_dart_removal = False
        self.game_over = False
        self.pred_queue = -np.ones((5, 3, 2))  # FIFO queue for last 5 frames' predictions
        self.pred_queue_count = 0
        repeat_threshold = 3  # frames a dart must be seen before committing

        if camera_manager.num_cameras() > 1:
            self._start_multi_camera_loop(GUI, camera_manager, repeat_threshold)
        else:
            # Single camera: use the original YOLO streaming approach
            primary = camera_manager.get_primary_camera()
            res = primary.resolution if resolution is None else resolution
            crop_size = min(res)
            crop_start = res / 2 - crop_size / 2

            prev_frame_time = 0
            new_frame_time = 0

            results = self._get_results_stream(primary)

            for result in results:
                if self.game_over:
                    break

                calibration_coords, dart_coords = self.predict.process_yolo_output(result)
                if np.count_nonzero(calibration_coords == -1) / 2 > 2:
                    continue
                calibration_coords, dart_coords = self._adjust_coords(
                    calibration_coords, dart_coords, res, crop_start, crop_size)
                calibration_coords = np.where(self.user_calibration == -1, calibration_coords, self.user_calibration)

                H_matrix = self.predict.find_homography(calibration_coords, crop_size)
                transformed_dart_coords = self.predict.transform_to_boardplane(H_matrix[0], dart_coords, crop_size)

                self._process_predictions(transformed_dart_coords, repeat_threshold)

                self.darts_in_visit, score = self.predict.score(np.array(self.dart_coords_in_visit))
                while len(self.darts_in_visit) < 3:
                    self.darts_in_visit.append('')

                score, remaining = self._assess_visit(self.darts_in_visit)

                new_frame_time = time.time()
                fps = round(1 / (new_frame_time - prev_frame_time), 1) if prev_frame_time > 0 else 0.0
                prev_frame_time = new_frame_time

                GUI._display_graphics(result, H_matrix, crop_start, crop_size,
                                      calibration_coords, dart_coords, score, remaining, fps)

        print(f'Number of user corrections: {self.num_corrections}')
        print(f'Number of darts thrown: {np.sum(self.scorer.num_dart_history)}')

if __name__ == "__main__":
    pass
    #game = GameLogic(ruleset='x01', player_names=['Ben'], x01=1001, num_legs=1)
    #AI_scorer = VideoDetection()
    #AI_scorer.start("192.168.0.68:8080", game, np.array((1200, 1600)))
    #AI_scorer.start("external", game)
    #AI_scorer.start("webcam", game)