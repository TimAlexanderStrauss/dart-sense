import cv2
import numpy as np


class CameraSource:
    """Represents a single camera source, either USB or IP-based."""

    TYPES = ('usb', 'ip')

    def __init__(self, source_type='ip', address='192.168.0.56:8080', device_index=0,
                 resolution=np.array((1200, 1600)), label='Camera 1'):
        if source_type not in self.TYPES:
            raise ValueError(f"source_type must be one of {self.TYPES}, got '{source_type}'")
        self.source_type = source_type
        self.address = address
        self.device_index = device_index
        self.resolution = resolution
        self.label = label

    def get_yolo_source(self):
        """Return the source string/int suitable for YOLO model prediction."""
        if self.source_type == 'ip':
            return 'http://' + self.address + '/video'
        else:
            return self.device_index

    def get_cv2_source(self):
        """Return the source suitable for cv2.VideoCapture."""
        if self.source_type == 'ip':
            return 'http://' + self.address + '/video'
        else:
            return self.device_index

    def get_actual_resolution(self):
        """Return the actual camera resolution as ``np.array((width, height))``.

        For USB cameras, reads width and height directly from the device via
        ``cv2.CAP_PROP_FRAME_WIDTH`` / ``cv2.CAP_PROP_FRAME_HEIGHT``.  Falls
        back to the stored ``resolution`` attribute when the device cannot be
        opened or reports invalid values.

        The returned array uses ``(width, height)`` order, consistent with the
        default ``np.array((1200, 1600))`` used elsewhere in the codebase, where
        ``resolution[0]`` scales the horizontal (x) axis and ``resolution[1]``
        scales the vertical (y) axis.
        """
        if self.source_type == 'usb':
            cap = cv2.VideoCapture(self.device_index)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if width > 0 and height > 0:
                    return np.array((width, height))
        return self.resolution

    def __repr__(self):
        if self.source_type == 'ip':
            return f"CameraSource(type=ip, address={self.address}, label={self.label})"
        else:
            return f"CameraSource(type=usb, index={self.device_index}, label={self.label})"


class CameraManager:
    """Manages up to 3 camera sources for the dart detection system."""

    MAX_CAMERAS = 3

    def __init__(self):
        self.cameras = []

    def add_camera(self, camera):
        if len(self.cameras) >= self.MAX_CAMERAS:
            raise ValueError(f"Cannot add more than {self.MAX_CAMERAS} cameras")
        self.cameras.append(camera)

    def remove_camera(self, index):
        if 0 <= index < len(self.cameras):
            self.cameras.pop(index)

    def get_camera(self, index):
        if 0 <= index < len(self.cameras):
            return self.cameras[index]
        return None

    def get_primary_camera(self):
        """Return the first configured camera as primary."""
        if self.cameras:
            return self.cameras[0]
        return None

    def num_cameras(self):
        return len(self.cameras)

    @staticmethod
    def detect_usb_cameras(max_check=5):
        """Detect available USB cameras by probing device indices."""
        available = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
            if cap is not None:
                cap.release()
        return available
