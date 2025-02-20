"""
    Encapsulates camera operations, such as capturing frames and converting them to the appropriate format for sending to the desktop

    Camera feed class should have methods for capturing frames, converting their format, and closing the camera feed
"""

try:
    import cv2
except Exception as e:
    print("Warning: OpenCV not installed. To use motion detection, make sure you've properly configured OpenCV.")
# from typing import Union, Optional, Tuple, List
from threading import Thread
# local imports
from ..utils import get_scaled_dims





class CameraFeed:
    def __init__(self, camera_port=0, max_dim_length=720):
        """
            :param camera_port: Camera port index for OpenCV.
            :param max_dim_length: Maximum size for the longest dimension of the resized frame.
        """
        self.camera_port = camera_port
        self.max_dim_length = max_dim_length
        self.capture = None
        self.resize_dims = None  # Will be set dynamically in __enter__

    def __enter__(self):
        self.capture = cv2.VideoCapture(self.camera_port)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open camera on port {self.camera_port}")
        # Compute resize dimensions dynamically if not already set
        if self.resize_dims is None:
            self.set_resize_dims()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Close the video capture context manager """
        self._close_feed()

    def set_resize_dims(self):
        """ Get initial frame dimensions from the camera to set the resize dimensions """
        # ret, frame = self.capture.read()
        # if not ret:
        #     raise RuntimeError("Failed to capture frame during initialization.")
        #H, W = frame[:2]
        H, W = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT), self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.resize_dims = get_scaled_dims((H, W), self.max_dim_length)

    def capture_frame(self):
        """ Capture a single frame from the camera """
        if self.capture is None:
            raise RuntimeError("Camera is not initialized. Use with `CameraFeed` context manager.")
        ret, frame = self.capture.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera.")
        return frame

    def convert_frame(self, frame):
        """ Convert the frame to a format suitable for desktop transmission """
        # TODO: add resizing logic later
        _, encoded_frame = cv2.imencode('.jpg', frame)
        return encoded_frame.tobytes()

    def _close_feed(self):
        """ Release the video capture and close all windows """
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()

    def display_live_feed(self, window_name="Live Feed", fps=30):
        """ Opens a window with live video. """
        print(f"Starting live video. Press 'q' to quit.")
        def run_feed():
            while True:
                frame = self.capture_frame()
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1000 // fps) & 0xFF == ord('q'): #checking for a q key press every 100 ms to break the video loop.
                    break
            self._close_feed()
        Thread(target=run_feed, daemon=True).start()


class CalibrationCameraFeed(CameraFeed):
    def __init__(self, camera_port=0, max_dim_length=720, calibration_overlay=None):
        super().__init__(camera_port, max_dim_length)
        self.calibration_overlay = calibration_overlay  # Optional overlay for guiding turret calibration

    def show_calibration_feed(self, window_name="Calibration Feed", fps=30):
        """Display the camera feed with calibration overlay."""
        print(f"Starting live video in 'calibration' mode. Press 'q' to quit.")
        while True:
            frame = self.capture_frame()
            # Add calibration overlay if provided
            if self.calibration_overlay:
                frame = self.calibration_overlay(frame)
            cv2.imshow(window_name, frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
                break
        self._close_feed()
