"""
    Encapsulates camera operations, such as capturing frames and converting them to the appropriate format for sending to the desktop

    Camera feed class should have methods for capturing frames, converting their format, and closing the camera feed
"""

import cv2
from typing import Optional, Callable
import time
from threading import Thread
import numpy as np
# local imports
from ..utils import get_scaled_dims
from ..config.types import CameraFeedType, CameraCalibratorType



class CameraFeed:
    def __init__(self, camera_port: int = 0, max_dim_length: int = 720) -> CameraFeedType:
        """ Camera feed context manager for capturing and processing video frames that closes the camera feed when done
            :param camera_port: Camera port index for OpenCV.
            :param max_dim_length: Maximum size for the longest dimension of the resized frame.
        """
        self.camera_port = camera_port
        self.max_dim_length = max_dim_length
        self.capture = None
        self.resize_dims = None  # Will be set dynamically in __enter__
        self.last_frame = None  # store last grabbed frame so external code can access

    def __enter__(self) -> CameraFeedType:
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
        H, W = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT), self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.resize_dims = get_scaled_dims((H, W), self.max_dim_length)

    def capture_frame(self) -> np.ndarray:
        """ Capture a single frame from the camera """
        if self.capture is None:
            raise RuntimeError("Camera is not initialized. Use with `CameraFeed` context manager.")
        ret, frame = self.capture.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera.")
        if self.resize_dims is not None:
            frame = cv2.resize(frame, (self.resize_dims[1], self.resize_dims[0]))
        return frame

    def convert_frame_to_bytes(self, frame: np.ndarray) -> bytes:
        """ Convert the frame to a format suitable for desktop transmission """
        # TODO: add resizing logic later
        _, encoded_frame = cv2.imencode('.jpg', frame)
        return encoded_frame.tobytes()

    def _close_feed(self):
        """ Release the video capture and close all windows """
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()

    def keypress_monitor(self, fps = 100, key_handler: Optional[Callable] = None):
        # NOTE: cv2.waitKey() is blocking and takes a millisecond argument
        keypress = cv2.waitKey(int(1000 / fps)) & 0xFF
        if keypress == ord('q'): # checking for a q key press every 100 ms to break the video loop
            return True
        if key_handler is not None and keypress != 255:
            # pass the key to your external logic (spacebar, etc.)
            key_handler(keypress)
        time.sleep(0.1)
        return False

    def display_live_feed(self, window_name: str = "Live Feed", fps: int = 30, key_handler: Optional[Callable] = None):
        """ Opens a window with live video. """
        print(f"Starting live video. Press 'q' to quit.")
        def feed_loop():
            while True:
                frame = self.capture_frame()
                self.last_frame = frame  # store the last frame for external access
                cv2.imshow(window_name, frame)
                break_flag = self.keypress_monitor(fps=fps, key_handler=key_handler)
                # TODO: need to add handling of KeyboardInterrupt being raised in other threads to exit then
                #~ IDEA: create threading.Thread subclass that allows interruptions by setting a shared flag to True
                #~ IDEA: use a queue to send messages from the main thread to the camera thread to break the loop
                    # would allow for more complex logic to be passed in, e.g. "stop the feed if motion is detected"
                    # and would allow passing keypresses from other threads to the camera thread in the case of calibration
                if break_flag:
                    break
            self._close_feed()
        # run the camera loop in a daemon thread so it won't block the main thread
        Thread(target=feed_loop, daemon=True).start()

    # TODO: add type hint for this eventually
    def finalize_calibration(self, calibrator: CameraCalibratorType) -> Callable:
        """ return a callable that runs after the Calibration state finishes to finalize and save the results to disk """
        def finalize():
            # user presumably pressed 'q' in the feed to exit the loop or 'q' from the turret keyboard thread
            W, H = self.resize_dims[1], self.resize_dims[0]
            ret, mtx, dist, rvecs, tvecs, error = calibrator.run_calibration((W, H))
            if ret:
                calibrator.save_to_json("calibration.json", ret, mtx, dist, rvecs, tvecs, error)
            else:
                print("[CALIB] Calibration not successful or not enough images.")
        return finalize

