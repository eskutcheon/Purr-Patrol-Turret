"""
    Encapsulates camera operations, such as capturing frames and converting them to the appropriate format for sending to the desktop

    Camera feed class should have methods for capturing frames, converting their format, and closing the camera feed
"""

try:
    import cv2
except Exception as e:
    print("Warning: OpenCV not installed. To use motion detection, make sure you've properly configured OpenCV.")
from typing import Union, Optional, Tuple, List, Callable
from threading import Thread
### planning to extract these to a new file for the calibration class later
# import os
import numpy as np
# from datetime import datetime
# from ..config import config as cfg
###
# local imports
from ..utils import get_scaled_dims




class CameraFeed:
    def __init__(self, camera_port: int = 0, max_dim_length: int = 720):
        """ Camera feed context manager for capturing and processing video frames that closes the camera feed when done
            :param camera_port: Camera port index for OpenCV.
            :param max_dim_length: Maximum size for the longest dimension of the resized frame.
        """
        self.camera_port = camera_port
        self.max_dim_length = max_dim_length
        self.capture = None
        self.resize_dims = None  # Will be set dynamically in __enter__
        self.last_frame = None  # store last grabbed frame so external code can access

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
        H, W = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT), self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.resize_dims = get_scaled_dims((H, W), self.max_dim_length)

    def capture_frame(self):
        """ Capture a single frame from the camera """
        if self.capture is None:
            raise RuntimeError("Camera is not initialized. Use with `CameraFeed` context manager.")
        ret, frame = self.capture.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera.")
        if self.resize_dims is not None:
            frame = cv2.resize(frame, (self.resize_dims[1], self.resize_dims[0]))
        return frame

    def convert_frame(self, frame: np.ndarray) -> bytes:
        """ Convert the frame to a format suitable for desktop transmission """
        # TODO: add resizing logic later
        _, encoded_frame = cv2.imencode('.jpg', frame)
        return encoded_frame.tobytes()

    def _close_feed(self):
        """ Release the video capture and close all windows """
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()

    def display_live_feed(self, window_name: str = "Live Feed", fps: int = 30, key_handler: Callable = None):
        """ Opens a window with live video. """
        print(f"Starting live video. Press 'q' to quit.")
        def feed_loop():
            while True:
                frame = self.capture_frame()
                self.last_frame = frame  # store the last frame for external access
                cv2.imshow(window_name, frame)
                keypress = cv2.waitKey(int(1000 / fps)) & 0xFF
                if keypress == ord('q'): # checking for a q key press every 100 ms to break the video loop
                    break
                if key_handler is not None and keypress != 255:
                    # pass the key to your external logic (spacebar, etc.)
                    key_handler(keypress)
            self._close_feed()
        # run the camera loop in a daemon thread so it won't block the main thread
        Thread(target=feed_loop, daemon=True).start()

    # TODO: add type hint for this eventually
    def finalize_calibration(self, calibrator):
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


# # TODO: separate this into a separate class for performing the actual calibration computation
# class CameraCalibrationFeed(CameraFeed):
#     """ Camera feed subclass for calibration. Captures images of a checkerboard whenever the user presses SPACE,
#         then runs the calibration solve once the feed ends (user presses 'q').
#     """
#     def __init__(self, camera_port: int = 0, max_dim_length: int = 720, checkerboard_size: Tuple[int] = (9, 6), save_dir: str="calibration_images"):
#         super().__init__(camera_port, max_dim_length)
#         # number of inner corners per chessboard row/column
#         self.checkerboard_size = checkerboard_size
#         self.save_dir = save_dir
#         os.makedirs(self.save_dir, exist_ok=True)
#         # For calibrateCamera, you typically need lists of 2D & 3D points:
#         self.objpoints = []  # 3d points in the real world (checkerboard plane)
#         self.imgpoints = []  # 2d points in the image plane (pixel coordinates)
#         # Precompute the “object points” for a canonical checkerboard
#         self._canonical_objp = self._create_canonical_obj_points(self.checkerboard_size)

#     def _create_canonical_obj_points(self, checkerboard_size: Tuple[int]) -> np.ndarray:
#         """ Create the canonical object points for a checkerboard of size=checkerboard_size.
#             e.g. for a 9x6 checkerboard, that’s 54 corners in a plane at z=0.
#         """
#         # Typically one squaresize=1.0 or in your real units. Adjust as needed.
#         squaresize = 1.0
#         w, h = checkerboard_size
#         objp = np.zeros((w * h, 3), np.float32)
#         # The 3D points are arranged in a grid: (x, y, 0)
#         objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
#         return objp * squaresize

#     def start_calibration_loop(self, window_name: str = "Calibration Feed", fps: int = 30):
#         """ Display the camera feed in a loop. The user can aim the turret via WASD in another thread
#             (CalibrationState), then press SPACE to capture the current frame, or 'q' to finish.
#         """
#         print(f"[CALIB] Starting calibration feed. Press SPACE to capture, 'q' to quit.")
#         while True:
#             frame = self.capture_frame()
#             # optionally resize if you like:
#             if self.resize_dims is not None:
#                 frame = cv2.resize(frame, (self.resize_dims[1], self.resize_dims[0]))
#             cv2.imshow(window_name, frame)
#             key = cv2.waitKey(1000 // fps) & 0xFF
#             #~~ pretty sure I could replace a lot of this with the `display_live_feed` method from CameraFeed if I abstract the key presses, but it doesn't make a thread - not sure if I should yet
#             if key == ord('q'):
#                 # Exit the loop => triggers __exit__ => final calibrate
#                 break
#             elif key == ord(' '):
#                 # Save this frame for calibration
#                 self._capture_checkerboard(frame)
#         # once this while loop ends, we return to the caller context
#         self._close_feed()

#     def _capture_checkerboard(self, frame):
#         """ Attempt to find the checkerboard corners in `frame`. If found, save corners to self.objpoints, self.imgpoints and write to disk """
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         found, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
#         if found:
#             # refine corner locations for subpixel accuracy
#             criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#             corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#             self.objpoints.append(self._canonical_objp)
#             self.imgpoints.append(corners_refined)
#             # Save image to disk with a timestamp
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#             filename = os.path.join(self.save_dir, f"calib_{timestamp}.png")
#             cv2.imwrite(filename, frame)
#             print(f"[CALIB] Captured checkerboard and saved {filename}")
#         else:
#             print("[CALIB] Checkerboard NOT detected. Please try again.")

#     def __exit__(self, exc_type, exc_value, exc_traceback):
#         """ When the calibration feed context is closed, attempt to run camera calibration if enough captures exist """
#         # NOTE: uses the superclass __enter__ to enter the context
#         self._close_feed()  # ensure feed is released
#         # require at least N frames for calibration
#         if len(self.objpoints) < 3:
#             print("[CALIB] Not enough checkerboard captures to run calibration.")
#             return
#         print("[CALIB] Running camera calibration solve...")
#         # run calibrateCamera
#         ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
#             self.objpoints,
#             self.imgpoints,
#             (self.resize_dims[1], self.resize_dims[0]),
#             None, None
#         )
#         if not ret:
#             print("[CALIB] CalibrateCamera failed to converge.")
#             return
#         self._save_calibration(cameraMatrix, distCoeffs)
#         print("[CALIB] Calibration complete!")

#     def _save_calibration(self, cameraMatrix, distCoeffs):
#         """ Write the new camera calibration data to calibration.json. You can add extrinsics (rvecs, tvecs) or other info as needed.
#         """
#         import json
#         focal_length = (cameraMatrix[0, 0], cameraMatrix[1, 1])
#         optical_center = (cameraMatrix[0, 2], cameraMatrix[1, 2])
#         radial = distCoeffs.ravel()[:2].tolist()  # up to k2 (or more if needed)
#         # etc. to match your existing calibration.json structure
#         data = {
#             "focal_length": focal_length,
#             "optical_center": optical_center,
#             "radial_distortion": radial,
#             "skew": 0.0,   # or cameraMatrix[0,1] if you want to store it
#             # ...
#         }
#         print(f"[CALIB] Saving updated calibration to {cfg.CALIBRATION_FILE}")
#         with open(cfg.CALIBRATION_FILE, "w") as f:
#             json.dump(data, f, indent=4)
