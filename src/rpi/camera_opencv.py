"""
    Encapsulates camera operations, such as capturing frames and converting them to the appropriate format for sending to the desktop
    Camera feed class should have methods for capturing frames, converting their format, and closing the camera feed
"""

import cv2
import matplotlib.pyplot as plt
from typing import Optional, Callable, Tuple
import time
from threading import Thread, Event
import numpy as np
# local imports
from ..utils import get_scaled_dims



class CameraFeedOpenCV:
    def __init__(self,
                 camera_port: int = 0,
                 max_dim_length: int = 720,
                 resize_dims: Optional[Tuple[int, int]] = None,
                 window_name: str = "Live Feed"):
        """ Camera feed context manager for capturing and processing video frames that closes the camera feed when done
            :param camera_port: Camera port index for OpenCV.
            :param max_dim_length: Maximum size for the longest dimension of the resized frame.
        """
        self.camera_port = camera_port
        self.max_dim_length = max_dim_length
        self.resize_dims = resize_dims  # Will be set dynamically in __enter__ if None
        self.window_name = window_name
        self.capture = None
        self.last_frame = None  # store last grabbed frame so external code can access
        self.consecutive_failures = 0

    def __enter__(self):
        self.capture = cv2.VideoCapture(self.camera_port)
        #self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # set buffer size to 1 to reduce latency
        self.capture.set(cv2.CAP_PROP_FPS, 10) # limit FPS to 10 to reduce load
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open camera on port {self.camera_port}")
        time.sleep(1) # adding a small delay to allow the camera to warm up
        # compute resize dimensions dynamically if not already set
        self.set_resize_dims(self.resize_dims)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Close the video capture context manager """
        self._close_feed()

    def _close_feed(self):
        """ release the video capture and close all windows when context manager exits """
        if self.capture:
            self.capture.release()
            self.capture = None
        cv2.destroyAllWindows()

    def set_resize_dims(self, resize_dims: Optional[Tuple[int, int]] = None):
        """ Get initial frame dimensions from the camera to set the resize dimensions """
        if resize_dims is None:
            H, W = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT), self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.resize_dims = get_scaled_dims((H, W), self.max_dim_length)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resize_dims[0])
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resize_dims[1])

    def at_max_capture_error(self, ret) -> bool:
        if not ret:
            if self.consecutive_failures > 5:
                print("[CAMERA] Too many consecutive failures. Exiting...")
                raise RuntimeError("Too many consecutive failures to capture frames.")
            print("[CAMERA] Failed to capture frame from camera. Skipping...")
            self.consecutive_failures += 1
            return True
        self.consecutive_failures = 0
        return False

    def _capture_frame(self) -> np.ndarray:
        """ Capture a single frame from the camera """
        if self.capture is None:
            raise RuntimeError("Camera is not initialized. Use with `CameraFeedOpenCV` context manager.")
        ret, self.last_frame = self.capture.read()
        while self.at_max_capture_error(ret):
            ret, self.last_frame = self.capture.read()
        # NOTE: cv2.imshow displays in BGR by default
        #self.last_frame = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        return self.last_frame

    def capture_frame(self) -> np.ndarray:
        """ Capture a single frame from the camera and return it as a numpy array """
        frame = self._capture_frame()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ###############################################################################################
    # change use of key_handler and even checking for keypresses to be in the controller
    # keep logic of monitoring for a stop flag in the camera feed

    def display_live_feed(self, stop_event: Event, render_delay: int = 0.1, use_plt=True):
        if use_plt:
            self._display_live_feed_plt(stop_event, render_delay)
        else:
            self._display_live_feed_opencv()

    def _display_live_feed_opencv(self, stop_event: Event, render_delay: int = 0.1):
        """ Opens a window with live video. """
        print(f"Starting live video. Press 'q' to quit.")
        # TODO: update this to be passed from the Controller as an argument and monitor it each iteration
        try:
            while not stop_event.is_set():
                frame = self._capture_frame()
                cv2.imshow(self.window_name, frame)
                cv2.waitKey(1) # 1 millisecond delay to allow for window events
                time.sleep(render_delay)
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Stopping live feed.")
            stop_event.set()
        except Exception as e:
            print(f"Error in live feed: {e}")
            raise e
        finally:
            self._close_feed()

    def _display_live_feed_plt(self, stop_event: Event, render_delay: int = 0.1):
        print(f"Starting live video feed (matplotlib). Press 'q' to quit.")
        # Create matplotlib figure
        fig, ax = plt.subplots()
        frame = self.capture_frame()
        img_display = ax.imshow(frame)
        with plt.ion():  # set interactive mode
            try:
                while not stop_event.is_set():
                    frame = self.capture_frame()
                    if frame is None:
                        continue
                    img_display.set_data(frame)  # Update frame data
                    plt.draw()
                    plt.pause(render_delay)  # Allow GUI events to process
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected. Stopping live feed.")
                stop_event.set()
            except Exception as e:
                print(f"Error in live feed: {e}")
                raise e
            finally:
                self._close_feed()
        #plt.ioff()  # Turn off interactive mode
        plt.close(fig)


    def convert_frame_to_bytes(self, frame: np.ndarray) -> bytes:
        """ Convert the frame to a format suitable for desktop transmission """
        # TODO: add resizing logic later
        _, encoded_frame = cv2.imencode('.jpg', frame)
        return encoded_frame.tobytes()
