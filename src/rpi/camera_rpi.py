"""
    Encapsulates camera operations using `libcamera` via `picamera2`, providing
    a drop-in replacement for OpenCV-based camera handling on Raspberry Pi.
"""
try:
    from picamera2 import Picamera2
except Exception as e:
    print("Warning: picamera2 module not found. Camera feed will not work.")
    print("If running on Windows, be sure to use `camera_opencv.py` instead.")
    raise e
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import Optional, Callable, Tuple
import time
from threading import Thread, Event


class CameraFeedRpi:
    def __init__(self,
                 camera_port: int = 0,
                 max_dim_length: int = 720,
                 resize_dims: Optional[Tuple[int, int]] = None,
                 window_name: str = "Live Feed"):
        """ Camera feed context manager using `libcamera` for capturing and processing video frames.
            :param max_dim_length: Maximum size for the longest dimension of the resized frame.
            :param resize_dims: Explicit resize dimensions (width, height). If None, defaults will be used.
            :param window_name: The title of the live video feed window.
        """
        self.max_dim_length = max_dim_length
        self.resize_dims = resize_dims  # Will be set dynamically in __enter__ if None
        self.window_name = window_name
        self.picam2 = None
        self.last_frame = None  # Store last grabbed frame so external code can access
        self.consecutive_failures = 0

    def __enter__(self):
        """ Initialize the camera on context manager entry. """
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1)  # Allow the camera to warm up
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Close the camera feed when context exits. """
        self.cleanup()

    def _close_feed(self):
        """ Release the camera when done. """
        if self.picam2:
            self.picam2.close()
            self.picam2 = None

    def cleanup(self, stop_event: Event = None):
        """ Cleanup the camera feed, destroy all windows, set stop event, etc. """
        if stop_event:
            stop_event.set()
        #self.picam2.stop_preview()
        plt.close()
        self._close_feed()

    def capture_frame(self) -> np.ndarray:
        """ Capture a frame using `libcamera` (picamera2). """
        try:
            #* REFERENCE: for other types of captures: https://github.com/raspberrypi/picamera2/tree/main/examples
            # may want to eventually implement a function using capture_stream or capture_to_buffer
            frame = self.picam2.capture_array("raw")
            return frame
        except Exception as e:
            print(f"[CAMERA] Capture error: {e}")
            return None

    def display_live_feed(self, stop_event: Event, render_delay: float = 0.1, use_plt=True):
        """ Opens a window with live video using `libcamera`. """
        print(f"[CAMERA] Starting live video. Press 'q' to quit.")
        try:
            #self.picam2.start_preview()
            while not stop_event.is_set():
                frame = self.capture_frame()
                if frame is None:
                    print("[CAMERA] Warning: Captured frame is None!")
                    continue
                plt.imshow(frame)
                plt.title(self.window_name)
                plt.pause(0.001)
                time.sleep(render_delay)
        except KeyboardInterrupt:
            print("[CAMERA] KeyboardInterrupt detected. Stopping live feed...")
            self.cleanup(stop_event)
            raise KeyboardInterrupt
        except Exception as e:
            print(f"[CAMERA] Error in live feed: {e}")
            self.cleanup(stop_event)
            raise e
        # finally:
        #     self.cleanup(stop_event)

    @staticmethod
    def convert_frame_to_bytes(frame: np.ndarray) -> bytes:
        """ Convert the frame to a format suitable for desktop transmission. """
        try:
            from PIL import Image
            from io import BytesIO
            img = Image.fromarray(frame)
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            return buffer.getvalue()
        except Exception as e:
            print(f"[CAMERA] Error converting frame to bytes: {e}")
            return b""
