# TODO: probably need to replace usage of this with `multiprocessing`
import threading
# using their recommendation here:
#https://github.com/adafruit/Adafruit_CircuitPython_MotorKit
from adafruit_motorkit import MotorKit
from config import *
from motion_tracking2 import MotionTracker
from object_detection import Detector
from turret import Turret


class TurretController:
    """ A class to control the turret in various operation modes.
        Modes:
        - Interactive: Direct control via keyboard input.
        - Motion Detection: Shoots at any detected motion.
        - Object Tracking: Shoots specific detected objects using an object detection neural network
    """

    def __init__(self, frame_update_freq=0.5, target_classes=None, friendly_mode=True):
        """ Initializes the turret controller.
            :param frame_update_freq: Frequency of frame updates in seconds.
            :param target_classes: List of object classes to track in Object Tracking mode.
            :param friendly_mode: Enables safety to prevent shooting in certain scenarios.
        """
        self.frame_update_freq = frame_update_freq
        self.target_classes = target_classes or []
        self.friendly_mode = friendly_mode
        # Components
        self.motion_tracker = MotionTracker()
        self.detector = Detector() if target_classes else None
        # Motors and Turret Components
        self.turret = Turret(friendly_mode=friendly_mode, specific_target=bool(target_classes))

    def interactive_mode(self):
        """ Runs the turret in interactive mode with keyboard input. """
        print('Commands: Pivot with (a) and (d). Tilt with (w) and (s). Fire with (Enter). Exit with (q)\n')
        self.turret.interactive()

    def motion_detection_mode(self, show_video=False):
        """ Runs the turret in motion detection mode. """
        print("Motion Detection Mode: Detecting and shooting motion.")
        self.turret.calibrate()
        self.motion_tracker.find_motion(callback=self.turret.move_axis, show_video=show_video)

    def object_tracking_mode(self):
        """ Runs the turret in object tracking mode using object detection. """
        print(f"Object Tracking Mode: Tracking objects {self.target_classes}.")
        self.turret.calibrate()
        while True:
            # Capture frame
            frame = self._capture_frame()
            # Detect objects
            detections = self.detector.detect(frame)
            # Check and fire if valid target detected
            should_fire, target_coord = self.turret.get_overlap(detections)
            if should_fire:
                self.turret.move_axis(None, frame)
            threading.sleep(self.frame_update_freq)

    def run(self, mode, show_video=False):
        """ Main entry point to run the turret in the specified mode.
            :param mode: The mode to run ('interactive', 'motion', 'tracking').
            :param show_video: Show video feed if available.
        """
        if mode == "interactive":
            self.interactive_mode()
        elif mode == "motion":
            self.motion_detection_mode(show_video)
        elif mode == "tracking":
            if not self.target_classes:
                print("Error: No target classes specified for object tracking mode.")
                return
            self.object_tracking_mode()
        else:
            print(f"Invalid mode: {mode}. Available modes are: 'interactive', 'motion', 'tracking'.")

    def _capture_frame(self):
        """ Captures a frame for object tracking.
            Replace this with your frame capture logic.
        """
        raise NotImplementedError("Frame capture logic should be implemented.")

    def configure(self, frame_update_freq=None, target_classes=None, friendly_mode=None):
        """ Configures the turret settings.
            :param frame_update_freq: Frequency of frame updates in seconds.
            :param target_classes: List of object classes to track.
            :param friendly_mode: Enables safety to prevent shooting in certain scenarios.
        """
        if frame_update_freq is not None:
            self.frame_update_freq = frame_update_freq
        if target_classes is not None:
            self.target_classes = target_classes
        if friendly_mode is not None:
            self.friendly_mode = friendly_mode