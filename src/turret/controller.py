from typing import List, Optional, Iterable
import time
# local imports
from src.config import config as cfg
from src.config.types import TurretControllerType, OperatorLike, StateLike, CommandLike, MotionDetectorType, DetectionPipelineType
from src.turret.state import IdleState, InteractiveState, CalibrationState, MotionTrackingOnlyState, MotionTrackingDetectionState
from src.rpi.camera import CameraFeed


class TurretController:
    """High-level control logic for the turret."""
    def __init__(self, operation: OperatorLike) -> TurretControllerType:
        self.operation = operation
        self.targeting_system = operation.targeting_system
        self.update_interval = cfg.MOTION_UPDATE_INTERVAL
        # TODO: integrate a "friendly" state object here that can't be overridden to avoid misfires
        self.current_state = IdleState()  # Start in Idle state
        self.command_queue: List[CommandLike] = []          # Queue for commands

    def set_state(self, state: StateLike):
        """ transition to the given state """
        self.current_state = state

    def handle_state(self, *args, **kwargs):
        """ delegate behavior to the current state - major method for handling state transitions and therefore actions """
        self.current_state.handle(self, *args, **kwargs)

    def execute_command(self, command: CommandLike):
        """ execute a single command through its internal execute() method """
        command.execute()

    def queue_command(self, command: CommandLike):
        """ add a command to the queue """
        self.command_queue.append(command)

    def process_commands(self):
        """ process all queued commands until the queue is empty """
        while self.command_queue:
            command = self.command_queue.pop(0)
            command.execute()

    def queue_then_process_commands(self, *commands: Iterable[CommandLike]):
        """ queue a command and immediately process the queue """
        for command in commands:
            self.queue_command(command)
        self.process_commands()


    def enter_interactive_mode(self, show_video: bool = False):
        """ launch the turret in interactive mode, with an optional live video feed and WASD controls """
        self.set_state(InteractiveState())
        if not show_video:
            # Immediately handle the new state
            self.handle_state(callback=self.operation.cleanup)
        else:
            with CameraFeed(cfg.CAMERA_PORT, max_dim_length=1080) as live_feed:
                live_feed.display_live_feed()
                self.handle_state(callback=self.operation.cleanup)

    def enter_calibration_mode(self):
        """ launch the turret in calibration mode, working like interactive mode but with a different state object and space bar response """
        from src.rpi.calibration import CameraCalibrator
        self.set_state(CalibrationState())
        calibrator = CameraCalibrator(checkerboard_size=cfg.CHECKERBOARD_SIZE, square_size=cfg.SQUARE_SIZE)
        # We'll show the same “live feed” but in calibration flavor:
        with CameraFeed(cfg.CAMERA_PORT, max_dim_length=1080) as feed:
            #! FIXME: to recognize keyboard input to capture frames, the window focus seems to need to be on the live feed window
                # because of the way that I set up `execute_spacebar_action` for interactive and calibration states, it has a reference to this controller
                #~ considering adding a threading.Thread subclass that allows interruptions by setting a shared flag to True
            # define a key handler that triggers calibrator.capture_checkerboard when pressing space
            def feed_key_handler(key):
                if key == ord(' '):
                    # Grab the last frame from the feed
                    frame = feed.last_frame
                    if frame is not None:
                        calibrator.capture_checkerboard(frame)
            # start camera feed for capturing images in a background thread
            feed.display_live_feed(window_name="Calibration Feed", fps=30, key_handler=feed_key_handler)
            # CalibrationState's handle() method simultaneously runs in another thread giving turret WASD control, etc.
            self.handle_state(callback=feed.finalize_calibration(calibrator))

    # these could definitely be abstracted into a common base method, but as top-level methods, these are more readable
    def enter_motion_tracking_only_mode(
        self,
        motion_detector: Optional[MotionDetectorType] = None,
        save_detections: bool = False,
    ):
        """ Start a loop capturing frames and pass them to a MonitoringState that fires on *any* motion. """
        if motion_detector is None:
            from src.host.tracking import MotionDetector
            motion_detector = MotionDetector(threshold=cfg.MOTION_THRESHOLD)
        # Create the specialized state:
        new_state = MotionTrackingOnlyState(motion_detector, debug=save_detections)
        self.set_state(new_state)
        # Open camera feed in the controller
        with CameraFeed(camera_port=cfg.CAMERA_PORT, max_dim_length=720) as feed:
            # Possibly a loop that checks user input to break:
            while True:
                frame = feed.capture_frame()
                # Pass the frame to the current state’s handle_frame method
                self.handle_state(frame=frame)
                # allow pressing 'q' to exit:
                if feed.keypress_monitor():
                    break
                time.sleep(self.update_interval) # only check every 10 seconds to reduce computational load
        # return to idle afterwards
        self.set_state(IdleState())
        self.handle_state()

    def enter_motion_tracking_detection_mode(
        self,
        # TODO: may want to add a factory method with defaults to create a DetectionPipeline in the controller and let it be an optional argument
        detection_pipeline: Optional[DetectionPipelineType] = None,
        motion_detector: Optional[MotionDetectorType] = None,
        save_detections: bool = False,
    ):
        """ Start a loop capturing frames and pass them to a MonitoringState that triggers object detection, and fires accordingly """
        # if not isinstance(detection_pipeline, DetectionPipelineType):
        #     raise TypeError(f"Invalid detection pipeline: {detection_pipeline}. Must be a DetectionPipeline object!")
        if detection_pipeline is None:
            from src.host.detection import DetectionPipeline
            detection_pipeline = DetectionPipeline.default_factory()
        if motion_detector is None:
            from src.host.tracking import MotionDetector
            motion_detector = MotionDetector(threshold=cfg.MOTION_THRESHOLD)
        # Create the specialized state:
        new_state = MotionTrackingDetectionState(motion_detector, detection_pipeline, debug=save_detections)
        self.set_state(new_state)
        # Open camera feed in the controller
        with CameraFeed(camera_port=cfg.CAMERA_PORT, max_dim_length=720) as feed:
            while True:
                frame = feed.capture_frame()
                # Pass the frame to the current state
                self.handle_state(frame=frame)
                # allow pressing 'q' to exit:
                if feed.keypress_monitor():
                    break
                time.sleep(self.update_interval) # only check every 10 seconds to reduce computational load
        # Return to idle afterwards
        self.set_state(IdleState())
        self.handle_state()