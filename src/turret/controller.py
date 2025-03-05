from typing import List, Optional, Iterable, ContextManager
import time
from threading import Event
# local imports
from src.config import config as cfg
from src.config.types import TurretControllerType, OperatorLike, StateLike, CommandLike, MotionDetectorType, DetectionPipelineType, CameraFeedLike
from src.turret.state import IdleState, InteractiveState, CalibrationState, MotionTrackingOnlyState, MotionTrackingDetectionState
# for importing the correct camera feed module
from src.utils import is_raspberry_pi, has_opencv

CameraFeed: CameraFeedLike = None
if is_raspberry_pi():
    from rpi.camera_rpi import CameraFeedRpi as CameraFeed
elif has_opencv():
    from rpi.camera_opencv import CameraFeedOpenCV as CameraFeed
else:
    raise ImportError("No camera module found! Please install OpenCV or libcamera (on Raspberry Pi Cam) to use the turret.")


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
        if not show_video:
            # NOTE: setting state without `trigger_action` will use the default trigger action of firing
            self.set_state(InteractiveState(cfg.DEGREES_PER_STEP)) # stick with default trigger action of firing
            # Immediately handle the new state
            self.handle_state(callback=self.operation.cleanup)
        else:
            self.stop_event = Event()
            with CameraFeed(cfg.CAMERA_PORT, max_dim_length=1080) as live_feed:
                def run_live_feed():
                    live_feed.display_live_feed(self.stop_event, cfg.LIVE_FEED_DELAY, use_plt=True)
                # NOTE: setting state without `trigger_action` will use the default trigger action of firing
                self.set_state(InteractiveState(display_func=run_live_feed, stop_event=self.stop_event))
                self.handle_state(callback=self.operation.cleanup)

    def enter_calibration_mode(self):
        """ launch the turret in calibration mode, working like interactive mode but with a different state object and space bar response """
        from src.rpi.calibration import CameraCalibrator
        calibrator = CameraCalibrator(checkerboard_size=cfg.CHECKERBOARD_SIZE, square_size=cfg.SQUARE_SIZE)
        self.stop_event = Event()
        # show the same live feed but with calibration objectives (spacebar takes a picture):
        with CameraFeed(cfg.CAMERA_PORT, max_dim_length=1080) as feed:
            #self.spawn_live_camera_thread(feed)
            def calibration_action(*args, **kwargs):
                # grab the last frame from the feed
                frame = feed.capture_frame()
                if frame is not None:
                    calibrator.capture_checkerboard(frame)
            # define a key handler that triggers calibrator.capture_checkerboard when pressing space
            def finalize_calibration():
                # user presumably pressed 'q' in the feed to exit the loop or 'q' from the turret keyboard thread
                calibrator.finalize(cfg.CALIBRATION_FILE, feed.resize_dims)
                self.operation.cleanup()
            def run_live_feed():
                feed.display_live_feed(self.stop_event, cfg.LIVE_FEED_DELAY, use_plt=True)
            self.set_state(CalibrationState(trigger_action=calibration_action, display_func=run_live_feed, stop_event=self.stop_event))
            # CalibrationState's handle() method runs in another thread giving turret WASD control, etc.
            try:
                self.handle_state(callback=finalize_calibration)
            except Exception as e:
                raise e


    # these could definitely be abstracted into a common base method, but as top-level methods, these are more readable
    def enter_motion_tracking_only_mode(
        self,
        motion_detector: Optional[MotionDetectorType] = None,
        save_detections: bool = False,
    ):
        """ start a loop capturing frames and pass them to a MonitoringState that fires on *any* motion """
        if motion_detector is None:
            from src.host.tracking import RefinedMotionDetector
            # TODO: add debugging flag to initialization here
            motion_detector = RefinedMotionDetector(
                threshold=cfg.MOTION_THRESHOLD,
                morph_disk_radius=9,
                background_update_interval=2 #cfg.MOTION_UPDATE_INTERVAL
            )
        # Create the specialized state:
        new_state = MotionTrackingOnlyState(motion_detector, debug=save_detections)
        self.set_state(new_state)
        # Open camera feed in the controller
        with CameraFeed(camera_port=cfg.CAMERA_PORT, max_dim_length=720) as feed:
            # loop that checks user input to break:
            while True:
                frame = feed.capture_frame()
                # pass the frame to the current state’s handle_frame method
                self.handle_state(frame=frame)
                time.sleep(self.update_interval) # only check every n seconds to reduce computational load
        # return to idle afterwards
        self.set_state(IdleState())
        try:
            self.handle_state()
        except Exception as e:
            raise e

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
            # BaseMotionDetector is the default motion detector used only for triggering object detection
            from src.host.tracking import BaseMotionDetector
            motion_detector = BaseMotionDetector(
                threshold=cfg.MOTION_THRESHOLD,
                background_update_interval=cfg.MOTION_UPDATE_INTERVAL
            )
        # Create the specialized state:
        new_state = MotionTrackingDetectionState(motion_detector, detection_pipeline, debug=save_detections)
        self.set_state(new_state)
        # Open camera feed in the controller
        with CameraFeed(camera_port=cfg.CAMERA_PORT, max_dim_length=720) as feed:
            while True:
                frame = feed.capture_frame()
                # pass the frame to the current state
                self.handle_state(frame=frame)
                time.sleep(self.update_interval) # only check every n seconds to reduce computational load
        # Return to idle afterwards
        self.set_state(IdleState())
        self.handle_state()