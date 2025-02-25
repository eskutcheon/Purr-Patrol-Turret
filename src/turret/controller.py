# local imports
from typing import List
from src.config.types import TurretControllerType, OperationLike, StateLike, CommandLike
from .state import IdleState, InteractiveState, CalibrationState
#from .targeting import TurretCoordinates
from ..config import config as cfg

class TurretController:
    """High-level control logic for the turret."""
    def __init__(self, operation: OperationLike):
        self.operation = operation
        self.targeting_system = operation.targeting_system
        # TODO: integrate a "friendly" state object here that can't be overridden to avoid misfires
        self.current_state = IdleState()  # Start in Idle state
        self.command_queue: List[CommandLike] = []          # Queue for commands

    def set_state(self, state: StateLike):
        """Transition to a new state."""
        self.current_state = state

    def handle_state(self, *args, **kwargs):
        """Delegate behavior to the current state."""
        self.current_state.handle(self, *args, **kwargs)

    def execute_command(self, command: CommandLike):
        """Execute a single command."""
        command.execute()

    def queue_command(self, command: CommandLike):
        """Add a command to the queue."""
        self.command_queue.append(command)

    def process_commands(self):
        """Process all queued commands."""
        while self.command_queue:
            command = self.command_queue.pop(0)
            command.execute()

    def enter_interactive_mode(self, show_video=False):
        self.set_state(InteractiveState())
        if not show_video:
            # Immediately handle the new state
            self.handle_state(callback=self.operation.cleanup)
        else:
            from src.rpi.camera import CameraFeed
            with CameraFeed(cfg.CAMERA_PORT, max_dim_length=1080) as live_feed:
                live_feed.display_live_feed()
                self.handle_state(callback=self.operation.cleanup)

    def enter_calibration_mode(self):
        """ launch the turret in calibration mode, working likw interactive mode but with a different state and space bar response """
        from src.rpi.camera import CameraFeed
        from src.rpi.calibration import CameraCalibrator
        self.set_state(CalibrationState())
        calibrator = CameraCalibrator(checkerboard_size=cfg.CHECKERBOARD_SIZE, square_size=1.0)
        # We'll show the same “live feed” but in calibration flavor:
        with CameraFeed(cfg.CAMERA_PORT, max_dim_length=1080) as feed:
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
            self.handle_state(callback=feed.finalize_calibration(calibrator, feed))