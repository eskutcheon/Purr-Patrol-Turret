from abc import ABC, abstractmethod
import os
import sys, time
import contextlib
from copy import deepcopy
# TODO: consider adding threading.Thread subclass with a shared flag to allow for exiting all threads with keypresses
from threading import Thread, Event
from typing import Union, Tuple, Optional, Callable
# local imports
from ..config.types import TurretControllerType, DetectionPipelineType, DetectionFeedbackType, MotionDetectorType, MotionFeedbackType
from .command import (
    FireCommand,
    MoveRelativeCommand,
    AimCommand,
    StopCommand,
    MotionTrackingCommand,
    DetectionCommand,
    ConditionalFireCommand
)

# NOTE: necessary since termios is only available on POSIX systems (Linux, macOS) and not on Windows
    # both termios and msvcrt are part of the Python standard library for the respective platforms
try:
    import termios
    # TODO: add to utils once I sort out dependencies between modules so that torch isn't needed on the Pi
    @contextlib.contextmanager
    def raw_mode(file):
        """ Magic function that allows key presses for Unix systems
            :param file:
        """
        old_attrs = termios.tcgetattr(file.fileno())
        new_attrs = deepcopy(old_attrs)
        new_attrs[3] &= ~(termios.ECHO | termios.ICANON)
        try:
            termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
            yield
        finally:
            termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)
    # method for reading keys in Unix simply accesses stdin directly
    def read_key():
        """ read a single key press from stdin (for Unix systems) """
        return sys.stdin.read(1)
except ModuleNotFoundError:
    import msvcrt
    @contextlib.contextmanager
    def raw_mode(file):
        """ Magic function that allows key presses on Windows
            :param file:
        """
        yield
    # also overriding the method for reading keys since msvcrt uses a different method
    def read_key():
        """ read a single key press on Windows and restrict to simple ASCII characters """
        ch = msvcrt.getch()
        if ch in {b'\x03', b'\x1a'}:  # handle Ctrl+C and Ctrl+Z
            raise KeyboardInterrupt
        elif ch in {b'\x00', b'\xe0'}:  # handle special keys (arrows, function keys)
            ch = msvcrt.getch()
            return None  # ignore for now, or return a mapping later
        return ch.decode('utf-8', errors="ignore")  # decode bytes to string for ASCII characters
except Exception as e:
    print(f"Error setting up raw mode: {e}")
    raise e


##################################################################################################################
############################################# Turret States ######################################################
##################################################################################################################

class TurretState(ABC):
    """ Abstract base class for turret states """
    @abstractmethod
    def handle(self, control: TurretControllerType, *args, **kwargs):
        """Handle the turret's behavior for the current state."""
        pass


class IdleState(TurretState):
    """ Idle state - turret is stationary and not targeting """
    def handle(self, control: TurretControllerType, *args, **kwargs):
        print("Turret is idle...")


class AimingState(TurretState):
    """ Aiming state - turret is moving to target coordinates """
    def __init__(self, target_coord):
        self.target_coord = target_coord

    def handle(self, control: TurretControllerType, *args, **kwargs):
        print(f"Aiming at target: {self.target_coord}...")
        # assume control has references to: operation, targeting_system, current_position
        cmd = AimCommand(
            control.operation,
            control.targeting_system,
            #control.current_position,
            self.target_coord
        )
        control.execute_command(cmd)
        control.set_state(FiringState())


class FiringState(TurretState):
    """ Firing state - The turret fires at the target """
    def handle(self, control: TurretControllerType, *args, **kwargs):
        print("Firing at target...")
        control.execute_command(FireCommand(control.operation))
        control.set_state(IdleState())

##################################################################################################################################
################################################ Interactive States ##############################################################
##################################################################################################################################

class BaseInteractiveState(ABC):
    """ Base class for interactive states - handles keyboard input and command execution """
    def __init__(self, trigger_action: Optional[Callable] = None, stop_event: Optional[Event] = None):
        """ Initialize the interactive state with the given step degrees and trigger action """
        self.stop_event = stop_event or Event()  # use given stop_event or create a new one
        self.spacebar_action: callable = self.set_spacebar_action(trigger_action)
        self.mode_abbrev = ("BASE", "base")
        self.action_desc = "do nothing (base action)"
        self.interactive_mapping = {
            "w": lambda op, dtheta: MoveRelativeCommand(op, dx_deg = 0, dy_deg = +dtheta),
            "s": lambda op, dtheta: MoveRelativeCommand(op, dx_deg = 0, dy_deg = -dtheta),
            "a": lambda op, dtheta: MoveRelativeCommand(op, dx_deg = -dtheta, dy_deg = 0),
            "d": lambda op, dtheta: MoveRelativeCommand(op, dx_deg = +dtheta, dy_deg = 0),
        }

    @abstractmethod
    def handle(self, control: TurretControllerType, *args, **kwargs):
        """ handle the state behavior - meant to mirror TurretState.handle() and MonitoringState.handle() """
        pass

    def cleanup(self, additional_callback: Optional[Callable] = None):
        """ cleanup method - called when exiting the state """
        print(f"[{self.mode_abbrev[0]}] Exiting from {self.mode_abbrev[1]} mode...")
        self.stop_event.set()
        if additional_callback:
            additional_callback()

    def set_spacebar_action(self, action: Callable = None):
        """ set the action to be executed when the space bar is pressed """
        if action is not None and isinstance(action, Callable):
            return action
        # else set default
        def default_action(ctrl: TurretControllerType):
            ctrl.queue_command(FireCommand(ctrl.operation))
        return default_action

    def execute_spacebar_action(self, control: TurretControllerType):
        """ return the command to execute for the current action for space bar input """
        self.spacebar_action(control)

    def _print_instructions(self):
        print(f"{self.mode_abbrev[1].capitalize()} Mode Controls:")
        print("  w/s/a/d => move turret up/down/left/right")
        print(f"  space   => ({self.action_desc})")
        print(f"  q       => quit {self.mode_abbrev[1]} mode")
        print("Press Ctrl+C to force exit if needed.\n")

    #~ pretty sure this could go in the base class - just need to further abstract mapping new key-action pairs
    def _loop(self, control: TurretControllerType):
        from ..config import config as cfg
        degrees_per_press = cfg.INTERACTIVE_STEP_MULT * cfg.DEGREES_PER_STEP
        with raw_mode(sys.stdin):
            while not self.stop_event.is_set():
                ch = read_key()
                if not ch:
                    print(f"[{self.mode_abbrev[0]}] Missing/Unrecognized key; Try again:")
                    continue
                if ch == 'q':
                    print(f"[{self.mode_abbrev[0]}] Exiting {self.mode_abbrev[1]} mode.")
                    self.stop_event.set()
                    break
                elif ch == ' ': # space => fire
                    self.execute_spacebar_action(control)
                elif ch in self.interactive_mapping:
                    cmd = self.interactive_mapping[ch](control.operation, degrees_per_press)
                    control.queue_command(cmd)
                else:
                    print(f"[{self.mode_abbrev[0]}] Invalid key: `{ch}`")
                    self._print_instructions()
                control.process_commands()
        # when done, go back to IdleState
        control.set_state(IdleState())
        control.handle_state()


class InteractiveState(BaseInteractiveState):
    """ Spawns a thread to listen for keyboard input; Creates MoveRelativeCommand or FireCommand based on user presses """
    def __init__(self,
                 trigger_action: Optional[Callable] = None,
                 display_func: Optional[Callable] = None,
                 stop_event: Optional[Event] = None):
        super().__init__(trigger_action, stop_event)
        self.display_func = display_func # optional live feed function
        self.mode_abbrev = ("INTER", "interactive")
        self.action_desc = "fire turret at current location"

    def handle(self, control: TurretControllerType, *args, **kwargs):
        print(f"[{self.mode_abbrev[0]}] Entering {self.__class__.__name__}...")
        self._print_instructions()
        keypress_thread = Thread(target=self._loop, args=[control], daemon=True)
        keypress_thread.start()
        def on_exit():
            keypress_thread.join()
            # if there's a callback (cleanup, etc.), call it
            if "callback" in kwargs and kwargs["callback"]:
                kwargs["callback"]()
        try:
            # Run the display function in the main thread (if provided)
            if self.display_func:
                self.display_func()
            else:
                while not self.stop_event.is_set():
                    time.sleep(1)
        finally:
            self.cleanup(on_exit)


class CalibrationState(BaseInteractiveState):
    """ Similar to InteractiveState, but used alongside the CameraFeed to capture frames for calibration """
    def __init__(self,
                 trigger_action: Optional[Callable] = None,
                 display_func: Callable = None,
                 stop_event: Optional[Event] = None):
        if not display_func:
            raise ValueError("CalibrationState requires a display_func to be provided.")
        super().__init__(trigger_action, stop_event)
        self.display_func = display_func
        self.action_desc = "capture and save image for calibration"
        self.mode_abbrev = ("CALIB", "calibration")

    def handle(self, control: TurretControllerType, *args, **kwargs):
        print(f"[{self.mode_abbrev[0]}] Entering {self.__class__.__name__}...")
        self._print_instructions()
        keypress_thread = Thread(target=self._loop, args=[control], daemon=True)
        keypress_thread.start()
        def on_exit():
            keypress_thread.join()
            # if there's a callback (cleanup, etc.), call it
            if "callback" in kwargs and kwargs["callback"]:
                kwargs["callback"]()
        try:
            self.display_func()
        finally:
            self.cleanup(on_exit)



class MonitoringState(ABC):
    """ base class for continuous monitoring
        accepts frames, checks for motion, and calls subclass method _on_motion_detected() when motion is detected
    """
    def __init__(self, motion_detector: MotionDetectorType, debug: Optional[bool] = False):
        self.motion_detector = motion_detector
        self.debug_mode = debug
        self.break_flag = False
        if self.debug_mode:
            self._setup_debug_mode()

    def _setup_debug_mode(self, prefix="base"):
        from ..config.config import VISUALIZATION_DIR
        self.output_dir = os.path.join(VISUALIZATION_DIR, prefix)
        os.makedirs(self.output_dir, exist_ok=True)

    def handle(self, control: TurretControllerType, *args, **kwargs):
        """ called by the controller to handle the current state - checks frames and calls handle_frame() for each new frame """
        if len(args) == 0 and "frame" not in kwargs:
            raise ValueError("MonitoringState.handle() requires a frame argument.")
        frame = args.pop(0) if len(args) > 0 else kwargs.pop("frame")
        self.handle_frame(control, frame, *args, **kwargs)

    @abstractmethod
    def handle_frame(self, control: TurretControllerType, frame, *args, **kwargs):
        """ called by the controller for each new camera frame """
        pass

    # # TODO: replace type hint for frames with something from config.types later
    # @abstractmethod
    # def visualize_target(self, frame, boundaries, target_coord: Tuple[int, int]):
    #     """ visualize the target on the frame with boundaries around the region of interest """
    #     pass


class MotionTrackingOnlyState(MonitoringState):
    """ If motion is detected, immediately queue a conditional fire (unless safe mode is on). """
    def __init__(self, motion_detector: MotionDetectorType, debug: Optional[bool] = False):
        super().__init__(motion_detector, debug)
        # Make sure we reset for fresh differencing
        self.motion_detector.reset()

    def _setup_debug_mode(self, prefix="motion_detect"):
        super()._setup_debug_mode(prefix)
        from ..utils import view_contours
        self.visualize_target = view_contours

    def handle_frame(self, control, frame, *args, **kwargs):
        # create a MotionTrackingCommand to check if there's motion
        # TODO: need to ensure this is an instance of RefinedMotionDetector
        track_cmd = MotionTrackingCommand(self.motion_detector, frame)
        control.queue_then_process_commands(track_cmd)
        # result for MotionTrackingCommand is a tuple of (found_motion: bool, target_coord: Union[CoordinatesLike, tuple])
        results: MotionFeedbackType = track_cmd.result
        if results is not None and results.motion_detected:
            # If motion was detected, aim to target_coord and fire
            print("[STATE] MotionTrackingOnlyState => motion => aiming & firing")
            target_coord = results.centroid
            aim_cmd = AimCommand(control.operation, control.operation.targeting_system, target_coord)
            fire_cmd = ConditionalFireCommand(control.operation)
            control.queue_then_process_commands(aim_cmd, fire_cmd)
            if self.debug_mode:
                output_path = os.path.join(self.output_dir, f"{time.strftime('%Y%m%d-%H%M%S')}.jpg")
                # visualize the target on the frame with boundaries around the region of interest
                self.visualize_target(frame, results.contour, target_coord, dest_path=output_path)
            # clear background and reset motion detector for next frame (otherwise it'll likely fire back to back)
            self.motion_detector.reset()
            # TODO: add checks for some escape condition - maybe a keypress or a timer (long term)


class MotionTrackingDetectionState(MonitoringState):
    """ If motion is detected, run detection. If detection says shoot_flag => aim + fire (unless safe mode is on). """
    def __init__(self, motion_detector: MotionDetectorType, detection_pipeline: DetectionPipelineType, debug: Optional[bool] = False):
        super().__init__(motion_detector, debug)
        self.detection_pipeline = detection_pipeline
        self.motion_detector.reset()

    def _setup_debug_mode(self, prefix="object_detect"):
        super()._setup_debug_mode(prefix)
        from ..utils import view_boxes
        self.visualize_target = view_boxes

    def handle_frame(self, control, frame, *args, **kwargs):
        # check for motion
        track_cmd = MotionTrackingCommand(self.motion_detector, frame)
        control.queue_then_process_commands(track_cmd)
        tracking_results = track_cmd.result
        if tracking_results is not None and tracking_results.motion_detected:
            print("[STATE] MotionTrackingDetectionState => motion => running detection")
            # if motion was detected, run object detection for specific classes
            detect_cmd = DetectionCommand(self.detection_pipeline, frame)
            control.queue_then_process_commands(detect_cmd)
            results: DetectionFeedbackType = detect_cmd.result
            # if detection says "shoot", aim then fire (when safe mode is off)
            if results and any(results.shoot_flag):
                print("[STATE] MotionTrackingDetectionState => detection => aiming & firing")
                # NOTE: detection returns center in pixel coords
                target_coord = results.resolve_target()
                aim_cmd = AimCommand(control.operation, control.operation.targeting_system, target_coord)
                fire_cmd = ConditionalFireCommand(control.operation)
                control.queue_then_process_commands(aim_cmd, fire_cmd)
                if self.debug_mode:
                    print("frame shape: ", frame.shape)
                    print("boxes: ", results.chosen_boxes)
                    print("labels: ", results.chosen_labels)
                    output_path = os.path.join(self.output_dir, f"{time.strftime('%Y%m%d-%H%M%S')}.jpg")
                    # visualize the target on the frame with boundaries around the region of interest
                    self.visualize_target(frame, results.chosen_boxes, results.chosen_labels, target_coord, dest_path=output_path)
                # clear background and reset motion detector for next frame (otherwise it'll likely fire back to back)
                self.motion_detector.reset()