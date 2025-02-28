from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple
# local imports
from src.turret.targeting import CameraCoordinates
from src.config.types import (
    OperatorLike,
    TargetingSystemType,
    MotionDetectorType,
    DetectionPipelineType,
    DetectionFeedbackType,
    CoordinatesLike
)



class Command(ABC):
    """ Abstract base class for turret commands """
    @abstractmethod
    def execute(self):
        pass


#* MoveCommand is unused in favor of MoveRelativeCommand or AimCommand, but I might want to keep it for future reference
class MoveCommand(Command):
    """ Command to move the turret to an absolute target (Cartesian) coordinate """
    def __init__(self, operation: OperatorLike, target_coord: CoordinatesLike):
        self.operation: OperatorLike = operation
        self.target_coord = target_coord

    def execute(self):
        self.operation.apply_angular_movement(self.target_coord)


class MoveRelativeCommand(Command):
    """ Command to move the turret by a certain delta in pan or tilt (in degrees). Useful for WASD-style incremental movement """
    def __init__(self, operation: OperatorLike, dx_deg=0.0, dy_deg=0.0):
        self.operation: OperatorLike = operation
        self.dx_deg = dx_deg
        self.dy_deg = dy_deg

    def execute(self):
        self.operation.apply_angular_movement(self.dx_deg, self.dy_deg)


class FireCommand(Command):
    """ Command to fire the turret """
    def __init__(self, operation: OperatorLike, duration: float = 3.0):
        self.operation: OperatorLike = operation
        self.duration = duration

    def execute(self):
        self.operation.fire(duration=self.duration)


# TODO: track down all uses and double check the type of target_coord
class AimCommand(Command):
    """Command to aim the turret at a given target (x, y), which is typically a pixel coordinate in the camera frame """
    #? NOTE: not sure yet whether I want to pass `target_coord` as a list or a TurretCoordinates object
    def __init__(self,
                 operation: OperatorLike,
                 targeting_system: TargetingSystemType,
                 target_coord: Union[CoordinatesLike, List[float], Tuple[float, float]]):
        # initialize AimCommand with the operation, targeting system, and target coordinate
        self.operation: OperatorLike = operation
        self.targeting_system: TargetingSystemType = targeting_system
        self.target_coord: CoordinatesLike = target_coord
        self.is_pixel = isinstance(target_coord, (tuple, list)) and len(target_coord) == 2
        if not self.is_pixel and not isinstance(target_coord, CoordinatesLike):
            raise ValueError(f"target_coord must be a length-2 tuple, list, or CoordinatesLike object; got {target_coord}")

    def execute(self):
        if self.is_pixel:
            # Convert from camera pixel coords to TurretCoordinates
            (u, v) = self.target_coord
            cam_coord = CameraCoordinates(u=float(u), v=float(v))
            self.target_coord = self.targeting_system.from_camera_to_turret(cam_coord)
        # compute needed degrees to rotate from current position to target
        dthx, dthy = self.targeting_system.compute_angular_displacement(self.target_coord)
        # move hardware
        self.operation.apply_angular_movement(dthx, dthy)
        # update current_position
        self.targeting_system.update_current_position([dthx, dthy])


#* Also unused everywhere - may or may not keep
class StopCommand(Command):
    """ Command to stop all turret operations """
    def __init__(self, operation: OperatorLike):
        self.operation: OperatorLike = operation

    def execute(self):
        self.operation.cleanup()


# TODO: need to integrate this into the cleanup process
class SetSafeModeCommand(Command):
    def __init__(self, operation: OperatorLike, on_or_off: bool):
        self.operation = operation
        self.value = on_or_off

    def execute(self):
        self.operation.set_safe_mode(self.value)
        print(f"[CMD] Setting SAFE_MODE to {self.value}")


class ConditionalFireCommand(Command):
    """ Command to fire only if SAFE_MODE is disabled """
    def __init__(self, operation: OperatorLike, duration=3):
        self.operation = operation
        self.duration = duration

    def execute(self):
        # Could be a global config, or a property on operation
        if not getattr(self.operation, "safe_mode", False):
            self.operation.fire(self.duration)
        else:
            print("[CMD] SAFE_MODE is ON => skip firing.")


class MotionTrackingCommand(Command):
    """ Runs motion detection on a single frame. result=True if motion found """
    # TODO: add type annotation for frame - might make a new np.ndarray type for this
    def __init__(self, motion_detector: MotionDetectorType, frame):
        self.motion_detector: MotionDetectorType = motion_detector
        self.frame = frame
        self.result = (False, (None, None))

    def execute(self):
        contour = self.motion_detector.process_frame(self.frame)
        if contour is not None:
            cx, cy = self.motion_detector.get_contour_centroid(contour)
            # store "True" plus the pixel coords
            self.result = (True, (cx, cy))
        # else result is already (False, None, None) - no need to set it again


class DetectionCommand(Command):
    """ Command that runs detection on a given frame or frame_id, sets an attribute on the operation or returns feedback in some queue """
    # TODO: add type annotation for frame - might make a new np.ndarray type for this
    def __init__(self, detection_pipeline: DetectionPipelineType, frame):
        self.detection_pipeline = detection_pipeline
        self.frame = frame
        self.result = None

    def execute(self):
        # TODO: figure out needed preprocessing here - may just want to ensure it's a tensor in the detection pipeline constructor
        feedback: DetectionFeedbackType = self.detection_pipeline.run_detection(self.frame)
        # store or return in some shared object or (MAYBE) pass it to some callback function
