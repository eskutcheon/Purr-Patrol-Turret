from abc import ABC, abstractmethod
# local imports
from ..config.types import OperationLike, TargetingSystemType, MotionDetectorType, DetectionPipelineType, DetectionFeedbackType

class Command(ABC):
    """ Abstract base class for turret commands """
    @abstractmethod
    def execute(self):
        pass


class MoveCommand(Command):
    """ Command to move the turret to an absolute target (Cartesian) coordinate """
    def __init__(self, operation: OperationLike, target_coord):
        self.operation: OperationLike = operation
        self.target_coord = target_coord

    def execute(self):
        self.operation.apply_angular_movement(self.target_coord)


class MoveRelativeCommand(Command):
    """ Command to move the turret by a certain delta in pan or tilt (in degrees). Useful for WASD-style incremental movement """
    def __init__(self, operation: OperationLike, dx_deg=0.0, dy_deg=0.0):
        self.operation: OperationLike = operation
        self.dx_deg = dx_deg
        self.dy_deg = dy_deg

    def execute(self):
        self.operation.apply_angular_movement(self.dx_deg, self.dy_deg)


class FireCommand(Command):
    """ Command to fire the turret """
    def __init__(self, operation: OperationLike, duration=3):
        self.operation: OperationLike = operation
        self.duration = duration

    def execute(self):
        self.operation.fire(duration=self.duration)


class AimCommand(Command):
    """Command to aim the turret at a given absolute target (x, y)."""
    #? NOTE: not sure yet whether I want to pass `target_coord` as a list or a TurretCoordinates object
    def __init__(self, operation: OperationLike, targeting_system: TargetingSystemType, target_coord):
        self.operation: OperationLike = operation
        self.targeting_system: TargetingSystemType = targeting_system
        self.target_coord = target_coord

    def execute(self):
        # TODO: should probably just replace this with another method later while leaving compute_angular_displacement for other uses
        # compute needed degrees
        dthx, dthy = self.targeting_system.compute_angular_displacement(self.target_coord)
        # move hardware
        self.operation.apply_angular_movement(dthx, dthy)
        # update current_position
        self.targeting_system.update_current_position([dthx, dthy])


class StopCommand(Command):
    """ Command to stop all turret operations """
    def __init__(self, operation: OperationLike):
        self.operation: OperationLike = operation

    def execute(self):
        self.operation.cleanup()


class SetSafeModeCommand(Command):
    def __init__(self, operation: OperationLike, on_or_off: bool):
        self.operation = operation
        self.value = on_or_off

    def execute(self):
        self.operation.set_safe_mode(self.value)
        print(f"[CMD] Setting SAFE_MODE to {self.value}")


class ConditionalFireCommand(Command):
    """ Command to fire only if SAFE_MODE is disabled """
    def __init__(self, operation: OperationLike, duration=3):
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
        self.result = False

    def execute(self):
        contour = self.motion_detector.process_frame(self.frame)
        self.result = (contour is not None)


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
