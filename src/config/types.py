# types.py
from typing import NewType, List, Dict, Any, Union, Callable, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # ensures imports are only for type-checking and not at runtime
    import torch
    import torchvision.tv_tensors as tv_tensors


########### Torch Type Hints ###########
Tensor = NewType("Tensor", "torch.Tensor")
FloatTensor = NewType("FloatTensor", "torch.FloatTensor")
IntTensor = NewType("IntTensor", "torch.IntTensor")
ByteTensor = NewType("ByteTensor", "torch.ByteStorage")
LongTensor = NewType("LongTensor", "torch.LongTensor")
# Frequently-used complex types for annotations
ImgMaskPair = Dict[str, Union[Tensor, "tv_tensors.Image", "tv_tensors.Mask", Any]]  # Example: A pair of image and mask tensors.

# Your custom classes (aliases referencing your modules)
#
if TYPE_CHECKING:
    from src.turret.command import Command, MoveCommand, MoveRelativeCommand, FireCommand, AimCommand, StopCommand
    from src.turret.controller import TurretController
    from src.turret.state import TurretState, IdleState, AimingState, FiringState, CalibrationState, InteractiveState
    from src.turret.operations import TurretOperator
    from src.turret.targeting import TurretCoordinates, CameraCoordinates, CalibrationParameters, TargetingSystem
    from src.turret.hardware import MotorHatInterface, PowerRelayInterface
    from src.host.tracking import MotionDetector, MotionDetectionFeedback
    from src.host.base_detectors import DetectionFeedback, BaseDetector, SSDDetector, FasterRCNNDetector, RetinaNetDetector #, YOLODetector
    from src.host.detection import DetectionPipeline
    from src.rpi.camera import CameraFeed
    from src.rpi.calibration import CameraCalibrator

# Turret specific types
# TODO: add new command and states for the new run modes
CommandLike = Union["Command", "MoveCommand", "MoveRelativeCommand", "FireCommand", "StopCommand", "AimCommand"]
StateLike = Union["TurretState", "IdleState", "AimingState", "FiringState", "CalibrationState", "InteractiveState"]
OperatorLike = NewType("OperatorLike", "TurretOperator")  # NewType is used to create a distinct type for clarity

TurretControllerType = NewType("TurretControllerType", "TurretController")
TargetingSystemType = NewType("TargetingSystemType", "TargetingSystem")
# TODO: after incorporating CameraCoordinates, I should make a CoordinatesLike type alias
TurretCoordinatesType = NewType("TurretCoordinatesType", "TurretCoordinates")
CameraCoordinatesType = NewType("CameraCoordinatesType", "CameraCoordinates")
CoordinatesLike = Union["TurretCoordinates", "CameraCoordinates"]  # Union of both coordinate types
CalibrationParamsType = NewType("CalibrationParamsType", "CalibrationParameters")

MotorInterfaceType = NewType("MotorInterfaceType"," MotorHatInterface")
PowerRelayType = NewType("PowerRelayType", "PowerRelayInterface")

MotionDetectorType = NewType("MotionDetectorType", "MotionDetector")
MotionFeedbackType = NewType("MotionFeedbackType", "MotionDetectionFeedback")  # NewType is used to create a distinct type for clarity
DetectorLike = Union["BaseDetector", "SSDDetector", "FasterRCNNDetector", "RetinaNetDetector"]  #, YOLODetector]"
DetectionFeedbackType = NewType("DetectionFeedbackType", "DetectionFeedback")
DetectionPipelineType = NewType("DetectionPipelineType", "DetectionPipeline")

CameraCalibratorType = NewType("CameraCalibratorType", "CameraCalibrator")
CameraFeedType = NewType("CameraFeedType", "CameraFeed")


__all__ = [
    "Tensor",
    "FloatTensor",
    "IntTensor",
    "ByteTensor",
    "LongTensor",
    "FeatureContainerType",
    "ImgMaskPair",
    "CommandLike",
    "StateLike",
    "OperatorLike",
    "TurretControllerType",
    "TargetingSystemType",
    "TurretCoordinatesType",
    "CalibrationParamsType",
    "MotorInterfaceType",
    "PowerRelayType",
    "CameraCoordinatesType",
    "CoordinatesLike",
    "MotionDetectorType",
    "DetectorLike",
    "DetectionFeedbackType",
    "DetectionPipelineType",
    "CameraCalibratorType",
    "CameraFeedType",
]