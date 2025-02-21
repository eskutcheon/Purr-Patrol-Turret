# types.py
from typing import NewType, List, Dict, Any, Union, Callable, Tuple, TypeAlias, TYPE_CHECKING

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
    from src.turret.operations import TurretOperation
    from src.turret.targeting import TargetingSystem, TurretCoordinates, CameraCoordinates, CalibrationParameters
    from src.turret.hardware import MotorHatInterface, PowerRelayInterface
    from src.host.base_detectors import DetectionFeedback, BaseDetector, SSDDetector, FasterRCNNDetector, RetinaNetDetector #, YOLODetector

# Turret specific types
CommandLike = Union["Command", "MoveCommand", "MoveRelativeCommand", "FireCommand", "StopCommand", "AimCommand"]
StateLike = Union["TurretState", "IdleState", "AimingState", "FiringState", "CalibrationState", "InteractiveState"]
OperationLike = NewType("OperationLike", "TurretOperation")  # NewType is used to create a distinct type for clarity

TurretControllerType = NewType("TurretControllerType", "TurretController")
TargetingSystemType = NewType("TargetingSystemType", "TurretController")
# TODO: after incorporating CameraCoordinates, I should make a CoordinatesLike type alias
TurretCoordinatesType = NewType("TurretCoordinatesType", "TurretCoordinates")
CalibrationParamsType = NewType("CalibrationParamsType", "CalibrationParameters")

MotorInterfaceType = NewType("MotorInterfaceType"," MotorHatInterface")
PowerRelayType = NewType("PowerRelayType", "PowerRelayInterface")
CameraCoordinatesType = NewType("CameraCoordinatesType", "CameraCoordinates")

DetectorLike = Union["BaseDetector", "SSDDetector", "FasterRCNNDetector", "RetinaNetDetector"]  #, YOLODetector]"
DetectionFeedbackType = NewType("DetectionFeedbackType", "DetectionFeedback")


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
    "OperationLike",
    "TurretControllerType",
    "TargetingSystemType",
    "TurretCoordinatesType",
    "CalibrationParamsType",
    "MotorInterfaceType",
    "PowerRelayType",
    "CameraCoordinatesType",
    "DetectorLike",
    "DetectionFeedbackType",
]