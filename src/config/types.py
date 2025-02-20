# types.py
from typing import NewType, List, Dict, Any, Union, Callable, Tuple, TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:  # ensures imports are only for type-checking and not at runtime
    import torch
    import torchvision.tv_tensors as tv_tensors


# custom types for external classes and libraries
Tensor: TypeAlias = "torch.Tensor"
FloatTensor: TypeAlias = "torch.FloatTensor"
IntTensor: TypeAlias = "torch.IntTensor"
ByteTensor: TypeAlias = "torch.ByteTensor"
LongTensor: TypeAlias = "torch.LongTensor"

# Frequently-used complex types for annotations
Batch: TypeAlias = Dict[str, Union[Tensor, Any]]  # Example: A batch might be a dictionary with tensors.
ImgMaskPair: TypeAlias = Dict[str, Union[Tensor, "tv_tensors.Image", "tv_tensors.Mask", Any]]  # Example: A pair of image and mask tensors.

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
CommandLike: TypeAlias = "Union[Command, MoveCommand, MoveRelativeCommand, FireCommand, StopCommand, AimCommand]"
StateLike: TypeAlias = "Union[TurretState, IdleState, AimingState, FiringState, CalibrationState, InteractiveState]"
OperationLike: TypeAlias = "TurretOperation"

TurretControllerType: TypeAlias = "TurretController"
TargetingSystemType: TypeAlias = "TargetingSystem"
TurretCoordinatesType: TypeAlias = "TurretCoordinates"
CalibrationParamsType: TypeAlias = "CalibrationParameters"

MotorInterfaceType: TypeAlias = "MotorHatInterface"
PowerRelayType: TypeAlias = "PowerRelayInterface"
CameraCoordinatesType: TypeAlias = "CameraCoordinates"

DetectorLike: TypeAlias = "Union[BaseDetector, SSDDetector, FasterRCNNDetector, RetinaNetDetector]"  #, YOLODetector]"
DetectionFeedbackType: TypeAlias = "DetectionFeedback"


# __all__ = [
#     "Tensor",
#     "FloatTensor",
#     "IntTensor",
#     "ByteTensor",
#     "LongTensor",
#     "FeatureContainerType",
#     "Batch",
#     "ImgMaskPair",
# ]