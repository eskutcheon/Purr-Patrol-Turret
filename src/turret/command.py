from abc import ABC, abstractmethod
# local imports
from ..config.types import OperationLike, TargetingSystemType

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
    def __init__(self, operation: OperationLike):
        self.operation: OperationLike = operation

    def execute(self):
        self.operation.fire()


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