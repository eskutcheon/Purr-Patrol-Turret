from abc import ABC, abstractmethod

class Command(ABC):
    """ Abstract base class for turret commands """
    @abstractmethod
    def execute(self):
        pass


class MoveCommand(Command):
    """ Command to move the turret to an absolute target (Cartesian) coordinate """
    def __init__(self, operation, target_coord):
        self.operation = operation
        self.target_coord = target_coord

    def execute(self):
        self.operation.move_to_target(self.target_coord)


class MoveRelativeCommand(Command):
    """ Command to move the turret by a certain delta in pan or tilt (in degrees). Useful for WASD-style incremental movement """
    def __init__(self, operation, dx_deg=0.0, dy_deg=0.0):
        self.operation = operation
        self.dx_deg = dx_deg
        self.dy_deg = dy_deg

    def execute(self):
        if self.dx_deg != 0:
            self.operation.move_x(self.dx_deg)
        if self.dy_deg != 0:
            self.operation.move_y(self.dy_deg)

class FireCommand(Command):
    """ Command to fire the turret """
    def __init__(self, operation):
        self.operation = operation

    def execute(self):
        self.operation.fire()


class AimCommand(Command):
    """Command to aim the turret at a given absolute target (x, y)."""
    def __init__(self, operation, targeting_system, current_position, target_coord):
        self.operation = operation
        self.targeting_system = targeting_system
        self.current_position = current_position
        self.target_coord = target_coord

    def execute(self):
        # compute needed degrees
        # !!! method doesn't exist yet !!!
        dthx, dthy = self.targeting_system.compute_degrees_for_target(
            self.current_position, self.target_coord
        )
        # move hardware
        self.operation.move_to_target(dthx, dthy)
        # update current_position
        self.current_position.x = self.target_coord[0]
        self.current_position.y = self.target_coord[1]
        self.current_position.theta_x += dthx
        self.current_position.theta_y += dthy



class StopCommand(Command):
    """ Command to stop all turret operations """
    def __init__(self, operation):
        self.operation = operation

    def execute(self):
        self.operation.cleanup()