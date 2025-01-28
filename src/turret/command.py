from abc import ABC, abstractmethod

class Command(ABC):
    """Abstract base class for turret commands."""
    @abstractmethod
    def execute(self):
        pass


class MoveCommand(Command):
    """Command to move the turret to a target coordinate."""
    def __init__(self, operation, target_coord):
        self.operation = operation
        self.target_coord = target_coord

    def execute(self):
        self.operation.move_to_target(self.target_coord)


class FireCommand(Command):
    """Command to fire the turret."""
    def __init__(self, operation):
        self.operation = operation

    def execute(self):
        self.operation.fire()


class StopCommand(Command):
    """Command to stop all turret operations."""
    def __init__(self, operation):
        self.operation = operation

    def execute(self):
        self.operation.stop_all()