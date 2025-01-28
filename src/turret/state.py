from abc import ABC, abstractmethod
from threading import Thread
from .command import FireCommand, MoveCommand

class TurretState(ABC):
    """Abstract base class for turret states."""
    @abstractmethod
    def handle(self, control, *args, **kwargs):
        """Handle the turret's behavior for the current state."""
        pass


class IdleState(TurretState):
    """Idle state: The turret is stationary and not targeting."""
    def handle(self, control, *args, **kwargs):
        print("Turret is idle. Waiting for instructions...")


class AimingState(TurretState):
    """Aiming state: The turret is moving to target coordinates."""
    def __init__(self, target_coord):
        self.target_coord = target_coord

    def handle(self, control, *args, **kwargs):
        print(f"Aiming at target: {self.target_coord}")
        control.execute_command(MoveCommand(control.operation, self.target_coord))
        control.set_state(FiringState())


class FiringState(TurretState):
    """Firing state: The turret fires at the target."""
    def handle(self, control, *args, **kwargs):
        print("Firing at target.")
        control.execute_command(FireCommand(control.operation))
        control.set_state(IdleState())


class CalibrationState(TurretState):
    """Calibration state: Allows the user to calibrate turret manually."""
    def handle(self, control, axis, step_size=5):
        print(f"Calibrating {axis}-axis. Use 'w'/'s' or 'a'/'d' to adjust position.")
        def calibration_loop():
            valid_keys = {'w', 's', 'a', 'd'}
            while True:
                key = input("Enter key (q to quit): ").strip()
                if key == 'q':
                    break
                if key in valid_keys:
                    if axis == 'x':
                        control.operation.move_x(key, step_size)
                    elif axis == 'y':
                        control.operation.move_y(key, step_size)
        Thread(target=calibration_loop).start()


# may move the commands to this file