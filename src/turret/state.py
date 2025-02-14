from abc import ABC, abstractmethod
import sys, time
import termios
import contextlib
from copy import deepcopy
from threading import Thread
from .command import FireCommand, MoveRelativeCommand, AimCommand

class TurretState(ABC):
    """ Abstract base class for turret states """
    @abstractmethod
    def handle(self, control, *args, **kwargs):
        """Handle the turret's behavior for the current state."""
        pass


class IdleState(TurretState):
    """ Idle state - turret is stationary and not targeting """
    def handle(self, control, *args, **kwargs):
        print("Turret is idle...")


class AimingState(TurretState):
    """ Aiming state - turret is moving to target coordinates """
    def __init__(self, target_coord):
        self.target_coord = target_coord

    def handle(self, control, *args, **kwargs):
        print(f"Aiming at target: {self.target_coord}")
        # assume control has references to: operation, targeting_system, current_position
        cmd = AimCommand(
            control.operation,
            control.targeting_system,
            control.current_position,
            self.target_coord
        )
        control.execute_command(cmd)
        control.set_state(FiringState())


class FiringState(TurretState):
    """ Firing state - The turret fires at the target """
    def handle(self, control, *args, **kwargs):
        print("Firing at target.")
        control.execute_command(FireCommand(control.operation))
        control.set_state(IdleState())


# TODO: not fully realized yet - intended for it to be the base functionality for the interactive mode but record position for calibration
class CalibrationState(TurretState):
    """ Calibration state - Allows the user to calibrate turret manually """
    def handle(self, control, axis, step_size=5):
        print(f"Calibrating {axis}-axis. Use 'w'/'s' or 'a'/'d' to adjust position.")
        def calibration_loop():
            raise NotImplementedError("Calibration loop not implemented yet.")
        Thread(target=calibration_loop).start()



@contextlib.contextmanager
def raw_mode(file):
    """ Magic function that allows key presses
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


class InteractiveState(TurretState):
    """ Spawns a thread to listen for keyboard input; Creates MoveRelativeCommand or FireCommand based on user presses """
    def __init__(self, step_degrees=5.0):
        super().__init__()
        self.interactive_mapping = {
            "w": lambda op, dtheta: MoveRelativeCommand(op, dx_deg = 0, dy_deg = +dtheta),
            "s": lambda op, dtheta: MoveRelativeCommand(op, dx_deg = 0, dy_deg = -dtheta),
            "a": lambda op, dtheta: MoveRelativeCommand(op, dx_deg = -dtheta, dy_deg = 0),
            "d": lambda op, dtheta: MoveRelativeCommand(op, dx_deg = +dtheta, dy_deg = 0),
        }
        self.step_degrees = step_degrees
        self.stop_interactive = False

    def _loop(self, control):
        from .controller import raw_mode
        from ..config import config as cfg
        degrees_per_press = cfg.INTERACTIVE_STEP_MULT * cfg.DEGREES_PER_STEP
        with raw_mode(sys.stdin):
            while not self.stop_interactive:
                ch = sys.stdin.read(1)
                if not ch:
                    break
                if ch == 'q':
                    print("Exiting interactive mode.")
                    self.stop_interactive = True
                    break
                elif ch == ' ': # space => fire
                    cmd = FireCommand(control.operation)
                    control.queue_command(cmd)
                elif ch in self.interactive_mapping:
                    cmd = self.interactive_mapping[ch](control.operation, degrees_per_press)
                    control.queue_command(cmd)
                control.process_commands()
        # when done, go back to IdleState
        control.set_state(IdleState())
        control.handle_state()

    def handle(self, control, *args, **kwargs):
        print("Entering InteractiveState. Press w/s/a/d to move, space to fire, or q to quit.")
        print("Use Ctrl+C to fully exit if needed.")
        Thread(target=self._loop, args=[control], daemon=True).start()
        try:
            while not self.stop_interactive:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting interactive mode...")
        finally:
            kwargs["callback"]()