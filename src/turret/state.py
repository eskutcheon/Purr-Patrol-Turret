from abc import ABC, abstractmethod
import sys, time
import termios
import contextlib
from copy import deepcopy
from threading import Thread
# local imports
from ..config.types import TurretControllerType
from .command import FireCommand, MoveRelativeCommand, AimCommand, StopCommand


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


class TurretState(ABC):
    """ Abstract base class for turret states """
    @abstractmethod
    def handle(self, control: TurretControllerType, *args, **kwargs):
        """Handle the turret's behavior for the current state."""
        pass


class IdleState(TurretState):
    """ Idle state - turret is stationary and not targeting """
    def handle(self, control: TurretControllerType, *args, **kwargs):
        print("Turret is idle...")


class AimingState(TurretState):
    """ Aiming state - turret is moving to target coordinates """
    def __init__(self, target_coord):
        self.target_coord = target_coord

    def handle(self, control: TurretControllerType, *args, **kwargs):
        print(f"Aiming at target: {self.target_coord}...")
        # assume control has references to: operation, targeting_system, current_position
        cmd = AimCommand(
            control.operation,
            control.targeting_system,
            #control.current_position,
            self.target_coord
        )
        control.execute_command(cmd)
        control.set_state(FiringState())


class FiringState(TurretState):
    """ Firing state - The turret fires at the target """
    def handle(self, control: TurretControllerType, *args, **kwargs):
        print("Firing at target...")
        control.execute_command(FireCommand(control.operation))
        control.set_state(IdleState())


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
        self.stop_flag = False
        self.action_desc = "fire turret at current location"
        self.mode_abbrev = ("INTER", "interactive")

    def execute_spacebar_action(self, control: TurretControllerType):
        """ Return the command to execute for the current action """
        cmd = FireCommand(self.control.operation)
        control.queue_command(cmd)

    def _print_instructions(self):
        print("Calibration Mode Controls:")
        print("  w/s/a/d => move turret up/down/left/right")
        print(f"  space   => ({self.action_desc})")
        print("  q       => quit calibration mode")
        print("Press Ctrl+C to force exit if needed.\n")

    def _loop(self, control: TurretControllerType):
        from ..config import config as cfg
        degrees_per_press = cfg.INTERACTIVE_STEP_MULT * cfg.DEGREES_PER_STEP
        with raw_mode(sys.stdin):
            while not self.stop_flag:
                ch = sys.stdin.read(1)
                if not ch:
                    break
                if ch == 'q':
                    print(f"[{self.mode_abbrev[0]}] Exiting {self.mode_abbrev[1]} mode.")
                    self.stop_flag = True
                    break
                elif ch == ' ': # space => fire
                    self.execute_spacebar_action()
                elif ch in self.interactive_mapping:
                    cmd = self.interactive_mapping[ch](control.operation, degrees_per_press)
                    control.queue_command(cmd)
                else:
                    print(f"[{self.mode_abbrev}] Invalid key: `{ch}`")
                    self._print_instructions()
                control.process_commands()
        # when done, go back to IdleState
        control.set_state(IdleState())
        control.handle_state()

    def handle(self, control: TurretControllerType, *args, **kwargs):
        print(f"[{self.mode_abbrev[0]}] Entering {self.__name__}...")
        self._print_instructions()
        Thread(target=self._loop, args=[control], daemon=True).start()
        try:
            while not self.stop_flag:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"[{self.mode_abbrev[0]}] Forcing exit from {self.mode_abbrev[1]} mode...")
            self.stop_flag = True
        finally:
            # If there's a callback (cleanup, etc.), call it
            if "callback" in kwargs and kwargs["callback"]:
                kwargs["callback"]()


class CalibrationState(InteractiveState):
    """ Similar to InteractiveState, but used alongside the CameraFeed to capture frames for calibration """
    def __init__(self, step_degrees=5.0):
        super().__init__(step_degrees)
        self.action_desc = "capture and save image for calibration"
        self.mode_abbrev = ("CALIB", "calibration")

    # TODO: will eventually want to extend this to simultaneously capture the current turret coordinates upon each capture
    def execute_spacebar_action(self, control: TurretControllerType):
        # for now, do nothing here since the camera feed loop catches space bar inputs
        pass
