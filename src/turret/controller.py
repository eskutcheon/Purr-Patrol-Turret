from threading import Thread
# local imports
from .state import IdleState
from .targeting import TurretCoordinates
from ..config import config as cfg

class TurretController:
    """High-level control logic for the turret."""
    def __init__(self, operation):
        self.operation = operation
        # TODO: integrate a "friendly" state object here that can't be overridden to avoid misfires
        self.current_state = IdleState()  # Start in Idle state
        self.command_queue = []          # Queue for commands
        self.interactive_mapping = {
            "w": self.operation.move_y,
            "s": lambda deg: self.operation.move_y(-deg),
            "a": self.operation.move_x,
            "d": lambda deg: self.operation.move_x(-deg),
        }

    def set_state(self, state):
        """Transition to a new state."""
        self.current_state = state

    def handle_state(self, *args, **kwargs):
        """Delegate behavior to the current state."""
        self.current_state.handle(self, *args, **kwargs)

    def execute_command(self, command):
        """Execute a single command."""
        command.execute()

    def queue_command(self, command):
        """Add a command to the queue."""
        self.command_queue.append(command)

    def process_commands(self):
        """Process all queued commands."""
        while self.command_queue:
            command = self.command_queue.pop(0)
            command.execute()


    # ! REMOVE LATER - just using this for testing for now, but I'd like to keep the new keyboard parsing methods
    def interactive_mode(self):
        """ Run the turret in interactive mode. """
        print("Interactive mode. Use 'w', 's', 'a', 'd' to move, 'Enter' to fire, 'q' to quit.")
        degrees_per_press = cfg.INTERACTIVE_STEP_MULT * cfg.DEGREES_PER_STEP
        def interactive_loop():
            valid_keys = {"w", "s", "a", "d"}
            while True:
                key = input("Enter command: ").rstrip('\r')  # Catch carriage returns too
                if key == "q":
                    print("Exiting interactive mode.")
                    break
                elif key == "":  # blank => user pressed Enter alone
                    self.operation.fire(duration=1)  # or whatever length you prefer
                elif key in valid_keys:
                    self.interactive_mapping[key](degrees_per_press)
                else:
                    print("Unknown command. Use w/s/a/d or Enter to fire, q to quit.")
        Thread(target=interactive_loop, daemon=True).start()

    def calibration_mode(self):
        """Run the turret in calibration mode."""
        print("Calibration mode. Use 'w', 's', 'a', 'd' to adjust and 'q' to save calibration.")
        def calibration_loop():
            #curr_position = TurretCoordinates(0, 0, 0, 0)
            valid_keys = {"w", "s", "a", "d"}
            while True:
                key = input("Enter command: ").strip()
                if key == "q":
                    self.operation.save_calibration()
                    print("Exiting calibration mode.")
                    break
                elif key in valid_keys:
                    if key in {"w", "s"}:
                        self.operation.move_y(key)
                    elif key in {"a", "d"}:
                        self.operation.move_x(key)
        Thread(target=calibration_loop, daemon=True).start()