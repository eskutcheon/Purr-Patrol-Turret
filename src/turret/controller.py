import sys
from threading import Thread
# local imports
from .state import IdleState, AimingState
from .targeting import TurretCoordinates
from ..config import config as cfg
from ..utils import raw_mode

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


    # probably also not keeping the following, but I wanted to test it with the current interactive mode
    def _parse_keyboard_input(self, degrees: float, valid_keys=None, additional_action=None):
        """ Generalized keyboard input parser with abstraction for context management and loop logic.
            :param motor: Motor object to control.
            :param is_reversed: Boolean indicating if the motor direction is reversed.
            :param valid_keys: Set of valid keys to parse. Defaults to {'w', 's', 'a', 'd'}.
            :param step_size: Number of steps for the motor movement.
            :param additional_action: Callable for any additional actions based on specific input.
        """
        if valid_keys is None:
            valid_keys = {'w', 's', 'a', 'd'}

        def process_input(ch):
            if ch in self.interactive_mapping:
                self.interactive_mapping[ch](degrees)
            elif additional_action and callable(additional_action):
                additional_action(ch)
        # actual loop logic for parsing input
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch or ch == "\n":
                        break
                    process_input(ch)
            except Exception as e:
                print(f"Exception in interactive mode: {e}\n Exiting interactive mode...")


    def interactive(self):
        """ Starts an interactive session. Key presses determine movement. """
        degrees_per_press = cfg.INTERACTIVE_STEP_MULT * cfg.DEGREES_PER_STEP
        def fire_action(ch):
            if ch == "q":
                sys.exit(0)
            elif ch == "\n":
                self.fire()
        Thread(target=self._parse_keyboard_input,
               args=[degrees_per_press],
               kwargs=dict(
                    additional_action=fire_action,
                    valid_keys={'w', 's', 'a', 'd', 'q', '\n'}),
               daemon=True).start()
        self.operation.cleanup()



    # ! REMOVE LATER - just using this for testing for now, but I'd like to keep the new keyboard parsing methods
    def interactive_mode(self):
        """ Run the turret in interactive mode. """
        print("Interactive mode. Use 'w', 's', 'a', 'd' to move, 'Enter' to fire, 'q' to quit.")
        degrees_per_press = cfg.INTERACTIVE_STEP_MULT * cfg.DEGREES_PER_STEP
        def interactive_loop():
            valid_keys = {"w", "s", "a", "d"}
            try:
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
            except Exception as e:
                print(f"Exception in interactive mode: {e}")
                print("Exiting interactive mode...")
        Thread(target=interactive_loop, daemon=True).start()
        self.operation.cleanup()