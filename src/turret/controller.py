import sys
import termios
import contextlib
from copy import deepcopy
from threading import Thread
# local imports
from .state import IdleState, InteractiveState
#from .targeting import TurretCoordinates
from ..config import config as cfg

class TurretController:
    """High-level control logic for the turret."""
    def __init__(self, operation):
        self.operation = operation
        # TODO: integrate a "friendly" state object here that can't be overridden to avoid misfires
        self.current_state = IdleState()  # Start in Idle state
        self.command_queue = []          # Queue for commands

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

    def enter_interactive_mode(self):
        self.set_state(InteractiveState())
        # Immediately handle the new state
        self.handle_state()



@contextlib.contextmanager
def raw_mode(file):
    """ Magic function that allows key presses.
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

class InteractiveTurretController(TurretController):
    def __init__(self, operation):
        super().__init__(operation)
        self.interactive_mapping = {
            "w": self.operation.move_y,
            "s": lambda deg: self.operation.move_y(-deg),
            "a": self.operation.move_x,
            "d": lambda deg: self.operation.move_x(-deg),
        }
        self.end_interactive = False

    # probably also not keeping the following, but I wanted to test it with the current interactive mode
    def _parse_keyboard_input(self, degrees: float, valid_keys=None, additional_action=None):
        """ Generalized keyboard input parser with abstraction for context management and loop logic.
            :param valid_keys: Set of valid keys to parse. Defaults to {'w', 's', 'a', 'd'}.
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
                    #if not ch or ch == "\n":
                    if ch not in valid_keys:
                        self.end_interactive = True
                        break
                    process_input(ch)
            except Exception as e:
                print(f"Exception in interactive mode: {e}\n Exiting interactive mode...")
                self.end_interactive = True

    def spawn_interactive_thread(self):
        degrees_per_press = cfg.INTERACTIVE_STEP_MULT * cfg.DEGREES_PER_STEP
        def fire_action(ch):
            if ch == "q":
                sys.exit(0)
            elif ch == " ": # space bar to fire
                self.operation.fire()
        Thread(target=self._parse_keyboard_input,
               args=[degrees_per_press],
               kwargs=dict(
                    additional_action=fire_action,
                    valid_keys={'w', 's', 'a', 'd', 'q', ' '}),
               daemon=True).start()

    def interactive(self):
        """ Starts an interactive session. Key presses determine movement. """
        import time
        self.spawn_interactive_thread()
        try:
            while not self.end_interactive:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting interactive mode...")
        finally:
            self.operation.cleanup()
