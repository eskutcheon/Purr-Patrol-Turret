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
        self.handle_state(callback=self.operation.cleanup)