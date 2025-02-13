import time
from typing import Dict, Union, List
import json
from copy import deepcopy
from dataclasses import dataclass
import RPi.GPIO as GPIO
# local imports
from ..config import config as cfg
from .motor import MotorHatInterface
from .relay import PowerRelayInterface
from .targeting import TurretCoordinates



# TODO: think over whether it would be more apt to call this `TurretOperator` given the way it's treated in interactive mode in the controller
class TurretOperation:
    """Handles low-level GPIO and motor operations."""
    def __init__(self, relay_pin, interactive=False):
        """ Initialize the turret operation with motor channels and GPIO relay pin.
            :param relay_pin: GPIO pin connected to the relay for firing mechanism.
        """
        # using an encapsulated class for both stepper motors
        self.motorkit = MotorHatInterface()
        # initial position, setting current orientation as the origin
        # TODO: load actual coordinates from a calibration file with last known (or just the default) coordinates
        self.current_position = TurretCoordinates(0, 0, 0, 0)
        self.initial_position = deepcopy(self.current_position)
        self.load_calibration()
        # GPIO setup for power relay module (the firing mechanism)
        self.power_relay = PowerRelayInterface(relay_pin)
        self.interactive_mode = interactive
        self.interactive_step_mult = cfg.INTERACTIVE_STEP_MULT


    @staticmethod
    def degrees_to_halfsteps(degrees):
        """ Convert degrees of movement into a number of motor (half) steps """
        # NOTE: *2 for the number of half steps overall (using stepper.INTERLEAVE)
        return 2 * int(degrees / cfg.DEGREES_PER_STEP)

    # !!! TODO: add safeguards to prevent moving the turret too far in either direction

    def move_x(self, degrees):
        """ move turret along the x-axis """
        halfsteps = self.degrees_to_halfsteps(abs(degrees))
        direction = 1 if degrees > 0 else -1
        print(f"Moving pan motor: {direction * degrees} degrees ({halfsteps//2} steps).")
        self.motorkit.move_pan(halfsteps, direction)

    def move_y(self, degrees):
        """ move turret along the y-axis """
        halfsteps = self.degrees_to_halfsteps(abs(degrees))
        direction = 1 if degrees > 0 else -1
        print(f"Moving tilt motor: {direction * degrees} degrees ({halfsteps//2} steps).")
        self.motorkit.move_tilt(halfsteps, direction)

    def move_to_target(self, target_coord):
        """ Move the turret to the specified coordinates """
        print(f"Moving turret to {target_coord}")
        # TODO: add some targeting code based on the calibration results to compute theta_x and theta_y to update current_position
            # need to research how the calibration is usually done first - think I'll have to compute the camera matrix and distortion coefficients
        degrees: List[float] = self.current_position.compute_dtheta(target_coord)
        if degrees[0] != 0:
            self.move_x(degrees[0])
        if degrees[1] != 0:
            self.move_y(degrees[1])
        # FIXME: compute theta_x and theta_y based on calibration data and target coordinates then update current_position
        # self.current_position = {"x": target_coord[0], "y": target_coord[1]}
        self.current_position.update(*target_coord, *degrees)
        time.sleep(0.5)  # Simulate movement delay

    def fire(self, duration=3):
        """ Fire the turret """
        print("Firing turret!")
        self.power_relay.run_n_seconds(duration)


    ######################################################################################################################^
    #^ REMOVE CALIBRATION CODE
    ######################################################################################################################^
    def save_calibration(self):
        """ Save the current position as the calibrated position """
        with open(cfg.CALIBRATION_FILE, "w") as f:
            json.dump(dict(self.current_position), f)
        print(f"Calibration data saved: {self.current_position}")

    def load_calibration(self):
        """ Load the calibration data from the file """
        try:
            with open(cfg.CALIBRATION_FILE, "r") as f:
                self.current_position = TurretCoordinates(**json.load(f))
            print(f"Calibration data loaded: {self.current_position}")
        except FileNotFoundError:
            print("No calibration data found. Using default position (0, 0).")
    ######################################################################################################################^

    def reset_to_initial(self):
        """ Reset the turret to the initial position. """
        print("Resetting turret to initial position...")
        self.current_position = deepcopy(self.initial_position)
        self.move_x(-self.current_position.theta_x)
        self.move_y(-self.current_position.theta_y)

    def cleanup(self):
        """ Clean up GPIO on shutdown """
        print("Cleaning up GPIO and resetting turret...")
        self.reset_to_initial()
        GPIO.cleanup()
