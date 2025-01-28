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
from .targeting import TurretPosition



class TurretOperation:
    """Handles low-level GPIO and motor operations."""
    def __init__(self, relay_pin):
        """ Initialize the turret operation with motor channels and GPIO relay pin.
            :param relay_pin: GPIO pin connected to the relay for firing mechanism.
        """
        # self.motorkit = MotorKit()
        # #self.relay_pin = relay_pin
        # # motors for pan (x-axis) and tilt (y-axis)
        # self.motor_x = self.motorkit.stepper1()
        # self.motor_y = self.motorkit.stepper2()
        self.motorkit = MotorHatInterface()
        # initial position, setting current orientation as the origin
        self.current_position = TurretPosition(0, 0, 0, 0)
        self.initial_position = deepcopy(self.current_position)
        self.load_calibration()
        # GPIO setup for power relay module (the firing mechanism)
        self.power_relay = PowerRelayInterface(relay_pin)
        # GPIO.setmode(GPIO.BCM)
        # GPIO.setup(self.relay_pin, GPIO.OUT)
        # GPIO.output(self.relay_pin, GPIO.LOW) # defaults to LOW (off)

    @staticmethod
    def degrees_to_halfsteps(degrees):
        """ Convert degrees of movement into a number of motor (half) steps """
        # NOTE: *2 for the number of half steps overall (using stepper.INTERLEAVE)
        return 2 * int(degrees / cfg.DEGREES_PER_STEP)

    def move_x(self, degrees):
        """ move turret along the x-axis """
        halfsteps = self.degrees_to_halfsteps(abs(degrees))
        direction = 1 if degrees > 0 else -1
        # direction = self.motor_x.FORWARD if degrees > 0 else self.motor_x.BACKWARD
        print(f"Moving pan motor: {direction * degrees} degrees ({halfsteps//2} steps).")
        # self.motor_x.step(steps, direction, MotorKit.INTERLEAVE)
        self.motorkit.pan_motor.step(halfsteps, direction)

    def move_y(self, degrees):
        """ move turret along the y-axis """
        halfsteps = self.degrees_to_halfsteps(abs(degrees))
        direction = 1 if degrees > 0 else -1
        # direction = MotorKit.FORWARD if degrees > 0 else MotorKit.BACKWARD
        print(f"Moving tilt motor: {direction * degrees} degrees ({halfsteps//2} steps).")
        # self.tilt_motor.step(steps, direction, MotorKit.INTERLEAVE)
        self.motorkit.tilt_motor.step(halfsteps, direction)

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
        """Fire the turret."""
        print("Firing turret!")
        self.power_relay.run_n_seconds(duration)
        # GPIO.output(self.relay_pin, GPIO.HIGH)
        # time.sleep(duration)  # duration of fire
        # GPIO.output(self.relay_pin, GPIO.LOW)

    def save_calibration(self):
        """Save the current position as the calibrated position."""
        with open(cfg.CALIBRATION_FILE, "w") as f:
            json.dump(dict(self.current_position), f)
        print(f"Calibration data saved: {self.current_position}")

    def load_calibration(self):
        """ Load the calibration data from the file. """
        try:
            with open(cfg.CALIBRATION_FILE, "r") as f:
                self.current_position = TurretPosition(**json.load(f))
            print(f"Calibration data loaded: {self.current_position}")
        except FileNotFoundError:
            print("No calibration data found. Using default position (0, 0).")

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
