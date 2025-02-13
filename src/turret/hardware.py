from typing import Literal, Dict
import board
import time
from dataclasses import dataclass, field
from adafruit_motorkit import MotorKit
from adafruit_motor import stepper

#import RPi.GPIO as GPIO
import digitalio



class PowerRelayInterface:
    """ Interface to a relay module connected to the Raspberry Pi GPIO. """
    def __init__(self, relay_pin: int):
        """ Initialize the relay module interface with a GPIO pin. """
        self.relay_pin = relay_pin
        # GPIO.setmode(GPIO.BCM)
        # GPIO.setup(self.relay_pin, GPIO.OUT)
        #& ALT: also adding approach that I confirmed working on the Pi:
        self.relay = digitalio.DigitalInOut(getattr(board, f"D{self.relay_pin}"))
        self.toggle_off()

    def toggle_on(self):
        """ Turn on the relay module. """
        # GPIO.output(self.relay_pin, GPIO.HIGH)
        self.relay.direction = digitalio.Direction.OUTPUT

    def toggle_off(self):
        """ Turn off the relay module. """
        # GPIO.output(self.relay_pin, GPIO.LOW)
        self.relay.direction = digitalio.Direction.INPUT

    def run_n_seconds(self, duration: float = 3):
        """ Fire the relay module for a specified duration. """
        self.toggle_on()
        time.sleep(duration) # duration of fire
        self.toggle_off()



class MotorHatInterface:
    """ Interface to the Adafruit Motor Hat. """
    def __init__(self):
        """ Initialize the motor hat interface. """
        self.motor_kit = MotorKit(i2c = board.I2C())
        self.pan_motor = StepperMotor(self._get_stepper_motor(1), 1)
        self.tilt_motor = StepperMotor(self._get_stepper_motor(2), 2)

    def _get_stepper_motor(self, motor_num: Literal[1, 2]) -> stepper.StepperMotor:
        """ get stepper motor object from the motor kit """
        if motor_num not in [1, 2]:
            raise ValueError(f"Invalid motor number: {motor_num}")
        return getattr(self.motor_kit, f"stepper{motor_num}")

    def move_pan(self, num_steps: int, direction: Literal[1, -1]):
        """ move turret along the x-axis """
        self.pan_motor.step(num_steps, direction)

    def move_tilt(self, num_steps: int, direction: Literal[1, -1]):
        """ move turret along the y-axis """
        self.tilt_motor.step(num_steps, direction)

@dataclass
class StepperMotor:
    """ Data class for a stepper motor with a motor kit and stepper motor object
        Docs: https://docs.circuitpython.org/projects/motor/en/latest/api.html#adafruit_motor.stepper.StepperMotor
    """
    motor: stepper.StepperMotor
    channel: int
    directions: Dict[int, int] = field(default_factory = lambda: {1: stepper.FORWARD, -1: stepper.BACKWARD})

    def step(self, steps: int, direction: Literal[1, -1]):
        """ step the motor a number of (half) steps in a direction and style """
        try:
            step_direction = self.directions[direction]
        except KeyError:
            raise ValueError(f"Invalid direction: {direction}. Must be one of {list(self.directions.keys())}")
        for _ in range(steps):
            self.motor.onestep(direction=step_direction, style=stepper.INTERLEAVE)
            time.sleep(0.01)  # Add a small delay to allow the motor to move smoothly
