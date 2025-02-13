from typing import Literal, Dict
from dataclasses import dataclass, field
from adafruit_motorkit import MotorKit
from adafruit_motor import stepper




class MotorHatInterface:
    """ Interface to the Adafruit Motor Hat. """
    def __init__(self):
        """ Initialize the motor hat interface. """
        self.motor_kit = MotorKit()
        self.pan_motor = StepperMotor(self.get_stepper_motor(1), 1)
        self.tilt_motor = StepperMotor(self.get_stepper_motor(2), 2)

    def get_stepper_motor(self, motor_num: Literal[1, 2]) -> stepper.StepperMotor:
        """ get stepper motor object from the motor kit """
        if motor_num not in [1, 2]:
            raise ValueError(f"Invalid motor number: {motor_num}")
        return getattr(self.motor_kit, f"stepper{motor_num}")()


@dataclass
class StepperMotor:
    """ Data class for a stepper motor with a motor kit and stepper motor object
        Docs: https://docs.circuitpython.org/projects/motor/en/latest/api.html#adafruit_motor.stepper.StepperMotor
    """
    motor: stepper.StepperMotor
    channel: int
    directions: Dict[int, int] = field(default = {1: stepper.FORWARD, -1: stepper.BACKWARD})

    def step(self, steps: int, direction: Literal[1, -1]):
        """ step the motor a number of (half) steps in a direction and style """
        try:
            step_direction = self.directions[direction]
        except KeyError:
            raise ValueError(f"Invalid direction: {direction}. Must be one of {list(self.directions.keys())}")
        for _ in range(steps):
            self.motor.onestep(step_direction, stepper.INTERLEAVE)



def test_motor_movement():
    motor_hat = MotorHatInterface()
    motor_hat.pan_motor.step(100, 1)  # Move pan motor forward
    motor_hat.tilt_motor.step(100, -1)  # Move tilt motor backward



if __name__ == "__main__":
    test_motor_movement()