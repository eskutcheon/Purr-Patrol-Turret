import time
from typing import Tuple
# local imports
#from ..config.types import TargetingSystemType
from ..config import config as cfg
from .hardware import MotorHatInterface, PowerRelayInterface
from .targeting import TargetingSystem



# TODO: think over whether it would be more apt to call this `TurretOperator` given the way it's treated in interactive mode in the controller
class TurretOperation:
    """ Handles low-level GPIO and motor operations """
    # NOTE: if this class is subordinate to a controller, it should probably receive more arguments to its constructor and be less independent
    def __init__(self, relay_pin: int, interactive: bool = False):
        """ Initialize the turret operation with motor channels and GPIO relay pin.
            :param relay_pin: GPIO pin connected to the relay for firing mechanism.
        """
        # using an encapsulated class for both stepper motors
        self.motorkit = MotorHatInterface()
        self.targeting_system = TargetingSystem()
        # TODO: load actual coordinates from a calibration file with last known (or just the default) coordinates
            # alternatively, it could be reset to the starting position each time, but an unexpected shutdown would mean it would need recalibrating
        #self.load_calibration()
        self.power_relay = PowerRelayInterface(relay_pin)
        self.interactive_mode = interactive
        self.rotation_range: Tuple[float, float] = cfg.ROTATION_RANGE


    @staticmethod
    def degrees_to_halfsteps(degrees: float) -> int:
        """ Convert degrees of movement into a number of motor (half) steps """
        # NOTE: *2 for the number of half steps overall (using stepper.INTERLEAVE)
        return 2 * int(degrees / cfg.DEGREES_PER_STEP)

    @staticmethod
    def halfsteps_to_degrees(halfsteps: int) -> float:
        """ Convert halfsteps into degrees of movement """
        return halfsteps * cfg.DEGREES_PER_STEP / 2

    def ensure_within_bounds(self, degrees: float, current_angle: float) -> bool:
        """ Ensure that the turret is within the rotation range """
        theta_lo, theta_hi = self.rotation_range
        in_bounds = theta_lo <= current_angle + degrees <= theta_hi
        if not in_bounds:
            print(f"Movement exceeds rotation range of {self.rotation_range} degrees. Movement cancelled.")
        return in_bounds

    def get_angle_steps(self, degrees: float) -> Tuple[float, int, int]:
        halfsteps = self.degrees_to_halfsteps(abs(degrees))
        direction = 1 if degrees > 0 else -1
        # rounding down degrees to reflect the actual movement by nearest halfstep
        degrees = round(direction * self.halfsteps_to_degrees(halfsteps), 4)
        return degrees, halfsteps, direction

    def move_x(self, degrees: float):
        """ move turret along the x-axis """
        degrees, halfsteps, direction = self.get_angle_steps(degrees)
        print(f"Moving pan motor: {degrees} degrees ({halfsteps//2} steps).")
        current_theta_x = self.targeting_system.current_position.theta_x
        if self.ensure_within_bounds(degrees, current_theta_x):
            self.motorkit.move_pan(halfsteps, direction)
            self.targeting_system.update_current_position([degrees, 0])

    def move_y(self, degrees: float):
        """ move turret along the y-axis """
        degrees, halfsteps, direction = self.get_angle_steps(degrees)
        print(f"Moving tilt motor: {degrees} degrees ({halfsteps//2} steps).")
        # if prospective movement would exceed the rotation range, don't move
        current_theta_y = self.targeting_system.current_position.theta_y
        if self.ensure_within_bounds(degrees, current_theta_y):
            self.motorkit.move_tilt(halfsteps, direction)
            self.targeting_system.update_current_position([0, degrees])

    def apply_angular_movement(self, dtheta_x: float, dtheta_y: float):
        """ Move the turret to the specified coordinates """
        # TODO: add some targeting code based on the calibration results to compute theta_x and theta_y to update current_position
            # need to look into how the calibration is usually done first - think I'll have to compute the camera matrix and distortion coefficients
        if dtheta_x != 0:
            self.move_x(dtheta_x)
        if dtheta_y != 0:
            self.move_y(dtheta_y)
        time.sleep(0.5)  # Simulate movement delay

    def fire(self, duration: float = 3):
        print("Firing turret!")
        self.power_relay.run_n_seconds(duration)


    def reset_to_initial(self):
        """ Reset the turret to the initial position. """
        print("Resetting turret to initial position...")
        dtheta_x, dtheta_y = self.targeting_system.compute_angular_displacement(self.targeting_system.initial_position)
        self.move_x(dtheta_x)
        self.move_y(dtheta_y)
        self.targeting_system.reset_to_initial()

    def cleanup(self):
        """ Clean up GPIO on shutdown """
        import RPi.GPIO as GPIO
        print("Cleaning up GPIO and resetting turret...")
        self.reset_to_initial()
        GPIO.cleanup()
