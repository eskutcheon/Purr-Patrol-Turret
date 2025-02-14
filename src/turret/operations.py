import time
from typing import Dict, Union, List, Tuple
from copy import deepcopy
# local imports
from ..config import config as cfg
from .hardware import MotorHatInterface, PowerRelayInterface
from .targeting import TurretCoordinates, TargetingSystem



# TODO: think over whether it would be more apt to call this `TurretOperator` given the way it's treated in interactive mode in the controller
class TurretOperation:
    """ Handles low-level GPIO and motor operations """
    # NOTE: if this class is subordinate to a controller, it should probably receive more arguments to its constructor and be less independent
    def __init__(self, relay_pin, interactive=False):
        """ Initialize the turret operation with motor channels and GPIO relay pin.
            :param relay_pin: GPIO pin connected to the relay for firing mechanism.
        """
        # using an encapsulated class for both stepper motors
        self.motorkit = MotorHatInterface()
        # TODO: think I should be storing current coordinates in the targeting system setting initial position with its own method
        self.targeting_system = TargetingSystem()
        # initial position, setting current orientation as the origin
        # TODO: load actual coordinates from a calibration file with last known (or just the default) coordinates
            # alternatively, it could be reset to the starting position each time, but an unexpected shutdown would mean it would need recalibrating
        self.current_position = TurretCoordinates(0, 0, 0, 0)
        self.initial_position = deepcopy(self.current_position)
        #self.load_calibration()
        self.power_relay = PowerRelayInterface(relay_pin)
        self.interactive_mode = interactive
        self.rotation_range: Tuple[float, float] = cfg.ROTATION_RANGE


    @staticmethod
    def degrees_to_halfsteps(degrees):
        """ Convert degrees of movement into a number of motor (half) steps """
        # NOTE: *2 for the number of half steps overall (using stepper.INTERLEAVE)
        return 2 * int(degrees / cfg.DEGREES_PER_STEP)

    @staticmethod
    def halfsteps_to_degrees(halfsteps):
        """ Convert halfsteps into degrees of movement """
        return halfsteps * cfg.DEGREES_PER_STEP / 2

    def ensure_within_bounds(self, degrees, current_angle):
        """ Ensure that the turret is within the rotation range """
        theta_lo, theta_hi = self.rotation_range
        in_bounds = theta_lo <= current_angle + degrees <= theta_hi
        if not in_bounds:
            print(f"Movement exceeds rotation range of {self.rotation_range} degrees. Movement cancelled.")
        return in_bounds


    def move_x(self, degrees):
        """ move turret along the x-axis """
        halfsteps = self.degrees_to_halfsteps(abs(degrees))
        direction = 1 if degrees > 0 else -1
        # rounding down degrees to reflect the actual movement by nearest halfstep
        degrees = direction * self.halfsteps_to_degrees(halfsteps)
        print(f"Moving pan motor: {direction * degrees} degrees ({halfsteps//2} steps).")
        if self.ensure_within_bounds(degrees, self.current_position.theta_x):
            self.motorkit.move_pan(halfsteps, direction)
            # TODO: still need to add integration of focal length from calibration files
            dx, dy = self.current_position.compute_displacement_from_angles([degrees, 0], cfg.FOCAL_LENGTH)
            self.current_position.update(self.current_position.x + dx, self.current_position.y + dy, degrees, 0)

    def move_y(self, degrees):
        """ move turret along the y-axis """
        halfsteps = self.degrees_to_halfsteps(abs(degrees))
        direction = 1 if degrees > 0 else -1
        # rounding down degrees to reflect the actual movement by nearest halfstep
        degrees = direction * self.halfsteps_to_degrees(halfsteps)
        print(f"Moving tilt motor: {direction * degrees} degrees ({halfsteps//2} steps).")
        # if prospective movement would exceed the rotation range, don't move
        if self.ensure_within_bounds(degrees, self.current_position.theta_y):
            self.motorkit.move_tilt(halfsteps, direction)
            dx, dy = self.current_position.compute_displacement_from_angles([0, degrees], cfg.FOCAL_LENGTH)
            self.current_position.update(self.current_position.x + dx, self.current_position.y + dy, 0, degrees)

    def move_to_target(self, target_coord):
        """ Move the turret to the specified coordinates """
        print(f"Moving turret to {target_coord}")
        # TODO: add some targeting code based on the calibration results to compute theta_x and theta_y to update current_position
            # need to look into how the calibration is usually done first - think I'll have to compute the camera matrix and distortion coefficients
        degrees: Tuple[float, float] = self.targeting_system(self.current_position, target_coord) #self.current_position.compute_dtheta(target_coord)
        if degrees[0] != 0:
            self.move_x(degrees[0])
        if degrees[1] != 0:
            self.move_y(degrees[1])
        self.current_position.update(*target_coord, *degrees)
        time.sleep(0.5)  # Simulate movement delay

    def fire(self, duration=3):
        print("Firing turret!")
        self.power_relay.run_n_seconds(duration)


    def reset_to_initial(self):
        """ Reset the turret to the initial position. """
        print("Resetting turret to initial position...")
        dtheta_x, dtheta_y = self.targeting_system.compute_angular_displacement(self.current_position, self.initial_position)
        self.current_position = deepcopy(self.initial_position)
        self.move_x(dtheta_x)
        self.move_y(dtheta_y)

    def cleanup(self):
        """ Clean up GPIO on shutdown """
        import RPi.GPIO as GPIO
        print("Cleaning up GPIO and resetting turret...")
        self.reset_to_initial()
        GPIO.cleanup()
