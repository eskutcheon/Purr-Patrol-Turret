import cv2
import time
import threading
import atexit
import sys
import termios
import contextlib
from copy import deepcopy
import numpy as np
import torch
# NOTE: Adafruit should still be the same since the motor driver is an Adafruit board
# TODO: revert back to Adafruit after moving back to the Raspberry Pi
import Jetson.GPIO as GPIO
#       ! This is apparently deprecated:
#       from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor, Adafruit_StepperMotor
# using their recommendation here:
#https://github.com/adafruit/Adafruit_CircuitPython_MotorKit
from adafruit_motorkit import MotorKit
from config.config import *
from motion_tracking import MotionTracker
from object_detection import Detector
from utils import get_user_confirmation


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




class Turret(object):
    """ Class used for turret functionality encapsulation. """
    def __init__(self, friendly_mode=True, specific_target=False):
        self.target_detector = Detector() if specific_target else None
        self.friendly_mode = friendly_mode
        # create a default object, no changes to I2C address or frequency
        #self.mh = Adafruit_MotorHAT()
        self.mh = MotorKit()
        atexit.register(self.__turn_off_motors)
        # Stepper motor 1
        self.sm_x = self.mh.getStepper(200, 1)      # 200 steps/rev, motor port #1
        self.sm_x.setSpeed(5)                       # 5 RPM
        self.current_x_steps = 0
        # Stepper motor 2
        self.sm_y = self.mh.getStepper(200, 2)      # 200 steps/rev, motor port #2
        self.sm_y.setSpeed(5)                       # 5 RPM
        self.current_y_steps = 0
        # Power relay to activate the sprayer
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(RELAY_PIN, GPIO.OUT)
        GPIO.output(RELAY_PIN, GPIO.LOW)


    def _parse_keyboard_input(self, motor, is_reversed: bool, valid_keys=None, step_size=5, additional_action=None):
        """ Generalized keyboard input parser with abstraction for context management and loop logic.
            :param motor: Motor object to control.
            :param is_reversed: Boolean indicating if the motor direction is reversed.
            :param valid_keys: Set of valid keys to parse. Defaults to {'w', 's', 'a', 'd'}.
            :param step_size: Number of steps for the motor movement.
            :param additional_action: Callable for any additional actions based on specific input.
        """
        if valid_keys is None:
            valid_keys = {'w', 's', 'a', 'd'}
        command_dict = {
            "w": lambda: self.move_forward(motor, step_size) if not is_reversed else self.move_backward(motor, step_size),
            "s": lambda: self.move_backward(motor, step_size) if not is_reversed else self.move_forward(motor, step_size),
            "a": lambda: self.move_backward(motor, step_size) if is_reversed else self.move_forward(motor, step_size),
            "d": lambda: self.move_forward(motor, step_size) if is_reversed else self.move_backward(motor, step_size),
        }
        def process_input(ch):
            if ch in command_dict:
                command_dict[ch]()
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
            except (KeyboardInterrupt, EOFError):
                print("Exiting input loop...")


    def calibrate(self):
        """ Waits for input to calibrate the turret's axis. """
        print("Please calibrate the tilt of the gun so that it is level. Commands: (w) moves up, " \
            "(s) moves down. Press (enter) to finish.\n")
        self.__calibrate_y_axis()
        print("Please calibrate the yaw of the gun so that it aligns with the camera. Commands: (a) moves left, " \
            "(d) moves right. Press (enter) to finish.\n")
        self.__calibrate_x_axis()
        print("Calibration finished.")

    def calibrate_axis(self, axis: str, motor, is_reversed: bool):
        """ Generalized axis calibration.
            :param axis: Name of the axis ('x' or 'y') being calibrated.
            :param motor: Motor object for the axis.
            :param is_reversed: Whether the motor direction is reversed.
        """
        print(f"Calibrating {axis}-axis. Use 'w', 's' (for y) or 'a', 'd' (for x). Press Enter to finish.")
        self._parse_keyboard_input(motor, is_reversed)

    def __calibrate_x_axis(self):
        """ Waits for input to calibrate the x axis (yaw) of the turret. """
        self.calibrate_axis('x', self.sm_x, MOTOR_X_REVERSED)

    def __calibrate_y_axis(self):
        """ Waits for input to calibrate the y axis (tilt) of the turret. """
        self.calibrate_axis('y', self.sm_y, MOTOR_Y_REVERSED)


    def motion_detection(self, show_video=False):
        """ Uses the camera to move the turret. OpenCV must be configured to use this """
        MotionTracker.find_motion(self.move_axis, show_video=show_video)


    def move_axis(self, contour: np.ndarray, frame: torch.Tensor):
        """ Moves the turret based on the detected object's position - used as a callback function in the MotionTracker """
        should_fire, target_coord = False, []
        (v_h, v_w) = frame.shape[-2:]
        # TODO: for the new modular version, this function should instead receive the target coordinates (as a list of np.ndarray) directly
        if self.target_detector is None:
            (x, y, w, h) = cv2.boundingRect(contour)
            target_coord = [x + w / 2, y + h / 2]
        else:
            should_fire, target_coord = self.target_detector.scan_frame(frame)
        self.move_to_target(target_coord, (v_w, v_h), should_fire)

    def move_to_target(self, target_coord, frame_dims, should_fire=False):
        """ Moves the turret to the specified target coordinates.
            :param target_coord: (x, y) pixel coordinates of the target in the frame.
            :param frame_dimensions: (width, height) of the frame.
        """
        v_w, v_h = frame_dims
        target_steps_x = (2 * MAX_STEPS_X * target_coord[0] / v_w) - MAX_STEPS_X
        target_steps_y = (2 * MAX_STEPS_Y * target_coord[1] / v_h) - MAX_STEPS_Y
        print(f"Target steps - x: {target_steps_x}, y: {target_steps_y}")
        thread_move_x = self.__move_motor(self.current_x_steps, target_steps_x, self.sm_x, MOTOR_X_REVERSED)
        thread_move_y = self.__move_motor(self.current_y_steps, target_steps_y, self.sm_y, MOTOR_Y_REVERSED)
        thread_fire = threading.Thread(target=self.fire) if self.__should_fire(target_steps_x, target_steps_y, should_fire) else None
        thread_move_x.start()
        thread_move_y.start()
        if thread_fire:
            thread_fire.start()
        thread_move_x.join()
        thread_move_y.join()
        if thread_fire:
            thread_fire.join()

    def __move_motor(self, current_steps, target_steps, motor, is_reversed) -> threading.Thread:
        delta = target_steps - current_steps
        if delta != 0:
            direction = self.move_backward if (delta < 0 and is_reversed) else self.move_forward
            return threading.Thread(target=direction, args=(motor, 2,))
        return threading.Thread()  # Returns a dummy thread if no movement is needed

    def __should_fire(self, target_steps_x: float, target_steps_y: float, valid_target = False) -> bool:
        """ Determines if the turret should fire based on the target's position. """
        # TODO: probably need to adjust this to account for the turret's orientation
        dx, dy = abs(target_steps_x - self.current_x_steps), abs(target_steps_y - self.current_y_steps)
        return not self.friendly_mode and dx <= 2 and dy <= 2 and valid_target


    def interactive(self):
        """ Starts an interactive session. Key presses determine movement. """
        def fire_action(ch):
            if ch == "q":
                sys.exit(0)
            elif ch == "\n":
                self.fire()
        # parse keyboard input in a continuous interactive session
        #self.move_forward(self.sm_x, 1)
        #self.move_forward(self.sm_y, 1)
        #print('Commands: Pivot with (a) and (d). Tilt with (w) and (s). Fire with (Enter). Exit with (q)\n')
        self._parse_keyboard_input(
            motor=None,  # Motor selection is dynamic based on input
            is_reversed=False,  # Dynamically chosen per input
            additional_action=fire_action,
            valid_keys={'w', 's', 'a', 'd', 'q', '\n'},
            step_size=5
        )


    def fire(self):
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(RELAY_PIN, GPIO.LOW)

    @staticmethod
    def move_forward(motor, steps):
        """ Moves the stepper motor forward the specified number of steps.
            :param motor: The motor to move.
            :param steps: The number of steps to move.
        """
        #motor.step(steps, Adafruit_MotorHAT.FORWARD,  Adafruit_MotorHAT.INTERLEAVE)
        motor.step(steps, MotorKit.FORWARD,  MotorKit.INTERLEAVE)


    @staticmethod
    def move_backward(motor, steps):
        """ Moves the stepper motor backward the specified number of steps
            :param motor: The motor to move.
            :param steps: The number of steps to move.
        """
        #motor.step(steps, Adafruit_MotorHAT.BACKWARD, Adafruit_MotorHAT.INTERLEAVE)
        motor.step(steps, MotorKit.BACKWARD, MotorKit.INTERLEAVE)

    def __turn_off_motors(self):
        """ Recommended for auto-disabling motors on shutdown """
        #self.mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
        #self.mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(1).run(MotorKit.RELEASE)
        self.mh.getMotor(2).run(MotorKit.RELEASE)


# TODO: will be removing this for the refactored version, but may create a new file for the main function
if __name__ == "__main__":
    t = Turret(friendly_mode=False, specific_target=False)
    user_input = input("Choose an input mode: (1) Motion Detection, (2) Interactive\n")
    while user_input not in ["1", "2"]:
        user_input = input("Unknown input mode. Please choose from (1) Motion Detection, (2) Interactive\n")
    if user_input == "1":
        print("Using Motion Detection mode...")
        t.calibrate()
        show_video = get_user_confirmation("Show live video feed?")
        t.motion_detection(show_video=show_video)
    elif user_input == "2":
        print("Using Interactive mode...")
        if get_user_confirmation("Show live video feed?"):
            #thread.start_new_thread(MotionTracker.live_video, ())
            # added this to eliminate old Python 2 threading
            cam_feed_thread = threading.Thread(target = MotionTracker.live_video, args = (CAMERA_PORT))
            cam_feed_thread.start()
        t.interactive()
