import cv2
import time
# TODO: probably need to replace usage of this with `multiprocessing`
import threading
import atexit
import sys
import termios
import contextlib
from copy import deepcopy
# TODO: replace this with the Jetson library and track down usage - Adafruit should still be the same since the motor driver is an Adafruit board
    # relevant methods only found in Turret class
#import RPi.GPIO as GPIO
import Jetson.GPIO as GPIO
#       This is apparently deprecated:
#       from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor, Adafruit_StepperMotor
# using their recommendation here:
#https://github.com/adafruit/Adafruit_CircuitPython_MotorKit
from adafruit_motorkit import MotorKit
from config import *
from motion_tracking2 import MotionTracker
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
    """
    Class used for turret control.
    """
    # TODO: figure out this stepper motor operation above all else
    # TODO: may also need to setup the sprayer pins here
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
        # Relay
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(RELAY_PIN, GPIO.OUT)
        GPIO.output(RELAY_PIN, GPIO.LOW)


    def _parse_keyboard_input(self, ch: str, motor, is_reversed: bool, step_size: int = 5):
        """ Parses a single character of keyboard input and applies the corresponding motor action.
            :param ch: Keyboard character input.
            :param motor: Motor object to control.
            :param is_reversed: Boolean indicating if the motor direction is reversed.
            :param step_size: Number of steps for the motor movement.
        """
        command_dict = {
            "w": lambda: self.move_forward(motor, step_size) if not is_reversed else self.move_backward(motor, step_size),
            "s": lambda: self.move_backward(motor, step_size) if not is_reversed else self.move_forward(motor, step_size),
            "a": lambda: self.move_backward(motor, step_size) if is_reversed else self.move_forward(motor, step_size),
            "d": lambda: self.move_forward(motor, step_size) if is_reversed else self.move_backward(motor, step_size),
        }
        if ch in command_dict:
            movement_func: callable = command_dict[ch]
            movement_func()


    def calibrate(self):
        """ Waits for input to calibrate the turret's axis. """
        print("Please calibrate the tilt of the gun so that it is level. Commands: (w) moves up, " \
            "(s) moves down. Press (enter) to finish.\n")
        self.__calibrate_y_axis()
        print("Please calibrate the yaw of the gun so that it aligns with the camera. Commands: (a) moves left, " \
            "(d) moves right. Press (enter) to finish.\n")
        self.__calibrate_x_axis()
        print("Calibration finished.")

    def __calibrate_x_axis(self):
        """ Waits for input to calibrate the x axis (yaw) of the turret. """
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch or ch == "\n":
                        break
                    # changed this to check the keys and just simplify the conditional chain
                    self._parse_keyboard_input(ch, self.sm_x, MOTOR_X_REVERSED, step_size=5)
            except (KeyboardInterrupt, EOFError):
                print("Error: Unable to calibrate turret. Exiting...")
                sys.exit(1)

    def __calibrate_y_axis(self):
        """ Waits for input to calibrate the y axis (tilt) of the turret. """
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch or ch == "\n":
                        break
                    # changed this to check the keys and just simplify the conditional chain
                    # NOTE: may need to be elif like the x-axis calibration
                    self._parse_keyboard_input(ch, self.sm_y, MOTOR_Y_REVERSED, step_size=5)
            except (KeyboardInterrupt, EOFError):
                print("Error: Unable to calibrate turret. Exiting...")
                sys.exit(1)

    def motion_detection(self, show_video=False):
        """ Uses the camera to move the turret. OpenCV ust be configured to use this """
        MotionTracker.find_motion(self.__move_axis, show_video=show_video)


    def __move_axis(self, contour, frame):
        """ Moves the turret based on the detected object's position - used as a callback function in the MotionTracker """
        should_fire, target_coord = False, []
        (v_h, v_w) = frame.shape[-2:]
        # TODO: need to ensure `frame` is a tensor object - really, if contour is a boolean array, it would be really really easy to recreate cv2.boundingRect
        if self.target_detector is None:
            (x, y, w, h) = cv2.boundingRect(contour)
            target_coord = [x + w / 2, y + h / 2]
        else:
            should_fire, target_coord = self.target_detector.scan_frame(frame)
        target_steps_x = (2 * MAX_STEPS_X * target_coord[0] / v_w) - MAX_STEPS_X
        target_steps_y = (2 * MAX_STEPS_Y * target_coord[1] / v_h) - MAX_STEPS_Y
        print("x: %s, y: %s" % (str(target_steps_x), str(target_steps_y)))
        print("current x: %s, current y: %s" % (str(self.current_x_steps), str(self.current_y_steps)))
        t_x = self.__move_motor(self.current_x_steps, target_steps_x, self.sm_x, MOTOR_X_REVERSED)
        t_y = self.__move_motor(self.current_y_steps, target_steps_y, self.sm_y, MOTOR_Y_REVERSED)
        t_fire = threading.Thread(target=Turret.fire) if self.__should_fire(target_steps_x, target_steps_y, should_fire) else None
        t_x.start()
        t_y.start()
        if t_fire:
            t_fire.start()
        t_x.join()
        t_y.join()
        if t_fire:
            t_fire.join()

    def __move_motor(self, current_steps, target_steps, motor, is_reversed) -> threading.Thread:
        delta = target_steps - current_steps
        if delta != 0:
            direction = self.move_backward if (delta < 0) ^ is_reversed else self.move_forward
            return threading.Thread(target=direction, args=(motor, 2,))
        return threading.Thread()  # Returns a dummy thread if no movement is needed

    def __should_fire(self, target_steps_x: float, target_steps_y: float, valid_target = False) -> bool:
        """ Determines if the turret should fire based on the target's position. """
        # TODO: may need to adjust this to account for the turret's orientation
        dx, dy = abs(target_steps_x - self.current_x_steps), abs(target_steps_y - self.current_y_steps)
        return not self.friendly_mode and dx <= 2 and dy <= 2 and valid_target


    def interactive(self):
        """ Starts an interactive session. Key presses determine movement. """
        self.move_forward(self.sm_x, 1)
        self.move_forward(self.sm_y, 1)
        print('Commands: Pivot with (a) and (d). Tilt with (w) and (s). Exit with (q)\n')
        # TODO: abstract this into some base functionality since it's so long - maybe replace the calibration code with it also
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch or ch == "q":
                        break
                    motor = self.sm_x if ch in ["a", "d"] else self.sm_y
                    is_reversed = MOTOR_X_REVERSED if ch in ["a", "d"] else MOTOR_Y_REVERSED
                    if ch == "\n":
                        Turret.fire()
                    else:
                        self._parse_keyboard_input(ch, motor, is_reversed)
            except (KeyboardInterrupt, EOFError):
                pass

    @staticmethod
    def fire():
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
        #self.mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
        #self.mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(1).run(MotorKit.RELEASE)
        self.mh.getMotor(2).run(MotorKit.RELEASE)
        self.mh.getMotor(3).run(MotorKit.RELEASE)
        self.mh.getMotor(4).run(MotorKit.RELEASE)


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
