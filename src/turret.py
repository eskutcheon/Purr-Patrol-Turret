try:
    import cv2
except Exception as e:
    print("Warning: OpenCV not installed. To use motion detection, make sure you've properly configured OpenCV.")
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
from motion_tracking import VideoUtils

@contextlib.contextmanager
def raw_mode(file):
    """
    Magic function that allows key presses.
    :param file:
    :return:
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
    def __init__(self, friendly_mode=True):
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

    def calibrate(self):
        """
        Waits for input to calibrate the turret's axis
        :return:
        """
        print("Please calibrate the tilt of the gun so that it is level. Commands: (w) moves up, " \
            "(s) moves down. Press (enter) to finish.\n")
        self.__calibrate_y_axis()
        print("Please calibrate the yaw of the gun so that it aligns with the camera. Commands: (a) moves left, " \
            "(d) moves right. Press (enter) to finish.\n")
        self.__calibrate_x_axis()
        print("Calibration finished.")

    def __calibrate_x_axis(self):
        """
        Waits for input to calibrate the x axis
        :return:
        """
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break
                    elif ch == "a":
                        if MOTOR_X_REVERSED:
                            Turret.move_backward(self.sm_x, 5)
                        else:
                            Turret.move_forward(self.sm_x, 5)
                    elif ch == "d":
                        if MOTOR_X_REVERSED:
                            Turret.move_forward(self.sm_x, 5)
                        else:
                            Turret.move_backward(self.sm_x, 5)
                    elif ch == "\n":
                        break
            except (KeyboardInterrupt, EOFError):
                print("Error: Unable to calibrate turret. Exiting...")
                sys.exit(1)

    def __calibrate_y_axis(self):
        """
        Waits for input to calibrate the y axis.
        :return:
        """
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break
                    if ch == "w":
                        if MOTOR_Y_REVERSED:
                            Turret.move_forward(self.sm_y, 5)
                        else:
                            Turret.move_backward(self.sm_y, 5)
                    elif ch == "s":
                        if MOTOR_Y_REVERSED:
                            Turret.move_backward(self.sm_y, 5)
                        else:
                            Turret.move_forward(self.sm_y, 5)
                    elif ch == "\n":
                        break
            except (KeyboardInterrupt, EOFError):
                print("Error: Unable to calibrate turret. Exiting...")
                sys.exit(1)

    def motion_detection(self, show_video=False):
        """
        Uses the camera to move the turret. OpenCV ust be configured to use this.
        :return:
        """
        VideoUtils.find_motion(self.__move_axis, show_video=show_video)

    def __move_axis(self, contour, frame):
        (v_h, v_w) = frame.shape[:2]
        (x, y, w, h) = cv2.boundingRect(contour)
        # find height
        target_steps_x = (2*MAX_STEPS_X * (x + w / 2) / v_w) - MAX_STEPS_X
        target_steps_y = (2*MAX_STEPS_Y*(y+h/2) / v_h) - MAX_STEPS_Y
        print("x: %s, y: %s" % (str(target_steps_x), str(target_steps_y)))
        print("current x: %s, current y: %s" % (str(self.current_x_steps), str(self.current_y_steps)))
        t_x = threading.Thread()
        t_y = threading.Thread()
        t_fire = threading.Thread()
        # move x
        if (target_steps_x - self.current_x_steps) > 0:
            self.current_x_steps += 1
            if MOTOR_X_REVERSED:
                t_x = threading.Thread(target=Turret.move_forward, args=(self.sm_x, 2,))
            else:
                t_x = threading.Thread(target=Turret.move_backward, args=(self.sm_x, 2,))
        elif (target_steps_x - self.current_x_steps) < 0:
            self.current_x_steps -= 1
            if MOTOR_X_REVERSED:
                t_x = threading.Thread(target=Turret.move_backward, args=(self.sm_x, 2,))
            else:
                t_x = threading.Thread(target=Turret.move_forward, args=(self.sm_x, 2,))
        # move y
        if (target_steps_y - self.current_y_steps) > 0:
            self.current_y_steps += 1
            if MOTOR_Y_REVERSED:
                t_y = threading.Thread(target=Turret.move_backward, args=(self.sm_y, 2,))
            else:
                t_y = threading.Thread(target=Turret.move_forward, args=(self.sm_y, 2,))
        elif (target_steps_y - self.current_y_steps) < 0:
            self.current_y_steps -= 1
            if MOTOR_Y_REVERSED:
                t_y = threading.Thread(target=Turret.move_forward, args=(self.sm_y, 2,))
            else:
                t_y = threading.Thread(target=Turret.move_backward, args=(self.sm_y, 2,))
        # fire if necessary
        if not self.friendly_mode:
            if abs(target_steps_y - self.current_y_steps) <= 2 and abs(target_steps_x - self.current_x_steps) <= 2:
                t_fire = threading.Thread(target=Turret.fire)
        t_x.start()
        t_y.start()
        t_fire.start()
        t_x.join()
        t_y.join()
        t_fire.join()

    def interactive(self):
        """
        Starts an interactive session. Key presses determine movement.
        :return:
        """
        Turret.move_forward(self.sm_x, 1)
        Turret.move_forward(self.sm_y, 1)
        print('Commands: Pivot with (a) and (d). Tilt with (w) and (s). Exit with (q)\n')
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch or ch == "q":
                        break
                    if ch == "w":
                        if MOTOR_Y_REVERSED:
                            Turret.move_forward(self.sm_y, 5)
                        else:
                            Turret.move_backward(self.sm_y, 5)
                    elif ch == "s":
                        if MOTOR_Y_REVERSED:
                            Turret.move_backward(self.sm_y, 5)
                        else:
                            Turret.move_forward(self.sm_y, 5)
                    elif ch == "a":
                        if MOTOR_X_REVERSED:
                            Turret.move_backward(self.sm_x, 5)
                        else:
                            Turret.move_forward(self.sm_x, 5)
                    elif ch == "d":
                        if MOTOR_X_REVERSED:
                            Turret.move_forward(self.sm_x, 5)
                        else:
                            Turret.move_backward(self.sm_x, 5)
                    elif ch == "\n":
                        Turret.fire()
            except (KeyboardInterrupt, EOFError):
                pass

    @staticmethod
    def fire():
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(RELAY_PIN, GPIO.LOW)

    @staticmethod
    def move_forward(motor, steps):
        """
        Moves the stepper motor forward the specified number of steps.
        :param motor:
        :param steps:
        :return:
        """
        #motor.step(steps, Adafruit_MotorHAT.FORWARD,  Adafruit_MotorHAT.INTERLEAVE)
        motor.step(steps, MotorKit.FORWARD,  MotorKit.INTERLEAVE)
        

    @staticmethod
    def move_backward(motor, steps):
        """
        Moves the stepper motor backward the specified number of steps
        :param motor:
        :param steps:
        :return:
        """
        #motor.step(steps, Adafruit_MotorHAT.BACKWARD, Adafruit_MotorHAT.INTERLEAVE)
        motor.step(steps, MotorKit.BACKWARD, MotorKit.INTERLEAVE)

    def __turn_off_motors(self):
        """
        Recommended for auto-disabling motors on shutdown!
        :return:
        """
        #self.mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
        #self.mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
        #self.mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
        #self.mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)

        self.mh.getMotor(1).run(MotorKit.RELEASE)
        self.mh.getMotor(2).run(MotorKit.RELEASE)
        self.mh.getMotor(3).run(MotorKit.RELEASE)
        self.mh.getMotor(4).run(MotorKit.RELEASE)

if __name__ == "__main__":
    t = Turret(friendly_mode=False)
    user_input = input("Choose an input mode: (1) Motion Detection, (2) Interactive\n")
    if user_input == "1":
        t.calibrate()
        if input("Live video? (y, n)\n").lower() == "y":
            t.motion_detection(show_video=True)
        else:
            t.motion_detection()
    elif user_input == "2":
        if input("Live video? (y, n)\n").lower() == "y":
            #thread.start_new_thread(VideoUtils.live_video, ())
            # added this to eliminate old Python 2 threading
            cam_feed_thread = threading.Thread(target = VideoUtils.live_video, args = (CAMERA_PORT))
            cam_feed_thread.start()
        t.interactive()
    else:
        print("Unknown input mode. Please choose a number (1) or (2)")
