import time
from adafruit_motorkit import MotorKit
#import Jetson.GPIO as GPIO
import RPi.GPIO as GPIO

kit = MotorKit()

for i in range(5):
    kit.stepper1.onestep()
    time.sleep(1)