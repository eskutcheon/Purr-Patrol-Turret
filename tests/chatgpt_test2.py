from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_StepperMotor

import time
import atexit

# Create a default object, no changes to I2C address or frequency
mh = Adafruit_MotorHAT()

# Recommended to automatically disable motors on shutdown!
def turn_off_motors():
    mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)

atexit.register(turn_off_motors)

# 200 steps/revolution (1.8 degree) as per your motor's specs
stepper = mh.getStepper(200, 1)  # motor port #1

stepper.setSpeed(60)  # 60 RPM

# Move the motor with a specific number of steps
stepper.step(100, Adafruit_MotorHAT.FORWARD,  Adafruit_MotorHAT.SINGLE)
time.sleep(2)
turn_off_motors()
