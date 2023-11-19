import board
import busio
from adafruit_motorkit import MotorKit
import time

# Create a default object, no changes to I2C address or frequency
kit = MotorKit(i2c=busio.I2C(board.SCL, board.SDA))

for i in range(100):
	kit.stepper1.onestep()
	time.sleep(0.01)

