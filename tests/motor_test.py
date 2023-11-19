import Jetson.GPIO as GPIO
import time

# gpio mode using specific pin numbering
GPIO.setmode(GPIO.TEGRA_SOC)
# gpio pins the motor shield is using
SDA_PIN = "SOC_GPIO_02" # D2 on the nano
SCL_PIN = "SOC_GPIO_03" # D3 on the nano
# setting up the gpio pins
GPIO.setup(SDA_PIN, GPIO.OUT)
GPIO.setup(SCL_PIN, GPIO.OUT)

# function to run the motor
def run_motor(duration):
	GPIO.output(SDA_PIN, GPIO.HIGH)
	GPIO.output(SCL_PIN, GPIO.HIGH)
	time.sleep(duration)
	GPIO.output(SDA_PIN, GPIO.LOW)
	GPIO.output(SCL_PIN, GPIO.LOW)

run_motor(2)
GPIO.cleanup()
