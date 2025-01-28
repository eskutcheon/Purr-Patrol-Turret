import RPi.GPIO as GPIO
import time

class PowerRelayInterface:
    """ Interface to a relay module connected to the Raspberry Pi GPIO. """
    def __init__(self, relay_pin: int):
        """ Initialize the relay module interface with a GPIO pin. """
        self.relay_pin = relay_pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.relay_pin, GPIO.OUT)
        GPIO.output(self.relay_pin, GPIO.LOW)  # defaults to LOW (off)

    def run_n_seconds(self, duration: float = 3):
        """ Fire the relay module for a specified duration. """
        GPIO.output(self.relay_pin, GPIO.HIGH)
        time.sleep(duration)  # duration of fire
        GPIO.output(self.relay_pin, GPIO.LOW)