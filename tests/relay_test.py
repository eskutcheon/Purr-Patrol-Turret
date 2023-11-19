import Jetson.GPIO as GPIO
import time

# Define the pin number where the relay is connected
RELAY_PIN = 7  # Change this to the pin you are using

def main():
    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup(RELAY_PIN, GPIO.OUT)  # Relay pin set as output

    print("Starting demo now! Press CTRL+C to exit")
    try:
        while True:
            print("Turning the relay on")
            GPIO.output(RELAY_PIN, GPIO.HIGH)  # Turn relay on
            time.sleep(2)  # Wait for 2 seconds

            print("Turning the relay off")
            GPIO.output(RELAY_PIN, GPIO.LOW)  # Turn relay off
            time.sleep(2)  # Wait for 2 seconds
    finally:
        GPIO.cleanup()  # Clean up GPIO settings

if __name__ == '__main__':
    main()
