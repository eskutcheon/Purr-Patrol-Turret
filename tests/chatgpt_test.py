import time
import Adafruit_GPIO.I2C as I2C

# Assuming default address for Adafruit Motor Shield
MOTOR_SHIELD_I2C_ADDR = 0x60

# Initialize the I2C connection to the motor shield
motor_shield = I2C.Device(MOTOR_SHIELD_I2C_ADDR, 1)

# Function to send command to motor shield
def set_motor_speed(speed):
    # You'll need to refer to the motor shield documentation for the correct command format
    # Here's an example placeholder for sending a speed command
    motor_shield.write8(0x80, speed)

# Function to run the motor for a given duration
def run_motor(duration):
    # Set the motor speed to the desired value
    set_motor_speed(255)  # Example speed value
    time.sleep(duration)
    set_motor_speed(0)  # Stop the motor

# Run the motor for 2 seconds
run_motor(2)
