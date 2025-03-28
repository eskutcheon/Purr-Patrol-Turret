""" Stores centralized configuration values for both the Pi and desktop server """

DEBUG_MODE = True # if True, execution will print debug messages to console instead of operating any hardware

### Turret Operational Parameters ###
SAFE_MODE = False  # if True, turret will not fire (good for debugging or in testing automated modes)
DEGREES_PER_STEP = 1.8
INTERACTIVE_STEP_MULT = 2
MAX_FIRE_DURATION = 3  # in seconds
# TODO: make this a length 4 tuple (xmin, ymin, xmax, ymax) for different axis limits
ROTATION_RANGE = (-45, 45)


# hardware config settings:
CAMERA_PORT = 0
RELAY_PIN = 4
MOTOR_X_REVERSED = False
MOTOR_Y_REVERSED = False

LIVE_FEED_DELAY = 0.1  # in seconds

#######################
#! TEMPORARY - REMOVE LATER after fixing calibration.json usage
FOCAL_LENGTH = (1, 1)

# tracking and detection config settings:
MOTION_THRESHOLD = 0.5  # in pixels
MOTION_UPDATE_INTERVAL = 5  # in seconds


# calibration config settings:
CHECKERBOARD_SIZE = (8,6) # actually the number of inner corners in the checkerboard pattern
SQUARE_SIZE = 25.0 # in mm
CALIBRATION_FILE = "src/config/calibration.json"
VISUALIZATION_DIR = "results/"
####################################################################################################
#~ new code for the server logic
####################################################################################################
"""
    SERVER_IP: IP address of the desktop server
    SERVER_PORT: Port for the Flask server
    CAMERA_PORT: Port for the Raspberry Pi camera
    MODES: Supported modes (e.g., interactive, network)
"""