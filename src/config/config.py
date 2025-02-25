""" Stores centralized configuration values for both the Pi and desktop server """

### User Parameters ###
DEGREES_PER_STEP = 1.8
INTERACTIVE_STEP_MULT = 2
# TODO: make this a length 4 tuple (xmin, ymin, xmax, ymax) for different axis limits
ROTATION_RANGE = (-60, 60)
MOTOR_X_REVERSED = False
MOTOR_Y_REVERSED = False
MAX_STEPS_X = 30
MAX_STEPS_Y = 15
RELAY_PIN = 4
# added this one in case we need it
CAMERA_PORT = 0
CALIBRATION_FILE = "calibration.json"
#######################
#! TEMPORARY - REMOVE LATER after fixing calibration.json usage
FOCAL_LENGTH = (1, 1)



CHECKERBOARD_SIZE = (9,6)
####################################################################################################
#~ new code for the server logic
####################################################################################################
"""
    Key Variables:
    SERVER_IP: IP address of the desktop server
    SERVER_PORT: Port for the Flask server
    CAMERA_PORT: Port for the Raspberry Pi camera
    MODES: Supported modes (e.g., interactive, network)
"""