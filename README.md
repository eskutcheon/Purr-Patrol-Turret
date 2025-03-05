# Purr-Patrol-Turret
Purr Patrol Turret is a robotics project designed to detect and deter pets (primarily cats) from tampering with hazardous houseplants using a blast of water. It originally began as a class project for my AI Robotics course but was never fully realized due to time and hardware constraints.

It integrates a simple turret system ~~with an infrared camera running on an NVIDIA Jetson Nano~~ running locally on a Raspberry Pi using a webcam, a client-server model that sends image captures to a desktop PC's Flask server connected via LAN, and advanced tracking and targeting using object detection and classification networks running on CUDA.

The camera system identifies pets and plants within a scene, detects overlap of bounding boxes of pets and plants that happens when pets start tampering with plants, targets the center of their bounding box, and activates a precision jet of water from a sprayer.

Though the project has been extended to the point that nothing remains, it was originally based on [*HackerShackOfficial*'s `Tracking-Turret` repository](https://github.com/HackerShackOfficial/Tracking-Turret) that triggered an airsoft turret based on simple motion detection.

## Features
The turret system has 4 primary modes of operation:
- **object detection** - the primary mode pursued under the original concept for this project, this searches for the overlap of pets and plants (or any other combination of classes) using an object detection model and performs targeting and fires (with a delay) on the center of their bounding box intersection
- **motion detection** - fires indiscriminately on motion detection found by calculating the difference in contours in a still frame over time; a more lightweight pipeline is used as the alarm to trigger the instance segmentation model
- **interactive** - controlled by keyboard input, allowing a user to manually move the turret and fire when the "safety" settings are off; allows an optional simultaneous live video feed
- **calibration** - allows users to interactively direct the turret to a position and take a screenshot of the current live video feed to capture checkboard patterns for calibration, then calibration results are saved to JSON


NOTE: Plans are in place to allow for some flexibility in specifying the object classes to be targeted and for setting other parameters via CLI/YAML input




## Install Guide

**!! UNDER CONSTRUCTION !!**

### HackerShackOfficial's (Mostly) Original Instructions:
[How to set up I2C on your Raspberry Pi](https://learn.adafruit.com/adafruits-raspberry-pi-lesson-4-gpio-setup/configuring-i2c)

Install the Adafruit stepper motor HAT library.
```bash
sudo pip install git+https://github.com/adafruit/Adafruit-Motor-HAT-Python-Library
```

*Create and activate virtual environment.*


Clone this repository
```
git clone https://github.com/eskutcheon/Purr-Patrol-Turret.git
```

Navigate to the directory
```
cd Purr-Patrol-Turret
```

### Additional Instructions
1. Install requirements for the Raspberry Pi and primary desktop PC respectively via
TODO: add separate requirements files later or add a setup.py since the installation of Pytorch may differ and I still need to give instructions to build OpenCV from source
```
    pip install -r requirements_rpi.txt
    pip install -r requirements_pc.txt
```




## Setting Parameters

The turret system has a couple parameters that can be set from the `config.py` file:
```python
### User Parameters ###
MOTOR_X_REVERSED = False
MOTOR_Y_REVERSED = False
MAX_STEPS_X = 30
MAX_STEPS_Y = 15
RELAY_PIN = 22
#######################
```


## Running the Turret

**!! UNDER CONSTRUCTION !!**
