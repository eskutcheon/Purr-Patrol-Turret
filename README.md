# Purr-Patrol-Turret
Purr Patrol Turret is a robotics project designed to detect and deter pets (primarily cats) from tampering with hazardous houseplants using a blast of water. It originally began as a class project for my AI Robotics course but was never fully realized due to time and hardware constraints.

It integrates a simple turret system with an infrared camera running on an NVIDIA Jetson Nano for advanced tracking and targeting using object detection and classification networks running on CUDA.

The camera system identifies pets and plants within a scene, detects overlap of bounding boxes of pets and plants that happens when pets start tampering with plants, targets the center of their bounding box, and activates a precision jet of water from a sprayer

Though the project has been heavily extended since inception, it was originally based on [*HackerShackOfficial*'s `Tracking-Turret` repository](https://github.com/HackerShackOfficial/Tracking-Turret) that activated an airsoft turret based on simple motion detection.


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


# Running the Turret

**!! UNDER CONSTRUCTION !!**
