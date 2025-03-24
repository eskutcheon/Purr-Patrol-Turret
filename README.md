# 🐾 Purr Patrol Turret
Purr Patrol Turret is a robotics project designed to detect and deter pets (primarily cats) from tampering with hazardous houseplants using a blast of water. Originally a class project for an AI Robotics course, the system has since evolved into a **modular, multi-mode turret platform** combining motion detection, deep learning object detection, real-time control, and physical actuation.

It integrates a simple turret system running locally on a Raspberry Pi using a webcam, a client-server model that sends image captures to a desktop PC's Flask server connected via LAN, and advanced tracking and targeting using object detection and classification networks running on CUDA. Using computer vision techniques and precision targeting logic, the system identifies when a pet is interacting with a plant and triggers a water spray deterrent.

Though the project has been extended to the point that nothing remains, it was originally based on [*HackerShackOfficial*'s `Tracking-Turret` repository](https://github.com/HackerShackOfficial/Tracking-Turret) that triggered an airsoft turret based on simple motion detection.


## Disclaimer
This project is still not fully realized due to some hardware constraints in the current turret. The codebase also needs some maintenance. Use your best judgement in tweaking things to your needs and feel free to create issues and discussion for help.


## Features

- **Instance Segmentation Mode**
    The primary mode of operation. Detects animals and plants using a deep learning model, calculates bounding box overlap and IoU scores, and targets the intersection for firing.

- **Motion Detection Mode**
    Monitors scene changes using background subtraction and average contour calculation and fires when motion is detected.

- **Interactive Mode**
    Manually control the turret via keyboard (WASD movement + spacebar fire), with an optional live camera feed.

- **Calibration Mode**
    Guides users to collect checkerboard images and calibrates the camera intrinsics matrix to improve aiming accuracy.



---

## System Architecture

```
+------------------+      +------------------------+
| Raspberry Pi     |      | Desktop PC             |
| (Turret Hardware)| <--> | Flask Server (Optional)|
+------------------+      +------------------------+
         |                            |
     [Camera Feed]            [Model Inference]
         |                            |
+------------------+      +------------------------+
| Motion Detection |      | Object Detection (CUDA)|
| Local Fire Logic |<---->| Detection Feedback     |
+------------------+      +------------------------+
```

The Raspberry Pi controls the physical turret and captures frames from a camera (either USB or Pi Camera v2). It can operate **autonomously** or offload detection to a **desktop server** via LAN for faster inference with CUDA-based models.

---

## Installation Instructions

### Getting Started
Clone this repository
```
git clone https://github.com/eskutcheon/Purr-Patrol-Turret.git
```

Navigate to the directory
```
cd Purr-Patrol-Turret
```

Create and activate virtual environment, e.g.
```bash
python -m venv .env
.env/Scripts/activate       # <-- WINDOWS
source .env/bin/activate    # <-- LINUX/MACOS
```


### Software Installation

#### On Raspberry Pi:
```bash
sudo apt update && sudo apt upgrade
sudo apt install python3-pip python3-opencv libatlas-base-dev
pip3 install -r requirements.txt
```

- For Raspberry Pi Camera Module v2 support, install `picamera2`:
```bash
sudo apt install -y libcamera-dev libcamera-apps
pip install picamera2
```

- If using a USB webcam:
```bash
sudo apt install python3-opencv
```

- Enable I2C for stepper motors:
[Adafruit I2C Setup Guide](https://learn.adafruit.com/adafruits-raspberry-pi-lesson-4-gpio-setup/configuring-i2c)

#### On Desktop Server:
```bash
pip install flask torch torchvision matplotlib
```

Then, launch the server:
```bash
python src/host/server.py
```

---

### Hardware Assembly

- Uses **Adafruit Motor HAT**, a **stepper motor turret base**, and a **relay-controlled water sprayer**.
- Hardware is adapted from [Tracking Turret by HackerShack](https://www.hackster.io/hackershack/raspberry-pi-motion-tracking-gun-turret-77fb0b).
- Assembly Video: [YouTube Link](https://www.youtube.com/watch?v=HoRPWUl_sF8)
- STL files for printing components are available in: `assets/STLs/`

> **⚠ Wiring instructions coming soon. Use common sense, GPIO safety practices, and double-check motor polarity and power requirements.**

---

## Run Modes

Launch the turret by running:

```bash
python main.py
```

Choose from the following modes:

- `interactive` – WASD keys + spacebar for manual testing
- `calibration` – capture checkerboard images for camera calibration
- `motion` – basic motion detection with optional firing
- `detect` – motion-triggered object detection + fire on plant-pet overlap

---

## Configuration Options

Edit `src/config/config.py` to modify runtime behavior:

| Variable            | Description                              | Default       |
|---------------------|------------------------------------------|---------------|
| `SAFE_MODE`         | Prevents firing (debug mode)             | `True`        |
| `MAX_FIRE_DURATION` | Max time for water spray (sec)           | `3`           |
| `MOTION_THRESHOLD`  | Motion mask sensitivity                  | `0.5`         |
| `CALIBRATION_FILE`  | Path to saved camera intrinsics          | `"..."`       |
| `CHECKERBOARD_SIZE` | Calibration checkerboard corners         | `(8, 6)`      |
| `SQUARE_SIZE`       | Real-world square size in mm             | `25.0`        |
| `CAMERA_PORT`       | USB webcam index                         | `0`           |
| `RELAY_PIN`         | GPIO pin controlling the sprayer         | `4`           |
| `ROTATION_RANGE`    | Pan/tilt angle bounds                    | `(-45, 45)`   |

**⚠ Remaining option still to come**

---

## Camera Setup and Testing

### Option A: USB Webcam

1. Plug in the webcam.
2. Test with:
```bash
python -m src.rpi.camera_opencv
```

### Option B: Pi Camera Module v2

1. Enable the camera:
```bash
sudo raspi-config
# Interface Options > Camera > Enable
```

2. Test with:
```bash
libcamera-hello
```

3. Or run turret’s feed viewer:
```bash
python -m src.rpi.camera_rpi
```

---


## Camera Calibration (Optional but Recommended)

1. Place a printed checkerboard in different positions.
2. Run:
```bash
python -m turret --mode calibration
```
3. Use `spacebar` to capture.
4. After a couple dozen captures at slightly different angles, press `q` to calibrate.
5. Parameters saved to `src/config/calibration.json`.

---

## Planned Features / Future Extensions

- YAML config loading for flexible runtime control
- Onboard YOLOv5 and other additional object detection networks; currently supported:
  - SSD Lite with a MobileNetv3 backbone
  - Faster RCNN with a ResNet50 backbone,
  - RetinaNet with ResNet50 backbone
- Multiple turret coordination (e.g. perimeter defense mode)
- REST API for mobile/web app control
- Logging/telemetry dashboard

---

## Debugging & Testing

Enable mock hardware to simulate motor/relay operations:
```python
DEBUG_MODE = True
```

Run interactive mode to test turret logic:
```bash
python main.py --mode interactive --debug
```

All turret operations respect the `SAFE_MODE` flag to avoid accidental firing.

---

## Project Structure

```
src/
├── rpi/                # Raspberry Pi camera & calibration modules
├── host/               # Server-side detection and tracking
├── turret/             # Core turret control, states, commands
├── config/             # Config files and constants
assets/STLs/            # STL files for turret hardware
results/                # Saved images and logs
```

---

## Credits

- Inspired by [HackerShack's Raspberry Pi Turret](https://www.hackster.io/hackershack/raspberry-pi-motion-tracking-gun-turret-77fb0b)
- Detection Models: PyTorch Hub (Faster R-CNN, RetinaNet, etc.)
- Hardware: Adafruit Motor HAT + Stepper motors + Relay module

---

## License

MIT License — use freely, but not for evil
