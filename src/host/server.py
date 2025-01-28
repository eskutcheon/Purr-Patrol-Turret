"""
A Flask server that listens for incoming frames, processes them, and sends back commands to the Pi.
- Coordinates motion detection, object detection, and turret control.

Key Routes:
/process_frame:
    - Receives a frame, runs motion detection, and sends a response (e.g., "prepare turret").
/target:
    - Runs object detection, calculates targeting coordinates, and sends them back to the Pi.
"""