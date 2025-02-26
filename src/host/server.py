"""
A Flask server that listens for incoming frames, processes them, and sends back commands to the Pi.
- Coordinates motion detection, object detection, and turret control.

Key Routes:
/process_frame:
    - Receives a frame, runs motion detection, and sends a response (e.g., "prepare turret").
/target:
    - Runs object detection, calculates targeting coordinates, and sends them back to the Pi.
"""





from flask import Flask

app = Flask(__name__)

@app.route('/api/detect', methods=['POST'])
def detect():
    # read image bytes from request
    # optionally run motion detection first if the Pi didn’t
    # if motion => run detection pipeline
    # return detection_feedback JSON
    raise NotImplementedError("Detection endpoint not implemented yet")