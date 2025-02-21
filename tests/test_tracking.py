import os, sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from src.rpi.camera import CameraFeed
from src.host.tracking import MotionDetection

def test_basic_motion_tracking():
    try:
        with CameraFeed(camera_port=0) as cam:
            motion_det = MotionDetection()
            # capture some frames -> feed them into the motion detector
            frame1 = cam.capture_frame()
            result1 = motion_det.process_frame(frame1)
            assert result1 is None, "First frame shouldn't detect motion"
            frame2 = cam.capture_frame()
            result2 = motion_det.process_frame(frame2)
            # wave your hand in front of the camera and you might get a contour
            print("Contours found:", result2)
    except RuntimeError as e:
        print("Skipping test because camera might be unavailable:", e)


test_basic_motion_tracking()