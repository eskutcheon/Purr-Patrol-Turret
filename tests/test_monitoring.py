# main_interactive.py
import os, sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from src.turret.operations import TurretOperator
from src.turret.controller import TurretController
from src.config.config import RELAY_PIN



def test_motion_tracking_default():
    """ testing the motion tracking mode by it setting defaults for the motion_detector """
    operation = TurretOperator(relay_pin=RELAY_PIN, interactive=True)
    controller = TurretController(operation)
    controller.enter_motion_tracking_only_mode(save_detections=True)

def test_object_detection_default():
    """ testing the object detection mode by it setting defaults for the motion detector and detection pipeline """
    operation = TurretOperator(relay_pin=RELAY_PIN, interactive=True)
    controller = TurretController(operation)
    controller.enter_motion_tracking_detection_mode(save_detections=True)


if __name__ == "__main__":
    test_motion_tracking_default()
    #test_object_detection_default()