# main_interactive.py
import os, sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from src.turret.operations import TurretOperator
from src.turret.controller import TurretController
from src.config.config import RELAY_PIN

def test_interactive(show_video: bool = False):
    operation = TurretOperator(relay_pin=RELAY_PIN, interactive=True)
    controller = TurretController(operation)
    controller.enter_interactive_mode(show_video=show_video)

def test_calibration():
    """ test the calibration mode """
    operation = TurretOperator(relay_pin=RELAY_PIN, interactive=True)
    controller = TurretController(operation)
    controller.enter_calibration_mode()

if __name__ == "__main__":
    #test_interactive(True)
    test_calibration()