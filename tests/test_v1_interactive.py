# main_interactive.py
import os, sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from src.turret.operations import TurretOperation
from src.turret.controller import TurretController
from src.config.config import RELAY_PIN

def main():
    operation = TurretOperation(relay_pin=RELAY_PIN, interactive=True)
    controller = TurretController(operation)
    controller.interactive_mode()
    # keep the script alive so that the user input loop can run
    # In practice you might do something more elegant
    import time
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        operation.cleanup()

if __name__ == "__main__":
    main()
