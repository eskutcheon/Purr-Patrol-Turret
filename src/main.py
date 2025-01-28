"""
Entry point for running the Raspberry Pi code
    Handles:
        - Loading configurations (e.g., mode selection)
        - Initializing components (e.g., camera, turret, messaging)
        - Running the selected mode (e.g., interactive, network)
    Key Responsibilities:
        - Parse user input (interactive vs network mode)
        - Start the event loop for network-controlled turret operation
        - should have an alternative launch mode for interactive control





"""

from turret.controller import TurretController
from turret.state import IdleState, AimingState, FiringState


# Testing for now
if __name__ == "__main__":
    # Mocking TurretOperation for illustration
    class MockOperation:
        def move_to_target(self, target_coord):
            print(f"Moving turret to {target_coord}")

        def fire(self):
            print("Firing turret!")

        def stop_all(self):
            print("Stopping all operations.")

    # Initialize operation and control
    operation = MockOperation()
    control = TurretController(operation)
    # Example Workflow
    try:
        print("Setting state to Aiming...")
        control.set_state(AimingState(target_coord=(100, 200)))
        control.handle_state()  # Aiming and transitioning to firing
        control.handle_state()  # Firing and transitioning to idle
    except KeyboardInterrupt:
        print("Exiting gracefully...")