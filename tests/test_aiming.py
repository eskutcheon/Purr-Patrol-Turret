import os, sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from src.turret.operations import TurretOperation
from src.turret.targeting import TargetingSystem, TurretCoordinates #, CalibrationParameters
from src.turret.command import AimCommand


def test_integration_aim_command_live():
    """ A cautionary test that tries to physically move the motors. Only run if your Pi is connected to the hardware! """
    #cal_params = CalibrationParameters(focal_length=[1.0, 200.0])
    targeting_system = TargetingSystem()
    current_pos = TurretCoordinates(0,0,0,0)
    op = TurretOperation(relay_pin=4, interactive=False)
    #cmd = AimCommand(op, targeting_system, current_pos, (0.8,-0.4))
    cmd = AimCommand(op, targeting_system, (0.8,-0.4))
    cmd.execute()
    # if the hardware actually moves, we watch it physically then check that final 'current_pos' is updated
    assert current_pos.x == 0.8
    assert current_pos.y == -0.4

test_integration_aim_command_live()