from copy import deepcopy
from typing import Dict, Union, List, Tuple
from dataclasses import dataclass
import math

# local imports
from ..config import config as cfg





@dataclass
class TurretCoordinates:
    x: Union[float, int]
    y: Union[float, int]
    theta_x: Union[float, int]
    theta_y: Union[float, int]

    def update(self, x: Union[float, int], y: Union[float, int], dtheta_x: Union[float, int], dtheta_y: Union[float, int]):
        self.x = x
        self.y = y
        self.theta_x += dtheta_x
        self.theta_y += dtheta_y

    def compute_displacement(self, target_coord: List[Union[float, int]]) -> List[Union[float, int]]:
        """ Compute the displacement needed to move to the target coordinates """
        return [target_coord[0] - self.x, target_coord[1] - self.y]

    def compute_displacement_from_angles(self, angles: List[Union[float, int]], focal_length: List[Union[float, int]]) -> List[float]:
        """ Compute the displacement needed to move by the given (in degrees) angular displacement """
        fx, fy = focal_length
        dthx, dthy = angles
        dx = fx * math.tan(math.radians(dthx))
        dy = fy * math.tan(math.radians(dthy))
        return dx, dy

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, theta_x={self.theta_x}, theta_y={self.theta_y})"


@dataclass
class CameraCoordinates(TurretCoordinates):
    pass
    # TODO: add an offset parameter (in world coordinates) between the turret muzzle and camera lens



@dataclass
class CalibrationParameters:
    """ Data class for calibration parameters for the camera attached to the turret """
    #* REFERENCE: https://www.mathworks.com/help/vision/ug/camera-calibration.html
    # focal length in terms of distance from the camera to the projection plane
        # to convert to pixel coordinates, I think you need to divide by width/height of a pixel on the projection plane
    focal_length: Tuple[Union[float, int]] = (1.0, 1.0)
    # optical center / principal point
    optical_center: List[Union[float, int]] = (1.0, 1.0) # = [0, 0]
    # radial distortion coefficients
    radial_distortion: List[Union[float, int]] = (1.0, 1.0) # = [0, 0, 0]
    # tangential distortion coefficients (when the lens and the image plane aren't parallel)
    tan_distortion: List[Union[float, int]] = (1.0, 1.0) # = [0, 0]
    # skew coefficient (0 for perpendicular axes)
    skew: Union[float, int] = 0

    # TODO read camera calibration parameters from a file elsewhere with the names above and pass the unpacked dict
    def load_from_json(self, filepath="calibration.json"):
        import json, os
        if not os.path.exists(filepath):
            print(f"No calibration file at {filepath}, using defaults.")
            return
        with open(filepath, "r") as f:
            data = json.load(f)
        self.focal_length = data.get("focal_length", self.focal_length)
        self.optical_center = data.get("optical_center", self.optical_center)
        # etc. for the rest
        print(f"Loaded calibration: {self}")

    def __dict__(self) -> Dict[str, Union[float, int]]:
        return {
            "focal_length": self.focal_length,
            "optical_center": self.optical_center,
            "radial_distortion": self.radial_distortion,
            "tan_distortion": self.tan_distortion,
            "skew": self.skew
        }

    def __repr__(self) -> str:
        kwargs = ", ".join([f"{key}={val}" for key, val in self.__dict__().items()])
        return f"{self.__class__.__name__}({kwargs})"



class TargetingSystem:
    """ encapsulate the targeting system with the world and camera coordinates and the projection between them """
    def __init__(self, current_position: TurretCoordinates = None, calibration: CalibrationParameters = None):
        if current_position is None:
            current_position = TurretCoordinates(0, 0, 0, 0)
        self.current_position = current_position
        self.initial_position = deepcopy(current_position)
        if calibration is None:
            # TODO: add default json file with calibration parameters
            calibration = CalibrationParameters()
        self.calibration = calibration

    def compute_angular_displacement(self, target_coord: TurretCoordinates) -> Tuple[float, float]:
        """ Using the turret's current position (x, y, theta_x, theta_y) and calibration fields (e.g. focal length),
            compute how many degrees we must move in pan (dx_deg) and tilt (dy_deg).
        """
        dx, dy = self.current_position.compute_displacement(target_coord)
        # e.g. focal_length might be [fx, fy]
        fx, fy = self.calibration.focal_length
        dthx = math.degrees(math.atan(dx / fx))
        dthy = math.degrees(math.atan(dy / fy))
        return (dthx, dthy)

    @staticmethod
    def compute_displacement_from_angles(angles: List[Union[float, int]], focal_length: List[Union[float, int]]) -> List[float]:
        """ Compute the displacement needed to move by the given (in degrees) angular displacement """
        fx, fy = focal_length
        dthx, dthy = angles
        dx = fx * math.tan(math.radians(dthx))
        dy = fy * math.tan(math.radians(dthy))
        return dx, dy

    def update_current_position(self, dtheta: List[Union[float, int]]) -> None:
        dx, dy = self.compute_displacement_from_angles(dtheta, self.calibration.focal_length)#cfg.FOCAL_LENGTH)
        self.current_position.update(self.current_position.x + dx, self.current_position.y + dy, *dtheta)

    def reset_to_initial(self) -> None:
        self.current_position = self.initial_position