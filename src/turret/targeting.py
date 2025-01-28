import time
from typing import Dict, Union, List
import json
from copy import deepcopy
from dataclasses import dataclass
import RPi.GPIO as GPIO
import numpy as np

# local imports
from ..config import config as cfg




@dataclass
class TurretPosition:
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

    def compute_dtheta(self, target_coord: List[Union[float, int]], focal_length: List[Union[float, int]]) -> List[float]:
        """ Compute the change in angles needed to move to the target coordinates """
        displacement = self.compute_displacement(target_coord)
        # FIXME: not finished - this should use camera parameters to compute the angles
        return [np.rad2deg(np.arctan(displacement[i] / focal_length[i])) for i in range(2)]

    def __dict__(self) -> Dict[str, Union[float, int]]:
        return {"x": self.x, "y": self.y, "theta_x": self.theta_x, "theta_y": self.theta_y}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, theta_x={self.theta_x}, theta_y={self.theta_y})"


@dataclass
class CalibrationParameters:
    """ Data class for calibration parameters for the camera attached to the turret """
    #* REFERENCE: https://www.mathworks.com/help/vision/ug/camera-calibration.html
    # focal length in terms of distance from the camera to the projection plane
        # to convert to pixel coordinates, I think you need to divide by width/height of a pixel on the projection plane
    focal_length: List[Union[float, int]] # = [0, 0]
    # optical center / principal point
    optical_center: List[Union[float, int]] # = [0, 0]
    # radial distortion coefficients
    radial_distortion: List[Union[float, int]] # = [0, 0, 0]
    # tangential distortion coefficients (when the lens and the image plane aren't parallel)
    tan_distortion: List[Union[float, int]] # = [0, 0]
    # skew coefficient (0 for perpendicular axes)
    skew: Union[float, int] = 0

    # TODO read camera calibration parameters from a file elsewhere with the names above and pass the unpacked dict
    def load_parameters_from_json(self):
        """ parse dictionary of calibration parameters passed in after reading from calibration.json """
        # still not entirely sure how I want to represent them in the json file - whether as their usual variable names or the more descriptive names above
        # NOTE: will need to convert them to a mutable type to update them
        pass

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


