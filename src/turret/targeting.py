from copy import deepcopy
from typing import Dict, Union, List, Tuple, Callable
from dataclasses import dataclass, asdict
import math
import numpy as np
from cv2 import undistortPoints
# look into later:
# from kornia.geometry import convert_points_to_homogeneous, convert_points_from_homogeneous

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, theta_x={self.theta_x}, theta_y={self.theta_y})"


# TODO: might actually want this in src/rpi/camera.py instead
@dataclass
class CameraCoordinates:
    """ represents a pixel or (u,v) coordinate in the camera image plane """
    u: float
    v: float
    # TODO: maybe store more fields (e.g., timestamp or confidence).

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(u={self.u}, v={self.v})"



@dataclass
class CalibrationParameters:
    """ Data class for calibration parameters (intrinsics, distortion, etc) for the camera attached to the turret """
    #* REFERENCE: https://www.mathworks.com/help/vision/ug/camera-calibration.html
    # Traditional intrinsics from OpenCV calibration:
    camera_matrix: np.ndarray = None   # shape (3,3)
    dist_coeffs: np.ndarray = None     # shape (N,) e.g. [k1, k2, p1, p2, k3,...]
    reprojection_error: float = 0.0
    # focal length in terms of distance from the camera to the projection plane
    focal_length: Tuple[Union[float, int]] = (1.0, 1.0)  # retrieve from camera_matrix[0,0], camera_matrix[1,1]
    # optical center / principal point
    optical_center: List[Union[float, int]] = (1.0, 1.0) # retrieve from from camera_matrix[0,2], camera_matrix[1,2]
    
    # radial distortion coefficients
    # radial_distortion: List[Union[float, int]] = (1.0, 1.0) # = [0, 0, 0]
    # # tangential distortion coefficients (when the lens and the image plane aren't parallel)
    # field_of_view: List[Union[float, int]] = (1.0, 1.0) # = [0, 0]
    
    # skew coefficient (0 for perpendicular axes)
    skew: Union[float, int] = 0

    # TODO read camera calibration parameters from a file elsewhere with the names above and pass the unpacked dict
    def load_from_json(self, filepath="calibration.json"):
        """ Load calibration parameters from a JSON file, hopefully created by the CameraCalibrator class """
        import json, os
        if not os.path.exists(filepath):
            print(f"[CALIB] No calibration file at {filepath}, retaining default values...")
            return
        with open(filepath, "r") as f:
            data = json.load(f)
        # camera_matrix is stored as a 3x3 list-of-lists, so convert to np.array
        cam_matrix = data.get("camera_matrix", None)
        if cam_matrix:
            self.camera_matrix = np.array(cam_matrix, dtype=np.float32)
            self.focal_length = (self.camera_matrix[0, 0], self.camera_matrix[1, 1]) # (fx, fy)
            self.optical_center = (self.camera_matrix[0, 2], self.camera_matrix[1, 2]) # (cx, cy)
        dc = data.get("dist_coeffs", None)
        if dc:
            self.dist_coeffs = np.array(dc, dtype=np.float32)
        self.reprojection_error = data.get("reprojection_error", 0.0)
        # If you store tangential/radial individually, parse them as well.
        print(f"[CALIB] Loaded calibration from {filepath} -> {self}")

    def __dict__(self) -> Dict[str, Union[float, int]]:
        return asdict(self)

    def __repr__(self) -> str:
        kwargs_str = ", ".join([f"{key}={val}" for key, val in self.__dict__().items()])
        return f"{self.__class__.__name__}({kwargs_str})"



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
            calibration.load_from_json(cfg.CALIBRATION_FILE)
        self.calibration = calibration

    def compute_angular_displacement(self, target_coord: TurretCoordinates) -> Tuple[float, float]:
        """ Using the turret's current position (x, y, theta_x, theta_y) and calibration fields (e.g. focal length),
            compute how many degrees we must move in pan (dx_deg) and tilt (dy_deg) to reach the target.
        """
        dx, dy = self.current_position.compute_displacement(target_coord)
        # e.g. focal_length might be [fx, fy]
        fx, fy = self.calibration.focal_length
        # since fx, fy from OpenCV calibration are typically pixel units, interpret them consistently for angles
        dthx = math.degrees(math.atan2(dx, fx))
        dthy = math.degrees(math.atan2(dy, fy))
        return dthx, dthy

    @staticmethod
    def compute_displacement_from_angles(
        angles: Tuple[float, float],
        focal_length: Tuple[float, float]
    ) -> Tuple[float, float]:
        """ (inverse of compute_angular_displacement) Compute the displacement needed to move by the given (in degrees) angular displacement """
        fx, fy = focal_length
        dthx, dthy = angles
        dx = fx * math.tan(math.radians(dthx))
        dy = fy * math.tan(math.radians(dthy))
        return dx, dy

    def update_current_position(self, dtheta: Tuple[float, float]) -> None:
        dx, dy = self.compute_displacement_from_angles(dtheta, self.calibration.focal_length)
        new_x = self.current_position.x + dx
        new_y = self.current_position.y + dy
        self.current_position.update(new_x, new_y, *dtheta)

    def reset_to_initial(self) -> None:
        """ set turret coordinates back to original position """
        self.current_position = deepcopy(self.initial_position)


    def from_camera_to_turret(self, cam_coord: CameraCoordinates) -> TurretCoordinates:
        """ convert pixel coordinates (u,v) into a TurretCoordinates object
            - If we have valid camera_matrix/dist_coeffs, undistort the point, compute angles from pinhole geometry, and return coordinates.
            - If we have no calibration loaded, fallback to known RPi V2 FOV parameters
        """
        if (self.calibration.camera_matrix is not None and
            self.calibration.dist_coeffs is not None and
            self.calibration.camera_matrix.size > 0 and
            self.calibration.dist_coeffs.size > 0):
            return self._undistort_and_convert(cam_coord)
        else:
            return self._simple_pixel_to_coords(cam_coord)

    def _undistort_and_convert(self, cam_coord: CameraCoordinates) -> TurretCoordinates:
        """
            1) Undistort (u,v) → normalized (xn, yn)
            2) angle_x = atan2(xn, 1), angle_y = atan2(yn, 1)   (in degrees)
            3) x = fx * tan(angle_x), y = fy * tan(angle_y)
            4) Return TurretCoordinates(x, y, angle_x, angle_y)
        """
        raw_pt = np.array([[[cam_coord.u, cam_coord.v]]], dtype=np.float32)  # shape (1,1,2)
        undist_pts = undistortPoints(raw_pt,
                                    cameraMatrix=self.calibration.camera_matrix,
                                    distCoeffs=self.calibration.dist_coeffs)
        # undist_pts shape => (1,1,2): (xn, yn)
        xn, yn = undist_pts[0, 0]
        angles = (math.degrees(math.atan2(xn, 1.0)), math.degrees(math.atan2(yn, 1.0)))
        # Now convert angles to turret system coordinates (assuming focal length is in pixels)
        dx, dy = self.compute_displacement_from_angles(angles, self.calibration.focal_length)
        return TurretCoordinates(x=dx, y=dy, theta_x=angles[0], theta_y=angles[1])

    def _simple_pixel_to_coords(self, cam_coord: CameraCoordinates) -> TurretCoordinates:
        """ Fallback if no real calibration is loaded using the camera described in docs/calibration_derivation.md
            - RPi Cam V2 FOV:  ~62.2° horizontally, 48.8° vertically
            - Full sensor resolution ~ 3280×2464
            Return turret coordinate with x=0, y=0, but angles = (deg_x, deg_y).
        """
        hfov = 62.2
        vfov = 48.8
        full_w, full_h = 3280.0, 2464.0
        # default: (cx, cy) = (1640, 1232)
        cx, cy = (full_w / 2.0), (full_h / 2.0)
        # du, dv = (u - cx, v - cy)
        du = cam_coord.u - cx
        dv = cam_coord.v - cy
        # fraction of the half-width => fraction * (HFOV/2)
        deg_x = (du / cx) * (hfov / 2.0)
        deg_y = (dv / cy) * (vfov / 2.0)
        # In fallback mode, just store angles so that x=0, y=0 => we interpret the angles as the entire aim offset from center
        return TurretCoordinates(x=0.0, y=0.0, theta_x=deg_x, theta_y=deg_y)