import os
import cv2
import json
import numpy as np
from datetime import datetime


class CameraCalibrator:
    """ Encapsulates logic for capturing checkerboard corners, running calibrateCamera, and saving parameters to JSON. """
    def __init__(
        self,
        checkerboard_size=(9, 6),
        square_size=1.0, # actual dimension of each square [arbitrary unit]
        save_dir="calibration_images"
    ):
        # 3D "object points" for a canonical checkerboard (z=0 plane)
        self.checkerboard_size = checkerboard_size
        self.objpoints = []   # list of 3D points for all images
        self.imgpoints = []   # list of 2D corner points for all images
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        w, h = self.checkerboard_size
        self._canonical_objp = np.zeros((w * h, 3), np.float32)
        # fill in (x,y) for each corner in the plane
        self._canonical_objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        # scale them by your actual real-world square size
        self._canonical_objp *= square_size

    def capture_checkerboard(self, frame):
        """ Detect a checkerboard in `frame`. If found, store corners to `self.imgpoints` and known 3D reference corners to `self.objpoints`.
            Then save the frame to disk.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        opencv_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        # findChessboardCorners returns (found, corners) where corners is a Nx1x2 array of pixel coordinates
        found, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, flags=opencv_flags)
        if not found:
            print("[CALIB] Checkerboard NOT detected. Try again.")
            return
        # refine the corner locations for better accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # store the 3D->2D correspondences
        self.objpoints.append(self._canonical_objp)
        self.imgpoints.append(corners_subpix)
        # save the image to disk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.save_dir, f"calib_{timestamp}.png")
        cv2.imwrite(filename, frame)
        print(f"[CALIB] Captured checkerboard and saved: {filename}")

    def run_calibration(self, frame_size):
        """ Once the user finishes capturing images (objpoints & imgpoints), call OpenCV’s calibrateCamera.
            Returns (as length-6 tuple):
                - ret: bool for success
                - mtx: 3x3 camera matrix
                - dist: distortion coefficients
                - rvecs, tvecs: extrinsic parameters for each image
                - error: avg reprojection error
        """
        if len(self.objpoints) < 3:
            print("[CALIB] Not enough images to run calibration. Need at least 3.")
            return False, None, None, None, None, None
        print(f"[CALIB] Running calibrateCamera on {len(self.objpoints)} images...")
        # frame_size is (width, height)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, frame_size, None, None)
        # compute reprojection error
        error = self._compute_reprojection_error(self.objpoints, self.imgpoints, rvecs, tvecs, mtx, dist)
        return ret, mtx, dist, rvecs, tvecs, error

    def _compute_reprojection_error(self, objpoints, imgpoints, rvecs, tvecs, cameraMatrix, distCoeffs):
        """ compute overall average reprojection error across all calibration images """
        total_error = 0
        total_points = 0
        for i in range(len(objpoints)):
            # project
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
            total_error += error**2
            total_points += len(objpoints[i])
        mean_error = np.sqrt(total_error / total_points)
        return mean_error


    def save_to_json(self, json_path, ret, mtx, dist, rvecs, tvecs, error):
        """ Save all relevant calibration parameters to JSON
            Saves:
                - camera matrix
                - distortion coefficients (k1, k2, p1, p2, k3, etc. if computed)
                - rotation/translation vectors for each calibration image
                - average reprojection error
        """
        if not ret:
            print("[CALIB] Calibration unsuccessful; not saving.")
            return
        data = {
            "camera_matrix": mtx.tolist(), # 3x3
            "dist_coeffs": dist.ravel().tolist(), # 1xN
            "reprojection_error": error,
            # extrinsic parameters
            "rotation_vectors": [rvec.tolist() for rvec in rvecs], # list of 3x1 vectors
            "translation_vectors": [tvec.tolist() for tvec in tvecs] # list of 3x1 vectors
        }
        print(f"[CALIB] Saving calibration to {json_path}...")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[CALIB] Done. Reprojection Error = {error:.4f}")
