# Attempting to use least-squares to fit the parameters of the model to the known camera parameters

import numpy as np
from scipy.optimize import least_squares

def compute_focal_length_pixels(focal_length_mm, pixel_size_mm):
    """ focal length (mm -> pixels) """
    return focal_length_mm / pixel_size_mm

def compute_radial_distortion(fx, fy, image_width_px, image_height_px):
    """ estimate k1, k2 from known camera specs using least squares approximation """
    # normalized coordinates of the edge pixels
    x_max = (image_width_px / 2) / fx
    y_max = (image_height_px / 2) / fy
    r_edge = np.sqrt(x_max**2 + y_max**2)
    # initial guess for distortion coefficients estimated from IMX219 cameras (~2% distortion at edges)
    k1_init, k2_init = -0.35, 0.25
    # assume distortion causes actual observed r' values to be slightly different
    r_observed = r_edge * (1 + (k1_init * r_edge**2) + (k2_init * r_edge**4))

    def distortion_model(params, r):
        """ radial distortion function r' = r(1 + k1 * r^2 + k2 * r^4) """
        k1, k2 = params
        return r * (1 + k1 * r**2 + k2 * r**4)

    def residuals(params):
        """ residual function for least squares minimization """
        return distortion_model(params, r_edge) - r_observed

    # solve for k1, k2 using least squares
    initial_guess = [-0.5, 0.4]
    result = least_squares(residuals, initial_guess, loss="soft_l1")
    return result.x

def get_rotation_matrix(axis, theta):
    """ Returns a 3x3 rotation matrix for a given axis ('x', 'y', or 'z') and angle in degrees. """
    theta = np.radians(theta)
    if axis == 'x':
        return np.array([[1, 0, 0],
                            [0, np.cos(theta), -np.sin(theta)],
                            [0, np.sin(theta), np.cos(theta)]])
    elif axis == 'y':
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                            [0, 1, 0],
                            [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 'z':
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
    else:
        raise ValueError("Invalid axis, must be 'x', 'y', or 'z'.")

def compute_extrinsics(yaw_deg, pitch_deg, roll_deg, translation):
    """ computes extrinsic matrix [R | t] given yaw, pitch, roll angles and translation vector """
    # Compute full rotation matrix R = R_y * R_x * R_z
    R = get_rotation_matrix('y', yaw_deg) @ get_rotation_matrix('x', pitch_deg) @ get_rotation_matrix('z', roll_deg)
    # Construct the extrinsic matrix [R | t]
    Rt = np.hstack((R, np.array(translation).reshape(3,1)))
    return Rt


if __name__ == "__main__":
    # known parameters
    focal_length_mm = 3.04  # mm
    pixel_size_mm = 1.12e-3  # mm
    image_width_px = 3280
    image_height_px = 2464
    # deriving pixel coordinates from known camera parameters
    fx = compute_focal_length_pixels(focal_length_mm, pixel_size_mm)
    fy = fx  # NOTE: assumes square pixels
    k1, k2 = compute_radial_distortion(fx, fy, image_width_px, image_height_px)
    print(f"Estimated k1: {k1}")
    print(f"Estimated k2: {k2}")

    # # now computing the extrinsic parameters from actual measurements of the turret system
    # # FAKE example usage with measured angles and translation
    # yaw_angle = 5   # degrees
    # pitch_angle = 2  # degrees
    # roll_angle = 0   # degrees
    # translation_vec = [2, 3, 5]  # in cm
    # Rt_matrix = compute_extrinsics(yaw_angle, pitch_angle, roll_angle, translation_vec)
    # print("Extrinsic Matrix [R | t]:\n", Rt_matrix)




