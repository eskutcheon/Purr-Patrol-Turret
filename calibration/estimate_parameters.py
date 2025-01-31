import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt


def display_gray_images(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def calibrate_camera(image_dir, pattern_size, square_size):
    """ Calibrates the camera using checkerboard images.
        Parameters:
        - image_dir: Directory containing checkerboard images.
        - pattern_size: Tuple (rows, cols) of checkerboard inner corners.
        - square_size: The size of a single square in real-world units (e.g., mm, cm).
        Returns:
        - K: 3x3 Intrinsic camera matrix
        - dist: Distortion coefficients (k1, k2)
        - Rt_list: List of 3x4 [R|t] matrices for each image
    """
    # Prepare object points (3D points of checkerboard corners)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale by square size
    obj_points = []  # 3D points in world space
    img_points = []  # 2D points in image plane
    # Load checkerboard images
    images = sorted(glob.glob(f"{image_dir}/*.jpg"))
    print(images)
    image_shape = None  # To store the resolution of the first valid image
    valid_images = 0
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Unable to load image {fname}. Skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray = cv2.equalizeHist(gray)
        #gray = cv2.GaussianBlur(gray, (5, 5), 0)
        display_gray_images(gray)
        # Find the checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS)
        print(f"corners for image '{os.path.basename(fname)}': ", corners)
        if ret:
            valid_images += 1
            if image_shape is None:
                image_shape = gray.shape[::-1]  # Set image resolution (width, height)
            elif gray.shape[::-1] != image_shape:
                print(f"Warning: Skipping {fname} due to mismatched image dimensions.")
                continue
            obj_points.append(objp)
            refined_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            img_points.append(refined_corners)
            # Optional: Draw and display corners
            cv2.drawChessboardCorners(img, pattern_size, refined_corners, ret)
            cv2.imshow('Corners', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    if valid_images < 1:
        raise ValueError("No valid checkerboard images were found for calibration.")
    # Perform camera calibration
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_shape, None, None)
    if not ret:
        raise RuntimeError("Camera calibration failed. Please check your images.")
    # Extract k1, k2 distortion coefficients
    k1, k2 = dist[0][:2]
    # Compute Rt matrices for each image
    Rt_list = []
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to matrix
        Rt = np.hstack((R, tvec))   # Create [R | t] (3×4 matrix)
        Rt_list.append(Rt)
    return K, (k1, k2), Rt_list


if __name__ == "__main__":
    # Example Usage
    image_directory = "E:\camera_calibration"
    pattern_size = (8,10)  # Adjust based on your checkerboard
    square_size = 25  # Adjust based on the real size of squares
    try:
        K, (k1, k2), Rt_list = calibrate_camera(image_directory, pattern_size, square_size)
        print("Intrinsic Matrix (K):\n", K)
        print("Distortion Coefficients (k1, k2):", k1, k2)
        print("Extrinsic Matrices (Rt):")
        for i, Rt in enumerate(Rt_list):
            print(f"Image {i+1} Rt:\n{Rt}\n")
    except Exception as e:
        print(f"Error: {e}")
