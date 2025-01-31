import os
import numpy as np
# import cv2
# from scipy import optimize
from homography import HomographyEstimator
from corners import GetCorners


# Compute radial distortion coefficients (k1, k2)
def get_distortion_coeffs(H, K):
    """ Estimates radial distortion parameters k1, k2 from homographies. """
    inv_K = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    r1 = inv_K @ h1
    r2 = inv_K @ h2
    #r3 = np.cross(r1, r2)
    # Enforce unit norm constraint
    r1 /= np.linalg.norm(r1)
    r2 /= np.linalg.norm(r2)
    #r3 /= np.linalg.norm(r3)
    # Compute ideal image points assuming zero distortion
    #ideal_pts = np.array([r1[:2] / r1[2], r2[:2] / r2[2]]).T
    # Radial distortion estimation
    u, v = h3[:2] / h3[2]
    r_squared = u**2 + v**2
    #return np.array([r_squared * u, r_squared * v])
    # Fit a distortion model: u_distorted = u * (1 + k1 * r^2 + k2 * r^4)
    # Overdetermined least squares solution
    A = np.vstack([r_squared, r_squared**2]).T
    b = np.vstack([u - u * (1 + r_squared), v - v * (1 + r_squared**2)])  # Shape (num_points,)
    b = b.reshape(1, b.size)
    print("shapes of A and b: ", A.shape, b.shape)
    #k_values, _, _, _ = np.linalg.lstsq(A, ideal_pts.reshape(1, ideal_pts.size), rcond=None)
    k_values, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return k_values



class CameraCalibrator:
    """ A class for performing camera calibration using Zhang's method.
        It estimates the intrinsic matrix K, distortion coefficients, and extrinsic parameters (R|t) for a given dataset.
    """
    def __init__(self, dataset_dir, num_horiz, num_vert, dist, radial_dist=True, fixed_img="Pic_11.jpg"):
        self.dataset_dir = dataset_dir
        self.num_horiz = num_horiz
        self.num_vert = num_vert
        self.dist = dist
        self.radial_dist = radial_dist
        self.fixed_img = fixed_img
        self.results_dir = os.path.join(dataset_dir, f"results_{fixed_img.split('.')[0].split('_')[1]}")
        os.makedirs(self.results_dir, exist_ok=True)
        self.imgs_dataset = sorted([x for x in os.listdir(self.dataset_dir) if x.endswith('.jpg')])
        self.fix_id = self.imgs_dataset.index(fixed_img)
        self.num_imgs_dataset = len(self.imgs_dataset)
        self.H = {}  # Homographies
        self.Rt = [[] for _ in range(self.num_imgs_dataset)]
        self.K = None
        self.k1_k2 = np.array([0, 0])

    def calculate_homographies(self):
        """ Calculates homographies for each image in the dataset. """
        corner_dir = os.path.join(self.results_dir, "output_corners")
        corner_extractor = GetCorners(corner_dir, self.num_horiz, self.num_vert, self.dist)
        for i, img_name in enumerate(self.imgs_dataset):
            img_path = os.path.join(self.dataset_dir, img_name)
            img_corners, world_corners = corner_extractor.run(img_path)
            if img_corners is not None:
                #print("number of source points: ", len(img_corners[:, :2]))
                #print("number of destination points: ", len(world_corners[:, :2]))
                # Ensure we pass only the properly filtered points
                filtered_img_corners = HomographyEstimator.filter_and_sort_corners(img_corners[:, :2], num_expected_corners=world_corners.shape[0])
                self.H[img_name] = HomographyEstimator.calculate_homography(filtered_img_corners, world_corners[:, :2])
                #print(f"Homography for {img_name} computed:\n ", self.H[img_name])

    def compute_intrinsic_parameters(self):
        """ Computes the intrinsic matrix K from homographies. """
        def stable_division(numerator, denominator):
            TOL = 1e-6
            if abs(denominator) < TOL:
                denominator = np.sign(denominator) * TOL
            return numerator/denominator
        V = []
        def build_v_matrix(H, i, j):
            return np.array([
                H[0][i - 1] * H[0][j - 1],
                H[0][i - 1] * H[1][j - 1] + H[1][i - 1] * H[0][j - 1],
                H[1][i - 1] * H[1][j - 1],
                H[2][i - 1] * H[0][j - 1] + H[0][i - 1] * H[2][j - 1],
                H[2][i - 1] * H[1][j - 1] + H[1][i - 1] * H[2][j - 1],
                H[2][i - 1] * H[2][j - 1]
            ])
        for H in self.H.values():
            V.append(build_v_matrix(H, 1, 2))
            V.append(build_v_matrix(H, 1, 1) - build_v_matrix(H, 2, 2))
        V = np.array(V)
        # _, _, Vt = np.linalg.svd(V)
        # b = Vt.T[:, -1]
        # omega = np.array([
        #     [b[0], b[1], b[3]],
        #     [b[1], b[2], b[4]],
        #     [b[3], b[4], b[5]]
        # ])
        #Compute omega using Cholesky decomposition
        try:
            #omega = np.linalg.solve(V.T @ V, np.eye(V.shape[1]))
            #omega = np.linalg.cholesky(V.T @ V)
            U, S, Vt = np.linalg.svd(V.T @ V, full_matrices=False)
            omega = Vt.T @ np.diag(1 / S) @ U.T  # Pseudo-inverse
        except np.linalg.LinAlgError:
            print("Warning: Cholesky decomposition failed, adding small regularization.")
            #omega = np.linalg.cholesky(V.T @ V + 1e-6 * np.eye(V.shape[1]))
            #omega = np.linalg.solve(V.T @ V + 1e-6 * np.eye(V.shape[1]), np.eye(V.shape[1]))
            U, S, Vt = np.linalg.svd(V.T @ V + 1e-6 * np.eye(V.shape[1]), full_matrices=False)
            omega = Vt.T @ np.diag(1 / S) @ U.T  # Pseudo-inverse
        v0 = stable_division(omega[0, 1] * omega[0, 2] - omega[0, 0] * omega[1, 2], omega[0, 0] * omega[1, 1] - omega[0, 1] ** 2)
        #v0 = (omega[0, 1] * omega[0, 2] - omega[0, 0] * omega[1, 2]) / (omega[0, 0] * omega[1, 1] - omega[0, 1] ** 2)
        lambda_ = omega[2, 2] - stable_division(omega[0, 2] ** 2 + v0 * (omega[0, 1] * omega[0, 2] - omega[0, 0] * omega[1, 2]), omega[0, 0])
        #lambda_ = omega[2, 2] - ((omega[0, 2] ** 2 + v0 * (omega[0, 1] * omega[0, 2] - omega[0, 0] * omega[1, 2])) / omega[0, 0])
        alpha = np.sqrt(stable_division(lambda_, omega[0, 0]))
        #alpha = np.sqrt(lambda_ / omega[0, 0])
        beta = np.sqrt(stable_division(lambda_ * omega[0, 0], omega[0, 0] * omega[1, 1] - omega[0, 1] ** 2))
        #beta = np.sqrt(lambda_ * omega[0, 0] / (omega[0, 0] * omega[1, 1] - omega[0, 1] ** 2))
        gamma = stable_division(-(omega[0, 1] * (alpha ** 2) * beta), lambda_)
        #gamma = -1 * (omega[0, 1] * (alpha ** 2) * beta) / lambda_
        u0 = stable_division(gamma * v0, beta) - stable_division(omega[0, 2] * (alpha ** 2), lambda_)
        #u0 = (gamma * v0 / beta) - (omega[0, 2] * (alpha ** 2) / lambda_)
        self.K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
        print("Intrinsic matrix K computed:\n ", self.K)

    def compute_distortion_coeffs(self):
        k_values = []
        for H in self.H.values():
            k_values.append(get_distortion_coeffs(H, self.K))
        self.k1_k2 = np.mean(k_values, axis=0)
        print("Radial distortion coefficients (k1, k2):", self.k1_k2)

    def compute_extrinsic_parameters(self):
        """ Computes extrinsic parameters R and t for each image. """
        for img_name, H in self.H.items():
            R_t = np.dot(np.linalg.inv(self.K), H)
            # Normalize rotation vectors
            r1 = R_t[:, 0] / np.linalg.norm(R_t[:, 0])
            r2 = R_t[:, 1] / np.linalg.norm(R_t[:, 1])
            r3 = np.cross(r1, r2)  # Ensure orthogonality
            r3 /= np.linalg.norm(r3)  # Normalize
            # Ensure a proper right-handed coordinate system
            if np.linalg.det(np.vstack([r1, r2, r3])) < 0:
                r3 *= -1
            # Construct proper [R | t] matrix
            Rt = np.column_stack((r1, r2, r3, R_t[:, 2]))  # Ensure proper structure
            self.Rt.append(Rt)
        self.Rt = [arr for arr in self.Rt if len(arr) > 1]
        print("Extrinsic parameters computed:\n ", np.array(self.Rt).mean(axis=0))

    def run(self):
        """ Executes the full camera calibration pipeline. """
        print("Starting camera calibration...")
        self.calculate_homographies()
        self.compute_intrinsic_parameters()
        #self.compute_distortion_coeffs()
        self.compute_extrinsic_parameters()
        print("Calibration completed.")


if __name__ == "__main__":
    dataset_dir = r"E:\camera_calibration\trial3_copies"
    num_horiz = 10
    num_vert = 8
    dist = 25
    fixed_img = "WIN_20250129_13_19_09_Pro.png"
    calibrator = CameraCalibrator(dataset_dir, num_horiz, num_vert, dist, radial_dist=True, fixed_img=fixed_img)
    calibrator.run()