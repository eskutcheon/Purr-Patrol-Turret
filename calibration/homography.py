import os
import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from scipy.spatial import distance_matrix
# from sklearn.cluster import KMeans
from scipy.spatial import Delaunay, cKDTree


class HomographyEstimator:
    """ A class for computing homography transformation matrices. """
    @staticmethod
    def filter_and_sort_corners(src_pts, num_expected_corners):
        """ Filters and sorts detected source points to match a grid pattern.
            :param src_pts: Raw detected corner points (N x 2).
            :param num_expected_corners: Number of expected corners (e.g., 80 for a 10x8 grid).
            :return: Sorted and filtered source points (num_expected_corners x 2).
        """
        if len(src_pts) < num_expected_corners:
            raise ValueError(f"Not enough detected corners: {len(src_pts)} found, {num_expected_corners} expected.")
        # Use k-means to find the best 80 corners
        # kmeans = KMeans(n_clusters=num_expected_corners, random_state=42, n_init=10)
        # clustered_corners = kmeans.fit(src_pts).cluster_centers_
        # # Sort points into a structured grid
        # sorted_corners = sorted(clustered_corners, key=lambda p: (p[1], p[0]))  # Sort by y first, then x
        # return np.array(sorted_corners)
        # Step 1: Use Delaunay Triangulation to enforce a structured grid pattern
        tri = Delaunay(src_pts)
        # Step 2: Select the points forming the largest structured grid
        valid_points = set()
        for simplex in tri.simplices:
            for idx in simplex:
                valid_points.add(tuple(src_pts[idx]))
        valid_points = np.array(list(valid_points))
        # Step 3: Sort based on grid-like structure
        if len(valid_points) > num_expected_corners:
            distances = np.linalg.norm(valid_points - np.mean(valid_points, axis=0), axis=1)
            sorted_indices = np.argsort(distances)[:num_expected_corners]
            final_corners = valid_points[sorted_indices]
        elif len(valid_points) < num_expected_corners:
            print(f"WARNING: fewer than {num_expected_corners} points ({len(valid_points)}) found: performing nearest-neighbor interpolation...")
            #raise ValueError(f"Filtered cluster has only {len(valid_points)} points, expected {num_expected_corners}")
            # Step 4: Recover missing points via Nearest Neighbor interpolation
            tree = cKDTree(valid_points)
            while len(valid_points) < num_expected_corners:
                missing_point = valid_points[np.random.choice(len(valid_points))]  # Select a reference
                _, nearest_idx = tree.query(missing_point, k=2)  # Find nearest neighbor
                new_point = (valid_points[nearest_idx[0]] + valid_points[nearest_idx[1]]) / 2  # Midpoint
                valid_points = np.vstack([valid_points, new_point])
            final_corners = valid_points[:num_expected_corners]
        else:
            final_corners = valid_points
        # Step 4: Sort the final points by (y, x) to align with expected grid order
        sorted_corners = sorted(final_corners, key=lambda p: (p[1], p[0]))
        return np.array(sorted_corners)

    @staticmethod
    def calculate_homography(src_pts, dst_pts):
        """ Compute the homography matrix using Direct Linear Transformation (DLT).
            :param src_pts: Source points in image plane
            :param dst_pts: Corresponding destination points in world coordinates
            :return: 3x3 Homography matrix
        """
        if isinstance(src_pts, list):
            src_pts = np.array(src_pts)
        if isinstance(dst_pts, list):
            dst_pts = np.array(dst_pts)
        # Ensure we use exactly the expected number of points
        src_pts = HomographyEstimator.filter_and_sort_corners(src_pts, num_expected_corners=dst_pts.shape[0])
        if src_pts.shape[0] != dst_pts.shape[0]:
            raise ValueError(f"Mismatch in number of points: src_pts={src_pts.shape[0]}, dst_pts={dst_pts.shape[0]}")
        A, b = HomographyEstimator.build_equations(src_pts, dst_pts)
        H = np.linalg.lstsq(A, b, rcond=None)[0]
        H = np.append(H, 1).reshape(3, 3)
        # normalize homographies for computing matrix V later:
        for i in range(3):
            H[:, i] /= np.linalg.norm(H[:, i])
        return H

    @staticmethod
    def build_equations(src_pts, dst_pts):
        """ Constructs the system of equations for homography estimation.
            :param src_pts: Source points in image plane
            :param dst_pts: Corresponding destination points in world coordinates
            :return: Matrix A and vector b for solving homography
        """
        num_pts = src_pts.shape[0]
        A = np.zeros((2 * num_pts, 8))
        b = np.zeros((2 * num_pts, 1))
        for i in range(num_pts):
            x, y = src_pts[i]
            X, Y = dst_pts[i]
            A[2 * i] = [X, Y, 1, 0, 0, 0, -X * x, -Y * x]
            A[2 * i + 1] = [0, 0, 0, X, Y, 1, -X * y, -Y * y]
            b[2 * i] = x
            b[2 * i + 1] = y
        return A, b



if __name__ == "__main__":
    src_points = [[100, 200], [150, 250], [200, 300], [250, 350]]
    dst_points = [[10, 20], [15, 25], [20, 30], [25, 35]]
    homography_matrix = HomographyEstimator.calculate_homography(src_points, dst_points)
    print("Computed Homography Matrix:")
    print(homography_matrix)
