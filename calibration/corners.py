import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil


def high_pass_filter(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
    # subtract the blurred image from the original
    high_pass = cv2.subtract(img, blurred)
    # add high-pass image back to the original
    sharpened = cv2.addWeighted(img, 1.0, high_pass, 1.0, 0)
    return sharpened


class GetCorners:
    """ A class for detecting checkerboard corners by finding intersections of Hough lines. """
    def __init__(self, results_dir, num_horiz, num_vert, dist):
        self.results_dir = results_dir
        self.num_horiz = num_horiz
        self.num_vert = num_vert
        self.dist = dist
        os.makedirs(results_dir, exist_ok=True)

    def get_horiz_vert_lines(self, img):
        """ Detect horizontal and vertical lines using Hough Transform. """
        edges = cv2.Canny(img, 300, 500, None, 3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)
        if lines is None:
            return None, None
        lines = np.squeeze(lines)
        horizontal = lines[(np.pi/6 <= lines[:, 1]) & (lines[:, 1] <= 5*np.pi/6)]
        vertical = lines[(lines[:, 1] < np.pi/6) | (lines[:, 1] > 5*np.pi/6)]
        return horizontal, vertical

    def get_intersections(self, horizontal, vertical):
        """ Compute intersections between horizontal and vertical lines. """
        corners = []
        for h in horizontal:
            for v in vertical:
                A = np.array([
                    [np.cos(h[1]), np.sin(h[1])],
                    [np.cos(v[1]), np.sin(v[1])]
                ])
                b = np.array([h[0], v[0]])
                if np.linalg.det(A) != 0:
                    corner = np.linalg.solve(A, b)
                    corners.append([corner[0], corner[1], 1])
        return np.array(corners)

    def generate_world_coordinates(self):
        """ Generate corresponding world coordinates for detected corners. """
        world_coords = [[j * self.dist, i * self.dist, 0, 1]
                        for i in range(self.num_horiz) for j in range(self.num_vert)]
        return np.array(world_coords)

    def run(self, img_path):
        """ Main function to detect and refine checkerboard corners. """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (3,3), 0, 0)
        #img = high_pass_filter(img)
        if img is None:
            print(f"Error: Unable to read image {img_path}")
            return None, None
        img_height, img_width = img.shape[:2]
        horizontal, vertical = self.get_horiz_vert_lines(img)
        if horizontal is None or vertical is None:
            print(f"No valid lines found in {img_path}")
            return None, None
        corners = self.get_intersections(horizontal, vertical)
        if len(corners) == 0:
            print(f"No intersections found in {img_path}")
            return None, None
        # Ensure all detected corners are within the valid image bounds
        valid_corners = []
        for corner in corners:
            x, y = corner[:2]
            if 0 <= x < img_width and 0 <= y < img_height:
                valid_corners.append(corner)
        if not valid_corners:
            print(f"No valid corners within image bounds for {img_path}")
            return None, None
        valid_corners = np.array(valid_corners)
        # world_coords = self.generate_world_coordinates()
        # if len(corners) == 0:
        #     print(f"No intersections found in {img_path}")
        #     return None, None
        # Refining corner locations
        refined_corners = cv2.cornerSubPix(
            img, valid_corners[:, :2].reshape(-1, 1, 2).astype(np.float32),
            #img, np.float32(corners[:, :2]).reshape(-1, 1, 2),
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        refined_corners = np.hstack((refined_corners.reshape(-1, 2), np.ones((refined_corners.shape[0], 1))))
        world_coords = self.generate_world_coordinates()
        return refined_corners, world_coords



if __name__ == "__main__":
    dataset_dir = "path/to/dataset"
    results_dir = os.path.join(dataset_dir, "results")
    corner_detector = GetCorners(results_dir, num_horiz=10, num_vert=8, dist=25)
    for img_name in os.listdir(dataset_dir):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(dataset_dir, img_name)
            corners, world_coords = corner_detector.run(img_path)
