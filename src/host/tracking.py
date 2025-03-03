from typing import Union, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
from skimage import filters, morphology, measure, color

import matplotlib.pyplot as plt

@dataclass
class MotionDetectionFeedback:
    motion_detected: bool
    motion_mask: Union[np.ndarray, None] = None # shape [H, W] or None
    contour: Union[np.ndarray, None] = None # shape [N, 2] or None
    centroid: Union[Tuple[int, int], None] = None     # (x, y) or None



def debug_value_histogram(arr: np.ndarray, num_bins: int = 100, top_n: int = 10):
    if top_n > num_bins:
        raise ValueError("top_n must be less than or equal to num_bins")
    hist, bin_edges = np.histogram(arr, bins=num_bins)
    largest_bins = sorted(enumerate(hist), key=lambda x: x[1], reverse=True)[:top_n]
    # NOTE: values returned above are (bin_index, count), so we need to convert to (bin_range, count)
    largest_bins = list(map(lambda pair: ((round(float(bin_edges[pair[0]]), 3),
                                            round(float(bin_edges[pair[0] + 1]), 3)),
                                          int(pair[1])), largest_bins))
    return largest_bins

def print_value_histogram(arr: np.ndarray, arr_name: str, num_bins: int = 100, top_n: int = 10):
    debug_values = debug_value_histogram(arr, top_n=top_n)
    print(f"[DEBUGGING] {arr_name}\n\t (min, max): {arr.min(), arr.max()}\n\t top {top_n} histogram pairs (value, count): {debug_values}")

def plot_debug_image(arr: np.ndarray, cmap="gray", title: Optional[str] = None):
    plt.imshow(arr, cmap=cmap)
    if title:
        plt.title(title)
    plt.show(block=False)
    plt.pause(5)
    plt.close()




# #def update_dynamic_alpha(stddev, lo = 0.001, hi = 0.5) -> float:
# def update_dynamic_alpha(stddev, slope=1.0, offset=0.0, lo = 0.001, hi = 0.1) -> float:
#     """ naive method to adapt alpha based on current difference distribution """
#     #~ might update this to use a nonlinear smoothing function like sigmoid later
#     # new_alpha = 0.1 / (stddev + lo)
#     # new_alpha = np.clip(new_alpha, lo, hi)
#     # return float(new_alpha)
#     # If 'std_val' is small => alpha near top plateau. If 'std_val' is large => alpha near bottom plateau.
#     x = slope * (stddev - offset)
#     sigma = 1.0 / (1.0 + np.exp(-x))  # standard logistic
#     return lo + (hi - lo) * sigma

def update_alpha(diff: np.ndarray, alpha_min=0.01, alpha_max=0.2, slope=10, midpoint=0.05):
    """ Uses a sigmoid function to scale alpha dynamically based on motion intensity. """
    motion_level = np.mean(diff)
    alpha = alpha_min + (alpha_max - alpha_min) / (1 + np.exp(-slope * (motion_level - midpoint)))
    return alpha




# #~ UPDATE: saying screw it and running this on CPU
# class MotionDetector:
#     def __init__(self,
#                  threshold: float = 0.5,
#                  alpha_init: float = 0.01,
#                  morph_disk_radius: int = 3):
#         self.diff_threshold = threshold
#         self.alpha = alpha_init
#         self.foreground_threshold = 10
#         self.running_std = 0.1  # arbitrary start; adjust as needed
#         # morphological structuring element for opening/closing; can also use `morphology.square`
#         self.struct_elem = morphology.disk(morph_disk_radius)
#         self.background = None
#         self.frame_count = 0

#     def _update_background(self, current_frame: np.ndarray) -> None:
#         """ updates the running-average background image combination in-place """
#         # compute difference for dynamic alpha
#         diff = np.abs(current_frame - self.background)
#         # updating running std with 90/10 rule
#         self.running_std = 0.9 * self.running_std + 0.1 * diff.std()
#         #self.alpha = update_dynamic_alpha(self.running_std)
#         self.alpha = update_alpha(diff)
#         print("[DEBUGGING] current alpha value: ", self.alpha)
#         # running average update
#         self.background = (1 - self.alpha)*self.background + self.alpha*current_frame

#     def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
#         """ preprocess the input frame by scaling, converting to grayscale then blurring """
#         frame_processed = frame.astype(np.float32)/255.0  # scale to [0..1]
#         frame_processed = color.rgb2gray(frame_processed)  # convert to grayscale
#         frame_processed = filters.gaussian(frame_processed, sigma=2.0)  # apply Gaussian blur
#         return frame_processed

#     def _get_largest_region(self, labeled: np.ndarray) -> Union[List['measure.RegionProperties'], None]:
#         """ returns the RegionProperties of the largest connected region by area, or None """
#         if labeled.max() < 1:
#             # no labeled regions implies no motion
#             return None
#         # find the largest labeled regions based on area (number of pixels in region) using regionprops
#         regions = measure.regionprops(labeled)
#         if not regions:
#             return None
#         largest_region = None
#         max_area = 0
#         # sequential search for largest region by area
#         for r in regions:
#             # NOTE: regionprops does lazy evaluation of properties
#             if r.area > max_area:
#                 max_area = r.area
#                 largest_region = r
#         #print(f"[DEBUGGING] largest region area: {max_area}")
#         return largest_region

#     def _get_largest_contour(self, mask: np.ndarray) -> Tuple[Union[np.ndarray, None], Union['measure.RegionProperties', None]]:
#         """ Label the mask, find the largest region, and use extract its largest contour
#             Returns:
#                 largest_contour, -> np.ndarray of shape [N,2] or None
#                 largest_region   -> RegionProperties or None
#         """
#         labeled = measure.label(mask, connectivity=2)
#         largest_region = self._get_largest_region(labeled)
#         if largest_region is None:
#             return (None, None)
#         # NOTE: find_contours returns a list of contours, each of which is a list of points (x,y) in the contour
#         #       so we need to find the largest contour by length and return it as a numpy array of shape (N,2)
#         region_mask = (labeled == largest_region.label).astype(np.float32)
#         contours = measure.find_contours(region_mask, 0.5)
#         if not contours:
#             return (None, largest_region)
#         # if there are multiple contours for the region, pick the largest in length
#         largest_contour = None
#         largest_contour_length = 0
#         for c in contours:
#             # c is shape [N, 2] and each row is (row, col)
#             if len(c) > largest_contour_length:
#                 largest_contour_length = len(c)
#                 largest_contour = c
#         return (largest_contour, largest_region)


#     def _get_negative_detection(self) -> MotionDetectionFeedback:
#         return MotionDetectionFeedback(motion_detected=False, contour=None, centroid=None)

#     def process_frame(self, frame: np.ndarray) -> MotionDetectionFeedback:
#         """ process a frame (hypothetically from the RPi camera feed) to detect motion """
#         self.frame_count += 1
#         # img frame passed by default as a numpy array of shape (480, 640, 3)
#         img: np.ndarray = self._preprocess_frame(frame) # now np.ndarray of shape (480, 640)
#         print(f"[MotionDetector] Processing frame {self.frame_count}")
#         DEBUG_BINCOUNT = 5
#         #print_value_histogram(img, "img", top_n=DEBUG_BINCOUNT)
#         if self.background is None:
#             self.background = img
#             return self._get_negative_detection()
#         if self.frame_count % 1 == 0: # setting to 1 while debugging for now
#             # update the background every 5 minutes (assuming 10 second delay in frame capture)
#             self._update_background(img)
#         # create motion mask by thresholding the difference
#         diff = np.abs(img - self.background)
#         plot_debug_image(diff, cmap='gray', title="Difference Image")
#         # convert boolean thresholded mask to bytes (0/1) for morphological operations
#         motion_mask = (diff > self.diff_threshold).astype(bool)
#         print("[DEBUGGING] motion mask values: ", np.unique(motion_mask, return_counts=True))
#         #plot_debug_image(motion_mask.astype(np.uint8), cmap='gray', title="Motion Mask")
#         # morphological cleanup using opening to remove small noise
#         # TODO: determine if closing is necessary and whether any morphological ops are necessary to begin with
#         motion_mask = morphology.binary_opening(motion_mask, self.struct_elem)
#         motion_mask = morphology.binary_closing(motion_mask, self.struct_elem)
#         print("[DEBUGGING] motion mask values: ", np.unique(motion_mask, return_counts=True))
#         #plot_debug_image(motion_mask.astype(np.uint8), cmap='gray', title="Motion Mask")
#         # check if motion_mask is basically empty based on the sum of foreground pixels (ones)
#         if motion_mask.sum() < self.foreground_threshold:
#             return self._get_negative_detection()
#         # label connected components and find the largest contour among the labeled regions
#         largest_contour, largest_region = self._get_largest_contour(motion_mask.astype(np.uint8))
#         if largest_contour is None or largest_region is None:
#             return self._get_negative_detection()
#         # return the final (non-empty) detection result
#         return MotionDetectionFeedback(
#             motion_detected=True,
#             contour=largest_contour,    # shape [N,2], in (row,col) format
#             centroid=largest_region.centroid # (row, col) indexing since that's what the targeting system and visualizer expect
#         )

#     def reset(self):
#         """ reset the background running average (e.g., after significant changes in the environment) """
#         self.background = None
#         self.frame_count = 0

def compute_cobel_edges(frame):
    # compute Sobel edge map to weight motion mask towards prioritizing change in edges
    sobel_x = filters.sobel_h(frame)  # Horizontal edges
    sobel_y = filters.sobel_v(frame)  # Vertical edges
    edge_map = np.hypot(sobel_x, sobel_y)
    return edge_map


class BaseMotionDetector:
    def __init__(
        self,
        threshold: float = 0.5,
        alpha_init: float = 0.1,
        morph_disk_radius: int = 7,
        background_update_interval: int = 1
    ):
        """ Basic motion detector without contour/centroid processing """
        self.diff_threshold = threshold
        self.alpha = alpha_init
        self.background_update_int = background_update_interval
        self.foreground_threshold = 10
        self.struct_elem = morphology.disk(morph_disk_radius)
        self.background = None
        self.frame_count = 0

    def _update_background(self, current_frame: np.ndarray) -> None:
        """ dynamically update the background using an adaptive exponential moving average """
        diff = np.abs(current_frame - self.background)
        self.alpha = update_alpha(diff)  # using sigmoid-based scaling
        print("[DEBUGGING] current alpha value: ", self.alpha)
        self.background = (1 - self.alpha) * self.background + self.alpha * current_frame

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """ preprocess the input frame by scaling, converting to grayscale then blurring """
        frame_processed = frame.astype(np.float32)/255.0  # scale to [0..1]
        frame_processed = color.rgb2gray(frame_processed)  # convert to grayscale
        frame_processed = filters.gaussian(frame_processed, sigma=2.0)
        return frame_processed #, edge_map

    def _compute_motion_mask(self, current_frame: np.ndarray) -> np.ndarray:
        """ compute motion mask by boosting weights where edges are more pronounced """
        TOL = 1e-6
        diff = np.abs(current_frame - self.background)
        texture_map = filters.rank.entropy((255*diff).astype(np.uint8), self.struct_elem)
        # normalize texture map to 0-1 scale with tolerance to avoid division by zero
        texture_weight = texture_map / (np.max(texture_map) + TOL)
        #plot_debug_image(texture_weight, cmap='gray', title="Edge Weight Map")
        motion_mask = diff * np.exp(2*texture_weight)  # boost weights where edges are more pronounced
        #print("[DEBUGGING] motion mask values: ", np.unique(motion_mask, return_counts=True))
        # threshold motion mask
        motion_mask = motion_mask > self.diff_threshold
        # morphological cleanup using opening to remove small holes/artifacts
        motion_mask = morphology.binary_closing(motion_mask, self.struct_elem)
        #motion_mask = morphology.binary_opening(motion_mask, self.struct_elem)
        return motion_mask.astype(np.uint8)

    def process_frame(self, frame: np.ndarray) -> MotionDetectionFeedback:
        """ process a frame (hypothetically from the RPi camera feed) to detect motion (without contours) """
        self.frame_count += 1
        print(f"[MotionDetector] Processing frame {self.frame_count}")
        # img frame passed by default as a numpy array of shape (480, 640, 3)
        img = self._preprocess_frame(frame)
        if self.background is None:
            self.background = img
            return MotionDetectionFeedback(motion_detected=False)
        if self.frame_count % self.background_update_int == 0:
            # update the background every 5 minutes (assuming 10 second delay in frame capture)
            self._update_background(img)
        motion_mask = self._compute_motion_mask(img)
        # check if motion_mask is basically empty based on the sum of foreground pixels (ones)
        if motion_mask.sum() < self.foreground_threshold:
            return MotionDetectionFeedback(motion_detected=False)
        plot_debug_image(motion_mask, cmap='gray', title="Motion Mask")
        # return the final (non-empty) detection result without contour information (used in base class)
        return MotionDetectionFeedback(motion_detected=True, motion_mask=motion_mask)

    def reset(self):
        """ reset the background running average (e.g., after significant changes in the environment) """
        self.background = None
        self.frame_count = 0


class RefinedMotionDetector(BaseMotionDetector):
    def _get_largest_region(self, labeled: np.ndarray) -> Optional['measure.RegionProperties']:
        """ returns the RegionProperties of the largest connected region by area, or None """
        if labeled.max() < 1:
            # no labeled regions implies no motion
            return None
        # find the largest labeled regions based on area (number of pixels in region) using regionprops
        regions = measure.regionprops(labeled)
        if not regions:
            return None
        return max(regions, key=lambda r: r.area, default=None)

    def _get_largest_contour(self, mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional['measure.RegionProperties']]:
        """ Label the mask, find the largest region, and use extract its largest contour
            Returns:
                largest_contour, -> np.ndarray of shape [N,2] or None
                largest_region   -> RegionProperties or None
        """
        labeled = measure.label(mask, connectivity=2)
        largest_region = self._get_largest_region(labeled)
        if largest_region is None:
            return None, None
        # NOTE: find_contours returns a list of contours, each of which is a list of points (x,y) in the contour
        #       so we need to find the largest contour by length and return it as a numpy array of shape (N,2)
        region_mask = (labeled == largest_region.label).astype(np.float32)
        contours = measure.find_contours(region_mask, 0.5)
        if not contours:
            return None, largest_region
        # if there are multiple contours for the region, pick the largest in length
        largest_contour = max(contours, key=len, default=None)
        return largest_contour, largest_region

    def process_frame(self, frame: np.ndarray) -> MotionDetectionFeedback:
        """ process a frame (hypothetically from the RPi camera feed) to detect motion (with contours and centroid) """
        base_feedback = super().process_frame(frame)
        if not base_feedback.motion_detected:
            return base_feedback
        motion_mask = base_feedback.motion_mask
        largest_contour, largest_region = self._get_largest_contour(motion_mask.astype(np.uint8))
        if largest_contour is None or largest_region is None:
            return MotionDetectionFeedback(motion_detected=False, motion_mask=motion_mask)
        return MotionDetectionFeedback(
            motion_detected=True,
            motion_mask=motion_mask,
            contour=largest_contour,
            centroid=largest_region.centroid
        )