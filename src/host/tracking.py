"""
- Handles motion detection on incoming frames.

detect_motion(frame): Detects motion using frame differencing or other techniques.
prepare_for_targeting(): Signals the Pi to prepare the turret for targeting.
"""


from typing import Union, Optional, Tuple, List
#import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as TT
import torch.nn.functional as F

# ! REMOVE LATER - for general idea of time bottlenecks:
from tqdm import tqdm


def to_cuda(img: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if not issubclass(type(img), torch.Tensor):
        try:
            img = TT.ToTensor()(img)
        except Exception as e:
            raise ValueError(f"Failed to convert image with type {type(img)} to tensor: {e}")
    if not img.is_cuda:
        img = img.to(device="cuda")
    return img



@torch.jit.script
def get_connectivity_kernel(kernel_size: int, max_distance: int) -> torch.Tensor:
    """ Create a connectivity kernel based on Manhattan distances.
        :param kernel_size: Size of the square kernel (must be odd).
        :param manhattan_distance: Maximum Manhattan distance to include neighbors.
        :return: A kernel of shape (1, 1, kernel_size, kernel_size) with 1s for neighbors.
    """
    assert kernel_size % 2 == 1, "Kernel size must be odd."
    center = kernel_size // 2
    # create grid of Manhattan distances
    grid_x, grid_y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size), indexing="ij")
    manhattan_distances = torch.abs(grid_x - center) + torch.abs(grid_y - center)
    # generate the kernel: 1 for neighbors within the distance, 0 otherwise
    kernel = (manhattan_distances <= max_distance).int()
    # remove the center pixel (not part of the neighbors)
    kernel[center, center] = 0
    return kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions


@torch.jit.script
def connected_components(tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """ Label connected components in a binary tensor with vectorized flood filling
        :param tensor: Binary tensor of shape (H, W) with 1s for foreground
        :return: Label tensor (where each connected component has a label) and the number of labels (components)
    """
    # NOTE: The original function had a safeguard to ensure the tensor was 2D, but torch uses empty batch dimensions like (1,H,W) for grayscale images
    tensor = tensor.clone().byte()  # copy tensor for modification and cast to byte
    labeled = torch.zeros_like(tensor, dtype=torch.int32)
    current_label = 0
    # initialize kernel to find 4-connected neighbors (NOTE: there may be a native kornia function for this)
    # kernel = torch.tensor([[0, 1, 0],
    #                        [1, 0, 1],
    #                        [0, 1, 0]], dtype=torch.uint8, device=tensor.device).unsqueeze(0).unsqueeze(0)
    kernel = get_connectivity_kernel(3, 1).to(device=tensor.device)
    # vectorized region labeling - flood fill until no unlabeled foreground pixels remain
    while tensor.any():
        # use first unlabeled foreground pixel as the seed for the next region
        seed = torch.nonzero(tensor == 1)[0:1]
        current_label += 1
        # create mask for the connected region
        region = torch.zeros_like(tensor)
        region[seed[:, 0], seed[:, 1]] = 1
        # flood fill using convolution
        while True:
            dilated = torch.conv2d(region.float(), kernel.float(), padding=1)
            dilated = (dilated > 0).byte() & tensor
            if torch.equal(dilated, region):
                break
            region = dilated
        # label the current region with the current label
        labeled[region.to(dtype=torch.bool)] = current_label
        # remove most recent labeled region from the original tensor for the loop termination condition
        tensor[region.to(dtype=torch.bool)] = 0
    return labeled, current_label

@torch.jit.script
def find_contours(thresh_mask: torch.Tensor, min_area: int = 5000):
    """ Extract contours from a binary mask tensor - updated from using cv2.findContours to pure PyTorch
        :param thresh_mask: Input binary mask (1 for foreground, 0 for background) of shape (H, W).
        :param min_area: Minimum contour area to include.
        :return: List of tensors containing the indices for each contour.
    """
    # label connected components
    labeled, num_labels = connected_components(thresh_mask)
    # extract contours for each label and save the contour indices
    contours = []
    for label in range(1, num_labels + 1):  # Skip label 0 (background)
        mask = labeled == label
        if mask.sum().item() >= min_area:  # Filter small regions
            # get boundary indices
            boundary = torch.logical_xor(mask, F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=1).to(dtype=torch.bool))
            contour_indices = boundary.nonzero()
            contours.append(contour_indices.cpu())
    return contours



class MotionDetection:
    def __init__(self, threshold=25, blur_size=(21, 21), min_contour_area=5000, device="cuda"):
        self.threshold = threshold
        self.blur_size = blur_size
        self.min_contour_area = min_contour_area
        self.first_frame = None
        self.preprocessor = TT.Compose([
            TT.Lambda(to_cuda) if device == "cuda" else TT.Lambda(lambda img: img),
            TT.Grayscale(num_output_channels=1),
            TT.GaussianBlur(self.blur_size, sigma=3.5) # 3.5 = what the original code's sigma came out to be with k=21 in cv2.GaussianBlur
        ])

    def process_frame(self, frame):
        """ process a frame (hypothetically from the RPi camera feed) to detect motion """
        img = self.preprocessor(frame)
        # initialize the first frame
        if self.first_frame is None:
            self.first_frame = img
            return None
        # get list of contour indices
        frame_delta = torch.abs(self.first_frame - img)
        mask = (frame_delta > self.threshold).float()
        contours = find_contours(mask, threshold=0.5, min_area=self.min_contour_area)
        # return largest contour
        if contours:
            return max(contours, key=lambda x: len(x))
        return None


    def reset(self):
        """ reset the first frame (e.g., after significant changes in the environment) """
        self.first_frame = None
