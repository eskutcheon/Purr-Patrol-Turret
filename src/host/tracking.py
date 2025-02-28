"""
- Handles motion detection on incoming frames.

detect_motion(frame): Detects motion using frame differencing or other techniques.
prepare_for_targeting(): Signals the Pi to prepare the turret for targeting.
"""


from typing import Union, Optional, Tuple, List
import sys
#import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as TT
import torch.nn.functional as F
import kornia.morphology as KM



def to_cuda(img: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if not issubclass(type(img), torch.Tensor):
        try:
            img = torch.from_numpy(img).permute(2, 0, 1).detach() #TT.ToTensor()(img)
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
def connected_components(tensor: torch.ByteTensor, kernel: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """ Label connected components in a binary tensor with vectorized flood filling
        :param tensor: Binary tensor of shape (H, W) with 1s for foreground
        :return: Label tensor (where each connected component has a label) and the number of labels (components)
    """
    # NOTE: The original function had a safeguard to ensure the tensor was 2D, but torch uses empty batch dimensions like (1,H,W) for grayscale images
    labeled = torch.zeros_like(tensor, dtype=torch.int32)
    current_label = 0
    # initialize kernel to find 4-connected neighbors (NOTE: there may be a native kornia function for this)
    # kernel = torch.tensor([[0, 1, 0],
    #                        [1, 0, 1],
    #                        [0, 1, 0]], dtype=torch.uint8, device=tensor.device).unsqueeze(0).unsqueeze(0)
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
            if torch.allclose(dilated, region, rtol=1e-4, atol=1e-6):
                break
            region = dilated
        # label the current region with the current label
        labeled[region.to(dtype=torch.bool)] = current_label
        # remove most recent labeled region from the original tensor for the loop termination condition
        tensor[region.to(dtype=torch.bool)] = 0
    return labeled, current_label

@torch.jit.script
def find_contours(thresh_mask: torch.ByteTensor, kernel: torch.Tensor, min_area: int = 5000):
    """ Extract contours from a binary mask tensor - updated from using cv2.findContours to pure PyTorch
        :param thresh_mask: Input binary mask (1 for foreground, 0 for background) of shape (H, W).
        :param min_area: Minimum contour area to include.
        :return: List of tensors containing the indices for each contour.
    """
    # label connected components
    labeled, num_labels = connected_components(thresh_mask, kernel)
    # labeled, num_labels = connected_components(thresh_mask.pin_memory().cpu(), kernel.cpu())  # Process on CPU
    # extract contours for each label and save the contour indices
    contours = []
    for label in range(1, num_labels + 1):  # skip label 0 (background)
        mask = labeled == label
        if mask.sum().item() >= min_area:  # filter small regions
            # get boundary indices
            boundary = torch.logical_xor(mask, F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=1).to(dtype=torch.bool))
            # append contour indices to the list of contours
            #contours.append(boundary.nonzero().cpu())
            contours.append(boundary.nonzero())
    return contours



class MotionDetector:
    def __init__(self, threshold=25, blur_size=(21, 21), min_contour_area=5000, device="cuda"):
        self.threshold = threshold
        # TODO: need to set this dynamically based on image sizes
        self.min_contour_area = min_contour_area
        self.kernel: torch.Tensor = get_connectivity_kernel(3, 1).to(device=device)  # 3x3 kernel for 4-connected neighbors
        self.first_frame = None
        self.preprocessor = TT.Compose([
            TT.Lambda(to_cuda) if device == "cuda" else TT.Lambda(lambda img: img),
            TT.Grayscale(num_output_channels=1),
            TT.GaussianBlur(blur_size, sigma=3.5) # 3.5 = what the original code's sigma came out to be with k=21 in cv2.GaussianBlur
        ])

    def process_frame(self, frame):
        """ process a frame (hypothetically from the RPi camera feed) to detect motion """
        # img frame passed by default as a numpy array of shape (480, 640, 3)
        img: torch.Tensor = self.preprocessor(frame) # np.ndarray => torch.Tensor of shape (1, 480, 640)
        # initialize the first frame
        if self.first_frame is None:
            self.first_frame = img.clone()
            return None
        # compute the absolute difference between the current frame and the first frame
        frame_delta = torch.abs(self.first_frame - img)
        # threshold the difference and convert to uint8
        mask = (frame_delta > self.threshold).unsqueeze(0) #.byte()
        # # 'opening' = erosion followed by dilation => remove noise
        mask = KM.opening(mask, self.kernel)
        # # 'closing' = dilation followed by erosion => fill holes
        mask = KM.closing(mask, self.kernel)
        # get list of contour indices for the current frame
        # contours = find_contours(mask, self.kernel, min_area=self.min_contour_area)
        # # return largest contour
        # if contours:
        #     del mask, self.first_frame
        #     torch.cuda.empty_cache()
        #     self.first_frame = img  # reset the first frame to the current frame
        #     return max(contours, key=lambda x: len(x))
        # TODO: need to set a timer of some sort to update the first frame after a certain amount of time has passed, whether motion was detected or not
        return None

    def get_contour_centroid(self, contour: torch.Tensor) -> Tuple[int, int]:
        """ given a single contour of shape (N,2), compute centroid (cx, cy)
            contour[:,0] = row, contour[:,1] = col, if you used (y,x) indexing
        """
        # called by MotionTrackingCommand.execute() to get the centroid of the largest contour
        if contour is None or len(contour) == 0:
            return None
        # if contour is (row, col) => (y, x), compute mean row, mean col
        mean_y = float(torch.mean(contour[:, 0]))
        mean_x = float(torch.mean(contour[:, 1]))
        return mean_x, mean_y

    def reset(self):
        """ reset the first frame (e.g., after significant changes in the environment) """
        self.first_frame = None
