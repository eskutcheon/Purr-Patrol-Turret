import time
import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes
import numpy as np
from typing import Union, List, Dict, Callable, Tuple


def get_user_confirmation(prompt):
    answers = {'y': True, 'n': False}
    # ! WARNING: only works in Python 3.8+
    while (response := input(f"[Y/n] {prompt} ").lower()) not in answers:
        print("Invalid input. Please enter 'y' or 'n' (not case sensitive).")
    return answers[response]


def view_boxes(img, *args, target=None, dest_path=None):
    # check if args contains a tuple of (box_coords, labels) or separate arguments
    box_coords, labels = None, None
    if len(args) == 2:
        box_coords, labels = args
    elif len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == 2:
        box_coords, labels = args[0]
    else:
        raise ValueError("Invalid arguments. Expected (box_coords, labels) as a tuple or separate arguments.")
    # TODO: need to update this to take an arbitrary number of box_coords, reference labels from a global dictionary, and pick colors randomly
    img_marked = draw_bounding_boxes(img, box_coords, labels=labels, width=2) #, colors=['red', 'green'])
    img_marked = img_marked.numpy().transpose([1,2,0]) # ndarray objects expect the channel dim to be the last one
    fig, ax = plt.subplots()
    ax.imshow(img_marked)
    ax.axis('off')  # Turns off axes
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Removes whitespace
    #plt.imshow(img_marked)
    if target is not None:
        ax.scatter([target[0]], [target[1]], color="red", marker="x", s=100)
    if dest_path is not None:
        print(f"Saving marked image to {dest_path}...")
        plt.savefig(dest_path, bbox_inches="tight", pad_inches=0)
        time.sleep(1)
        plt.close(fig)
    plt.show()


def view_contours(img, contours, target=None, dest_path=None):
    """ visualize contours on an image
        :param img: Base image as a numpy array.
        :param contours: Contours as a numpy array of shape (N, 2).
        :param target: Target as a tuple (row, column) index.
        :param dest_path: Optional path to save the image.
    """
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.plot(contours[:, 1], contours[:, 0], linewidth=2)  # contours[:, 1] is x, contours[:, 0] is y
    ax.axis('off')  # Turns off axes
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Removes whitespace
    if target is not None:
        ax.scatter([target[1]], [target[0]], color="red", marker="x", s=100)  # target[1] is x, target[0] is y
    if dest_path is not None:
        print(f"Saving marked image to {dest_path}...")
        plt.savefig(dest_path, bbox_inches="tight", pad_inches=0)
        time.sleep(1)
        plt.close(fig)
    plt.show()

# borrowed all the image utility functions from my other projects

def tensor_to_ndarray(tensor: torch.Tensor) -> np.ndarray:
    ''' convert pytorch tensor in shape (N,C,H,W) or (C,H,W) to ndarray of shape (N,H,W,C) or (H,W,C) '''
    assert isinstance(tensor, torch.Tensor), f"input must be a torch.Tensor object; got {type(tensor)}"
    if tensor.dim() not in [3,4]:
        return tensor.numpy()
    is_batch = tensor.dim() == 4
    new_dims = (0,2,3,1) if is_batch else (1,2,0)
    # NOTE: might be faster to do torch.permute(new_dims).numpy()
    np_array = np.transpose(tensor.numpy(), new_dims)
    return np_array

def ndarray_to_tensor(arr: np.ndarray) -> torch.Tensor:
    assert isinstance(arr, np.ndarray), f"input must be a numpy.ndarray object; got {type(arr)}"
    if arr.ndim not in [3,4]:
        return torch.from_numpy(arr)
    is_batch = (arr.ndim == 4)
    new_dims = (0,3,1,2) if is_batch else (2,0,1)
    tensor = torch.from_numpy(np.transpose(arr, new_dims))
    return tensor

def enforce_type(target_type):
    def decorator(func):
        def wrapper(img, *args, **kwargs):
            if target_type == "tensor":
                if isinstance(img, np.ndarray):
                    img = ndarray_to_tensor(img)
            elif target_type == "ndarray":
                if torch.is_tensor(img):
                    img = tensor_to_ndarray(img.cpu())
            else:
                raise ValueError("Unsupported target type, must be either 'tensor' or 'ndarray'")
            return func(img, *args, **kwargs)
        return wrapper
    return decorator

def is_int_dtype(img: torch.Tensor):
    return img.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)

def is_float_dtype(img: torch.Tensor):
    return img.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)

def get_normalized_image(img: torch.Tensor) -> torch.FloatTensor:
    TOL = 10e-8
    if is_int_dtype(img) and torch.max(img) > 1:
        img = img.to(dtype=torch.float32)/255
    elif is_float_dtype(img) and torch.max(img) > 1+TOL:
        img /= 255
    return img


def get_scaled_dims(input_shape: Tuple[int, int], max_size: int) -> Tuple[int, int]:
    """ get dimensions of input such that longest dimension <= max_size; scales the shorter dimension by the same factor
        :param input_shape: Tuple (H, W) representing the input frame dimensions
        :param max_size: Maximum size for the longest dimension
        :return: Tuple (H, W) with the scaled dimensions
    """
    H, W = input_shape
    long_dim = max(H, W)
    if long_dim > max_size:
        scale = max_size / float(long_dim)
        H *= scale
        W *= scale
    return int(H), int(W)


#######################################################################################################################
# ~ new code for the major refactor - mostly the decorators for now
#######################################################################################################################
"""
TODO: write decorators for reusable logic, such as error handling, logging, and mode-specific behavior

retry_on_failure: Retries failed operations (e.g., sending data to the server)
log_event: Logs key events like turret movement or firing
mode_specific: Adapts behavior for interactive vs network modes
"""