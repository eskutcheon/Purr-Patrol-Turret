import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision as TV
from typing import Union, List, Dict, Callable



def view_boxes(img, box_coords, labels):
    # TODO: need to update this to take an arbitrary number of box_coords, reference labels from a global dictionary, and pick colors randomly
    img_marked = TV.utils.draw_bounding_boxes(img, box_coords, labels=labels) #, colors=['red', 'green'])
    img_marked = img_marked.numpy().transpose([1,2,0]) # ndarray objects expect the channel dim to be the last one
    plt.imshow(img_marked)
    plt.show()

# borrowed all the image utility functions from my other projects

def is_batch_tensor(tensor: Union[torch.Tensor, np.ndarray]):
    ''' test if shape is (N,C,H,W) or (C,H,W) '''
    return int(len(tensor.shape) == 4) # 0 if shape is (C,H,W), 1 if shape is (N,C,H,W)

def tensor_to_ndarray(tensor: torch.Tensor) -> np.ndarray:
    ''' convert pytorch tensor in shape (N,C,H,W) or (C,H,W) to ndarray of shape (N,H,W,C) or (H,W,C) '''
    assert isinstance(tensor, torch.Tensor), f"input must be a torch.Tensor object; got {type(tensor)}"
    if tensor.dim() not in [3,4]:
        return tensor.numpy()
    # TODO: check if tensor is already permuted to shape below
    is_batch = is_batch_tensor(tensor)
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