try:
    import cv2
except Exception as e:
    print("Warning: OpenCV not installed. To use motion detection, make sure you've properly configured OpenCV.")
import time
from typing import Union, Optional, Tuple, List
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as TT
import torch.nn.functional as F
from utils import enforce_type



def to_cuda(img: Union[torch.Tensor, np.ndarray, Image.Image]) -> torch.Tensor:
    if not img.is_cuda:
        img = img.to(device="cuda")
    return img


class MotionTracker(object):
    def __init__(self, camera_port, device="cuda"):
        self.device = device
        self.get_grayscale_tensor = TT.Compose([
            TT.ToTensor(),
            TT.Lambda(to_cuda) if device == "cuda" else TT.Lambda(lambda img: img),
            TT.Resize(size=(360, 480), interpolation=TT.InterpolationMode.BILINEAR, antialias=True),
            TT.Grayscale(num_output_channels=1),
            TT.GaussianBlur((21,21), sigma=3.5) # 3.5 = what the original code's sigma came out to be with k=21 in cv2.GaussianBlur
        ])
        self.vid_capture = cv2.VideoCapture(camera_port)
        self.first_frame = None
        self.new_frame = None
        self.count = 0

    def live_video(self):
        """ Opens a window with live video. """
        while True:
            # TODO: probably want to add some time.sleep(n) statements here to take a frame every `n` seconds
            # Capture frame-by-frame
            time.sleep(5) # sleep for 5 seconds
            ret, frame = self.vid_capture.read()
            # Display the resulting frame
            cv2.imshow('Video', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'): #checking for a q key press every 100 ms to break the video loop.
                break
        self._close_camera_feed()

    def _close_camera_feed(self):
        # When everything is done, release the capture
        self.vid_capture.release()
        cv2.destroyAllWindows()

    #######################################################################################################################
    #~ thresholding code - may take another approach later
    #######################################################################################################################
    @enforce_type("tensor")
    def _init_first_frame(self, img):
        if self.new_frame is None:
            self.new_frame = img
            return None
        delta = torch.abs(self.new_frame, img)
        self.new_frame = img
        tst = self.apply_threshold(delta, 5)
        # missing dilation
        if self.count > 30:
            if not torch.count_nonzero(tst):
                return img
        else:
            self.count += 1
            return None

    @staticmethod
    def apply_threshold(frame, threshold_value):
        return (frame > threshold_value/255).type(torch.float)
    #######################################################################################################################


    @staticmethod
    @enforce_type("tensor")
    def resize_frame(frame, new_width=500):
        """Resizes frame based on the original aspect ratio."""
        H, W = frame.shape[-2:]  # Get height and width
        new_height = int(H * (new_width / float(W))) # Calculate new height based on aspect ratio
        frame = F.interpolate(frame.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)
        return frame

    @staticmethod
    @enforce_type("tensor")
    def dilate_tensor(tensor, kernel_size=3, num_iter=2):
        # Create a dilation kernel
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=tensor.device)
        while tensor.dim() < 4:
            tensor = tensor.unsqueeze(0)  # Add batch and channel dimensions
        # Apply dilation
        for _ in range(num_iter):
            tensor = F.conv2d(tensor, kernel, padding=kernel_size//2)
        tensor = torch.clamp(tensor, 0, 1)  # Ensure values are within range [0, 1]
        return tensor.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

    @staticmethod
    def _get_best_contour(mask, threshold):
        # Convert tensor to numpy array and back to uint8
        mask_np = mask.cpu().numpy().astype(np.uint8)
        _, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_area = threshold
        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > best_area:
                best_area = area
                best_cnt = cnt
        return best_cnt


    def find_motion(self, callback, show_video=False):
        time.sleep(0.25)
        # loop over the frames of the video
        while True:
            # grab the current frame and initialize the occupied/unoccupied text
            grabbed, frame = self.vid_capture.read()
            # if the frame could not be grabbed, then we have reached the end of the video
            if not grabbed:
                break
            # covert from BGR to RGB so that all the torchvision stuff works right
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.resize_frame(frame)
            # probably want to decompose this further into tensor creation and resizing and a separate grayscaling
            gray_img = self.get_grayscale_tensor(frame)
            # if the first frame is None, initialize it with the grayscale image
            if self.first_frame is None:
                self.first_frame = self._init_first_frame(gray_img)
                if self.first_frame is None:
                    continue
            frame_delta = torch.abs(self.first_frame - gray_img)
            thresh = self.apply_threshold(frame_delta, 25)
            thresh = self.dilate_tensor(thresh)
            # Find contour and process
            contour = self._get_best_contour(thresh, 5000)
            if contour is not None:
                # TODO: need to add all the targeting stuff here
                # Draw bounding box and callback
                # bbox = calculate_bounding_box(contour)
                # frame = draw_bounding_box(frame, bbox)
                callback(contour, frame)
            #######################################################################################################################
            # show the frame and record if the user presses a key
            if show_video:
                cv2.imshow("Security Feed", frame)
                key = cv2.waitKey(1) & 0xFF
                # if the `q` key is pressed, break from the lop
                if key == ord("q"):
                    break
        self._close_camera_feed()