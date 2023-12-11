try:
    import cv2
except Exception as e:
    print("Warning: OpenCV not installed. To use motion detection, make sure you've properly configured OpenCV.")
import time
import torch
import numpy as np
import torchvision.transforms as TT
import torch.nn.functional as F
import utils

class ToCuda(object):
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return pic.to(device="cuda")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class VideoUtils(object):
    def __init__(self, camera_port):
        self.get_grayscale_tensor = TT.Compose([
            TT.ToTensor(),
            ToCuda(),
            TT.Resize(size=(360, 480), interpolation=TT.InterpolationMode.BILINEAR, antialias=True),
            TT.Grayscale(num_output_channels=1),
            TT.GaussianBlur((21,21), sigma=3.5) # 3.5 = what the original sigma came out to be with k=21 in cv2.GaussianBlur
        ])
        self.vid_capture = cv2.VideoCapture(camera_port)
        self.first_frame = None
        self.new_frame = None
        self.count = 0
    """
    Helper functions for video utilities.
    """
    def live_video(self):
        """ Opens a window with live video. """
        while True:
            # TODO: probably want to add some time.sleep(n) statements here to take a frame every `n` seconds
            # Capture frame-by-frame
            time.sleep(0.128) #sleeping for 128 milliseconds, checking roughly every 4 frames.
            ret, frame = self.vid_capture.read()
            # Display the resulting frame
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): #checking for a q key press every millisecond to break the video loop.
                break
        self.close_camera_feed()

    def close_camera_feed(self):
        # When everything is done, release the capture
        self.vid_capture.release()
        cv2.destroyAllWindows()

    def init_first_frame(self, img):
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

    @staticmethod
    def dilate_tensor(tensor, kernel_size=3, num_iter=2):
        # Create a dilation kernel
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=tensor.device)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        # Apply dilation
        for _ in range(num_iter):
            tensor = F.conv2d(tensor, kernel, padding=kernel_size//2)
        return tensor.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

    @staticmethod
    def get_best_contour(mask, threshold):
        # Convert tensor to numpy array and back to uint8
        mask_np = mask.cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_area = threshold
        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > best_area:
                best_area = area
                best_cnt = cnt
        return best_cnt

    # TODO: decompose this into more functions for better flexibility
    def find_motion(self, callback, show_video=False):
        time.sleep(0.25)
        # loop over the frames of the video
        while True:
            # grab the current frame and initialize the occupied/unoccupied text
            (grabbed, frame) = self.vid_capture.read()
            # if the frame could not be grabbed, then we have reached the end of the video
            if not grabbed:
                break
            # covert from BGR to RGB so that all the torchvision stuff works right
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # probably want to decompose this further into tensor creation and resizing and a separate grayscaling
            gray_img = self.get_grayscale_tensor(frame)
            # if the first frame is None, initialize it
            if self.first_frame is None:
                self.first_frame = self.init_first_frame(gray_img)
                if self.first_frame is None:
                    continue
            frame_delta = torch.abs(self.first_frame, gray_img)
            thresh = self.apply_threshold(frame_delta, 25)

            # Find contour and process
            contour = self.get_best_contour(thresh, 5000)
            if contour is not None:
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
        self.close_camera_feed()

    def get_best_contour(self, imgmask, threshold):
        im, contours, hierarchy = cv2.findContours(imgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_area = threshold
        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > best_area:
                best_area = area
                best_cnt = cnt
        return best_cnt