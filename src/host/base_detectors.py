from typing import Tuple, List, Union, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch
import torchvision as TV
# local imports
from ..utils import get_normalized_image, enforce_type, view_boxes

@dataclass
class DetectionResult:
    """ single detection result instance with bounding boxes, labels, and scores """
    boxes: torch.Tensor   # shape (N,4) in [xmin, ymin, xmax, ymax]
    labels: torch.Tensor  # shape (N,) integer label IDs
    scores: torch.Tensor  # shape (N,) detection confidences

@dataclass
class DetectionFeedback:
    """ Final, high-level detection outcome to pass back to the callering location
        - shoot_flag: whether a valid target (e.g. cat overlapping with plant) was found
        - target_center: (cx, cy) in camera pixel coords
        - overlap_iou: IoU of the relevant bounding boxes if an overlap was found
        - chosen_boxes: the boxes that participated in the final overlap check, with class labels
        - notes: arbitrary info for debugging or future expansions
    """
    shoot_flag: bool
    target_center: Tuple[int, int] = (0, 0)
    overlap_iou: float = 0.0
    chosen_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    chosen_labels: List[str] = field(default_factory=list)
    notes: str = ""



class BaseDetector(ABC):
    """ abstract base class for bounding box detectors """
    @abstractmethod
    def detect(self, frame: torch.Tensor) -> DetectionResult:
        """ given input frame of shape (C,H,W), returns bounding boxes, labels, and scores after thresholding + NMS, etc """
        pass


class SSDDetector(BaseDetector):
    """ bounding-box detector using a PyTorch SSDLite320_MobileNet_V3_Large model """
    def __init__(self, score_threshold: float = 0.1, device="cuda"):
        device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        weights = TV.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = TV.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
        self.model.to(device=device)
        self.model.eval()
        # Store metadata for label mapping
        self.classes = weights.meta["categories"]
        # detection confidence and IoU (bounding box overlap) thresholds, respectively
        self.score_threshold = score_threshold

    @enforce_type("tensor")
    def detect(self, frame: torch.Tensor) -> DetectionResult:
        """ inference on a single frame;  expected to be a [C,H,W] or (H,W,C) that gets converted. """
        frame = get_normalized_image(frame)  # scale to [0..1]
        with torch.no_grad():
            raw_result = self.model([frame])[0]  # returns dict of 'boxes','labels','scores'
        # apply score threshold
        #print("shape of scores before thresholding: ", raw_result["scores"].shape)
        mask = raw_result["scores"] > self.score_threshold
        #print("shape of scores after thresholding: ", raw_result["scores"][mask].shape)
        boxes = raw_result["boxes"][mask]
        labels = raw_result["labels"][mask]
        scores = raw_result["scores"][mask]
        # TODO: consider adding something to the base class to drop all results using TV.ops.remove_small_boxes
        return DetectionResult(boxes=boxes, labels=labels, scores=scores)


class FasterRCNNDetector(BaseDetector):
    def __init__(self, score_threshold: float = 0.1, device="cuda"):
        device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        weights = TV.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = TV.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        self.model.to(device=device)
        self.model.eval()
        # NOTE: animal classes: 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
        self.classes = weights.meta["categories"]
        self.score_threshold = score_threshold
        """ All object classes for reference:
            detector classes:  ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
            'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']
        """

    @enforce_type("tensor")
    def detect(self, frame: torch.Tensor) -> DetectionResult:
        frame = get_normalized_image(frame)
        with torch.no_grad():
            raw = self.model([frame])[0]
        keep_mask = raw["scores"] > self.score_threshold
        boxes = raw["boxes"][keep_mask]
        labels = raw["labels"][keep_mask]
        scores = raw["scores"][keep_mask]
        # no NMS here => pipeline can handle it
        return DetectionResult(boxes=boxes, labels=labels, scores=scores)


class RetinaNetDetector(BaseDetector):
    def __init__(self, score_threshold: float = 0.1, device="cuda"):
        device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        weights = TV.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
        self.model = TV.models.detection.retinanet_resnet50_fpn(weights=weights)
        self.model.to(device=device)
        self.model.eval()
        self.classes = weights.meta["categories"]
        self.score_threshold = score_threshold

    @enforce_type("tensor")
    def detect(self, frame: torch.Tensor) -> DetectionResult:
        frame = get_normalized_image(frame)
        with torch.no_grad():
            raw = self.model([frame])[0]
        keep_mask = raw["scores"] > self.score_threshold
        boxes = raw["boxes"][keep_mask]
        labels = raw["labels"][keep_mask]
        scores = raw["scores"][keep_mask]
        return DetectionResult(boxes=boxes, labels=labels, scores=scores)