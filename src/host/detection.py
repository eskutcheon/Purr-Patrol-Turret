from typing import Tuple, List, Union, Dict, Optional, Set
import numpy as np
import torch
import torchvision as TV
# local imports
from ..utils import view_boxes
from ..config.types import DetectorLike
from .base_detectors import DetectionResult, DetectionFeedback


def sanitize_label_inputs(labels, valid_labels):
    if labels is None:
        labels = valid_labels
    elif isinstance(labels, (list, tuple, set)):
        labels = [label for label in labels if label in valid_labels]
        if len(labels) == 0:
            raise ValueError(f"None of the provided labels are valid. Valid labels are: {valid_labels}")
    elif labels and isinstance(labels, str):
        labels = [labels]
    else:
        raise ValueError(f"Invalid label input: {labels}. Must be a list, tuple, set, or None.")
    return labels

class DetectionPipeline:
    """ coordinates detection steps:
        1. Use a chosen bounding-box detector (Strategy design pattern).
        2. Do label-based overlap checks, or pick the highest-scoring bounding box.
        3. Return a final DetectionFeedback dataclass with various values of interest.
    """
    # TODO: may want to add a factory method with defaults to create a DetectionPipeline in the controller and let it be an optional argument
    def __init__(
        self,
        # TODO: allow detector input to be a string indicating the detector type (probably should save keys and class names in a global dict)
        detector: DetectorLike,
        overlap_threshold: float = 0.1,
        animal_labels: Optional[List[str]] = None,
        plant_labels: Optional[Union[List[str], str]] = None,
    ):
        self.detector = detector
        # Optionally store label maps or overlap thresholds
        self.overlap_threshold = overlap_threshold
        self.animal_labels = sanitize_label_inputs(animal_labels, ["cat", "dog"])
        self.animal_ids = [self.detector.classes.index(label) for label in self.animal_labels]
        self.plant_labels = sanitize_label_inputs(plant_labels, ["potted plant", "vase", "bowl"])
        self.plant_ids = [self.detector.classes.index(label) for label in self.plant_labels]

    def run_detection(self, frame: Union[np.ndarray, torch.Tensor]) -> DetectionFeedback:
        """ run the detection pipeline on a single frame:
            1) Run bounding-box detection
            2) Group boxes by label
            3) Check for overlap (e.g. animal vs plant)
            4) Return a comprehensive DetectionFeedback object
        """
        # def debug_results(nms_results: DetectionResult):
        #     label_list = [self.detector.classes[int(label.item())] for label in nms_results.labels]
        #     view_boxes(frame, nms_results.boxes, label_list)
        # results is a dictionary of each detected object’s bounding box, label, and score
        if not isinstance(frame, torch.Tensor):
            frame = torch.from_numpy(frame).permute(2, 0, 1) if isinstance(frame, np.ndarray) else torch.stack(frame)
            frame = frame.to(device = self.detector.device)
        results = self.detector.detect(frame)
        # if no boxes, return empty feedback
        if results.boxes.shape[0] == 0:
            return DetectionFeedback(shoot_flag=[False], notes="No bounding boxes after confidence score thresholding")
        nms_results: DetectionResult = self._apply_labelwise_nms(results.boxes, results.labels, results.scores)
        if len(nms_results.boxes) == 0:
            return DetectionFeedback(shoot_flag=[False], notes="All boxes suppressed by NMS")
        # debug_results(nms_results)
        # Group boxes by "animal" vs "plant" (or other categories) for potential overlap checks
        # NOTE: _group_boxes can be simplified since _merge_labels is used to collapse to a single plant and animal label
        animal_boxes, plant_boxes = self._group_boxes(nms_results)
        # try to find an overlap between animal_boxes & plant_boxes
        feedback = self._compute_animal_plant_overlap(animal_boxes, plant_boxes)
        return feedback


    def _merge_labels(self, labels: torch.Tensor, ids_to_merge: Set[str]) -> torch.Tensor:
        """ Convert e.g. 'vase' => 'potted plant' by rewriting label IDs (referencing self.detector.classes) in a new tensor """
        target_id = ids_to_merge[0]
        ids_to_merge = torch.tensor(ids_to_merge, device=labels.device)
        label_mask = torch.isin(labels, ids_to_merge)
        labels[label_mask] = target_id
        return labels

    def _apply_labelwise_nms(self, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor) -> DetectionResult:
        """ Apply NMS to each label category separately """
        # merge plant labels then animal labels into (distinct) unified categories before NMS
        labels = self._merge_labels(labels, self.plant_ids)
        labels = self._merge_labels(labels, self.animal_ids)
        # "Batched" labelwise non-maximum suppression - treats `labels` as the 'idxs' so that boxes w/ different labels won't suppress each other
        keep_idx = TV.ops.batched_nms(boxes, scores, labels, 2*self.overlap_threshold)
        return DetectionResult(boxes=boxes[keep_idx], labels=labels[keep_idx], scores=scores[keep_idx])

    def _group_boxes(self, detection_result: DetectionResult) -> Tuple[List[torch.Tensor]]:
        """ separating bounding boxes into 3 categories:
            - Animal: label in self.animal_labels
            - Plant: label == self.plant_labels
            - Others: everything else
        """
        boxes = detection_result.boxes
        labels = detection_result.labels
        animal_ids = torch.tensor(self.animal_ids, device=labels.device)
        plant_ids = torch.tensor(self.plant_ids, device=labels.device)
        animal_boxes = boxes[torch.isin(labels, animal_ids)]
        plant_boxes = boxes[torch.isin(labels, plant_ids)]
        #other_boxes = boxes[~torch.isin(labels, animal_ids) & ~torch.isin(labels, plant_ids)]
        return animal_boxes, plant_boxes

    def _compute_animal_plant_overlap(self, animal_boxes: torch.Tensor, plant_boxes: torch.Tensor) -> DetectionFeedback:
        """ For each animal box vs each plant box, compute IoU and return all pairs that exceed self.overlap_threshold. """
        all_feedback = None
        # If either list is empty, we’re done
        if len(animal_boxes) == 0 or len(plant_boxes) == 0:
            return DetectionFeedback(shoot_flag=[False], notes="No animal or plant boxes detected")
        for ani_box in animal_boxes:
            for plant_box in plant_boxes:
                feedback = self._compute_overlap_feedback(ani_box, plant_box)
                if all_feedback is None and feedback: # if feedback is not None, we found a valid overlap to add
                    all_feedback = feedback
                elif feedback: # if feedback is not None, we found a valid overlap to add
                    all_feedback.append(feedback) # append the new feedback to the existing feedback object
        if all_feedback is None: # if the all_feedback was never initialized, no valid overlaps were found
            all_feedback = DetectionFeedback(shoot_flag=[False], notes="No valid overlap found")
        return all_feedback

    def _compute_overlap_feedback(self, ani_box: torch.Tensor, plant_box: torch.Tensor) -> DetectionFeedback:
        """ Compute IoU for a given pair of animal and plant boxes and return DetectionFeedback if it exceeds the threshold."""
        iou = TV.ops.box_iou(ani_box.unsqueeze(0), plant_box.unsqueeze(0))  # shape [1,1]
        iou_val = float(iou.item())
        if iou_val > self.overlap_threshold:
            # bounding box order: (xmin, ymin, xmax, ymax)
            center_x = int((ani_box[0] + ani_box[2]) // 2)
            center_y = int((ani_box[1] + ani_box[3]) // 2)
            # package final data
            chosen_boxes = torch.stack([ani_box, plant_box])
            chosen_labels = ["animal_box", "plant_box"]
            return DetectionFeedback([True], [(center_x, center_y)], [iou_val], chosen_boxes, chosen_labels,
                                    notes=[f"Overlap found with IoU={iou_val} with threshold {self.overlap_threshold}"])
        return None

    @staticmethod
    def default_factory(
        detector: Optional[DetectorLike] = None,
        overlap_threshold: Optional[float] = 0.1,
        animal_labels: Optional[List[str]] = None,
        plant_labels: Optional[Union[List[str], str]] = None
    ) -> 'DetectionPipeline':
        """ Factory method to create a DetectionPipeline with default values for all arguments """
        if detector is None:
            from .base_detectors import FasterRCNNDetector
            detector = FasterRCNNDetector() # default detector with default score threshold
        if animal_labels is None:
            animal_labels = ["cat", "dog"]
        if plant_labels is None:
            plant_labels = ["potted plant"]
        return DetectionPipeline(detector, overlap_threshold, animal_labels, plant_labels)