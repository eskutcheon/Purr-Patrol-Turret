from typing import Tuple, List, Union, Dict, Optional, Set
import torch
import torchvision as TV
# local imports
from ..utils import view_boxes
from ..config.types import DetectorLike
from .base_detectors import DetectionResult, DetectionFeedback


class DetectionPipeline:
    """ coordinates detection steps:
        1. Use a chosen bounding-box detector (Strategy design pattern).
        2. Do label-based overlap checks, or pick the highest-scoring bounding box.
        3. Return a final DetectionFeedback dataclass with various values of interest.
    """
    # TODO: add DetectorLike type hint for this in config.types later
    def __init__(
        self,
        # TODO: allow detector input to be a string indicating the detector type (probably should save keys and class names in a global dict)
        detector: DetectorLike,
        # TODO: remove overlap_threshold argument and get it from the detector instead later (unless I change things to create the detector here)
        overlap_threshold: float = 0.1,
        animal_labels: Optional[List[str]] = None,
        plant_labels: Optional[Union[List[str], str]] = None #["potted plant", "vase"]
    ):
        self.detector = detector
        # Optionally store label maps or overlap thresholds
        self.overlap_threshold = overlap_threshold
        # TODO: create a small helper function for this that also ensures it's an iterable in case a string is ever passed
        self.animal_labels = set(animal_labels) if animal_labels else {"cat", "dog"}
        if plant_labels and isinstance(plant_labels, str):
            plant_labels = [plant_labels]
        # TODO: look through the other MobileNet object classes and see if there are any other pertinent labels like flowers or fruits
        self.plant_labels = set(plant_labels) if plant_labels else {"potted plant", "vase", "bowl"}
        self.protected_label = "potted plant"  # TODO: add a config option for this later

    def run_detection(self, frame: torch.Tensor) -> DetectionFeedback:
        """ run the detection pipeline on a single frame:
            1) Run bounding-box detection
            2) Group boxes by label
            3) Check for overlap (e.g. animal vs plant)
            4) Return a comprehensive DetectionFeedback object
        """
        # results is a dictionary of each detected object’s bounding box, label, and score
        results = self.detector.detect(frame)
        # if no boxes, return empty feedback
        if results.boxes.shape[0] == 0:
            return DetectionFeedback(shoot_flag=False, notes="No bounding boxes after threshold + NMS")
        # merge plant labels into a unified category before NMS
        results.labels = self._merge_labels(results.labels, self.plant_labels)
        nms_results: DetectionResult = self._apply_labelwise_nms(results.boxes, results.labels, results.scores)
        if len(nms_results.boxes) == 0:
            return DetectionFeedback(shoot_flag=False, notes="All boxes suppressed by NMS")
        #label_list = [self.detector.classes[int(label.item())] for label in nms_results.labels]
        #view_boxes(frame, nms_results.boxes, label_list)
        # Group boxes by "animal" vs "plant" (or other categories) for potential overlap checks
        animal_boxes, plant_boxes = self._group_boxes(nms_results)
        # try to find an overlap between animal_boxes & plant_boxes
        feedback = self._compute_animal_plant_overlap(animal_boxes, plant_boxes)
        return feedback


    def _merge_labels(self, labels: torch.Tensor, labels_to_merge: Set[str]) -> torch.Tensor:
        """ Convert e.g. 'vase' => 'potted plant' by rewriting label IDs (referencing self.detector.classes) in a new tensor """
        ids_to_merge = [self._class_to_id(label) for label in labels_to_merge]
        # NOTE: in the future, it shouldn't hypothetically matter if we just set target_id = ids_to_merge[0] since they get reset to "plant_box" either way
        target_id = self._class_to_id(self.protected_label)
        label_mask = torch.isin(labels, torch.tensor(ids_to_merge, device=labels.device))
        labels[label_mask] = target_id
        return labels


    def _class_to_id(self, class_str: str) -> int:
        """ Convert label string -> label ID using self.detector.classes or a fallback """
        if class_str in self.detector.classes:
            return self.detector.classes.index(class_str)
        else:
            raise ValueError(f"Label '{class_str}' not found in detector classes")

    def _apply_labelwise_nms(self, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor) -> DetectionResult:
        """ Apply NMS to each label category separately """
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
        # Assuming the detector has “classes” to map label IDs -> label strings
        # SSDDetector uses class_name = self.detector.classes[label_id]; with different ones, I'll need to unify that approach
        animal_boxes = []
        plant_boxes = []
        #other_boxes = []
        for i in range(len(labels)):
            label_id = int(labels[i].item())
            class_name = self.detector.classes[label_id]
            if class_name in self.animal_labels:
                animal_boxes.append(boxes[i])
            elif class_name in self.plant_labels:
                plant_boxes.append(boxes[i])
            # else:
            #     other_boxes.append(box)
        return animal_boxes, plant_boxes #, other_boxes

    def _compute_animal_plant_overlap(self, animal_boxes: List[torch.Tensor], plant_boxes: List[torch.Tensor]) -> DetectionFeedback:
        """ For each animal box vs each plant box, compute IoU and return the first pair that exceeds self.overlap_threshold.
            Once found, compute center + gather final data.
            TODO: might want to return all pairs that exceed the threshold, not just the first one
        """
        # If either list is empty, we’re done
        if len(animal_boxes) == 0 or len(plant_boxes) == 0:
            return DetectionFeedback(shoot_flag=False, notes="No animal or plant boxes detected")
        for ani_box in animal_boxes:
            for plant_box in plant_boxes:
                iou = TV.ops.box_iou(ani_box.unsqueeze(0), plant_box.unsqueeze(0))  # shape [1,1]
                iou_val = float(iou.item())
                if iou_val > self.overlap_threshold:
                    #? NOTE: somewhat wondering whether it might be worth having a boundingbox namedtuple used only in this file
                        # alternatively, I suspect that torchvision.tv_tensors.BoundingBox might be more efficient through their optimizations
                    # bounding box order: (xmin, ymin, xmax, ymax)
                    center_x = int((ani_box[0] + ani_box[2]) // 2)
                    center_y = int((ani_box[1] + ani_box[3]) // 2)
                    # package final data
                    chosen_boxes = torch.stack([ani_box, plant_box])#[box[:4].int().tolist() for box in [ani_box, plant_box]]
                    chosen_labels = ["animal_box", "plant_box"]
                    return DetectionFeedback(True, (center_x, center_y), iou_val, chosen_boxes, chosen_labels,
                                             notes=f"Overlap found with IoU={iou_val} with threshold {self.overlap_threshold}")
        # If we never find an overlap, use defaults for DetectionFeedback:
        return DetectionFeedback(shoot_flag=False, notes="No valid overlap found")

