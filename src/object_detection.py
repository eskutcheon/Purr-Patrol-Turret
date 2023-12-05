import torch
import torchvision as TV
import numpy as np
import utils # utils.py in this project
import os
from pprint import pprint
from typing import Union, List, Dict, Callable



class Detector(object):
    def __init__(self):
        # https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html
        # DEFAULT weights are SSDLite320_MobileNet_V3_Large_Weights.COCO_V1 with 91 classes and 3440060 params
        weights = TV.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        # view categories where each index corresponds to the class labels returned by the model
        # TODO: need to figure out how to restrict output classes to just the ones we care about within the model
        self.classes = weights.meta["categories"]
        self.class_labels = {self.classes.index('cat'): 'cat',
                            self.classes.index('dog'): 'dog',
                            self.classes.index('potted plant'): 'plant'}
        #self.animal_labels = {self.classes.index('cat'): 'cat', self.classes.index('dog'): 'dog'}
        # TODO: may extend this and do some more training on a plant dataset if time permits
        #self.plant_labels = {self.classes.index('potted plant'): 'plant'}
        print(self.classes)
        self.overlap_threshold = 0.2
        self.score_threshold = 0.25
        # TODO: still probably need to specify weights_backbone, num_classes, and norm_layer
        self.model = TV.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
        self.model.eval()
        # still have to test the outputs of this model's forward method


    def detect(self, img):
        ''' from the model weights docs:
            The inference transforms are available at SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.transforms and perform the following preprocessing operations:
            Accepts PIL.Image, batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects. The images are rescaled to [0.0, 1.0].
            recipe: https://github.com/pytorch/vision/tree/main/references/detection#ssdlite320-mobilenetv3-large
        '''
        # inputs are expected to be normalized float tensors
        input_img = img.to(dtype=torch.float32)/255
        # forward method expects a list of torch.Tensor objects with shape (3,H,W)
        results = self.model([input_img])[0]
        return self.filter_results(results)

    def filter_results(self, results):
        # TODO: move to using Non-Maximum Suppression: https://machinelearningspace.com/non-maximum-suppression-in-pytorch-how-to-select-the-correct-bounding-box/
        # essentially have most of this but need to account for multiple instances by IOU scores
        results_pruned = {key: [] for key in results.keys()}
        for i in range(len(results["labels"])):
            label = int(results["labels"][i])
            if results["scores"][i] < self.score_threshold:
                break
            if label in self.class_labels.keys():
                results_pruned["boxes"].append(results["boxes"][i])
                results_pruned["labels"].append(self.class_labels[label])
        return results_pruned

    def get_overlap(self, animal_boxes: torch.Tensor, plant_boxes: torch.Tensor):
        # pretty sure this can take batch tensors just fine, so may just do that
        # bounding box order MUST be (xmin, ymin, xmax, ymax)
        iou = TV.ops(animal_boxes, plant_boxes, reduction="sum")
        if iou > self.overlap_threshold:
            pass
        # pass whatever results with class bounding boxes to this to get the percent overlap of cat and plant boxes
        # maybe return the center of the cat bounding box if overlap > threshold in pixel coordinates
        # pass this center to a filter that corrects for the offset between the camera and the sprayer tip
            # probably needs to be done in the turret.py file or a new targeting.py file
            # this will depend on receiving the current orientation of the turret
                # just assume the camera is always at the same angle and maybe physically mark the default angle on the camera mount
        pass

        @utils.enforce_type(torch.Tensor)
        def scan_frame(self, capture):
            # NOTE: may need more preprocessing done here, e.g., grayscale to ensure lower computation time from the model
            results = self.detect(capture)


''' general pipeline order:
    motion_tracking watches for its trigger, then an image is passed as a torch.Tensor or np.ndarray to a scan_frame method of its Detector class member
        add check for image array type later
    scan_frame -> detect -> filter_boxes -> get_overlap -> ?
        need to also get the centroid of the bounding box intersection to return the (unprocessed) coordinate to the target to motion_tracking
            probably just return None if the bounding box overlap isn't high enough to trigger things
'''


# using this just for testing for now
if __name__ == "__main__":
    test_img = TV.io.read_image(os.path.join("tests", "catonaplant.png"), TV.io.ImageReadMode.RGB) # read as uint8 tensor of shape (3,H,W)
    print(f"test_img shape: {test_img.shape}")
    detector = Detector()
    results = detector.detect(test_img)
    for key, vals in results.items():
        # should return boxes (2D float16 Tensor of shape (N,4)), scores (1D float16 Tensor of size N), labels (1D int Tensor of size N)
        print(f"{key}:\n{vals}")
    boxes = torch.stack(results["boxes"])
    utils.view_boxes(test_img, boxes, results["labels"])
