import torch
import torchvision as TV
import numpy as np
import utils # utils.py in this project
import os, sys
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
        # NOTE: may actually want to add the vase class to this list since it keeps detecting the plant pots as one
        # should be faster than doing this on the fly
        self.class_int_labels = torch.Tensor(list(self.class_labels.keys()))
        self.animal_labels = {self.classes.index('cat'): 'cat', self.classes.index('dog'): 'dog'}
        # TODO: may extend this and do some more training on a plant dataset if time permits
        self.plant_labels = {self.classes.index('potted plant'): 'plant'}
        #print(self.classes)
        self.overlap_threshold = 0.1
        # NOTE: seems that some images may give very low confidence scores so constants may be a problem
        self.score_threshold = 0.1
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
        with torch.no_grad():
            results = self.model([input_img])[0]
            '''for key, vals in results.items():
                # should return boxes (2D float16 Tensor of shape (N,4)), scores (1D float16 Tensor of size N), labels (1D int Tensor of size N)
                print(f"{key}:\n{vals}")'''
            return self.filter_results(results)


    def filter_results(self, results):
        thresh_mask = torch.where(results["scores"] > self.score_threshold)
        results_pruned = {key: val[thresh_mask] for key, val in results.items()}
        all_boxes = []
        all_labels = []
        is_animal = []
        unique_labels = torch.unique(results_pruned["labels"])
        valid_indices = torch.isin(unique_labels, self.class_int_labels)
        # NOTE: may apparently be able to replace this loop with torchvision.ops.boxes.batches_nms
            # actually has a lot I could use: http://pytorch.org/vision/stable/_modules/torchvision/ops/boxes.html
        for label in unique_labels[valid_indices]:
            label_indices = torch.where(results_pruned["labels"] == label)[0]
            label_boxes = results_pruned["boxes"][label_indices]
            label_scores = results_pruned["scores"][label_indices]
            # non-maximum suppression to prune less confident results
            nms_indices = TV.ops.nms(label_boxes, label_scores, iou_threshold=0.2)
            all_boxes.append(label_boxes[nms_indices])
            all_labels.extend([int(label.item()) for _ in range(len(nms_indices))])
            # check over each new label to see if it's an animal
            is_animal.extend([label.item() in self.animal_labels.keys() for _ in range(len(nms_indices))])
        return {"boxes": torch.round(torch.cat(all_boxes, dim=0)),
                "labels": torch.tensor(all_labels),
                "is_animal": torch.tensor(is_animal)}


    def get_overlap(self, detections: dict):
        plant_indices = torch.where(~detections["is_animal"])[0]
        for animal_idx in torch.where(detections["is_animal"])[0]:
            for plant_idx in plant_indices:
                # bounding box order MUST be (xmin, ymin, xmax, ymax)
                ani_box = detections["boxes"][animal_idx].unsqueeze(0)
                plant_box = detections["boxes"][plant_idx].unsqueeze(0)
                iou = TV.ops.box_iou(ani_box, plant_box)
                if float(iou) > self.overlap_threshold:
                    center = [int((ani_box[0,0] + ani_box[0,2])//2), int((ani_box[0,1] + ani_box[0,3])//2)]
                    return True, center
        return False, [0,0]
        # pass whatever results with class bounding boxes to this to get the percent overlap of cat and plant boxes
        # maybe return the center of the cat bounding box if overlap > threshold in pixel coordinates
        # pass this center to a filter that corrects for the offset between the camera and the sprayer tip
            # probably needs to be done in the turret.py file or a new targeting.py file
            # this will depend on receiving the current orientation of the turret
                # just assume the camera is always at the same angle and maybe physically mark the default angle on the camera mount

    @utils.enforce_type("tensor")
    def scan_frame(self, capture):
        # NOTE: may need more preprocessing done here, e.g., grayscale to ensure lower computation time from the model
        detected_dict = self.detect(capture)
        shoot_flag, target_coord = self.get_overlap(detected_dict)
        return shoot_flag, target_coord
        #print(shoot_flag)
        #print(target_coord)


''' general pipeline order:
    motion_tracking watches for its trigger, then an image is passed as a torch.Tensor or np.ndarray to a scan_frame method of its Detector class member
        add check for image array type later
    scan_frame -> detect -> filter_boxes -> get_overlap -> ?
        need to also get the centroid of the bounding box intersection to return the (unprocessed) coordinate to the target to motion_tracking
            probably just return None if the bounding box overlap isn't high enough to trigger things
'''

def run_targeting_tests():
    all_test_images = os.listdir("test_images")
    if not os.path.exists("results"):
        os.makedirs("results")
    for test in all_test_images:
        test_img = TV.io.read_image(os.path.join("test_images", test), TV.io.ImageReadMode.RGB) # read as uint8 tensor of shape (3,H,W)
        #print(f"test_img shape: {test_img.shape}")
        detector = Detector()
        #detector.scan_frame(test_img)
        results = detector.detect(test_img)
        shoot_flag, target_coord = detector.get_overlap(results)
        print(shoot_flag)
        print(target_coord)
        print(results)
        # FIXME: add check for when results["boxes"] is empty
        new_filename = f"{os.path.splitext(test)[0]}_t{detector.score_threshold}_targeted.png"
        output_path = os.path.join("results", new_filename)
        box_labels = [detector.class_labels[int(label)] for label in results["labels"]]
        if not np.array_equal(target_coord, [0,0]):
            utils.view_boxes(test_img, results["boxes"], box_labels, target=target_coord, dest_path=output_path)


# using this just for testing for now
'''if __name__ == "__main__":
    run_targeting_tests()'''
