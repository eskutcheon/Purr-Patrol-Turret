import torch
import torchvision.models as Models
import torchvision.utils as Utils



class Detector(object):
    def __init__(self):
        # https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html
        # DEFAULT weights are SSDLite320_MobileNet_V3_Large_Weights.COCO_V1 with 91 classes and 3440060 params
        weights = Models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        # view categories where each index corresponds to the class labels returned by the model
        # TODO: need to figure out how to restrict output classes to just the ones we care about - should reduce computation time
        self.classes = weights.meta["categories"]
        self.animal_labels = {'cat': self.classes.index('cat'), 'dog': self.classes.index('dog')}
        # TODO: may extend this and do some more training on a plant dataset if time permits
        self.plant_labels = {'plant': self.classes.index('potted plant')}
        print(self.classes)
        # TODO: still probably need to specify weights_backbone, num_classes, and norm_layer
        self.model = Models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
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
        return self.model([input_img])

    def get_overlap(self, results):
        # pass whatever results with class bounding boxes to this to get the percent overlap of cat and plant boxes
        # maybe return the center of the cat bounding box if overlap > threshold in pixel coordinates
        # pass this center to a filter that corrects for the offset between the camera and the sprayer tip
            # probably needs to be done in the turret.py file or a new targeting.py file
            # this will depend on receiving the current orientation of the turret
                # just assume the camera is always at the same angle and maybe physically mark the default angle on the camera mount
        pass


def view_boxes(img, box_coords):
    import matplotlib.pyplot as plt
    img_marked = Utils.draw_bounding_boxes(img, box_coords, labels=['cat', 'plant'], colors=['yellow', 'green'])
    img_marked = img_marked.numpy().transpose([1,2,0]) # ndarray objects expect the channel dim to be the last one
    plt.imshow(img_marked)
    plt.show()


def get_significant_boxes(results):
    # TODO: need to add some checks elsewhere to ensure the highest categories the detector finds are among the class labels in Detector
    bbox_dict = {'boxes': [], 'labels': []}
    for idx, score in enumerate(results['scores']):
        # they're sorted by default so this logic should be fine
        if score < 0.5:
            break
        bbox_dict['boxes'].append(results['boxes'][idx])
        bbox_dict['labels'].append(results['labels'][idx])
    return bbox_dict


# using this just for testing for now
if __name__ == "__main__":
    import os
    from pprint import pprint
    import torchvision.io as IO
    test_img = IO.read_image(os.path.join("tests", "catonaplant.png"), IO.ImageReadMode.RGB) # read as uint8 tensor of shape (3,H,W)
    print(f"test_img shape: {test_img.shape}")
    detector = Detector()
    results = detector.detect(test_img)[0]
    for key, vals in results.items():
        print(f"{key}:\n{vals}")
    print(results['boxes'][0])
    # TODO: need criteria for selecting bounding boxes - selecting a variable number of those with scores > threshold (like 0.5) may be good enough
    boxes = torch.zeros((2,4))      #, dtype=torch.int)
    boxes[0] = results['boxes'][0]  #.to(dtype=torch.int)
    boxes[1] = results['boxes'][1]  #.to(dtype=torch.int)
    view_boxes(test_img, boxes)
    # TODO: after getting relevant boxes from get_significant boxes, return the boxes (in the relevant order - write those down later - think it's [xmin, ymin, xmax, ymax])