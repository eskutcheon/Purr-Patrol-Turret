import os
import sys
import time
from torchvision.io import read_image, ImageReadMode
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
# local imports
from src.utils import view_boxes
from src.host.base_detectors import SSDDetector, FasterRCNNDetector, RetinaNetDetector
from src.host.detection import DetectionPipeline


IOU_THRESHOLD = 0.1
SCORE_THRESHOLD = 0.2

def run_targeting_tests(test_img_dir, results_dir):
    time_sum = 0
    test_filenames = os.listdir(test_img_dir)
    num_test_imgs = len(test_filenames)
    #detector = SSDDetector(score_threshold=SCORE_THRESHOLD)
    #detector = FasterRCNNDetector(score_threshold=SCORE_THRESHOLD)
    detector = RetinaNetDetector(score_threshold=SCORE_THRESHOLD)
    pipeline = DetectionPipeline(detector, overlap_threshold=IOU_THRESHOLD) #, animal_labels=["cat", "dog"])
    for filename in test_filenames:
        print(f"Running detection test on {filename}...")
        new_filename = f"refactored_{os.path.splitext(filename)[0]}_t{SCORE_THRESHOLD}_targeted.png"
        output_path = os.path.join(results_dir, new_filename)
        test_img_path = os.path.join(test_img_dir, filename)
        test_img = read_image(test_img_path, ImageReadMode.RGB) # read as uint8 tensor of shape (3,H,W)
        start_time = time.time()
        results = pipeline.run_detection(test_img.to(device="cuda"))
        time_sum += time.time() - start_time
        print("final boxes:", results.chosen_boxes)
        print("final labels:", results.chosen_labels)
        print("final target_center:", results.target_center)
        if results.shoot_flag:
            print(f"Target detected with IoU overlap {results.overlap_iou}!")
            view_boxes(test_img, results.chosen_boxes, results.chosen_labels, target=results.target_center, dest_path=output_path)
        print()
    print("avg time per image: ", time_sum/num_test_imgs)

if __name__ == "__main__":
    results_dir = os.path.realpath(r".\results")
    os.makedirs(results_dir, exist_ok=True)
    test_img_dir = os.path.realpath(r".\test_images")
    if not os.path.exists(test_img_dir) or len(os.listdir(test_img_dir)) == 0:
        raise FileNotFoundError("No test images found in the test_images directory")
    run_targeting_tests(test_img_dir, results_dir)