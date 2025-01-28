"""
- Runs instance segmentation on frames using GPU acceleration (CUDA)
- Interfaces with the pytorch detection model

detect(frame): Runs instance segmentation on a frame and returns detection results
filter_results(results): Filters and processes segmentation results
calculate_target_coord(detections): Determines target coordinates for the turret

"""