import os, sys
import torch
import torchvision.io as IO
import torchvision.transforms.v2 as TT
import kornia.filters as KF
#import kornia.morphology as KM
# import kornia.color as KC
# import kornia.enhance as KE
from sklearn.cluster import KMeans
from skimage.segmentation import slic
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist



def get_mean_and_std(img):
    return [
        img.float().mean(dim=(2,1)),
        img.float().std(dim=(2,1))
    ]


def slic_superpixel_segmentation(img, n_segments=500, compactness=20):
    """ Applies SLIC superpixel segmentation to preserve edges before K-Means. """
    img_np = img.permute(1, 2, 0).numpy()  # Convert to (H, W, C)
    # Perform SLIC segmentation
    segments = slic(img_np, n_segments=n_segments, compactness=compactness, start_label=0)
    return segments


def get_top_colors(img: np.ndarray, num_clusters):
    # hard-coding in black and white as 2 initial centroid colors for now
    centroid_colors = [np.array([0, 0, 0], dtype=np.float32), np.array([255, 255, 255], dtype=np.float32)]
    if num_clusters == len(centroid_colors):
        return np.array(centroid_colors)
    # Find the most frequent colors for centroid initialization
    color_counts = Counter(map(tuple, img))
    top_colors = np.array([color for color, _ in color_counts.most_common(2*num_clusters)], dtype=np.float32)
    #print("top 10 colors before filtering: ", top_colors[:10])
    # iterate over the hard-coded colors and find the average pairwise difference between it and each color in `top_colors`
    # Compute pairwise distances between hard-coded centroids and top colors
    dists = cdist(np.array(centroid_colors), top_colors, metric="euclidean")  # (2, N) distance matrix
    min_dists = np.min(dists, axis=0)  # Get min distance to either black or white
    # Select the `num_clusters - 2` most distinct colors
    distinct_indices = np.argsort(min_dists)[::-1][:num_clusters - 2]  # Sort by max distance
    distinct_colors = top_colors[distinct_indices]
    # Append them to the hardcoded centroids
    centroid_colors.extend(distinct_colors)
    #print("final centroid colors: ", centroid_colors)
    return np.array(centroid_colors, dtype=np.float32)


def kmeans_compression(img: torch.Tensor, n_clusters=4, max_iter=200, use_spatial=True):
    """ Applies KMeans clustering with spatial weighting to reduce the number of colors in an image.
        Args:
            img (torch.Tensor): RGB image tensor of shape (C, H, W), values in [0, 255].
            n_clusters (int): Number of color clusters.
            max_iter (int): Maximum iterations for KMeans.
            use_spatial (bool): Whether to use (x, y) coordinates for spatial consistency.
        Returns:
            torch.Tensor: Compressed image with reduced colors.
    """
    # Reshape the image into (num_pixels, 3)
    c, h, w = img.shape
    img_flat = img.permute(1, 2, 0).reshape(-1, 3).numpy()
    # Superpixel segmentation
    segments = slic_superpixel_segmentation(img)
    segment_flat = segments.flatten()
    # Find the most frequent colors for centroid initialization (as byte to reduce search space)
    top_colors = get_top_colors(img_flat.astype(np.uint8), n_clusters)
    # Spatial weighting: Append (x, y) coordinates to color values
    if use_spatial:
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        spatial_features = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1) / max(h, w)  # Normalize
        img_features = np.hstack((img_flat, spatial_features))
        # If spatial coordinates are used, extend the centroid features to 5D with avg spatial coordinates of pixels near each color centroid
        spatial_mean = np.mean(spatial_features, axis=0)
        spatial_centroids = np.tile(spatial_mean, (n_clusters, 1))  # Assign avg spatial positions to centroids
        top_colors = np.hstack((top_colors, spatial_centroids))  # Now centroids are (N_clusters, 5)
    else:
        img_features = img_flat
    # Apply KMeans clustering with frequent colors as initial centroids
    kmeans = KMeans(n_clusters=n_clusters, init=top_colors, max_iter=max_iter, n_init=1)
    labels = kmeans.fit_predict(img_features)
    centroids = kmeans.cluster_centers_[:, :3].astype(np.uint8) # Keep only RGB; discard spatial info
    # Replace each pixel with its centroid color
    compressed_img_flat = centroids[labels]
    # Reshape back to (H, W, C) and convert to tensor
    compressed_img = torch.tensor(compressed_img_flat, dtype=torch.uint8).reshape(h, w, c).permute(2, 0, 1)
    # Enforce cluster consistency within superpixels
    for i in range(n_clusters):
        mask = segment_flat == i
        if np.any(mask):
            compressed_img_flat[mask] = np.median(compressed_img_flat[mask], axis=0)
    return compressed_img




def process_image(img: torch.Tensor):
    # get tensor to a 4D shape in range [0,1] for preprocessing
    img = img.unsqueeze(0)/255
    img = TT.functional.adjust_saturation(img, saturation_factor=0.4)
    img = TT.functional.autocontrast(img)
    img = KF.gaussian_blur2d(img, (5,5), (1.5, 1.5))
    #img = KF.median_blur(img, kernel_size=5)
    img = KF.bilateral_blur(img, kernel_size=7, sigma_color=0.1, sigma_space=(1.5, 1.5))
    #print("post filtering mean and std of img: ", get_mean_and_std(img))
    #img = 255*img.squeeze(0)
    #print("adjusted mean and std of img: ", get_mean_and_std(img))
    # brighten lighter regions while darkening black squares
    img = TT.functional.adjust_contrast(img, contrast_factor=1.5)  # Increase contrast
    #img = TT.functional.adjust_brightness(img, brightness_factor=0.8)  # Lighten background
    img = 255*img.squeeze(0)
    print("final mean and std of img: ", get_mean_and_std(img), "\n")
    return img



if __name__ == "__main__":
    data_dir = r"E:\camera_calibration\trial3"
    copy_dir = r"E:\camera_calibration\trial3_copies2"
    template_path = r"E:\camera_calibration\calibration_pattern.png"
    # template = IO.read_image(template_path, IO.ImageReadMode.RGB)
    # print("template mean and std: ", get_mean_and_std(template))
    os.makedirs(copy_dir, exist_ok=True)
    all_filenames = sorted([p for p in os.listdir(data_dir) if p.endswith((".jpg", ".png"))])
    for filename in all_filenames:
        file_path = os.path.join(data_dir, filename)
        file_basename = os.path.splitext(filename)[0]
        copy_path = os.path.join(copy_dir, f"{file_basename}.png")
        img = IO.read_image(file_path, IO.ImageReadMode.RGB) #.repeat([3,1,1])
        print("initial mean and std of img: ", get_mean_and_std(img))
        img = process_image(img)
        num_clusters = 4
        img = kmeans_compression(img, n_clusters=num_clusters)
        IO.write_png(img.byte(), copy_path, compression_level=2)



