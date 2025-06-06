import numpy as np
import pandas as pd
import os
from PIL import Image
from skimage.filters import threshold_otsu
from porespy.metrics import boxcount
from typing import List
from tqdm import tqdm

def manual_box_counting(binary_image: np.ndarray, box_sizes: List[int]) -> tuple:
    """A manual implementation of box counting."""
    counts = []
    for size in box_sizes:
        if size > binary_image.shape[0] or size > binary_image.shape[1]:
            continue
        
        # Reduce the image by taking the max value in each block of 'size'
        reduced_image = binary_image.reshape(
            binary_image.shape[0] // size, size,
            binary_image.shape[1] // size, size
        ).max(axis=(1, 3))
        
        counts.append(np.sum(reduced_image))
        
    valid_sizes = [size for size in box_sizes if size <= min(binary_image.shape)]
    return counts, valid_sizes

def compute_fractal_dimension_on_image(image_path: str, box_sizes: List[int]) -> float:
    """
    Computes the fractal dimension of a single image using box-counting.

    Args:
        image_path: The path to the image file.
        box_sizes: A list of box sizes to use for the calculation.

    Returns:
        The estimated fractal dimension (slope of log-log plot).
    """
    try:
        img = Image.open(image_path).convert('L')
        arr = np.array(img)
        
        # Binarize the image. The boundary is between red (diverged) and blue (converged).
        # Otsu's method should find a good threshold between them.
        thresh = threshold_otsu(arr)
        binary = arr > thresh
        
        # Use PoreSpy's boxcount if available, otherwise use a manual method.
        try:
            results = boxcount(binary, bins=len(box_sizes), scaling_factor=None)
            counts, sizes = results.count, results.size
        except Exception:
            # Fallback to manual box-counting
            counts, sizes = manual_box_counting(binary, box_sizes)

        if len(counts) < 2:
            return np.nan

        # The fractal dimension is the slope of log(count) vs log(1/size)
        coeffs = np.polyfit(np.log(1 / np.array(sizes)), np.log(counts), 1)
        return coeffs[0]
        
    except Exception as e:
        print(f"Could not process {image_path}: {e}")
        return np.nan

def estimate_box_counting_dimension(image_dir: str, box_sizes: List[int]) -> pd.DataFrame:
    """
    Estimates the fractal dimension for a sequence of zoom images.

    Args:
        image_dir: Directory containing the zoom images.
        box_sizes: A list of box sizes to use for box-counting.

    Returns:
        A pandas DataFrame with the results.
    """
    data = []
    files = sorted([f for f in os.listdir(image_dir) if f.startswith("zoom_") and f.endswith(".png")])

    print(f"Estimating fractal dimension for {len(files)} images...")
    for fname in tqdm(files, desc="Fractal Dimension Estimation"):
        level = int(fname.split("_")[1].split(".")[0])
        path = os.path.join(image_dir, fname)
        fd = compute_fractal_dimension_on_image(path, box_sizes)
        data.append({'level': level, 'estimated_dimension': fd})
        
    df = pd.DataFrame(data).sort_values(by='level').reset_index(drop=True)
    return df
