import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio
from tqdm import tqdm
from typing import Tuple, List

def generate_zoom_images(status_grid: np.ndarray,
                           lr0_vals: np.ndarray,
                           lr1_vals: np.ndarray,
                           center: Tuple[float, float],
                           num_levels: int,
                           zoom_factor: float,
                           image_size: int,
                           output_dir: str):
    """
    Generates a sequence of zoomed-in images from a status grid.

    Args:
        status_grid: The full 2D numpy array of convergence statuses.
        lr0_vals: The array of learning rates for the W0 layer.
        lr1_vals: The array of learning rates for the W1 layer.
        center: A tuple (log10(lr_w0), log10(lr_w1)) for the zoom center.
        num_levels: The number of zoom levels to generate.
        zoom_factor: The magnification factor for each level.
        image_size: The output size (width and height) of each zoom image.
        output_dir: The directory to save the generated PNG images.
    """
    os.makedirs(output_dir, exist_ok=True)
    grid_size = status_grid.shape[0]
    log_lr0_vals = np.log10(lr0_vals)
    log_lr1_vals = np.log10(lr1_vals)

    # Find the grid index closest to the desired center
    i_center = np.argmin(np.abs(log_lr0_vals - center[0]))
    j_center = np.argmin(np.abs(log_lr1_vals - center[1]))

    # Custom colormap
    blues = plt.cm.get_cmap('Blues_r', 256)(np.linspace(0.4, 0.8, 128))
    reds = plt.cm.get_cmap('Reds', 256)(np.linspace(0.4, 0.8, 128))
    cmap = ListedColormap(np.vstack((blues, reds)))

    print(f"Generating {num_levels} zoom images in '{output_dir}'...")
    for level in tqdm(range(num_levels)):
        # Calculate the window size for the current zoom level
        scale = zoom_factor ** level
        window_size = int(grid_size / scale)
        if window_size < 2:
            print("Zoom window is too small, stopping.")
            break

        half_w = window_size // 2
        i_min = max(i_center - half_w, 0)
        i_max = min(i_center + half_w, grid_size)
        j_min = max(j_center - half_w, 0)
        j_max = min(j_center + half_w, grid_size)

        subgrid = status_grid[i_min:i_max, j_min:j_max]

        # Upsample the subgrid to the target image size using nearest-neighbor
        if subgrid.shape[0] > 0 and subgrid.shape[1] > 0:
            zoomed = np.kron(subgrid,
                               np.ones((image_size // subgrid.shape[0],
                                        image_size // subgrid.shape[1])))
        else:
            continue # Skip if subgrid is empty

        # Plot and save the zoomed image
        fig, ax = plt.subplots(figsize=(12, 12), dpi=image_size/12)
        ax.imshow(zoomed.T, origin="lower", cmap=cmap, interpolation='none')
        ax.axis("off")
        save_path = os.path.join(output_dir, f"zoom_{level:03d}.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

def stitch_zoom_to_video(image_dir: str, output_video: str, fps: int = 10):
    """
    Stitches a sequence of images into a video file.

    Args:
        image_dir: Directory containing the zoom images (e.g., zoom_000.png).
        output_video: Path to save the output MP4 video.
        fps: Frames per second for the video.
    """
    files = sorted([os.path.join(image_dir, f)
                    for f in os.listdir(image_dir)
                    if f.startswith("zoom_") and f.endswith(".png")])

    if not files:
        print(f"No images found in {image_dir} to create a video.")
        return

    print(f"Stitching {len(files)} images into video: {output_video}")
    with imageio.get_writer(output_video, fps=fps) as writer:
        for fpath in tqdm(files, desc="Creating video"):
            img = imageio.imread(fpath)
            writer.append_data(img)
    print("Video creation complete.")
