import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Tuple

def plot_status_heatmap(status_grid: np.ndarray,
                          lr_grid: Tuple[np.ndarray, np.ndarray],
                          axis_labels: Tuple[str, str],
                          title: str,
                          save_path: str,
                          resolution: int = 1024):
    """
    Plots a heatmap of training status and saves it to a file.

    Args:
        status_grid: A 2D numpy array with status codes (0 or 1).
        lr_grid: A tuple containing the 1D arrays for x and y axes.
        axis_labels: A tuple of strings for the x and y axis labels.
        title: The title for the plot.
        save_path: The file path to save the PNG image to.
        resolution: The resolution of the output image.
    """
    plt.style.use('default')
    dpi = resolution / 10  # 10x10 inch figure
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)

    cmap = ListedColormap(['#4682B4', '#DC143C']) # SteelBlue and Crimson

    lr_x_vals, lr_y_vals = lr_grid
    im = ax.imshow(status_grid.T, origin='lower', cmap=cmap,
                   extent=(np.log10(lr_x_vals[0]), np.log10(lr_x_vals[-1]),
                           np.log10(lr_y_vals[0]), np.log10(lr_y_vals[-1])),
                   aspect='auto', interpolation='none')

    ax.set_xlabel(f"log10({axis_labels[0]})")
    ax.set_ylabel(f"log10({axis_labels[1]})")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Converged', 'Diverged'])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
