import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_status_heatmap(status_grid: np.ndarray,
                          lr0_vals: np.ndarray,
                          lr1_vals: np.ndarray,
                          save_path: str,
                          figsize: tuple = (12, 12),
                          dpi: int = 341):
    """
    Plots a heatmap of training status and saves it to a file.

    Generates a high-resolution heatmap where converged runs (0) are blue
    and diverged runs (1) are red.

    Args:
        status_grid: A 2D numpy array with status codes (0 or 1).
        lr0_vals: 1D array of learning rates for the x-axis.
        lr1_vals: 1D array of learning rates for the y-axis.
        save_path: The file path to save the PNG image to.
        figsize: The figure size in inches.
        dpi: The resolution of the figure in dots per inch.
             (12 * 341.33 gives ~4096 pixels)
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Custom colormap: blue for converged (0), red for diverged (1)
    # Using a slice of standard colormaps for better aesthetics
    blues = plt.cm.get_cmap('Blues_r', 256)(np.linspace(0.4, 0.8, 128))
    reds = plt.cm.get_cmap('Reds', 256)(np.linspace(0.4, 0.8, 128))
    colors = np.vstack((blues, reds))
    cmap = ListedColormap(colors)

    # The grid is indexed (i, j) which corresponds to (lr0, lr1).
    # Matplotlib's imshow displays (row, col), so we need to transpose
    # the grid to have lr0 on the x-axis and lr1 on the y-axis.
    # origin='lower' puts (0,0) at the bottom-left.
    im = ax.imshow(status_grid.T, origin='lower', cmap=cmap,
                   extent=(np.log10(lr0_vals[0]), np.log10(lr0_vals[-1]),
                           np.log10(lr1_vals[0]), np.log10(lr1_vals[-1])),
                   aspect='auto', interpolation='none')

    ax.set_xlabel("log10(Learning Rate for W0)")
    ax.set_ylabel("log10(Learning Rate for W1)")
    ax.set_title("Training Stability (Converged vs. Diverged)")

    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Converged', 'Diverged'])
    cbar.set_label("Status")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
