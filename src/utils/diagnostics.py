import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

def check_grid_validity(status_grid: np.ndarray) -> bool:
    """
    Checks if a status grid contains both converged (0) and diverged (1) points.

    Args:
        status_grid: A 2D numpy array of training statuses.

    Returns:
        True if both 0 and 1 are present, False otherwise.
    """
    unique_statuses = np.unique(status_grid)
    return 0 in unique_statuses and 1 in unique_statuses

def plot_sample_trajectories(loss_trajectories: Dict[tuple, List[float]],
                               status_grid: np.ndarray,
                               num_samples: int = 5):
    """
    Plots a few sample loss trajectories for converged and diverged runs.

    Args:
        loss_trajectories: A dictionary mapping grid indices (i, j) to loss trajectories.
        status_grid: The 2D numpy array of final statuses.
        num_samples: The number of sample trajectories to plot for each status.
    """
    converged_indices = np.argwhere(status_grid == 0)
    diverged_indices = np.argwhere(status_grid == 1)

    plt.figure(figsize=(12, 5))

    # Plot converged samples
    ax1 = plt.subplot(1, 2, 1)
    if len(converged_indices) > 0:
        sample_indices = converged_indices[np.random.choice(len(converged_indices), num_samples)]
        for i, j in sample_indices:
            if (i,j) in loss_trajectories:
                ax1.plot(loss_trajectories[(i,j)], alpha=0.7)
    ax1.set_title(f"{num_samples} Converged Trajectories")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_yscale('log')

    # Plot diverged samples
    ax2 = plt.subplot(1, 2, 2)
    if len(diverged_indices) > 0:
        sample_indices = diverged_indices[np.random.choice(len(diverged_indices), num_samples)]
        for i, j in sample_indices:
            if (i,j) in loss_trajectories:
                ax2.plot(loss_trajectories[(i,j)], alpha=0.7)
    ax2.set_title(f"{num_samples} Diverged Trajectories")
    ax2.set_xlabel("Training Step")
    ax2.setyscale('log')

    plt.tight_layout()
    plt.show()
