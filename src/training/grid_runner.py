import numpy as np
import torch
from tqdm import tqdm
import os
from typing import Tuple

from src.config import (LR0_MIN, LR0_MAX, LR1_MIN, LR1_MAX,
                        GRID_POINTS, OUTPUT_DIR, DEVICE, SEED)
from src.models.simple_net import SimpleNet
from src.utils.data_generator import generate_synthetic_data
from src.training.train_utils import train_one_run
from src.utils.plotting import plot_status_heatmap

def run_grid_scan(activation: str,
                    width: int,
                    num_steps: int,
                    batch_size: int,
                    dataset_mode: str,
                    alpha0: float,
                    alpha1: float):
    """
    Runs a grid scan over learning rates (lr_w0, lr_w1).

    For each point in the hyperparameter grid, it trains a `SimpleNet`
    and records whether the training converged or diverged. The results
    are saved as a NumPy array and a high-resolution heatmap image.

    Args:
        activation: The activation function to use ('tanh', 'relu', 'linear').
        width: The width of the hidden layer.
        num_steps: The number of training steps.
        batch_size: The batch size for training.
        dataset_mode: The mode for synthetic data generation.
        alpha0: The scaling factor for the first layer.
        alpha1: The scaling factor for the second layer.
    """
    # Set seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Prepare storage for results
    status_grid = np.zeros((GRID_POINTS, GRID_POINTS), dtype=np.uint8)

    # Create log-spaced hyperparameter grid
    lr0_vals = np.logspace(np.log10(LR0_MIN), np.log10(LR0_MAX), GRID_POINTS)
    lr1_vals = np.logspace(np.log10(LR1_MIN), np.log10(LR1_MAX), GRID_POINTS)

    # Ensure output directories exist
    img_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Iterate over the grid of learning rates
    for i, lr0 in enumerate(tqdm(lr0_vals, desc=f"Grid scan for '{activation}' (lr_w0)")):
        for j, lr1 in enumerate(lr1_vals):
            # Initialize network
            net = SimpleNet(input_dim=width,
                            hidden_dim=width,
                            activation=activation,
                            alpha0=alpha0,
                            alpha1=alpha1)
            net.initialize_weights(mean_offset=0.0, std=1.0)

            # Generate data
            X, y = generate_synthetic_data(net,
                                           mode=dataset_mode,
                                           device=DEVICE)
            # Train the network
            result = train_one_run(net, X, y,
                                   lr_w0=lr0,
                                   lr_w1=lr1,
                                   num_steps=num_steps,
                                   batch_size=batch_size,
                                   device=DEVICE)

            status_code = 0 if result['status'] == 'converged' else 1
            status_grid[i, j] = status_code

    # Save the status grid array
    npy_path = os.path.join(img_dir, f"grid_status_{activation}_n{width}.npy")
    np.save(npy_path, status_grid)
    print(f"Saved grid status to {npy_path}")

    # Plot and save the heatmap
    img_path = os.path.join(img_dir, f"grid_status_{activation}_n{width}.png")
    plot_status_heatmap(status_grid,
                        lr0_vals,
                        lr1_vals,
                        img_path)
    print(f"Saved grid heatmap to {img_path}")
