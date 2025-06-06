import torch
import numpy as np
from typing import Tuple
from src.models.simple_net import SimpleNet

def generate_synthetic_data(net: SimpleNet,
                            mode: str = "matching",
                            device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random (X, y) pairs drawn i.i.d. from N(0,1).

    Args:
        net: The network for which to generate data.
        mode: 'matching' or 'single_example'.
              - "matching": dataset_size = total_number_of_parameters(net).
              - "single_example": dataset_size = 1.
        device: The device to place the tensors on.

    Returns:
        A tuple of (X, y) tensors.
        X: (dataset_size, input_dim)
        y: (dataset_size,)
    """
    # Count total number of trainable parameters
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    if mode == "matching":
        dataset_size = total_params
    elif mode == "single_example":
        dataset_size = 1
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    input_dim = net.input_dim
    # Create random data on CPU, then move to device
    X_np = np.random.randn(dataset_size, input_dim).astype(np.float32)
    y_np = np.random.randn(dataset_size).astype(np.float32)

    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)

    return X, y
