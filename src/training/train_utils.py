import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
from src.models.simple_net import SimpleNet

def train_one_run(net: SimpleNet,
                    X: torch.Tensor,
                    y: torch.Tensor,
                    lr_w0: float,
                    lr_w1: float,
                    num_steps: int,
                    batch_size: int,
                    device: str = "cpu") -> Dict:
    """
    Train a SimpleNet for a single hyperparameter configuration.

    Trains the network for `num_steps` or until divergence is detected.
    W0 uses learning rate `lr_w0`; W1 uses `lr_w1`.

    Args:
        net: The SimpleNet model to train.
        X: Training data inputs.
        y: Training data targets.
        lr_w0: Learning rate for the first layer weights (W0).
        lr_w1: Learning rate for the second layer weights (W1).
        num_steps: The number of training steps to perform.
        batch_size: The batch size for training. If less than the dataset
                    size, mini-batch training is used.
        device: The device to run training on ('cpu' or 'cuda').

    Returns:
        A dictionary containing the training status and results:
        {
            'status': 'converged' or 'diverged',
            'final_loss': float or None,
            'loss_trajectory': List[float],
            'weight_norm_trajectory': List[float],
        }
    """
    net.to(device)
    net.train()

    # Set up optimizer with two parameter groups for different learning rates
    optimizer = optim.SGD([
        {'params': net.W0, 'lr': lr_w0},
        {'params': net.W1, 'lr': lr_w1}
    ])

    criterion = nn.MSELoss()
    dataset_size = X.size(0)
    loss_trajectory: List[float] = []
    weight_norm_trajectory: List[float] = []

    for step in range(num_steps):
        if batch_size < dataset_size:
            # Mini-batch training
            indices = torch.randperm(dataset_size, device=device)[:batch_size]
            X_batch, y_batch = X[indices], y[indices]
        else:
            # Full-batch training
            X_batch, y_batch = X, y

        optimizer.zero_grad()
        preds = net(X_batch)
        loss = criterion(preds, y_batch)
        loss_value = loss.item()
        loss_trajectory.append(loss_value)

        # Check for divergence
        norm_w0 = net.W0.norm().item()
        norm_w1 = net.W1.norm().item()
        weight_norm_trajectory.append((norm_w0 + norm_w1) / 2.0)

        if torch.isnan(loss) or torch.isinf(loss) or norm_w0 > 1e6 or norm_w1 > 1e6:
            return {
                'status': 'diverged',
                'final_loss': None,
                'loss_trajectory': loss_trajectory,
                'weight_norm_trajectory': weight_norm_trajectory
            }

        loss.backward()
        optimizer.step()

    final_loss = loss_trajectory[-1] if loss_trajectory else None
    return {
        'status': 'converged',
        'final_loss': final_loss,
        'loss_trajectory': loss_trajectory,
        'weight_norm_trajectory': weight_norm_trajectory
    }
