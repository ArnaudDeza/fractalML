import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List
from src.models.mlp import MLP

def train_one_run(model: MLP,
                    X: torch.Tensor,
                    y: torch.Tensor,
                    hparams: Dict,
                    num_steps: int,
                    batch_size: int,
                    device: str = "cpu") -> Dict:
    """
    Train a model for a single hyperparameter configuration.

    Args:
        model: The MLP model to train.
        X: Training data inputs.
        y: Training data targets.
        hparams: Dictionary of hyperparameters including optimizer, LRs, etc.
        num_steps: The number of training steps.
        batch_size: Batch size for training.
        device: The device to run training on.

    Returns:
        A dictionary with training status and results.
    """
    model.to(device)
    model.train()

    # --- Optimizer Setup ---
    optimizer_name = hparams.get('optimizer', 'sgd').lower()
    learning_rates = hparams['learning_rates']
    
    # Get layer-specific learning rates
    lr_map = {
        0: learning_rates.get('lr_w0', 1e-3),
        1: learning_rates.get('lr_w1', 1e-3),
    }
    lr_mid = learning_rates.get('lr_mid', 1e-3)
    
    # Create parameter groups for the optimizer
    param_groups = []
    for group in model.get_parameter_groups():
        idx = group['layer_idx']
        lr = lr_map.get(idx, lr_mid)
        param_groups.append({'params': group['params'], 'lr': lr})

    # Configure and instantiate the optimizer
    optimizer_args = {
        'weight_decay': hparams.get('weight_decay', 0.0),
    }
    if optimizer_name == 'sgd':
        optimizer_args['momentum'] = hparams.get('momentum', 0.0)
        optimizer = optim.SGD(param_groups, **optimizer_args)
    elif optimizer_name == 'adam':
        optimizer_args['betas'] = hparams.get('betas', (0.9, 0.999))
        optimizer = optim.Adam(param_groups, **optimizer_args)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # --- Training Loop ---
    criterion = nn.MSELoss()
    loss_trajectory = []
    for step in range(num_steps):
        if batch_size < X.size(0):
            indices = torch.randperm(X.size(0), device=device)[:batch_size]
            X_batch, y_batch = X[indices], y[indices]
        else:
            X_batch, y_batch = X, y

        optimizer.zero_grad()
        preds = model(X_batch).squeeze()
        loss = criterion(preds, y_batch)
        loss_val = loss.item()
        loss_trajectory.append(loss_val)

        if np.isnan(loss_val) or np.isinf(loss_val) or loss_val > 1e6:
            return {'status': 'diverged', 'final_loss': None}

        loss.backward()
        optimizer.step()

    return {'status': 'converged', 'final_loss': loss_trajectory[-1]}
