import argparse
import yaml
import numpy as np
import torch
import os
import itertools
from datetime import datetime
from tqdm import tqdm
import math

from src.models.mlp import build_mlp
from src.training.train_utils import train_one_run
from src.utils.plotting import plot_status_heatmap

# --- Data Generator ---
def generate_synthetic_data(model: torch.nn.Module, mode: str, device: str):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dataset_size = total_params if mode == "matching" else 1
    # The model expects input_dim, which is set to width
    input_dim = model.input_dim
    X = torch.randn(dataset_size, input_dim, device=device)
    y = torch.randn(dataset_size, device=device)
    return X, y

def get_axis_vals(axis_name: str, config: dict):
    """Creates a 1D array of values for a given hyperparameter axis."""
    grid_points = config['grid_points']
    key_prefix = axis_name.lower().replace("lr_", "lr") # e.g. LR_MID -> lrmid
    min_val = config.get(f'{key_prefix}_min', 0.0)
    max_val = config.get(f'{key_prefix}_max', 1.0)
    
    # Use log scale for learning rates and weight decay
    if "LR" in axis_name or "WEIGHT_DECAY" in axis_name:
        return np.logspace(np.log10(min_val), np.log10(max_val), grid_points)
    else: # Linear scale for momentum, betas
        return np.linspace(min_val, max_val, grid_points)

# --- Grid Runner ---
def run_grid_scan(config: dict):
    """Orchestrates the grid scan for a given experiment configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))

    scan_axes = config['scan_axes']
    axis_x_vals = get_axis_vals(scan_axes[0], config)
    axis_y_vals = get_axis_vals(scan_axes[1], config)

    status_grid = np.zeros((config['grid_points'], config['grid_points']), dtype=np.uint8)

    for i, x_val in enumerate(tqdm(axis_x_vals, desc=f"Scanning {config['name']}")):
        for j, y_val in enumerate(axis_y_vals):
            model = build_mlp(
                depth=config['depth'], width=config['width'],
                activation=config['activation'], residual=config['residual'],
                input_dim=config['width'], output_dim=1
            ).to(device)
            
            X, y = generate_synthetic_data(model, config['dataset_mode'], device)
            
            hparams = {
                'optimizer': config.get('optimizer', 'sgd'),
                'weight_decay': config.get('weight_decay', 0.0),
                'momentum': config.get('momentum', 0.0),
                'betas': tuple(config.get('betas', (0.9, 0.999))),
                'lr_w0': config.get('lr0', 1e-3),
                'lr_w1': config.get('lr1', 1e-3),
            }
            
            # Override with scanned values
            for axis_name, value in [(scan_axes[0], x_val), (scan_axes[1], y_val)]:
                key = axis_name.lower().replace("lr_", "lr")
                if key == 'lr0': hparams['lr_w0'] = value
                elif key == 'lr1': hparams['lr_w1'] = value
                elif key == 'beta2': hparams['betas'] = (hparams['betas'][0], value)
                else: hparams[key] = value

            # Derive non-scanned LRs
            hparams['lr_mid'] = math.sqrt(hparams['lr_w0'] * hparams['lr_w1'])
            
            final_hparams = {k: v for k, v in hparams.items() if not k.startswith('lr_w')}
            final_hparams['learning_rates'] = {
                'lr_w0': hparams['lr_w0'],
                'lr_w1': hparams['lr_w1'],
                'lr_mid': hparams['lr_mid'],
            }

            result = train_one_run(
                model=model, X=X, y=y, hparams=final_hparams,
                num_steps=config['num_steps'], batch_size=config['batch_size'], device=device
            )
            status_grid[i, j] = 1 if result['status'] == 'diverged' else 0

    # Save results
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join('outputs', 'depth', date_str)
    os.makedirs(output_dir, exist_ok=True)
    
    title = f"D={config['depth']}{'-Res' if config['residual'] else ''} | {config['optimizer'].upper()} | Act={config['activation']}"
    safe_name = config['name'].replace(' ', '_').replace('=', '')
    save_path = os.path.join(output_dir, f"{safe_name}.png")

    plot_status_heatmap(
        status_grid, lr_grid=(axis_x_vals, axis_y_vals), axis_labels=scan_axes,
        title=title, save_path=save_path, resolution=config['grid_points']
    )
    print(f"Saved heatmap to {save_path}")

# --- Main CLI ---
def main():
    parser = argparse.ArgumentParser(description="Fractal Trainability Scan")
    parser.add_argument('--config', type=str, default='configs/depth_experiments.yaml', help='Path to YAML config file.')
    parser.add_argument('--exp_name', type=str, help='Run a single experiment by name from the config file.')
    parser.add_argument('--scan_axes', nargs=2, help='Override scan axes, e.g., LR0 MOMENTUM.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)

    base_config = yaml_config['experiment_defaults']
    
    final_configs = []
    for exp_template in yaml_config['experiments']:
        # If a specific experiment is requested, only run that one
        if args.exp_name and exp_template['name'] != args.exp_name:
            continue
            
        # Combine base and experiment-specific configs
        cfg = {**base_config, **exp_template}
        
        # Determine scan axes
        if args.scan_axes:
            cfg['scan_axes'] = args.scan_axes
        elif 'scan_axes' not in cfg:
            cfg['scan_axes'] = ['LR0', 'LR1'] # Default scan
        
        # product over other dimensions like depth, residual
        keys = ['depth', 'residual']
        value_sets = [cfg.get(k, [None]) for k in keys]
        
        for values in itertools.product(*value_sets):
            run_cfg = cfg.copy()
            params = dict(zip(keys, values))
            run_cfg.update(params)

            if any(v is None for v in params.values()): continue

            res_str = f"_res{run_cfg['residual']}"
            depth_str = f"_d{run_cfg['depth']}"
            run_cfg['name'] = f"{cfg['name']}{depth_str}{res_str}"
            final_configs.append(run_cfg)

    if not final_configs:
        print("No experiments matched the provided arguments.")
        return
        
    for config in final_configs:
        run_grid_scan(config)


if __name__ == "__main__":
    main()
