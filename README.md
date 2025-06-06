# Fractal Trainability (Advanced)

This repository provides a Python implementation to reproduce and extend the experiments from the paper **"Fractal Trainability: The Hyperparameter Fractal of Neural Network Training"** (arXiv:2402.06184v1).

This updated version supports creating MLPs of arbitrary depth, with optional residual connections, and provides a flexible system for running experiments defined in YAML configuration files.

## Installation

1.  Clone the repository and navigate into the directory.
2.  Create and activate a Python 3.10+ virtual environment.
3.  Install dependencies and the project in editable mode:
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

## Usage

The main entry point is `fractal_scan.py`, which reads experiment configurations from a YAML file and can be customized with command-line arguments.

### Configuration (`configs/depth_experiments.yaml`)

Experiments are defined in a YAML file. You can specify the optimizer, and set up scans over learning rates, momentum, weight decay, and Adam's betas.

```yaml
experiment_defaults:
  width: 128
  optimizer: "sgd"
  # ... other defaults

experiments:
  - name: "sgd_momentum_scan"
    activation: "tanh"
    depth: [5]
    optimizer: "sgd"
    scan_axes: ["LR0", "MOMENTUM"]
    lr0_min: 1.0e-4
    lr0_max: 1.0e-1
    momentum_min: 0.0
    momentum_max: 0.99
```

### Running Experiments

To run all experiments defined in the config:

```bash
python fractal_scan.py
```

To run a single, named experiment from the config:

```bash
python fractal_scan.py --exp_name sgd_momentum_scan
```

To override the scanned axes for an experiment:

```bash
python fractal_scan.py --exp_name adam_beta2_scan --scan_axes LR0 BETA2
```

### CLI Flags

-   `--config`: Path to the YAML configuration file.
-   `--exp_name <str>`: Run only the experiment with this name from the config file.
-   `--scan_axes <str> <str>`: Override the hyperparameters to scan on the X and Y axes.
    Supported axes: `LR0`, `LR1`, `LR_MID`, `MOMENTUM`, `WEIGHT_DECAY`, `BETA2`.

### Outputs

Results are saved in `outputs/depth/{date}/{experiment_name}.png`.
The plot titles are descriptive of the experiment configuration (e.g., optimizer, depth, activation, residual status).
