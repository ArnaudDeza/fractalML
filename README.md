# Fractal Trainability

This repository provides a Python implementation to reproduce the experiments from the paper **"Fractal Trainability: The Hyperparameter Fractal of Neural Network Training"** (arXiv:2402.06184v1).

The code trains simple one-hidden-layer neural networks across a grid of hyperparameters, identifies regions of convergence versus divergence, generates high-resolution "fractal" images of the trainability boundary, produces zoom animations into the fractal, and computes box-counting fractal dimension estimates.

## Repository Structure

```
fractalML/
├── README.md
├── requirements.txt
├── setup.py
├── data/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── models/
│   │   └── simple_net.py
│   ├── training/
│   │   ├── train_utils.py
│   │   └── grid_runner.py
│   ├── analysis/
│   │   ├── fractal_dimension.py
│   │   └── zoom_generator.py
│   ├── utils/
│   │   ├── data_generator.py
│   │   ├── plotting.py
│   │   └── diagnostics.py
│   └── scripts/
│       ├── run_full_experiment.py
│       ├── generate_zoom_sequence.py
│       └── estimate_fractal_dimension.py
└── outputs/
    ├── images/
    ├── zooms/
    └── fractal_metrics/
```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/fractal_trainability.git
    cd fractal_trainability
    ```

2.  Create a Python 3.10+ virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install the dependencies and the project in editable mode:
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

## Usage

### 1. Run the full grid experiment

This script runs the hyperparameter grid scan for the specified activation functions.

```bash
python src/scripts/run_full_experiment.py \
    --activations tanh relu linear \
    --width 16 \
    --num_steps 500 \
    --batch_size 16 \
    --dataset_mode matching
```

This will produce `.npy` status arrays and high-resolution PNG heatmaps in `outputs/images/`.

### 2. Generate a zoom sequence

This script generates a sequence of zoomed-in images around a chosen center in the hyperparameter space.

```bash
python src/scripts/generate_zoom_sequence.py \
    --status_npy outputs/images/grid_status_tanh_n16.npy \
    --center_lr0 -3.0 \
    --center_lr1 -2.0 \
    --num_levels 50 \
    --zoom_factor 2.0 \
    --image_size 4096 \
    --output_dir outputs/zooms/tanh_n16
```

The zoomed PNGs will be stored under `outputs/zooms/tanh_n16/`. This script can also create an MP4 video of the zoom sequence.

### 3. Stitch zoom images into a video

You can create a video from the generated zoom images using this command:

```bash
python -c "from src.analysis.zoom_generator import stitch_zoom_to_video; stitch_zoom_to_video(image_dir='outputs/zooms/tanh_n16', output_video='outputs/zooms/tanh_n16_zoom.mp4', fps=10)"
```

### 4. Estimate fractal dimension

This script estimates the fractal dimension of the boundary from the generated zoom images.

```bash
python src/scripts/estimate_fractal_dimension.py \
    --images_dir outputs/zooms/tanh_n16 \
    --output_csv outputs/fractal_metrics/tanh_n16_fractal_dims.csv
```

The resulting CSV file will contain per-level dimension estimates.

### 5. Inspect results

-   **High-resolution heatmaps**: `outputs/images/grid_status_{activation}_n{width}.png`
-   **Zoom video**: `outputs/zooms/{activation}_n{width}_zoom.mp4`
-   **Fractal dimension CSVs**: `outputs/fractal_metrics/{activation}_n{width}_fractal_dims.csv`
