import os
import torch

# Paths
# Correctly determine ROOT_DIR as the 'fractal_trainability' directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

# Default network settings
INPUT_DIM = 16  # Hidden width = 16 (so total params ~ 16^2 + 16 = 272)
HIDDEN_DIM = 16
ACTIVATIONS = ["tanh", "relu", "linear"]

# Default training settings
NUM_STEPS = 500
BATCH_SIZE = 16
SEED = 42

# Hyperparameter grid
LR0_MIN, LR0_MAX = 1e-6, 1e-1
LR1_MIN, LR1_MAX = 1e-6, 1e-1
GRID_POINTS = 1024  # final grid = 1024 x 1024

# Zoom settings
ZOOM_LEVELS = 50
ZOOM_FACTOR = 2.0
ZOOM_IMAGE_SIZE = 4096

# Fractal estimation settings
BOX_SIZES = [2**i for i in range(1, 12)]  # example box sizes for box-counting

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
