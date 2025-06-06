import torch
import torch.nn as nn
import math

# Activation function mapping
ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "linear": nn.Identity,
}

class ResidualBlock(nn.Module):
    """A residual block with two linear layers."""
    def __init__(self, width: int, activation_fn: nn.Module):
        super().__init__()
        self.is_residual_block = True
        self.activation = activation_fn()
        self.layer1 = nn.Linear(width, width, bias=False)
        self.layer2 = nn.Linear(width, width, bias=False)
        # Scaling factor to maintain variance
        self.sqrt2_inv = 1 / math.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.activation(self.layer1(x))
        out = self.layer2(out)
        out = self.activation(out)
        # Add skip connection
        return (identity + out) * self.sqrt2_inv

class MLP(nn.Module):
    """A multi-layer perceptron with configurable depth and residual connections."""
    def __init__(self, layers: list, input_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def get_parameter_groups(self):
        """Separates parameters for different learning rates."""
        groups = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (nn.Linear, ResidualBlock)):
                groups.append({'params': layer.parameters(), 'layer_idx': i})
        return groups

def build_mlp(depth: int, width: int, activation: str, residual: bool,
              input_dim: int = None, output_dim: int = 1) -> MLP:
    """
    Builds a multi-layer perceptron with specified properties.

    Args:
        depth: The number of hidden layers.
        width: The number of neurons in each hidden layer.
        activation: The activation function to use ('relu', 'tanh', 'linear').
        residual: Whether to use residual connections.
        input_dim: The dimension of the input layer. Defaults to `width`.
        output_dim: The dimension of the output layer.

    Returns:
        An MLP model.
    """
    if input_dim is None:
        input_dim = width
        
    act_fn = ACTIVATION_FUNCTIONS.get(activation.lower())
    if not act_fn:
        raise ValueError(f"Unsupported activation: {activation}")

    layers = []
    current_dim = input_dim
    
    # Input layer
    layers.append(nn.Linear(current_dim, width, bias=False))
    layers.append(act_fn())
    current_dim = width
    
    # Hidden layers
    d = 1
    while d < depth:
        if residual and (d + 1 < depth):
            # Add a residual block (consumes 2 depth levels)
            layers.append(ResidualBlock(width, act_fn))
            d += 2
        else:
            # Add a standard linear layer
            layers.append(nn.Linear(width, width, bias=False))
            layers.append(act_fn())
            d += 1
            
    # Output layer
    layers.append(nn.Linear(width, output_dim, bias=False))
    
    # Initialize weights
    model = MLP(layers, input_dim)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    return model
