import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    """A simple one-hidden-layer network with configurable activation."""
    def __init__(self, input_dim: int, hidden_dim: int, activation: str,
                 alpha0: float = None, alpha1: float = None, use_bias: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation.lower()
        self.use_bias = use_bias

        # Mean-field scaling: if not provided, set defaults for tanh/relu
        if alpha0 is None or alpha1 is None:
            if self.activation in ("tanh", "relu"):
                # α0 = sqrt(2 / n_in), α1 = 1 / n_hid
                self.alpha0 = (2.0 / input_dim)**0.5
                self.alpha1 = 1.0 / hidden_dim
            else:  # linear
                self.alpha0 = 1.0
                self.alpha1 = 1.0
        else:
            self.alpha0 = alpha0
            self.alpha1 = alpha1

        # Define weight matrices (no biases by default)
        self.W0 = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.W1 = nn.Parameter(torch.randn(1, hidden_dim))

        # Activation module
        if self.activation == "tanh":
            self.act = torch.tanh
        elif self.activation == "relu":
            self.act = torch.relu
        elif self.activation in ("linear", "identity"):
            self.act = lambda x: x
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SimpleNet.
        x: (batch_size, input_dim)
        """
        h = self.alpha0 * (x @ self.W0.t())
        h = self.act(h)
        out = self.alpha1 * (h @ self.W1.t())
        return out.squeeze(-1)  # shape: (batch_size,)

    def initialize_weights(self, mean_offset: float = 0.0, std: float = 1.0):
        """Initializes weights from Normal(mean_offset, std)."""
        nn.init.normal_(self.W0, mean=mean_offset, std=std)
        nn.init.normal_(self.W1, mean=mean_offset, std=std)
