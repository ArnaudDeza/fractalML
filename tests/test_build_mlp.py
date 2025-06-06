import pytest
import torch
from src.models.mlp import build_mlp

@pytest.mark.parametrize("depth", [1, 2, 3, 5, 9])
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_build_mlp_output_shape(depth, residual, activation):
    """
    Tests if the build_mlp function produces a model with the correct output shape.
    """
    width = 64
    batch_size = 4
    model = build_mlp(
        depth=depth,
        width=width,
        activation=activation,
        residual=residual,
        input_dim=width,
        output_dim=1
    )
    
    # Create a dummy input tensor
    input_tensor = torch.randn(batch_size, width)
    
    # Get model output
    output = model(input_tensor)
    
    # Check if the output shape is correct (batch_size, output_dim)
    assert output.shape == (batch_size, 1), \
        f"Failed for depth={depth}, residual={residual}, activation={activation}"

@pytest.mark.parametrize("depth", [1, 3, 5])
def test_residual_connections_exist(depth):
    """
    Tests if residual connections are correctly added to the model.
    """
    if depth < 2:
        pytest.skip("Residual connections are not applicable for depth < 2.")

    model = build_mlp(depth=depth, width=32, activation='relu', residual=True)
    
    # Check for attributes that indicate a residual connection
    has_residual_layers = any(hasattr(mod, 'is_residual_block') for mod in model.layers)
    
    assert has_residual_layers, "Model should have residual blocks, but none were found."
