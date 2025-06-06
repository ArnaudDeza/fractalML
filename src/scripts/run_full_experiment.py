import argparse
from src.training.grid_runner import run_grid_scan
from src.config import INPUT_DIM, HIDDEN_DIM

def main():
    """
    Main function to run the full grid scan experiment.
    
    Parses command-line arguments and calls the `run_grid_scan` function
    for each specified activation function.
    """
    parser = argparse.ArgumentParser(
        description="Run full grid scan for fractal trainability."
    )
    parser.add_argument(
        "--activations", nargs='+', default=["tanh"],
        choices=["tanh", "relu", "linear"],
        help="List of activation functions to test."
    )
    parser.add_argument(
        "--width", type=int, default=HIDDEN_DIM,
        help="Width of the hidden layer."
    )
    parser.add_argument(
        "--num_steps", type=int, default=500,
        help="Number of training steps."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for training."
    )
    parser.add_argument(
        "--dataset_mode", type=str, default="matching",
        choices=["matching", "single_example"],
        help="Dataset generation mode."
    )
    parser.add_argument(
        "--alpha0", type=float, default=None,
        help="Scaling factor for the first layer."
    )
    parser.add_argument(
        "--alpha1", type=float, default=None,
        help="Scaling factor for the second layer."
    )

    args = parser.parse_args()

    for activation in args.activations:
        print(f"--- Running experiment for activation: {activation} ---")
        run_grid_scan(
            activation=activation,
            width=args.width,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            dataset_mode=args.dataset_mode,
            alpha0=args.alpha0,
            alpha1=args.alpha1
        )
    print("--- All experiments complete ---")

if __name__ == "__main__":
    main()
