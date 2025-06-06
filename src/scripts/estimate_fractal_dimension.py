import argparse
import os
import pandas as pd
from src.analysis.fractal_dimension import estimate_box_counting_dimension
from src.config import BOX_SIZES, OUTPUT_DIR

def main():
    """
    Main function to estimate the fractal dimension from zoom images.
    
    Parses command-line arguments, calls the estimation function,
    and saves the results to a CSV file.
    """
    parser = argparse.ArgumentParser(
        description="Estimate fractal dimension from a directory of zoom images."
    )
    parser.add_argument(
        "--images_dir", type=str, required=True,
        help="Directory containing the zoom PNG images."
    )
    parser.add_argument(
        "--output_csv", type=str, required=True,
        help="Path to save the output CSV file with fractal dimension estimates."
    )

    args = parser.parse_args()

    # Ensure the images directory exists
    if not os.path.isdir(args.images_dir):
        print(f"Error: Images directory not found at {args.images_dir}")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Estimate fractal dimension
    df = estimate_box_counting_dimension(
        image_dir=args.images_dir,
        box_sizes=BOX_SIZES
    )

    # Save results to CSV
    df.to_csv(args.output_csv, index=False)
    print(f"Fractal dimension estimates saved to {args.output_csv}")

    # Print median dimension
    median_dim = df['estimated_dimension'].median()
    print(f"Median estimated fractal dimension: {median_dim:.4f}")

if __name__ == "__main__":
    main()
