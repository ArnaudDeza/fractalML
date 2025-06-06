import argparse
import numpy as np
import os
from src.analysis.zoom_generator import generate_zoom_images, stitch_zoom_to_video
from src.config import ZOOM_LEVELS, ZOOM_FACTOR, ZOOM_IMAGE_SIZE, LR0_MIN, LR0_MAX, LR1_MIN, LR1_MAX, GRID_POINTS

def main():
    """
    Main function to generate a zoom sequence.
    
    Parses command-line arguments and calls the `generate_zoom_images`
    and `stitch_zoom_to_video` functions.
    """
    parser = argparse.ArgumentParser(
        description="Generate a zoom sequence from a grid status file."
    )
    parser.add_argument(
        "--status_npy", type=str, required=True,
        help="Path to the .npy file containing the grid status."
    )
    parser.add_argument(
        "--center_lr0", type=float, required=True,
        help="Center of the zoom in log10(lr_w0) space."
    )
    parser.add_argument(
        "--center_lr1", type=float, required=True,
        help="Center of the zoom in log10(lr_w1) space."
    )
    parser.add_argument(
        "--num_levels", type=int, default=ZOOM_LEVELS,
        help="Number of zoom levels to generate."
    )
    parser.add_argument(
        "--zoom_factor", type=float, default=ZOOM_FACTOR,
        help="Zoom factor for each level."
    )
    parser.add_argument(
        "--image_size", type=int, default=ZOOM_IMAGE_SIZE,
        help="Size of the output zoom images."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the zoom images and video."
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="If set, do not generate a video from the zoom images."
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="FPS for the generated video."
    )

    args = parser.parse_args()

    # Load status grid
    if not os.path.exists(args.status_npy):
        print(f"Error: Status file not found at {args.status_npy}")
        return
    status_grid = np.load(args.status_npy)

    # Recreate the learning rate grids
    lr0_vals = np.logspace(np.log10(LR0_MIN), np.log10(LR0_MAX), GRID_POINTS)
    lr1_vals = np.logspace(np.log10(LR1_MIN), np.log10(LR1_MAX), GRID_POINTS)

    # Generate zoom images
    generate_zoom_images(
        status_grid=status_grid,
        lr0_vals=lr0_vals,
        lr1_vals=lr1_vals,
        center=(args.center_lr0, args.center_lr1),
        num_levels=args.num_levels,
        zoom_factor=args.zoom_factor,
        image_size=args.image_size,
        output_dir=args.output_dir
    )

    # Stitch images to video
    if not args.no_video:
        video_path = os.path.join(args.output_dir, "zoom_sequence.mp4")
        stitch_zoom_to_video(
            image_dir=args.output_dir,
            output_video=video_path,
            fps=args.fps
        )

if __name__ == "__main__":
    main()
