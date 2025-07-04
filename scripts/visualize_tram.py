import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/..")

import argparse

from lib.pipeline import visualize_tram
from lib.pipeline.video_frame_iterator import VideoFrameIterator


def main(file: str, bin_size: int = -1, floor_scale: int = 3) -> None:
    # File and folders
    root = os.path.dirname(file)
    seq = os.path.basename(file).split(".")[0]

    seq_folder = f"results/{seq}"
    video_iterator = VideoFrameIterator(file)

    ##### Combine camera & human motion #####
    # Render video
    print("Visualize results ...")
    visualize_tram(
        video_iterator, seq_folder, floor_scale=floor_scale, bin_size=bin_size
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=str, default="./example_video.mov", help="input video"
    )
    parser.add_argument(
        "--bin_size",
        type=int,
        default=-1,
        help="rasterization bin_size; set to [64,128,...] to increase speed",
    )
    parser.add_argument("--floor_scale", type=int, default=3, help="size of the floor")
    args = parser.parse_args()

    main(args.video, args.bin_size, args.floor_scale)
