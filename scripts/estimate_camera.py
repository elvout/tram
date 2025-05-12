import sys
import os

sys.path.insert(0, os.path.dirname(__file__) + "/..")

import torch
import argparse
import numpy as np
from glob import glob
from pycocotools import mask as masktool

from lib.pipeline import video2frames, detect_segment_track, visualize_tram
from lib.pipeline.tools import VideoFrameIterator
from lib.camera import run_metric_slam, calibrate_intrinsics, align_cam_to_world


parser = argparse.ArgumentParser()
parser.add_argument(
    "--video", type=str, default="./example_video.mov", help="input video"
)
parser.add_argument(
    "--static_camera", action="store_true", help="whether the camera is static"
)
parser.add_argument(
    "--visualize_mask", action="store_true", help="save deva vos for visualization"
)
args = parser.parse_args()

# File and folders
file = args.video
root = os.path.dirname(file)
seq = os.path.basename(file).split(".")[0]

seq_folder = f"results/{seq}"
# img_folder = f"{seq_folder}/images"
os.makedirs(seq_folder, exist_ok=True)
# os.makedirs(img_folder, exist_ok=True)

##### Extract Frames #####
# print("Extracting frames ...")
# nframes = video2frames(file, img_folder)
video_iterator = VideoFrameIterator(file)

##### Detection + SAM + DEVA-Track-Anything #####
print("Detect, Segment, and Track ...")
boxes_, masks_, tracks_ = detect_segment_track(
    video_iterator.reset(),
    seq_folder,
    thresh=0.25,
    min_size=100,
    save_vos=args.visualize_mask,
)

##### Run Masked DROID-SLAM #####
print("Masked Metric SLAM ...")
masks = np.array([masktool.decode(m) for m in masks_])
masks = torch.from_numpy(masks)


if "view0" in file:
    cam_int = np.array([753.647766, 753.647766, 460.39897955, 172.36292921])
    wd_cam_R = np.array(
        [
            [-0.25353999, -0.2134666, 0.94347734],
            [-0.02448514, -0.97361815, -0.22686599],
            [0.96701497, -0.08062077, 0.24162438],
        ],
        dtype=np.float32,
    )
    wd_cam_T = np.array([-5.28583237, 2.76902223, -1.43603922], dtype=np.float32)
elif "view1" in file:
    cam_int = np.array([690.65456922, 690.65456922, 519.1134486, 301.3375964])
    wd_cam_R = np.array(
        [
            [0.48540121, 0.38949092, -0.78274037],
            [-0.03991211, -0.88448029, -0.46486733],
            [-0.87338004, 0.25688799, -0.4137824],
        ],
        dtype=np.float32,
    )
    wd_cam_T = np.array([5.17125901, 2.71406991, 1.27483565], dtype=np.float32)
# else:
#     cam_int, is_static = calibrate_intrinsics(
#         video_iterator.reset(), masks, is_static=args.static_camera
#     )
#     cam_R, cam_T = run_metric_slam(
#         video_iterator.reset(), masks=masks, calib=cam_int, is_static=is_static
#     )
#     wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(video_iterator[0], cam_R, cam_T)

camera = {
    # "pred_cam_R": cam_R.numpy(),
    # "pred_cam_T": cam_T.numpy(),
    # "world_cam_R": wd_cam_R.numpy(),
    # "world_cam_T": wd_cam_T.numpy(),
    "world_cam_R": np.tile(wd_cam_R[np.newaxis], (len(video_iterator), 1, 1)),
    "world_cam_T": np.tile(wd_cam_T[np.newaxis], (len(video_iterator), 1)),
    "img_focal": cam_int[0],
    "img_center": cam_int[2:],
    # "spec_focal": spec_f,
}

np.save(f"{seq_folder}/camera.npy", camera)
np.save(f"{seq_folder}/boxes.npy", boxes_)
np.save(f"{seq_folder}/masks.npy", masks_)
np.save(f"{seq_folder}/tracks.npy", tracks_)
