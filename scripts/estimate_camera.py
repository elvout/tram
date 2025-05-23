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
    cam_int = np.array([755.72794387, 755.72794387, 456.87370896, 177.63166597])
    wd_cam_R = np.array(
        [
            [-0.24858075, -0.22008981,  0.94327519],
            [-0.02512474, -0.9720502 , -0.23342485],
            [ 0.96828527, -0.08172447,  0.23610327],
        ],
        dtype=np.float32,
    )
    wd_cam_T = np.array([-5.28583237, 2.76902223, -1.43603922], dtype=np.float32)
elif "view1" in file:
    cam_int = np.array([712.46871849, 712.46871849, 507.52318343, 288.23119081])
    wd_cam_R = np.array(
        [
            [ 0.47298317,  0.37591698, -0.79685215],
            [-0.04569512, -0.89272936, -0.44827029],
            [-0.87988572,  0.24843656, -0.40506839],
        ],
        dtype=np.float32,
    )
    wd_cam_T = np.array([5.18368024, 2.72539556, 1.28666089], dtype=np.float32)
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
