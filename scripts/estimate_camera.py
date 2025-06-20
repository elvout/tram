import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/..")

import argparse

import numpy as np
import torch
from pycocotools import mask as masktool

from lib.camera import align_cam_to_world, calibrate_intrinsics, run_metric_slam
from lib.pipeline import detect_segment_track, video2frames
from lib.pipeline.tools import VideoFrameIterator


def main(file: str, static_camera: bool = False, visualize_mask: bool = False) -> None:
    # File and folders
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
        save_vos=visualize_mask,
    )

    ##### Run Masked DROID-SLAM #####
    print("Masked Metric SLAM ...")
    masks = np.array([masktool.decode(m) for m in masks_])
    masks = torch.from_numpy(masks)

    if "view0" in file:
        cam_int = np.array([750.89197431, 750.89197431, 441.54551998, 229.49044537])
        wd_cam_R = np.array(
            [
                [-0.23182008, -0.28239837, 0.93086552],
                [-0.03448204, -0.95394666, -0.29798785],
                [0.97214734, -0.10117771, 0.21140631],
            ],
            dtype=np.float32,
        )
        wd_cam_T = np.array([-5.21771088, 2.73573661, -1.4177108], dtype=np.float32)
    elif "view1" in file:
        cam_int = np.array([741.66069556, 741.66069556, 536.35351899, 245.83074597])
        wd_cam_R = np.array(
            [
                [0.50106555, 0.3273517, -0.80110809],
                [-0.03575623, -0.91707699, -0.39710362],
                [-0.86467034, 0.22761955, -0.44781083],
            ],
            dtype=np.float32,
        )
        wd_cam_T = np.array([5.32422618, 2.8042409, 1.3339204], dtype=np.float32)
    # else:
    #     cam_int, is_static = calibrate_intrinsics(
    #         video_iterator.reset(), masks, is_static=static_camera
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


if __name__ == "__main__":
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

    main(args.video, args.static_camera, args.visualize_mask)
