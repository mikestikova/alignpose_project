#!/usr/bin/env python
#
# Released under the MIT License.
# Created on: 2024-09-25
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
"""Fix the drifting camera poses in the YCB-V test split.

In the original YCB-V test data, the camera poses (cam_R_w2c, cam_t_w2c in scene_camera.json) are not
consistent with the ground-truth object poses (cam_R_m2c, cam_t_m2c in scene_gt.json): although the objects
in each scene are static, their poses expressed in the world frame drift over time. Multi-view refinement
relies on the relative camera poses, so this drift has to be removed.

For every test scene the script:
  1. Computes the world pose of each object from the first frame (camera pose @ GT object pose).
  2. For every frame, takes the first object in its GT list (that is also in the first frame) and recomputes the camera pose so that this object
     stays at its initial world pose: T_world_cam = T_world_obj @ inv(T_cam_obj).
  3. Saves the corrected scene_camera.json (and an unchanged copy of scene_gt.json) to
     {out_dir}/{scene_id}/. The original dataset is left untouched.

The output scene_camera.json files are meant to replace the original ones in $BOP_PATH/ycbv/test
(see README).

Usage:
    python src/scripts/fix_ycbv_drift.py --bop-test-dir $BOP_PATH/ycbv/test --out-dir data/ycbv/fixed_drift
"""

import argparse
import os
from pathlib import Path
import json
import numpy as np


def pose_wc_from_scene_cam(scene_cam):
    pose_w2c = np.eye(4)
    pose_w2c[:3, :3] = np.array(scene_cam['cam_R_w2c']).reshape(3, 3)
    pose_w2c[:3, 3] = np.array(scene_cam['cam_t_w2c']) * 1e-3
    pose_wc = np.linalg.inv(pose_w2c)
    return pose_wc


def pose_co_from_obj(obj):
    pose_m2c = np.eye(4)
    pose_m2c[:3, :3] = np.array(obj['cam_R_m2c']).reshape(3, 3)
    pose_m2c[:3, 3] = np.array(obj['cam_t_m2c']) * 1e-3
    pose_cm = pose_m2c
    return pose_cm


parser = argparse.ArgumentParser()
parser.add_argument(
    "--bop-test-dir",
    type=Path,
    default=Path(os.environ.get("BOP_PATH", ".")) / "ycbv" / "test",
    help="Original YCB-V test folder.",
)
parser.add_argument(
    "--out-dir",
    type=Path,
    default=Path("./data/ycbv/fixed_drift"),
    help="Output folder for the fixed scene_camera.json and scene_gt.json files.",
)
args = parser.parse_args()

input_path = args.bop_test_dir
out_path_parent = args.out_dir

for d in sorted(input_path.iterdir()):
    if d.is_file():
        continue
    gt = json.load(open(d / "scene_gt.json"))
    cam = json.load(open(d / "scene_camera.json"))
    cam_out = json.load(open(d / "scene_camera.json"))
    out_dir = out_path_parent / d.name
    out_dir.mkdir(parents=True, exist_ok=True)

    timesteps = list(gt.keys())

    # Get initial object poses in world coordinates
    initial_obj_poses = {}
    scene_gt = gt[timesteps[0]]
    scene_cam = cam[timesteps[0]]
    pose_wc = pose_wc_from_scene_cam(scene_cam)
    for obj in scene_gt:
        pose_co = pose_co_from_obj(obj)
        pose_wo = pose_wc @ pose_co
        initial_obj_poses[obj["obj_id"]] = pose_wo

    # For each timestamp compute the new camera by assuming objects are static
    for t in timesteps:
        scene_gt = gt[t]
        for obj in scene_gt:
            if obj["obj_id"] not in initial_obj_poses:
                continue
            pose_co = pose_co_from_obj(obj)
            pose_wc_fixed = initial_obj_poses[obj["obj_id"]] @ np.linalg.inv(pose_co)
            pose_cw_fixed = np.linalg.inv(pose_wc_fixed)
            cam_out[t]["cam_R_w2c"] = pose_cw_fixed[:3, :3].flatten().tolist()
            cam_out[t]["cam_t_w2c"] = (pose_cw_fixed[:3, 3] * 1e3).tolist()
            break
    json.dump(cam_out, open(out_dir / "scene_camera.json", "w"), indent=2)
    json.dump(gt, open(out_dir / "scene_gt.json", "w"), indent=2)
    print(f"Saved fixed cameras to {out_dir}")
